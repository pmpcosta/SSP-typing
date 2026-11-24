"""
 * Copyright 2024 University of Edinburgh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.

Binary segmentation mask -> discrete gel bands -> lanes.

``mask_to_bands`` mirrors the QuPath extension's ``ModelRunner.findSplitROIs``:
connected-component labelling (8-connectivity, threshold 0.5) turns the binary
mask into individual band objects. Bands are reported in a global
left-to-right, top-to-bottom centroid order. (The extension's ``BandSorter``
instead groups bands into lanes by bounding-box x-overlap and sorts by y
within each lane; since lane assignment happens downstream here, the global
order is sufficient and deterministic.)

Lane assignment is intentionally simple and deterministic (one band per lane is
the norm for SSP gels): either supply explicit lane boundaries
(``assign_lanes_by_edges``) or cluster band x-centroids by gap
(``cluster_lanes_by_gap``). This replaces the friend-chain column grouping in
``gelgenie.segmentation_postprocessing``, which targets dense multi-band
polyacrylamide gels rather than SSP layouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.measure import label, regionprops


@dataclass
class Band:
    """A single connected gel band."""

    RESERVED_KEYS = ("id", "area", "centroid_y", "centroid_x", "bbox", "lane", "extra")

    id: int
    area: int
    centroid_y: float
    centroid_x: float
    bbox: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    lane: Optional[int] = None
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        collisions = set(self.extra) & set(self.RESERVED_KEYS)
        if collisions:
            raise ValueError(
                f"extra keys collide with reserved Band fields: {sorted(collisions)}"
            )
        return {
            "id": self.id,
            "area": int(self.area),
            "centroid_y": float(self.centroid_y),
            "centroid_x": float(self.centroid_x),
            "bbox": [int(v) for v in self.bbox],
            "lane": self.lane,
            **self.extra,
        }


def mask_to_bands(
    mask: np.ndarray,
    min_area: int = 1,
    connectivity: int = 2,
) -> List[Band]:
    """Extract bands from a binary mask.

    Parameters
    ----------
    mask:
        ``(H, W)`` array; non-zero (>0.5) pixels are foreground.
    min_area:
        Discard components with fewer pixels than this.
    connectivity:
        ``skimage.measure.label`` connectivity. Default 2 = 8-connectivity,
        matching ``findSplitROIs``' ``conn8``.

    Returns
    -------
    List of :class:`Band`, sorted by x-centroid then y-centroid, with ``id``
    assigned in that sorted order (0-based).
    """
    binary = np.asarray(mask) > 0.5
    if not binary.any():
        return []

    labelled = label(binary.astype(np.uint8), connectivity=connectivity)
    bands: List[Band] = []
    for region in regionprops(labelled):
        if region.area < min_area:
            continue
        cy, cx = region.centroid  # (row, col) == (y, x)
        bands.append(
            Band(
                id=-1,  # assigned after sorting
                area=int(region.area),
                centroid_y=float(cy),
                centroid_x=float(cx),
                bbox=tuple(int(v) for v in region.bbox),
            )
        )

    bands.sort(key=lambda b: (b.centroid_x, b.centroid_y))
    for i, b in enumerate(bands):
        b.id = i
    return bands


def assign_lanes_by_edges(bands: Sequence[Band], lane_edges: Sequence[float]) -> None:
    """Assign each band to a lane using explicit vertical boundaries.

    ``lane_edges`` are interior x-boundaries; N edges define N+1 lanes indexed
    0..N. A band belongs to the lane its x-centroid falls in. Mutates ``bands``
    in place (sets ``band.lane``).
    """
    edges = np.asarray(list(lane_edges), dtype=float)
    for b in bands:
        b.lane = int(np.searchsorted(edges, b.centroid_x, side="right"))


def cluster_lanes_by_gap(xs: Sequence[float], gap_threshold: float) -> List[int]:
    """Cluster 1D x-centroids into lanes by splitting on gaps larger than
    ``gap_threshold``.

    Returns a lane label (0-based, in ascending x order) for each input value,
    in the original input order. Deterministic given the threshold.
    """
    xs = list(xs)
    if not xs:
        return []
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    labels = [0] * len(xs)
    current = 0
    prev_x = xs[order[0]]
    labels[order[0]] = current
    for idx in order[1:]:
        x = xs[idx]
        if x - prev_x > gap_threshold:
            current += 1
        labels[idx] = current
        prev_x = x
    return labels


def split_merged_bands(
    mask: np.ndarray,
    signal: np.ndarray,
    height_ratio: float = 1.6,
    min_extra: int = 6,
    min_peers: int = 5,
    min_contrast: float = 0.5,
    min_step: float = 0.2,
    min_dip: float = 0.1,
) -> np.ndarray:
    """Cut mask blobs that hold two touching bands, e.g. a product just
    below its control (Gel_01 lane 70: 180 bp under the 256 bp control),
    which the model segments as one blob. Returns a new mask.

    ``signal`` is the image with bands bright, in [0, 1]. A blob is cut when
    all of these hold:

    * it is at least ``height_ratio`` times (and ``min_extra`` px) taller
      than the median blob at its height across the gel (``min_peers`` or
      more such blobs): the bands of one row are about equally tall, while
      primer-dimer smears are tall in every lane and so are not outliers;
    * it is a strong band: its peak is at least ``min_contrast`` of the
      gel's strong bands (90th percentile), as faint diffuse smears have
      noisy profiles;
    * its profile, in its middle 40%, shows two bands: a row that dips
      ``min_dip`` (of the peak) below both sides, cut there; or else a step
      where the halves' levels differ by ``min_step`` (one band riding on
      the other's shoulder), cut at the steepest row. A uniformly
      overloaded band is neither, and stays whole.

    The cut removes that row of the blob, so the halves become two bands.
    """
    binary = np.asarray(mask) > 0.5
    out = binary.copy()
    if not binary.any():
        return out
    sig = np.asarray(signal, dtype=np.float32)
    lab = label(binary.astype(np.uint8), connectivity=2)
    regs = regionprops(lab, intensity_image=sig)
    bg = float(np.median(sig[~binary])) if (~binary).any() else 0.0
    strong = [r.intensity_max - bg for r in regs if r.area >= 50]
    if not strong:
        return out
    ref = float(np.percentile(strong, 90))
    heights = np.array([r.bbox[2] - r.bbox[0] for r in regs], dtype=float)
    cys = np.array([r.centroid[0] for r in regs])
    big = np.array([r.area >= 20 for r in regs])
    for k, r in enumerate(regs):
        t, l, b, rr = r.bbox
        h = b - t
        peers = big & (np.abs(cys - cys[k]) <= h / 2)
        peers[k] = False
        if peers.sum() < min_peers:
            continue
        typical = float(np.median(heights[peers]))
        if h < height_ratio * typical or h - typical < min_extra:
            continue
        peak = float(r.intensity_max) - bg
        if ref <= 0 or peak < min_contrast * ref:
            continue
        # vertical profile over the blob's middle columns
        w = rr - l
        c0, c1 = w // 4, w - w // 4
        inside = r.image[:, c0:c1]
        vals = sig[t:b, l + c0:l + c1]
        rows = inside.any(axis=1)
        prof = np.where(rows, (vals * inside).sum(axis=1) / np.maximum(inside.sum(axis=1), 1), bg)
        prof = gaussian_filter1d(prof, 1.0)
        lo, hi = int(round(0.3 * h)), int(round(0.7 * h))
        # two bands show as a dip between them (cut at its bottom) or as
        # one riding on the other's shoulder (cut where the level steps)
        cut = lo + int(np.argmin(prof[lo:hi + 1]))
        top, bottom = prof[:cut], prof[cut + 1:]
        if not len(top) or not len(bottom):
            continue
        if (min(top.max(), bottom.max()) - prof[cut]) / peak < min_dip:
            grad = np.abs(np.diff(prof))      # grad[i]: between rows i, i+1
            cut = lo + int(np.argmax(grad[lo:hi]))
            if abs(prof[:cut + 1].mean() - prof[cut + 1:].mean()) / peak < min_step:
                continue
        out[t + cut, l:rr] &= ~r.image[cut]
    return out
