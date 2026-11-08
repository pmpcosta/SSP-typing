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
mask into individual band objects. Bands are reported left-to-right, top-to-
bottom, matching ``BandSorter`` / ``CentroidCompareX`` ordering.

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
from skimage.measure import label, regionprops


@dataclass
class Band:
    """A single connected gel band."""

    id: int
    area: int
    centroid_y: float
    centroid_x: float
    bbox: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    lane: Optional[int] = None
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
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
