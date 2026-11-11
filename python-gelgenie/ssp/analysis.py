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

Gel analysis between segmentation and typing: lanes, controls, sizes.

Pipeline (mirrors the validated G_26160 analysis recipe):

1. :func:`cluster_lanes` groups bands into lanes by x-gaps, inserting empty
   lanes for wholly missing ones (gap ~2x median pitch) so lane index i always
   maps to tray mix i+1 (left-to-right numbering).
2. Panel detection: stitched gel-doc composites have physical seams (gaps
   > 1.6x median pitch); migration drift ("smile") resets per panel.
3. Per (panel, control-bp) group, the control row is found as the densest
   50-px y-window across the group's bands, then each lane picks its closest
   band within tolerance; a quadratic smile fit (or median for small groups)
   refines the picks over two rounds.
4. Fragment sizes are estimated per lane relative to its own control:
   ``bp_est = bp_ctrl * exp((y_ctrl - y_band) / b)`` where b (px per ln(bp))
   is auto-calibrated when one panel hosts two different control sizes
   (e.g. the 256bp B lanes and 800bp C lanes sharing panel 3), else defaults
   to :data:`DEFAULT_BP_SLOPE`.
5. Bands are classified: control / product (matches a sheet size within
   tolerance) / dimer (below the smallest credible product) / hmw artifact /
   unexplained.

Known limitation: on stitched composites, a panel seam and a missing lane are
distinguished only by gap ratio (~2x = missing lane, ~>2.4x = seam). A wholly
empty lane adjacent to a seam could confuse numbering; the lane count is
validated against the tray, and mismatches raise rather than silently shift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ssp.bands import Band
from ssp.trays import Tray

DEFAULT_BP_SLOPE = 81.0   # px per ln(bp); calibrated on FLC gel panel 3
DIMER_MAX_BP = 160.0      # below this: primer-dimer / dye front zone
HMW_MIN_BP = 1300.0       # above this: high-molecular-weight artifact
_MISSING_LANE_RATIO = (1.6, 2.4)  # gap/median-pitch signature of one empty lane
_PANEL_SEAM_RATIO = 1.6           # gap above this splits panels


def as_band(b: Union[Band, Dict]) -> Band:
    """Accept a Band or its to_dict() form."""
    if isinstance(b, Band):
        return b
    known = ("id", "area", "centroid_y", "centroid_x", "bbox", "lane")
    kw = {k: b[k] for k in known if k in b}
    kw["extra"] = {k: v for k, v in b.items() if k not in known}
    return Band(**kw)


def estimate_bp(y_band: float, y_ctrl: float, ctrl_bp: float, slope_b: float) -> float:
    """Fragment-size estimate from migration relative to the lane's control."""
    return ctrl_bp * math.exp((y_ctrl - y_band) / slope_b)


def cluster_lanes(
    bands: Sequence[Union[Band, Dict]],
    expected_count: Optional[int] = None,
    gap: float = 20.0,
) -> List[List[Band]]:
    """Group bands into lane clusters by centroid-x gaps.

    With ``expected_count``, inserts empty lanes at gaps whose size matches a
    single missing lane (~2x median pitch, window [1.6, 2.4)x). Raises
    ``ValueError`` when the lane count cannot be reconciled.
    """
    bl = sorted((as_band(b) for b in bands), key=lambda b: b.centroid_x)
    if not bl:
        if expected_count:
            return [[] for _ in range(expected_count)]
        return []
    clusters: List[List[Band]] = [[bl[0]]]
    for b in bl[1:]:
        if b.centroid_x - clusters[-1][-1].centroid_x > gap:
            clusters.append([b])
        else:
            clusters[-1].append(b)
    if expected_count is None:
        return clusters

    if len(clusters) == expected_count:
        return clusters
    if len(clusters) > expected_count:
        raise ValueError(
            f"found {len(clusters)} lane clusters, expected {expected_count}; "
            "consider raising --min-area or checking lane splitting"
        )
    # too few clusters: try inserting empty lanes at missing-lane-signature gaps
    centers = [float(np.mean([b.centroid_x for b in c])) for c in clusters]
    spacings = np.diff(centers)
    median_pitch = float(np.median(spacings))
    shortfall = expected_count - len(clusters)
    candidates = []
    for i, s in enumerate(spacings):
        ratio = s / median_pitch
        if _MISSING_LANE_RATIO[0] <= ratio < _MISSING_LANE_RATIO[1]:
            n_missing = int(round(ratio)) - 1
            candidates.append((abs(ratio - 2.0), i, n_missing))
    candidates.sort()
    inserted: Dict[int, int] = {}
    remaining = shortfall
    for _, i, n_missing in candidates:
        take = min(n_missing, remaining)
        if take > 0:
            inserted[i + 1] = inserted.get(i + 1, 0) + take
            remaining -= take
        if remaining == 0:
            break
    if remaining > 0:
        raise ValueError(
            f"found {len(clusters)} lanes, expected {expected_count}, and only "
            f"{shortfall - remaining} could be re-inserted from gap geometry"
        )
    out: List[List[Band]] = []
    for i, c in enumerate(clusters):
        if i in inserted:
            out.extend([] for _ in range(inserted[i]))
        out.append(c)
    return out


@dataclass
class BandCall:
    """One detected band with its size estimate and classification."""

    band: Band
    bp_est: Optional[float]
    kind: str  # control | product | dimer | hmw | unexplained | unclassified
    matched_bp: Optional[int] = None

    def to_dict(self) -> Dict:
        d = self.band.to_dict()
        d.update({
            "bp_est": round(self.bp_est, 1) if self.bp_est is not None else None,
            "kind": self.kind,
            "matched_bp": self.matched_bp,
        })
        return d


@dataclass
class LaneCall:
    """Per-lane analysis result."""

    mix: int
    center_x: float
    control_bp: int
    valid: bool                 # control band identified
    positive: bool              # size-matched product band present
    control_y: Optional[float]
    calls: List[BandCall] = field(default_factory=list)

    @property
    def products(self) -> List[BandCall]:
        return [c for c in self.calls if c.kind == "product"]

    def to_dict(self) -> Dict:
        return {
            "mix": self.mix, "center_x": round(self.center_x, 1),
            "control_bp": self.control_bp, "valid": self.valid,
            "positive": self.positive, "control_y": self.control_y,
            "bands": [c.to_dict() for c in self.calls],
        }


@dataclass
class GelAnalysis:
    lanes: List[LaneCall]
    slope_b: float
    panels: List[Tuple[int, int]]   # inclusive mix ranges
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "slope_b": round(self.slope_b, 2),
            "panels": [list(p) for p in self.panels],
            "warnings": self.warnings,
            "n_lanes": len(self.lanes),
            "invalid_lanes": [l.mix for l in self.lanes if not l.valid],
            "positive_lanes": [l.mix for l in self.lanes if l.positive],
            "lanes": [l.to_dict() for l in self.lanes],
        }


def _densest_window_row(ys: np.ndarray, lanes: np.ndarray, width: float) -> Optional[float]:
    """Median y of the densest window; ties prefer more distinct lanes."""
    if len(ys) == 0:
        return None
    order = np.argsort(ys)
    ys_s, lanes_s = ys[order], lanes[order]
    best = (0, 0, None)
    for i in range(len(ys_s)):
        j = np.searchsorted(ys_s, ys_s[i] + width, side="left")
        count = j - i
        distinct = len(set(lanes_s[i:j].tolist()))
        if (count, distinct) > (best[0], best[1]):
            best = (count, distinct, float(np.median(ys_s[i:j])))
    return best[2]


def _panel_ranges(centers: Sequence[float]) -> List[Tuple[int, int]]:
    """Inclusive index ranges split at seam gaps (> 1.6x median pitch)."""
    n = len(centers)
    if n < 2:
        return [(0, n - 1)] if n else []
    spacings = np.diff(np.asarray(centers, dtype=float))
    median_pitch = float(np.median(spacings))
    bounds = [0]
    for i, s in enumerate(spacings):
        if s > _PANEL_SEAM_RATIO * median_pitch:
            bounds.append(i + 1)
    bounds.append(n)
    return [(bounds[k], bounds[k + 1] - 1) for k in range(len(bounds) - 1)]


def _fit_controls(
    lane_bands: List[List[Band]],
    lane_mixes: List[int],
    ctrl_bps: List[int],
    centers: List[float],
) -> Tuple[Dict[int, Optional[Band]], float, List[Tuple[int, int]]]:
    """Pick control bands per lane; returns (picks, slope_b, panels)."""
    panels = _panel_ranges(centers)
    picks: Dict[int, Optional[Band]] = {}

    groups: List[Tuple[int, List[int]]] = []  # (control_bp, lane indices)
    for p0, p1 in panels:
        idxs = list(range(p0, p1 + 1))
        by_bp: Dict[int, List[int]] = {}
        for i in idxs:
            by_bp.setdefault(ctrl_bps[i], []).append(i)
        for bp, group in sorted(by_bp.items()):
            groups.append((bp, group))

    for bp, group in groups:
        # rough control row: densest y-window over the group's bands
        ys, ls = [], []
        for i in group:
            for b in lane_bands[i]:
                ys.append(b.centroid_y)
                ls.append(i)
        row = _densest_window_row(np.asarray(ys), np.asarray(ls), 50.0)
        if row is None:
            for i in group:
                picks[i] = None
            continue
        pred_fn = lambda m, v=row: v  # noqa: E731
        for _ in range(2):  # pick -> refit -> re-pick
            for i in group:
                near = [b for b in lane_bands[i] if abs(b.centroid_y - pred_fn(lane_mixes[i])) <= 14.0]
                picks[i] = max(near, key=lambda b: b.area) if near else None
            got = [(lane_mixes[i], picks[i].centroid_y) for i in group if picks[i]]
            if len(got) >= 8:
                coef = np.polyfit([g[0] for g in got], [g[1] for g in got], 2)
                pred_fn = lambda m, c=coef: float(np.polyval(c, m))  # noqa: E731
            elif got:
                med = float(np.median([g[1] for g in got]))
                pred_fn = lambda m, v=med: v  # noqa: E731

    # slope calibration: panels hosting >=2 control sizes with >=3 picks each
    slope_estimates = []
    for p0, p1 in panels:
        idxs = [i for i in range(p0, p1 + 1) if picks.get(i) is not None]
        by_bp: Dict[int, List[float]] = {}
        for i in idxs:
            by_bp.setdefault(ctrl_bps[i], []).append(picks[i].centroid_y)
        usable = sorted((bp, float(np.median(ys))) for bp, ys in by_bp.items() if len(ys) >= 3)
        for (bp_lo, y_lo), (bp_hi, y_hi) in zip(usable, usable[1:]):
            if bp_hi > bp_lo:
                slope_estimates.append((y_lo - y_hi) / (math.log(bp_hi) - math.log(bp_lo)))
    slope_b = float(np.mean(slope_estimates)) if slope_estimates else DEFAULT_BP_SLOPE
    return picks, slope_b, panels


def analyse_gel(
    bands: Sequence[Union[Band, Dict]],
    tray: Tray,
    lane_gap: float = 20.0,
    slope_b: Optional[float] = None,
    match_tol: float = 0.25,
) -> GelAnalysis:
    """Full post-segmentation analysis against a tray definition.

    Lane index i (left-to-right) is mapped to tray mix i+1; the caller-provided
    ``slope_b`` overrides auto-calibration when given.
    """
    mixes = [l.mix for l in sorted(tray.lanes, key=lambda l: l.mix)]
    if mixes != list(range(1, len(mixes) + 1)):
        raise ValueError("tray mixes must be consecutive 1..N for lane alignment")
    ctrl_bps_all = {l.mix: l.control_bp for l in tray.lanes}
    products_all = {l.mix: l.product_bp for l in tray.lanes}

    lane_bands = cluster_lanes(bands, expected_count=len(mixes), gap=lane_gap)
    centers: List[float] = []
    for i, lb in enumerate(lane_bands):
        if lb:
            centers.append(float(np.mean([b.centroid_x for b in lb])))
        else:  # empty lane: midpoint of neighbours (interpolated if run of empties)
            prev = next((centers[j] for j in range(i - 1, -1, -1) if lane_bands[j]), None)
            nxt = next((float(np.mean([b.centroid_x for b in lane_bands[j]]))
                        for j in range(i + 1, len(lane_bands)) if lane_bands[j]), None)
            centers.append(np.mean([v for v in (prev, nxt) if v is not None])
                           if (prev is not None or nxt is not None) else 0.0)

    picks, slope_auto, panels = _fit_controls(
        lane_bands, mixes, [ctrl_bps_all[m] for m in mixes], centers)
    slope = float(slope_b) if slope_b is not None else slope_auto
    panels_mix = [(mixes[p0], mixes[p1]) for p0, p1 in panels]

    lanes_out: List[LaneCall] = []
    warnings: List[str] = []
    if slope_b is None and not panels:
        warnings.append("no slope calibration possible; using default")
    for i, mix in enumerate(mixes):
        ctrl = picks.get(i)
        ctrl_bp = ctrl_bps_all[mix]
        calls: List[BandCall] = []
        positive = False
        for b in sorted(lane_bands[i], key=lambda b: b.centroid_y):
            if ctrl is not None and b is ctrl:
                calls.append(BandCall(b, float(ctrl_bp), "control", int(ctrl_bp)))
                continue
            if ctrl is None:
                calls.append(BandCall(b, None, "unclassified"))
                continue
            bp_est = estimate_bp(b.centroid_y, ctrl.centroid_y, ctrl_bp, slope)
            sizes = products_all[mix]
            hit = min(sizes, key=lambda s: abs(bp_est - s) / s) if sizes else None
            if hit is not None and abs(bp_est - hit) / hit <= match_tol:
                calls.append(BandCall(b, bp_est, "product", hit))
                positive = True
            elif bp_est < DIMER_MAX_BP:
                calls.append(BandCall(b, bp_est, "dimer"))
            elif bp_est > HMW_MIN_BP:
                calls.append(BandCall(b, bp_est, "hmw"))
            else:
                calls.append(BandCall(b, bp_est, "unexplained"))
        lanes_out.append(LaneCall(
            mix=mix, center_x=centers[i], control_bp=ctrl_bp,
            valid=ctrl is not None, positive=positive,
            control_y=ctrl.centroid_y if ctrl else None, calls=calls,
        ))

    invalid = [l.mix for l in lanes_out if not l.valid]
    if len(invalid) > 0.25 * len(lanes_out):
        warnings.append(f"more than 25% of lanes lack a control band: {invalid}")
    return GelAnalysis(lanes=lanes_out, slope_b=slope, panels=panels_mix,
                       warnings=warnings)
