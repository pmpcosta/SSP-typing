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
   refines the picks over two rounds. The primer-dimer front runs ahead of
   everything, so when it nearly ties with a row just above it, that row is
   the control (the front is in every lane, controls can be missing).
4. Fragment sizes are estimated per lane relative to its own control:
   ``bp_est = bp_ctrl * exp((y_ctrl - y_band) / b)`` where b (px per ln(bp))
   is auto-calibrated when one panel hosts two different control sizes
   (e.g. the 256bp B lanes and 800bp C lanes sharing panel 3), else defaults
   to :data:`DEFAULT_BP_SLOPE`.
5. Bands are classified: control / product (matches a sheet size within
   tolerance) / dimer (below the smallest credible product) / hmw artifact /
   unexplained. A listed product expected nearer the panel's dimer front than
   the control is matched by position instead (nearest of control, product,
   front), since sizes run large there and the dimer itself would match.

Known limitation: on stitched composites, a panel seam and a missing lane are
distinguished only by gap ratio (~2x = missing lane, ~>2.4x = seam). A wholly
empty lane adjacent to a seam could confuse numbering; the lane count is
validated against the tray, and mismatches raise rather than silently shift.

When the lane geometry is known (a :class:`LaneMap`, built from the manifest
that :mod:`ssp.layout` writes when it joins a raw gel photo), steps 1-2 are
replaced: bands are assigned to their lane slot by x position and each joined
strip is its own panel, so neither empty lanes nor seams need guessing.
"""

from __future__ import annotations

import itertools
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
    if len(clusters) < 2:
        raise ValueError(
            f"found a single lane cluster but expected {expected_count}; "
            "cannot infer lane pitch (segmentation may have merged lanes)")
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
class LaneMap:
    """Known lane geometry: one x-slot per mix, plus panel boundaries."""

    slots: List[Tuple[float, float]]   # (x0, x1) per lane, in mix order
    panels: List[Tuple[int, int]]      # inclusive 0-based lane-index ranges
    # strips start at y=0 just above their wells; bands above this y are in
    # the well zone (well marks, loading smudges) and never taken as controls
    well_zone: float = 0.0

    @property
    def centers(self) -> List[float]:
        return [(a + b) / 2.0 for a, b in self.slots]


def assign_to_slots(
    bands: Sequence[Union[Band, Dict]],
    slots: Sequence[Tuple[float, float]],
) -> Tuple[List[List[Band]], List[Band]]:
    """Put each band in the slot containing its x-centroid; a band just
    outside a slot (within half a slot width) goes to the nearest one. Returns
    ``(lane_bands, outside)`` where ``outside`` are bands in no slot's reach
    (e.g. in the gap between joined strips)."""
    lane_bands: List[List[Band]] = [[] for _ in slots]
    outside: List[Band] = []
    for b in (as_band(b) for b in bands):
        x = b.centroid_x
        best, best_d = None, math.inf
        for i, (x0, x1) in enumerate(slots):
            if x0 <= x < x1:  # half-open, like assign_lanes_by_edges
                best, best_d = i, 0.0
                break
            d = x0 - x if x < x0 else x - x1
            if d < best_d:
                best, best_d = i, d
        if best is not None and best_d <= 0.5 * (slots[best][1] - slots[best][0]):
            lane_bands[best].append(b)
        else:
            outside.append(b)
    return lane_bands, outside


@dataclass
class BandCall:
    """One detected band with its size estimate and classification."""

    band: Band
    bp_est: Optional[float]
    kind: str  # control | product | dimer | hmw | unexplained | unclassified
    matched_bp: Optional[int] = None
    manual: bool = False  # kind set by the user, not by the size match

    def to_dict(self) -> Dict:
        d = self.band.to_dict()
        d.update({
            "bp_est": round(self.bp_est, 1) if self.bp_est is not None else None,
            "kind": self.kind,
            "matched_bp": self.matched_bp,
            "manual": self.manual,
        })
        return d


# kinds a user may assign to a band (``band_kinds`` in :func:`analyse_gel`)
MANUAL_KINDS = ("control", "product", "dimer", "hmw", "unexplained")


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


def _densest_window(ys_s: np.ndarray, lanes_s: np.ndarray, width: float,
                    min_lanes: float) -> Optional[Tuple[int, int, int, int]]:
    """(count, distinct lanes, i, j) of the y-sorted bands' densest window
    ``ys_s[i:j]`` (ties prefer more distinct lanes, then the higher window),
    or None."""
    best = None
    for i in range(len(ys_s)):
        j = int(np.searchsorted(ys_s, ys_s[i] + width, side="left"))
        count, distinct = j - i, len(set(lanes_s[i:j].tolist()))
        if distinct >= min_lanes and (best is None or (count, distinct) > best[:2]):
            best = (count, distinct, i, j)
    return best


def _window_rows(ys: np.ndarray, lanes: np.ndarray, width: float,
                 min_lanes: float = 0, k: int = 3,
                 heights: Optional[np.ndarray] = None) -> List[Tuple[float, int, float]]:
    """Up to ``k`` candidate rows, best first, as (y, distinct lanes, median
    band height): the median y of the densest y-window (ties prefer more
    distinct lanes), then the densest window of the bands left, and so on.
    Only windows with bands from at least ``min_lanes`` lanes count.

    A window holding ~2 bands per lane spans two rows, and its median falls
    between them, where no lane has a band (Gel_06 lanes 49-64: controls and
    the dimer front 43 px apart). Such a window is split into its densest
    half-width rows, and the candidates are then ranked by their own band
    counts, the higher row first on a tie."""
    ys, lanes = np.asarray(ys, dtype=float), np.asarray(lanes)
    hs = np.zeros(len(ys)) if heights is None else np.asarray(heights, dtype=float)
    found: List[Tuple[int, int, float, float]] = []  # (count, distinct, y, height)
    split = False

    def row(ys_s, hs_s, count, distinct, i, j):
        return (count, distinct, float(np.median(ys_s[i:j])), float(np.median(hs_s[i:j])))

    while len(ys) and len(found) < k:
        order = np.argsort(ys)
        ys_s, lanes_s, hs_s = ys[order], lanes[order], hs[order]
        best = _densest_window(ys_s, lanes_s, width, min_lanes)
        if best is None:
            break
        count, distinct, i, j = best
        if count >= 1.5 * distinct:
            split = True
            ws, wl, wh = ys_s[i:j], lanes_s[i:j], hs_s[i:j]
            while len(ws):
                sub = _densest_window(ws, wl, width / 2, min_lanes)
                if sub is None:
                    break
                found.append(row(ws, wh, *sub))
                keep = np.ones(len(ws), dtype=bool)
                keep[sub[2]:sub[3]] = False
                ws, wl, wh = ws[keep], wl[keep], wh[keep]
        else:
            found.append(row(ys_s, hs_s, count, distinct, i, j))
        keep = (ys < ys_s[i]) | (ys > ys_s[j - 1])
        ys, lanes, hs = ys[keep], lanes[keep], hs[keep]
    if split:
        found.sort(key=lambda r: (-r[0], -r[1], r[2]))
    return [(y, distinct, h) for _, distinct, y, h in found[:k]]


def _dimer_front_last(rows: List[Tuple[float, int, float]], ctrl_bp: int,
                      tie: float = 0.8, diffuse: float = 1.25) -> List[float]:
    """Candidate row ys, best first, with the primer-dimer front moved behind
    the control row it outvotes. Dimers run ahead of every product and smear
    into taller bands, so when the best row is ``diffuse`` (its bands at
    least that much taller than those of the runner-up), and the runner-up
    sits in nearly as many lanes (``tie``) just above it, close enough that
    the best row would size as a dimer from there, the runner-up is the
    control row and the best row is the dimer front (Gel_04 lanes 33-64: the
    dimer row has 18 px bands in 31 lanes, the 256 bp control row 12 px bands
    in 29). A row of sharp bands below the controls is a product row and
    keeps its rank."""
    ys = [r[0] for r in rows]
    if len(rows) < 2:
        return ys
    (y0, n0, h0), (y1, n1, h1) = rows[0], rows[1]
    if (y1 < y0 and n1 >= tie * n0 and h0 >= diffuse * h1
            and estimate_bp(y0, y1, ctrl_bp, DEFAULT_BP_SLOPE) < DIMER_MAX_BP):
        ys[0], ys[1] = ys[1], ys[0]
    return ys


def _ordered_rows(cands: List[Tuple[int, List[float]]]) -> List[Optional[float]]:
    """One row per control size of a panel, from each size's ranked
    candidates: bigger products run less far, so a bigger control must sit
    higher (smaller y). Picks the ordered combination with the best ranks
    (G_26160: the 800 bp C controls tie with the dimer front, which lies below
    the panel's 256 bp controls). Falls back to each size's best row."""
    best_each = [c[0] if c else None for _, c in cands]
    if len(cands) < 2 or any(not c for _, c in cands):
        return best_each
    best = None
    for combo in itertools.product(*[range(len(c)) for _, c in cands]):
        ys = [c[r] for (_, c), r in zip(cands, combo)]
        # cands are sorted by bp ascending, so y must strictly decrease
        if all(a > b for a, b in zip(ys, ys[1:])):
            key = (sum(combo), combo)
            if best is None or key < best[0]:
                best = (key, ys)
    return best[1] if best else best_each


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
    panels: Optional[List[Tuple[int, int]]] = None,
    well_zone: float = 0.0,
    forced: Optional[Dict[int, Band]] = None,
) -> Tuple[Dict[int, Optional[Band]], float, List[Tuple[int, int]], bool]:
    """Pick control bands per lane; returns (picks, slope_b, panels, calibrated).

    ``panels`` (inclusive lane-index ranges) overrides seam detection.
    Bands with y < ``well_zone`` are not control candidates, unless the rest
    cannot give a control row in at least half of a group's lanes: a row of
    well marks is in every lane and would otherwise outvote a control row
    with one empty lane. ``forced`` (lane index -> band) are the user's
    control choices: they replace the lane's pick and count for calibration."""
    forced = forced or {}
    if panels is None:
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

    # rough control rows: ranked densest y-windows per group
    group_cands: List[Dict[int, List[Band]]] = []
    group_rows: List[List[float]] = []
    for bp, group in groups:
        def rows_of(cands: Dict[int, List[Band]], min_lanes: float = 0) -> List[float]:
            ys = [b.centroid_y for i in group for b in cands[i]]
            ls = [i for i in group for _ in cands[i]]
            hs = [b.bbox[2] - b.bbox[0] for i in group for b in cands[i]]
            return _dimer_front_last(_window_rows(
                np.asarray(ys), np.asarray(ls), 50.0, min_lanes, heights=np.asarray(hs)), bp)

        cands = {i: lane_bands[i] for i in group}
        rows = rows_of(cands)
        if well_zone > 0:
            # densest below-zone rows among those spanning enough lanes (one
            # lane's cluster of bands must not trigger the fallback)
            below = {i: [b for b in lane_bands[i] if b.centroid_y >= well_zone] for i in group}
            rows_below = rows_of(below, min_lanes=max(2, len(group) / 2))
            if rows_below:
                cands, rows = below, rows_below
        group_cands.append(cands)
        group_rows.append(rows)

    # within a panel, control rows must be ordered by size
    chosen: List[Optional[float]] = [None] * len(groups)
    for p0, p1 in panels:
        gi = [g for g, (_, group) in enumerate(groups) if p0 <= group[0] <= p1]
        for g, row in zip(gi, _ordered_rows([(groups[g][0], group_rows[g]) for g in gi])):
            chosen[g] = row

    for (bp, group), cands, row in zip(groups, group_cands, chosen):
        if row is None:
            for i in group:
                picks[i] = None
            continue
        pred_fn = lambda m, v=row: v  # noqa: E731
        # pick -> refit -> re-pick until the picks settle: on a curved gel
        # the first fit only sees the middle lanes, and its extrapolation can
        # miss the end lanes that a refit over more picks reaches (G_26160)
        prev = None
        for _ in range(6):
            for i in group:
                near = [b for b in cands[i] if abs(b.centroid_y - pred_fn(lane_mixes[i])) <= 14.0]
                picks[i] = max(near, key=lambda b: b.area) if near else None
            now = [id(picks[i]) for i in group]
            if now == prev:
                break
            prev = now
            got = [(lane_mixes[i], picks[i].centroid_y) for i in group if picks[i]]
            if len(got) >= 8:
                coef = np.polyfit([g[0] for g in got], [g[1] for g in got], 2)
                pred_fn = lambda m, c=coef: float(np.polyval(c, m))  # noqa: E731
            elif got:
                med = float(np.median([g[1] for g in got]))
                pred_fn = lambda m, v=med: v  # noqa: E731
    for i, b in forced.items():
        picks[i] = b

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
                slope = (y_lo - y_hi) / (math.log(bp_hi) - math.log(bp_lo))
                # a bigger control at or below a smaller one is a mispick: a
                # zero slope cannot size bands, a negative one sizes them backwards
                if slope > 0:
                    slope_estimates.append(slope)
    slope_b = float(np.mean(slope_estimates)) if slope_estimates else DEFAULT_BP_SLOPE
    return picks, slope_b, panels, bool(slope_estimates)


def _dimer_fronts(
    lane_bands: List[List[Band]],
    picks: Dict[int, Optional[Band]],
    ctrl_bps: List[int],
    panels: List[Tuple[int, int]],
    slope_b: float,
    reach: int = 4,
) -> Dict[int, float]:
    """Expected y of the primer-dimer front per lane (lane index -> y).

    Each lane with a control contributes its lowest band below
    :data:`DIMER_MAX_BP` (the front runs ahead of everything, so a small
    product sits above it); a lane's front is the median of those from the
    other lanes of its panel within ``reach`` lanes, or of all the panel's
    other lanes when fewer than three are near. A lane never counts its own
    band, which is the one being judged. Lanes of a panel with fewer than
    three dimer bands get no front."""
    seen: Dict[int, float] = {}
    for i, lb in enumerate(lane_bands):
        ctrl = picks.get(i)
        if ctrl is None:
            continue
        low = [b for b in lb if b is not ctrl and estimate_bp(
            b.centroid_y, ctrl.centroid_y, ctrl_bps[i], slope_b) < DIMER_MAX_BP]
        if low:
            seen[i] = max(b.centroid_y for b in low)
    fronts: Dict[int, float] = {}
    for p0, p1 in panels:
        inside = {i: y for i, y in seen.items() if p0 <= i <= p1}
        if len(inside) < 3:
            continue
        for i in range(p0, p1 + 1):
            others = {j: y for j, y in inside.items() if j != i}
            near = [y for j, y in others.items() if abs(j - i) <= reach]
            fronts[i] = float(np.median(near if len(near) >= 3 else list(others.values())))
    return fronts


def _cluster_centers(lane_bands: List[List[Band]]) -> List[float]:
    """Mean band x per lane; an empty lane takes the midpoint of its nearest
    non-empty neighbours (interpolated across a run of empties)."""
    centers: List[float] = []
    for i, lb in enumerate(lane_bands):
        if lb:
            centers.append(float(np.mean([b.centroid_x for b in lb])))
            continue
        prev = next((centers[j] for j in range(i - 1, -1, -1) if lane_bands[j]), None)
        nxt = next((float(np.mean([b.centroid_x for b in lane_bands[j]]))
                    for j in range(i + 1, len(lane_bands)) if lane_bands[j]), None)
        near = [v for v in (prev, nxt) if v is not None]
        centers.append(float(np.mean(near)) if near else 0.0)
    return centers


def analyse_gel(
    bands: Sequence[Union[Band, Dict]],
    tray: Tray,
    lane_gap: float = 20.0,
    slope_b: Optional[float] = None,
    match_tol: float = 0.25,
    lane_map: Optional[LaneMap] = None,
    band_kinds: Optional[Dict[int, str]] = None,
) -> GelAnalysis:
    """Full post-segmentation analysis against a tray definition.

    Lane index i (left-to-right) is mapped to tray mix i+1; the caller-provided
    ``slope_b`` overrides auto-calibration when given. With ``lane_map``,
    lanes and panels come from the known geometry instead of band gaps
    (``lane_gap`` is then unused).

    ``band_kinds`` (band id -> one of :data:`MANUAL_KINDS`) are the user's
    classifications, which win over the size match: a ``product`` makes its
    lane positive whatever its estimated size; a ``control`` becomes the
    lane's control (the largest, if several are marked); a band given any
    other kind is never picked as a control.
    """
    band_kinds = {int(k): v for k, v in (band_kinds or {}).items()}
    bad = sorted({v for v in band_kinds.values() if v not in MANUAL_KINDS})
    if bad:
        raise ValueError(f"unknown band kind(s) {bad}; use one of {list(MANUAL_KINDS)}")
    mixes = [l.mix for l in sorted(tray.lanes, key=lambda l: l.mix)]
    if mixes != list(range(1, len(mixes) + 1)):
        raise ValueError("tray mixes must be consecutive 1..N for lane alignment")
    ctrl_bps_all = {l.mix: l.control_bp for l in tray.lanes}
    products_all = {l.mix: l.product_bp for l in tray.lanes}

    warnings: List[str] = []
    known_panels: Optional[List[Tuple[int, int]]] = None
    well_zone = 0.0
    if lane_map is not None:
        well_zone = lane_map.well_zone
        if len(lane_map.slots) != len(mixes):
            raise ValueError(
                f"lane layout has {len(lane_map.slots)} lanes but the tray has "
                f"{len(mixes)} mixes")
        lane_bands, outside = assign_to_slots(bands, lane_map.slots)
        if outside:
            warnings.append(
                f"{len(outside)} band(s) lie outside every lane slot of the layout "
                "(e.g. between joined strips) and were ignored")
        known_panels = [tuple(p) for p in lane_map.panels]
        centers = lane_map.centers
    else:
        lane_bands = cluster_lanes(bands, expected_count=len(mixes), gap=lane_gap)
        centers = _cluster_centers(lane_bands)

    # the user's kinds: marked controls are forced picks; bands marked as
    # anything else are not control candidates
    forced: Dict[int, Band] = {}
    ctrl_cands: List[List[Band]] = []
    for i, lb in enumerate(lane_bands):
        marked = [b for b in lb if band_kinds.get(b.id) == "control"]
        if marked:
            forced[i] = max(marked, key=lambda b: b.area)
        ctrl_cands.append([b for b in lb if b.id not in band_kinds])
    picks, slope_auto, panels, calibrated = _fit_controls(
        ctrl_cands, mixes, [ctrl_bps_all[m] for m in mixes], centers, known_panels,
        well_zone, forced)
    slope = float(slope_b) if slope_b is not None else slope_auto
    panels_mix = [(mixes[p0], mixes[p1]) for p0, p1 in panels]

    fronts = _dimer_fronts(lane_bands, picks, [ctrl_bps_all[m] for m in mixes],
                           panels, slope)

    lanes_out: List[LaneCall] = []
    if slope_b is None and not calibrated:
        warnings.append(
            f"bp slope not calibratable (no panel hosts two control sizes); "
            f"using default {DEFAULT_BP_SLOPE} px/ln(bp)")
    for i, mix in enumerate(mixes):
        ctrl = picks.get(i)
        ctrl_bp = ctrl_bps_all[mix]
        calls: List[BandCall] = []
        positive = False
        for b in sorted(lane_bands[i], key=lambda b: b.centroid_y):
            if ctrl is not None and b is ctrl:
                calls.append(BandCall(b, float(ctrl_bp), "control", int(ctrl_bp),
                                      manual=i in forced))
                continue
            bp_est = (estimate_bp(b.centroid_y, ctrl.centroid_y, ctrl_bp, slope)
                      if ctrl is not None else None)
            sizes = products_all[mix]
            manual = band_kinds.get(b.id)
            if manual not in (None, "control"):
                matched = None
                if manual == "product":
                    positive = True
                    if sizes:  # the tray size it stands for: nearest, or the only one
                        matched = (min(sizes, key=lambda s: abs(bp_est - s) / s)
                                   if bp_est is not None else
                                   (sizes[0] if len(sizes) == 1 else None))
                calls.append(BandCall(b, bp_est, manual, matched, manual=True))
                continue
            if ctrl is None or bp_est is None:
                calls.append(BandCall(b, None, "unclassified"))
                continue
            # near the dimer front, sizes from the control run large and the
            # dimer itself can fall within tolerance of a small product, so a
            # small product (expected nearer the front than the control) is
            # matched by position: a band between the control and the front
            # is that product when it lies nearer the product's spot than
            # either (Gel_04 lanes 54, 57, 69 hold only a dimer on the front;
            # G_26160 lane 84 and Gel_01 lane 57 a product above it that
            # sizes 25-33% too large)
            front = fronts.get(i)
            small: List[int] = []
            if front is not None and b.centroid_y > ctrl.centroid_y:
                at = {s: ctrl.centroid_y - slope * math.log(s / ctrl_bp) for s in sizes}
                small = [s for s in sizes if at[s] > (ctrl.centroid_y + front) / 2]
                if small:
                    s = min(small, key=lambda s: abs(b.centroid_y - at[s]))
                    d = abs(b.centroid_y - at[s])
                    if (b.centroid_y < front and d < front - b.centroid_y
                            and d < b.centroid_y - ctrl.centroid_y):
                        calls.append(BandCall(b, bp_est, "product", s))
                        positive = True
                        continue
            sizes = [s for s in sizes if s not in small]
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
