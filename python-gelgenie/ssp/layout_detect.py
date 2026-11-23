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

Automatic first guess of a gel :class:`~ssp.layout.Layout` from a raw photo.

The result is a *suggestion* for the user to confirm on the overlay, never a
silent decision. Classical image analysis only (no model):

1. Normalise polarity so bands are bright, then keep thin horizontal
   structure (``band``: signal minus its local vertical mean).
2. 6x16: split the photo at the strongest vertical seam in its middle third.
3. Per gel piece, the control-band lines are the strongest peaks of the row
   profile, one per row of wells.
4. All rows of a piece come from the same comb, so one lane grid (offset +
   pitch for the known lane count) is fitted to the summed column profile of
   every row. Summed intensity cannot always tell "first lane empty" from
   "last lane faint", so whole-lane shifts are settled separately by lane
   *occupancy*: the grid moves one lane only if that adds a lane with bands
   and drops one without. Each row then refines its own offset and pitch
   (shear, perspective) by least squares on band centres.
5. Wells: empty holes are *dark* features in the bands-bright image on
   both gel types, so a vertical black top-hat (closing minus image) marks
   them without the halo a mean filter leaves around bright bands. Above
   each control line (and below the previous one), the well line is the y
   where most lanes are dark (median over lanes, so a few odd lanes cannot
   win). Rows are chosen *together*: one candidate per row, each between
   its control line and the previous one, scoring strength minus uneven
   spacing (the comb rows are evenly spaced). This rejects the plate edge
   and spare comb rows. Tilt: the lane profiles are compared along lines of
   slightly different slopes (up to ~3 degrees), keeping the slope along
   which the most lanes line up.
6. The piece outline joins the lane-grid edges with the first row's cut line
   and the last row's bottom.

:func:`detect_layout` returns the layout plus warnings describing anything
that looked unreliable.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d, grey_closing, uniform_filter1d
from scipy.signal import find_peaks

from ssp.layout import PRESETS, Block, Layout, LayoutError, Row


@dataclass
class Detection:
    layout: Layout
    warnings: List[str] = field(default_factory=list)


def _gray01(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1) if arr.shape[-1] >= 3 else arr[..., 0]
    lo, hi = np.percentile(arr, [1, 99.5])
    return np.clip((arr - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(arr)


def _bands_bright(g: np.ndarray) -> np.ndarray:
    """Flip light-background gels so bands are bright."""
    return g if np.median(g) < 0.5 else 1.0 - g


def _band_signal(g: np.ndarray, scale: float) -> np.ndarray:
    """Thin horizontal structure: g minus its local vertical mean, >= 0."""
    win = max(5, int(round(scale * 2.5)) | 1)
    return np.clip(g - uniform_filter1d(g, size=win, axis=0), 0, None)


def find_seam(g: np.ndarray) -> Optional[int]:
    """x of the strongest full-height vertical edge in the middle third."""
    H, W = g.shape
    col = gaussian_filter1d(g.mean(axis=0), 2)
    step = np.abs(col[6:] - col[:-6])  # step[i] is centred on x = i + 3
    lo, hi = W // 3, 2 * W // 3
    mids = np.arange(len(step)) + 3
    cand = np.flatnonzero((mids >= lo) & (mids < hi))
    if len(cand) == 0:
        return None
    return int(mids[cand[np.argmax(step[cand])]])


def _fit_grid(profile: np.ndarray, n: int, p_range: Tuple[float, float]) -> Tuple[float, float, float]:
    """Best (offset, pitch, score) for n equally spaced lanes on a 1-D profile."""
    L = len(profile)
    best = (0.0, p_range[0], -np.inf)
    ks = np.arange(n)
    for p in np.arange(p_range[0], p_range[1], 0.1):
        span = (n - 1) * p
        if span >= L - 1:
            continue
        offs = np.arange(0, L - 1 - span, 0.5)
        idx = np.rint(offs[:, None] + ks[None, :] * p).astype(int)
        scores = profile[idx].sum(axis=1)
        i = int(np.argmax(scores))
        if scores[i] > best[2]:
            best = (float(offs[i]), float(p), float(scores[i]))
    return best


def _contrast(profile: np.ndarray, p_guess: float) -> np.ndarray:
    """Peaky, background-free version of a column profile."""
    sm = gaussian_filter1d(profile, max(1.0, p_guess / 8))
    base = gaussian_filter1d(profile, p_guess * 1.5)
    return np.clip(sm - base, 0, None)


def _dark_features(gb: np.ndarray, pitch: float) -> np.ndarray:
    """Vertical black top-hat: thin features darker than their neighbourhood."""
    size = max(3, int(pitch * 0.5) | 1)
    return grey_closing(gb, size=(size, 1)) - gb


ROW_ANCHOR = 0.75  # max end-lane move of a row's grid from the piece grid (lanes)
OCCUPIED_RATIO = 2.0  # lane strip vs neighbouring gaps (99th percentiles)


def _lane_occupancy(band: np.ndarray, centers: np.ndarray, pitch: float,
                    rows: Sequence[float], spacing: float) -> np.ndarray:
    """Per lane: strongest (over rows) ratio of band signal in the lane strip
    to the gaps beside it. Lanes outside the image score 0."""
    H, W = band.shape
    q = max(1, int(pitch / 6))
    out = np.zeros(len(centers))
    for k, c in enumerate(centers):
        ci = int(round(c))
        if ci - q < 0 or ci + q >= W:
            continue
        for yc in rows:
            y0, y1 = max(0, int(yc - 0.6 * spacing)), min(H, int(yc + 0.35 * spacing))
            if y1 <= y0:
                continue
            lane = np.percentile(band[y0:y1, ci - q:ci + q + 1], 99)
            gaps = [np.percentile(band[y0:y1, int(round(g)) - q:int(round(g)) + q + 1], 99)
                    for g in (c - pitch / 2, c + pitch / 2) if q <= round(g) < W - q]
            ref = max(float(np.mean(gaps)) if gaps else 0.0, 1e-3)
            out[k] = max(out[k], lane / ref)
    return out


def _settle_lane_shift(band: np.ndarray, off: float, pitch: float, n: int,
                       rows: Sequence[float], spacing: float) -> Tuple[int, np.ndarray]:
    """Whole-lane correction for a coarse grid: returns (shift, occupancy of
    the n lanes after shifting). Moves by one lane only when the lane gained
    has bands and the lane dropped has none; otherwise keeps the grid."""
    occ = _lane_occupancy(band, off + np.arange(-1, n + 1) * pitch, pitch, rows, spacing)
    on = occ >= OCCUPIED_RATIO
    left_out, first, last, right_out = on[0], on[1], on[n], on[n + 1]
    if right_out and not first and not (left_out and not last):
        return 1, occ[2:n + 2]
    if left_out and not last and not (right_out and not first):
        return -1, occ[0:n]
    return 0, occ[1:n + 1]


def _lsq_grid(profile: np.ndarray, off: float, pitch: float, n: int) -> Tuple[float, float]:
    """Refine a coarse lane grid by least squares on per-lane band peaks.

    A coarse comb fit is ambiguous on flat-topped bands (the grid can shift
    or stretch until its points sit on band edges); peak positions are not.
    Each lane's band is searched within +-0.45 lane of the current grid and
    located as the centroid of its part above half height (bands are
    flat-topped, so the arg-max alone is biased); bands touching the window
    edge (likely a neighbour's) and weak lanes are skipped, outliers dropped. Iterates while
    the grid moves. If the result moves an end lane by more than half a
    lane from the grid it was given, that grid is returned instead (callers
    anchor to their own settled grid as well)."""
    o, p = off, pitch
    for _ in range(10):
        ks, xs, hs = [], [], []
        for k in range(n):
            xc = o + k * p
            lo = int(max(0, np.floor(xc - 0.45 * p)))
            hi = int(min(len(profile), np.ceil(xc + 0.45 * p) + 1))
            w = profile[lo:hi]
            if len(w) < 3:
                continue
            j = int(np.argmax(w))
            if w[j] <= 0:
                continue
            # bands are flat-topped: centre of the part above half height
            half = 0.5 * w[j]
            a, b = j, j
            while a > 0 and w[a - 1] > half:
                a -= 1
            while b < len(w) - 1 and w[b + 1] > half:
                b += 1
            if a == 0 or b == len(w) - 1:
                continue  # touches the window edge: likely a neighbour's band
            top = w[a:b + 1] - half
            ks.append(k)
            xs.append(lo + a + float((np.arange(len(top)) * top).sum() / top.sum()))
            hs.append(float(w[j]))
        if len(ks) < max(4, n // 3):
            break
        ks_a, xs_a, hs_a = map(np.asarray, (ks, xs, hs))
        keep = hs_a > 0.2 * np.median(hs_a)
        ok = True
        for _ in range(2):
            if keep.sum() < max(4, n // 3):
                ok = False
                break
            p_new, o_new = np.polyfit(ks_a[keep], xs_a[keep], 1)
            keep &= np.abs(xs_a - (o_new + p_new * ks_a)) <= 0.25 * p
        if not ok:
            break
        moved = max(abs(o_new - o), abs((o_new + (n - 1) * p_new) - (o + (n - 1) * p)))
        o, p = float(o_new), float(p_new)
        if moved < 0.1:
            break
    if abs(o - off) > 0.5 * pitch or abs((o + (n - 1) * p) - (off + (n - 1) * pitch)) > 0.5 * pitch:
        return off, pitch  # wandered off: keep the coarse fit
    return o, p


def _lane_profiles(dark: np.ndarray, centers: np.ndarray, pitch: float) -> np.ndarray:
    """Dark-feature strength down each lane centre: (n_lanes, H)."""
    half = max(1, int(pitch / 4))
    W = dark.shape[1]
    cols = [np.clip(int(round(c)), half, W - half - 1) for c in centers]
    return np.stack([dark[:, c - half:c + half].mean(axis=1) for c in cols])


def _tilted_median(per_lane: np.ndarray, centers: np.ndarray, slope: float) -> np.ndarray:
    """Median over lanes after shifting each lane by ``slope`` * (x - x_mid),
    i.e. the well-line profile along a tilted line, indexed by y at x_mid."""
    xm = float(np.mean(centers))
    H = per_lane.shape[1]
    y = np.arange(H, dtype=float)
    shifted = np.stack([np.interp(y + slope * (c - xm), y, p, left=0.0, right=0.0)
                        for c, p in zip(centers, per_lane)])
    return gaussian_filter1d(np.median(shifted, axis=0), 1.5)


def _well_profile(dark: np.ndarray, centers: np.ndarray, pitch: float,
                  n_rows: int) -> Tuple[np.ndarray, float]:
    """Best tilted well profile and its slope (dy/dx). Gels are photographed
    at most a few degrees off, so slopes up to ~3 degrees are tried; the
    slope whose strongest peaks are highest wins (lanes line up)."""
    per_lane = _lane_profiles(dark, centers, pitch)
    best = (None, 0.0, -np.inf)
    for slope in np.linspace(-0.05, 0.05, 41):
        prof = _tilted_median(per_lane, centers, slope)
        pk, _ = find_peaks(prof, distance=max(1, int(pitch)))
        score = float(np.sort(prof[pk])[-n_rows:].sum()) if len(pk) else 0.0
        if score > best[2] + 1e-9 or (abs(score - best[2]) <= 1e-9 and abs(slope) < abs(best[1])):
            best = (prof, float(slope), score)
    return best[0], best[1]


def _choose_wells(prof: np.ndarray, windows: List[Tuple[int, int]],
                  pitch: float) -> Tuple[Optional[List[float]], List[int]]:
    """One well line per row: strong, inside its window, evenly spaced.

    A row whose wells are not visible (faint, or cut off at the photo edge)
    may be left out and placed from the others' spacing, provided at least
    two rows were found. Returns ``(wells, inferred_row_indices)``."""
    n = len(windows)
    if prof.max() <= 1e-4:
        return None, []
    peaks, _ = find_peaks(prof, distance=max(1, int(pitch * 0.8)),
                          prominence=0.15 * prof.max())
    options = [[int(p) for p in peaks if lo <= p <= hi] + [None] for lo, hi in windows]
    need = n if n < 3 else 2
    best, best_score = None, -np.inf
    for combo in itertools.product(*options):
        found = [(k, y) for k, y in enumerate(combo) if y is not None]
        if len(found) < need:
            continue
        ks = np.array([k for k, _ in found], dtype=float)
        ys = np.array([y for _, y in found], dtype=float)
        if np.any(np.diff(ys) <= 0):
            continue
        strength = sum(prof[int(y)] for y in ys) / (prof.max() * n)
        if len(found) >= 2:
            b, a = np.polyfit(ks, ys, 1)
            if b <= 0:
                continue
            uneven = float(np.std(ys - (a + b * ks)) / b) if len(found) > 2 else 0.0
        else:
            uneven = 0.0
        score = strength - 2.0 * uneven - 0.1 * (n - len(found))
        if score > best_score:
            best, best_score = combo, score
    if best is None:
        return None, []
    found = [(k, y) for k, y in enumerate(best) if y is not None]
    if len(found) == n:
        return [float(y) for y in best], []
    ks, ys = zip(*found)
    b, a = np.polyfit(ks, ys, 1)
    missing = [k for k, y in enumerate(best) if y is None]
    return [float(y) if y is not None else float(a + b * k) for k, y in enumerate(best)], missing




def _detect_block(g: np.ndarray, x0: int, x1: int, n_rows: int, n_lanes: int,
                  warnings: List[str], tag: str) -> Block:
    H = g.shape[0]
    sub = g[:, x0:x1]
    W = sub.shape[1]
    p_guess = W / (n_lanes + 2)
    band = _band_signal(sub, p_guess)

    # control lines: strongest row-profile peaks, one per row of wells
    rowprof = gaussian_filter1d(band.mean(axis=1), max(1.0, p_guess / 10))
    peaks, props = find_peaks(rowprof, distance=max(1, int(0.55 * H / n_rows)),
                              prominence=0)
    if len(peaks) < n_rows:
        raise LayoutError(f"{tag}: found {len(peaks)} band rows, expected {n_rows}")
    top = np.sort(peaks[np.argsort(props["prominences"])[-n_rows:]])
    spacing = float(np.median(np.diff(top))) if n_rows > 1 else H / 2.0

    # one lane grid for the whole piece (same comb for every row)
    profs = []
    for c in top:
        r0, r1 = int(c - 0.6 * spacing), int(c + 0.35 * spacing)
        profs.append(band[max(0, r0):min(H, r1)].mean(axis=0))
    total = _contrast(np.sum(profs, axis=0), p_guess)
    p_lo, p_hi = W / (n_lanes + 8), W / (n_lanes - 0.5)
    off, pitch, _ = _fit_grid(total, n_lanes, (p_lo, p_hi))
    shift, occ = _settle_lane_shift(band, off, pitch, n_lanes, top, spacing)
    off += shift * pitch
    for k, label in ((0, "first"), (n_lanes - 1, "last")):
        if occ[k] < OCCUPIED_RATIO:
            warnings.append(f"{tag}: the {label} lane shows little band signal; "
                            "confirm the lane numbering at that edge")
    # per-row refinement of offset and pitch (shear / perspective). The row
    # search alone could stretch the far lane by over a lane, so every row is
    # anchored to the settled grid (least-squares refined: the comb fit is
    # stretched on flat-topped bands): neither end lane may move ROW_ANCHOR
    # lanes or more from it, else the row keeps that grid. The limit is not
    # half a lane because real shear goes beyond it (G_26160: 0.55 lane
    # between rows); a one-lane jump shows as about 1.
    anc_off, anc_pitch = _lsq_grid(total, off, pitch, n_lanes)
    anc_end = anc_off + (n_lanes - 1) * anc_pitch
    row_off, row_pitch = [], []
    ks = np.arange(n_lanes)
    for prof in profs:
        c = _contrast(prof, pitch)
        best = (off, pitch, -np.inf)
        for p in np.arange(0.97 * pitch, 1.03 * pitch, 0.05):
            cands = np.arange(off - 0.35 * pitch, off + 0.35 * pitch, 0.5)
            cands = cands[(cands >= 0) & (cands + (n_lanes - 1) * p < W - 1)]
            if len(cands) == 0:
                continue
            idx = np.rint(cands[:, None] + ks[None, :] * p).astype(int)
            sc = c[idx].sum(axis=1)
            i = int(np.argmax(sc))
            if sc[i] > best[2]:
                best = (float(cands[i]), float(p), float(sc[i]))
        o, p = _lsq_grid(c, best[0], best[1], n_lanes)
        if (abs(o - anc_off) >= ROW_ANCHOR * pitch
                or abs(o + (n_lanes - 1) * p - anc_end) >= ROW_ANCHOR * pitch):
            o, p = anc_off, anc_pitch
        row_off.append(o)
        row_pitch.append(p)

    # well lines: chosen jointly, then refined per half (tilt)
    dark = _dark_features(g, pitch)
    centers = x0 + float(np.mean(row_off)) + np.arange(n_lanes) * pitch
    windows = []
    for ri, c in enumerate(top):
        lo = int(c - 0.95 * spacing)
        if ri > 0:
            lo = max(lo, int(top[ri - 1] + 0.25 * spacing))
        windows.append((lo, int(c - 0.1 * spacing)))
    prof, slope = _well_profile(dark, centers, pitch, n_rows)
    xm = float(np.mean(centers))
    wells, inferred = _choose_wells(prof, windows, pitch)
    if wells is None:
        warnings.append(f"{tag}: wells not found; row cuts placed by band spacing")
        wells = [c - 0.75 * spacing for c in top]
        inferred = []
    for k in inferred:
        warnings.append(f"{tag} row {k + 1}: wells not visible; cut placed from the "
                        "other rows' spacing")
    # cut just above each well: walk up from the well signal's peak to where
    # it fades (wells differ in height), then leave a quarter lane
    def _well_top(y: float) -> float:
        i = int(round(y))
        if not 0 <= i < len(prof):
            return y
        floor = 0.3 * prof[i]
        j = i
        while j > 0 and prof[j - 1] > floor and i - j < 0.5 * pitch:
            j -= 1
        return float(j)

    wells = [w if k in inferred else _well_top(w) for k, w in enumerate(wells)]
    margin = 0.4 * pitch
    # well line y at the piece's left / right lane edges, along the slope
    x_left = centers[0] - pitch / 2
    x_right = centers[-1] + pitch / 2
    tops_l = [w + slope * (x_left - xm) - margin for w in wells]
    tops_r = [w + slope * (x_right - xm) - margin for w in wells]
    under = max(0.0, -min(tops_l[0], tops_r[0]))  # keep the top edge parallel
    tops_l[0] += under
    tops_r[0] += under

    # outline: lane-grid edges (linear in y across rows) x cut lines
    ys = np.asarray(top, dtype=float)
    rp = np.asarray(row_pitch)
    lefts = x0 + np.asarray(row_off) - rp / 2
    rights = lefts + n_lanes * rp
    fl = np.polyfit(ys, lefts, 1) if n_rows > 1 else (0.0, lefts[0])
    fr = np.polyfit(ys, rights, 1) if n_rows > 1 else (0.0, rights[0])
    well_spacing = float(np.median(np.diff(wells))) if n_rows > 1 else spacing
    # bottom edge parallel to the top (a non-parallel pair would make the
    # projective map squeeze the middle lanes); shift both up if needed
    bot_l, bot_r = tops_l[-1] + well_spacing, tops_r[-1] + well_spacing
    over = max(0.0, max(bot_l, bot_r) - (H - 1.0))
    bot_l, bot_r = bot_l - over, bot_r - over
    yt_l, yt_r = tops_l[0], tops_r[0]
    quad = [
        (float(np.polyval(fl, yt_l)), yt_l),
        (float(np.polyval(fr, yt_r)), yt_r),
        (float(np.polyval(fr, bot_r)), bot_r),
        (float(np.polyval(fl, bot_l)), bot_l),
    ]
    # each band row must start below its own cut: cuts that merely increase
    # can still put two rows' bands in one strip. (The bottom edge is not
    # checked against the last peak: on real gels the strongest band row of
    # the last row can be the dye front below the gel, e.g. G_26160.)
    cuts = [(yt_l + yt_r) / 2] + [(tl + tr) / 2 for tl, tr in zip(tops_l[1:], tops_r[1:])]
    for ri, c in enumerate(top):
        nxt = cuts[ri + 1] if ri + 1 < n_rows else np.inf
        if not cuts[ri] < c < nxt:
            raise LayoutError(f"{tag} row {ri + 1}: its band row (y={c:.0f}) falls outside "
                              f"the row cut (y={cuts[ri]:.0f} to {nxt:.0f})")
    # row cuts as fractions of the rectified block height (mean of both sides)
    hl, hr = bot_l - yt_l, bot_r - yt_r
    fracs = [0.0] + [float(np.clip(((tl - yt_l) / hl + (tr - yt_r) / hr) / 2, 0, 1))
                     for tl, tr in zip(tops_l[1:], tops_r[1:])] + [1.0]
    rows = [Row(top=round(fracs[i], 4), bottom=round(fracs[i + 1], 4), lanes=n_lanes)
            for i in range(n_rows)]
    return Block(quad=[(round(x, 1), round(y, 1)) for x, y in quad], rows=rows)


def detect_layout(image: np.ndarray, preset: str) -> Detection:
    """Suggest a layout for a raw gel photo of the given preset."""
    if preset not in PRESETS:
        raise LayoutError(f"automatic detection supports {', '.join(PRESETS)}, not {preset!r}")
    n_blocks, n_rows, n_lanes = PRESETS[preset]
    g = _gray01(image)
    H, W = g.shape
    gb = _bands_bright(g)
    warnings: List[str] = []
    if n_blocks == 1:
        spans = [(0, W)]
    else:
        seam = find_seam(g)
        if seam is None:
            raise LayoutError("could not find the boundary between the two gel pieces")
        spans = [(0, seam), (seam, W)]
    blocks = [
        _detect_block(gb, a, b, n_rows, n_lanes, warnings, f"piece {i + 1}")
        for i, (a, b) in enumerate(spans)
    ]
    layout = Layout(blocks=blocks, image_size=(W, H), preset=preset)
    try:
        warnings += layout.validate(n_blocks * n_rows * n_lanes)
    except LayoutError as e:
        raise LayoutError(f"detected layout is unusable: {e}") from None
    return Detection(layout=layout, warnings=warnings)
