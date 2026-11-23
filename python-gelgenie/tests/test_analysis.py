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
"""

import json
import math
import os
from pathlib import Path

import pytest

from ssp.analysis import (
    DEFAULT_BP_SLOPE,
    GelAnalysis,
    LaneMap,
    analyse_gel,
    assign_to_slots,
    cluster_lanes,
    estimate_bp,
)
from ssp.bands import Band
from ssp.trays import Tray, TrayLane

TESTDATA = Path(os.environ.get(
    'SSP_TESTDATA_DIR',
    str(Path(__file__).resolve().parents[2] / 'testdata')))


def _band(x, y, area=300):
    return Band(id=0, area=area, centroid_y=float(y), centroid_x=float(x),
                bbox=(int(y) - 3, int(x) - 8, int(y) + 3, int(x) + 8))


def _tray(n, control_bp=256, products=None):
    products = products or {}
    return Tray(name='t', lanes=[
        TrayLane(mix=i + 1, locus='A', product_bp=products.get(i + 1, [500]),
                 control_bp=control_bp, serotypes='', specificities='A*01:01')
        for i in range(n)
    ])


# ---------- cluster_lanes ----------

def test_cluster_lanes_basic():
    bands = [_band(10, 100), _band(12, 300), _band(60, 100), _band(110, 100)]
    clusters = cluster_lanes(bands, gap=20.0)
    assert len(clusters) == 3
    assert len(clusters[0]) == 2


def test_cluster_lanes_inserts_missing_lane():
    # 5 lanes at 50px pitch, lane 3 absent -> one 100px gap
    xs = [10, 60, 160, 210, 260]
    bands = [_band(x, 100) for x in xs]
    clusters = cluster_lanes(bands, expected_count=6, gap=20.0)
    assert len(clusters) == 6
    assert clusters[2] == []  # empty lane inserted at the 2x gap


def test_cluster_lanes_seam_not_treated_as_missing():
    # 2 panels of 3 lanes; seam gap is ~2.6x pitch -> no phantom insertion
    xs = [10, 60, 110, 250, 300, 350]
    bands = [_band(x, 100) for x in xs]
    clusters = cluster_lanes(bands, expected_count=6, gap=20.0)
    assert len(clusters) == 6


def test_cluster_lanes_unrepairable_mismatch_raises():
    bands = [_band(10, 100), _band(60, 100)]
    with pytest.raises(ValueError):
        cluster_lanes(bands, expected_count=9, gap=20.0)


def test_cluster_lanes_too_many_raises():
    bands = [_band(10, 100), _band(60, 100), _band(110, 100)]
    with pytest.raises(ValueError, match='found 3 lane clusters'):
        cluster_lanes(bands, expected_count=2, gap=20.0)


def test_cluster_lanes_single_cluster_expected_more_raises():
    # one merged blob, no pitch information -> clean ValueError (no nan maths)
    bands = [_band(10, 100), _band(12, 300)]
    with pytest.raises(ValueError, match='single lane cluster'):
        cluster_lanes(bands, expected_count=4, gap=20.0)


# ---------- estimate_bp ----------

def test_estimate_bp_math():
    # band 81px above control with slope 81 -> e times smaller bp... check exact
    assert estimate_bp(300.0, 400.0, 256, 81.0) == pytest.approx(256 * math.exp(100 / 81))
    assert estimate_bp(400.0, 400.0, 256, 81.0) == pytest.approx(256)
    # below control -> larger bp estimate? NO: below control = smaller fragment
    assert estimate_bp(450.0, 400.0, 256, 81.0) == pytest.approx(256 * math.exp(-50 / 81))


# ---------- analyse_gel on synthetic mini-gels ----------

def _synth_lanes(n=8, x0=20.0, pitch=50.0, ctrl_y=400.0, smile=2.0):
    """control per lane + optional extra bands; returns (bands, truth)."""
    bands, truth = [], {}
    for i in range(n):
        x = x0 + i * pitch
        cy = ctrl_y + smile * i
        bands.append(_band(x, cy, area=400))
        truth[i + 1] = {'ctrl_y': cy}
    return bands, truth


def test_analyse_gel_controls_and_positivity():
    # 8 lanes; lane 3 has a 500bp product 55px above control (b=81:
    # 256*exp(55/81)=504bp ~ sheet 500bp -> positive)
    bands, _ = _synth_lanes(8)
    bands.append(_band(120, 400 + 2 * 2 - 55, area=350))  # lane 3 product
    tray = _tray(8, products={i: [500] for i in range(1, 9)})
    ga = analyse_gel(bands, tray)
    assert isinstance(ga, GelAnalysis)
    assert len(ga.lanes) == 8
    assert all(l.valid for l in ga.lanes)
    l3 = ga.lanes[2]
    assert l3.positive
    assert l3.products and abs(l3.products[0].bp_est - 500) / 500 < 0.25
    assert not ga.lanes[0].positive


def test_analyse_gel_missing_control_invalid():
    bands, _ = _synth_lanes(8)
    # remove lane 5's control band
    bands = [b for b in bands if abs(b.centroid_x - (20 + 4 * 50)) > 1]
    tray = _tray(8)
    ga = analyse_gel(bands, tray)
    assert not ga.lanes[4].valid
    assert ga.lanes[4].control_y is None
    assert all(ga.lanes[i].valid for i in range(8) if i != 4)


def test_analyse_gel_classifies_dimer_hmw_unexplained():
    bands, _ = _synth_lanes(6)
    x1, cy1 = 20.0, 400.0
    bands.append(_band(x1, cy1 + 60, area=500))    # ~120bp est -> dimer
    bands.append(_band(x1, cy1 - 250, area=500))   # huge bp -> HMW
    bands.append(_band(x1, cy1 - 20, area=500))    # ~350bp est, sheet=500 -> unexplained
    tray = _tray(6, products={i: [500] for i in range(1, 7)})
    ga = analyse_gel(bands, tray)
    kinds = {c.kind for c in ga.lanes[0].calls}
    assert {'control', 'dimer', 'hmw', 'unexplained'} <= kinds
    assert not ga.lanes[0].positive


def test_analyse_gel_estimates_slope_from_two_control_sizes():
    # one panel: lanes 1-4 ctrl 256bp @ y400, lanes 5-8 ctrl 800bp @ y300
    bands = []
    for i in range(4):
        bands.append(_band(20 + i * 50, 400, area=400))
    for i in range(4):
        bands.append(_band(220 + i * 50, 300, area=400))
    tray = Tray(name='t', lanes=[
        TrayLane(mix=i + 1, locus='B', product_bp=[500], control_bp=256,
                 serotypes='', specificities='') for i in range(4)
    ] + [
        TrayLane(mix=i + 5, locus='C', product_bp=[500], control_bp=800,
                 serotypes='', specificities='') for i in range(4)
    ])
    ga = analyse_gel(bands, tray)
    expected_b = 100.0 / (math.log(800) - math.log(256))  # ~87.7
    assert ga.slope_b == pytest.approx(expected_b, rel=0.05)
    assert all(l.valid for l in ga.lanes)


def test_analyse_gel_default_slope_single_control_size():
    bands, _ = _synth_lanes(5)
    ga = analyse_gel(bands, _tray(5))
    assert ga.slope_b == DEFAULT_BP_SLOPE
    # the fallback must be surfaced to the user
    assert any('calibrat' in w for w in ga.warnings)


def test_analyse_gel_slope_override():
    bands, _ = _synth_lanes(5)
    ga = analyse_gel(bands, _tray(5), slope_b=50.0)
    assert ga.slope_b == 50.0
    assert not any('calibrat' in w for w in ga.warnings)  # explicit override: no warning


def test_analyse_gel_nonconsecutive_mixes_raise():
    tray = Tray(name='t', lanes=[
        TrayLane(mix=1, locus='A', product_bp=[500], control_bp=256,
                 serotypes='', specificities='A*01:01'),
        TrayLane(mix=3, locus='A', product_bp=[500], control_bp=256,
                 serotypes='', specificities='A*03:01'),
    ])
    bands, _ = _synth_lanes(2)
    with pytest.raises(ValueError, match='consecutive'):
        analyse_gel(bands, tray)


def test_analyse_gel_accepts_dicts():
    bands, _ = _synth_lanes(3)
    dicts = [b.to_dict() for b in bands]
    ga = analyse_gel(dicts, _tray(3))
    assert len(ga.lanes) == 3 and all(l.valid for l in ga.lanes)


# ---------- integration: recorded G_26160 detection vs FLC sheet ----------

FLC = TESTDATA / 'FLClasseI.xlsx'
G_BANDS = TESTDATA / 'ssp_bands_G26160.json'
requires_data = pytest.mark.skipif(not (FLC.exists() and G_BANDS.exists()),
                                   reason='G_26160 testdata not present')


@requires_data
def test_g26160_full_analysis():
    from ssp.trays import load_tray_xlsx
    tray = load_tray_xlsx(FLC)
    with open(G_BANDS) as f:
        bands = json.load(f)['bands']
    ga = analyse_gel(bands, tray)
    assert len(ga.lanes) == 96
    invalid = [l.mix for l in ga.lanes if not l.valid]
    assert invalid == [39, 49, 90, 96]
    positives = [l.mix for l in ga.lanes if l.positive]
    assert positives == [19, 21, 23, 26, 40, 41, 63, 70, 76, 84]
    assert ga.slope_b == pytest.approx(81, abs=6)


# ---------- known lane geometry (LaneMap from a layout manifest) ----------

def _two_panel_map(n_per=4, pitch=50.0, panel_start=(0.0, 300.0)):
    slots, panels = [], []
    for p, x0 in enumerate(panel_start):
        slots += [(x0 + i * pitch, x0 + (i + 1) * pitch) for i in range(n_per)]
        panels.append((p * n_per, (p + 1) * n_per - 1))
    return LaneMap(slots=slots, panels=panels)


def test_lane_map_handles_empty_lane_next_to_seam():
    # panel 2's first lane (mix 5) is empty: by band gaps this is
    # indistinguishable from a wider seam, so gap clustering cannot recover it
    lm = _two_panel_map()
    bands = [_band(c, 400, area=400) for i, c in enumerate(lm.centers) if i != 4]
    tray = _tray(8)
    with pytest.raises(ValueError):
        analyse_gel(bands, tray)
    ga = analyse_gel(bands, tray, lane_map=lm)
    assert [l.mix for l in ga.lanes if not l.valid] == [5]
    assert ga.panels == [(1, 4), (5, 8)]
    assert [l.center_x for l in ga.lanes] == lm.centers


def test_lane_map_panels_override_seam_detection():
    # uniform pitch (no seam) but the map says two panels: controls are fitted
    # per panel, so a step in control height between panels is tolerated
    lm = _two_panel_map(panel_start=(0.0, 200.0))
    bands = [_band(c, 400 if i < 4 else 430, area=400) for i, c in enumerate(lm.centers)]
    ga = analyse_gel(bands, _tray(8), lane_map=lm)
    assert ga.panels == [(1, 4), (5, 8)]
    assert all(l.valid for l in ga.lanes)
    assert ga.lanes[4].control_y == 430


def test_lane_map_ignores_bands_between_strips_with_warning():
    lm = _two_panel_map()
    bands = [_band(c, 400, area=400) for c in lm.centers] + [_band(250, 380)]
    ga = analyse_gel(bands, _tray(8), lane_map=lm)
    assert all(l.valid for l in ga.lanes)
    assert any("outside every lane slot" in w for w in ga.warnings)


def test_well_marks_in_every_lane_do_not_outvote_the_control_row():
    # Gel_05: faint marks under the wells are segmented in every lane; the
    # control row misses one lane, so by count alone the marks would win and
    # every band would be sized against them
    lm = _two_panel_map()
    bands = [_band(c, 30, area=150) for c in lm.centers]                 # well marks
    bands += [_band(c, 300, area=300) for i, c in enumerate(lm.centers) if i != 2]
    bands += [_band(lm.centers[1], 240, area=300)]                       # a product
    lm.well_zone = 0.0
    assert analyse_gel(bands, _tray(8), lane_map=lm).lanes[0].control_y == 30
    lm.well_zone = 100.0
    ga = analyse_gel(bands, _tray(8), lane_map=lm)
    assert [l.control_y for l in ga.lanes] == [300, 300, None, 300, 300, 300, 300, 300]
    kinds = [(c.kind, c.band.centroid_y) for c in ga.lanes[1].calls]
    assert kinds[0][0] == "hmw" and ("control", 300) in kinds


def test_one_lane_cluster_below_zone_does_not_trigger_fallback():
    # four bands from a single lane are the densest below-zone window, but
    # the control row (three lanes) still qualifies and must be chosen
    lm = LaneMap(slots=[(i * 50.0, (i + 1) * 50.0) for i in range(4)], panels=[(0, 3)],
                 well_zone=100.0)
    c = lm.centers
    bands = [_band(x, 30, area=150) for x in c]
    bands += [_band(x, 300) for x in c[:3]]
    bands += [_band(c[3], y) for y in (150, 154, 158, 162)]
    ga = analyse_gel(bands, _tray(4), lane_map=lm)
    assert [l.control_y for l in ga.lanes] == [300, 300, 300, None]


def test_well_zone_falls_back_when_controls_sit_inside_it():
    # a very short run: the only band row is inside the zone; it is still used
    lm = _two_panel_map()
    lm.well_zone = 100.0
    bands = [_band(c, 60, area=300) for c in lm.centers]
    ga = analyse_gel(bands, _tray(8), lane_map=lm)
    assert all(l.valid and l.control_y == 60 for l in ga.lanes)


def test_manifest_lane_map_sets_well_zone():
    from ssp.layout import WELL_ZONE_PITCHES, LaneSlot, Manifest, Strip
    man = Manifest(image_size=(200, 100),
                   strips=[Strip(block=0, row=0, x0=0, x1=160, height=100,
                                 first_mix=1, last_mix=4)],
                   lanes=[LaneSlot(mix=i + 1, block=0, row=0, x0=40.0 * i, x1=40.0 * (i + 1))
                          for i in range(4)])
    assert man.to_lane_map().well_zone == pytest.approx(WELL_ZONE_PITCHES * 40)


def test_lane_map_lane_count_must_match_tray():
    with pytest.raises(ValueError, match="8 lanes but the tray has 6"):
        analyse_gel([], _tray(6), lane_map=_two_panel_map())


def test_assign_to_slots_edges_and_tolerance():
    slots = [(0.0, 50.0), (50.0, 100.0), (200.0, 250.0)]
    bands = [_band(x, 100) for x in (49.9, 50.0, -20.0, 124.0, 126.0, 160.0)]
    lane_bands, outside = assign_to_slots(bands, slots)
    assert [b.centroid_x for b in lane_bands[0]] == [49.9, -20.0]
    assert [b.centroid_x for b in lane_bands[1]] == [50.0, 124.0]
    assert lane_bands[2] == []
    assert [b.centroid_x for b in outside] == [126.0, 160.0]


def test_bigger_control_row_must_sit_above_the_smaller_one():
    # G_26160 panel 3: lanes 65-70 have a 256 bp control, 71-96 an 800 bp one.
    # In the 800 bp lanes the real control row ties with the dimer front, and
    # one product next to the front tips the count; but an 800 bp control
    # cannot run below the panel's 256 bp controls
    bands = []
    xs = [20 + i * 50 for i in range(12)]
    for i, x in enumerate(xs):
        if i < 4:
            bands.append(_band(x, 340, area=400))       # 256 bp controls
        elif i != 5:
            bands.append(_band(x, 250, area=400))       # 800 bp controls
        if i != 9:
            bands.append(_band(x, 410, area=300))       # dimer / front
    bands.append(_band(xs[6], 380, area=300))           # a product near the front
    tray = Tray(name='t', lanes=[
        TrayLane(mix=i + 1, locus='B' if i < 4 else 'C', product_bp=[500],
                 control_bp=256 if i < 4 else 800, serotypes='', specificities='')
        for i in range(12)])
    ga = analyse_gel(bands, tray)
    assert [l.control_y for l in ga.lanes] == [340] * 4 + [250, None] + [250] * 6
    assert ga.slope_b == pytest.approx(90 / math.log(800 / 256), rel=0.01)


def test_control_picks_follow_a_curved_row_to_the_panel_ends():
    # G_26160 panel 1: the control row curves down ~50 px across 32 lanes; the
    # first fit sees only the middle lanes and misses the last ones
    # (the session's control heights; lane 27 had no bands)
    ys = [370, 370, 370, 370, 371, 372, 373, 374, 376, 377, 379, 381, 383, 384,
          387, 389, 393, 396, 398, 400, 403, 405, 407, 410, 413, 415, None, 417,
          419, 420, 422, 423]
    lm = LaneMap(slots=[(i * 50, i * 50 + 40) for i in range(32)], panels=[(0, 31)])
    bands = [_band(20 + i * 50, y, area=400) for i, y in enumerate(ys) if y]
    ga = analyse_gel(bands, _tray(32), lane_map=lm)
    assert [l.control_y for l in ga.lanes] == ys


def test_control_sizes_on_one_row_fall_back_to_the_default_slope():
    # both control sizes of a panel settle on y=290: the estimate would be a
    # zero slope, and sizing any other band would divide by it
    bands = [_band(20 + i * 50, 290, area=400) for i in range(8)] + [_band(20, 250)]
    tray = Tray(name='t', lanes=[
        TrayLane(mix=i + 1, locus='B' if i < 4 else 'C', product_bp=[500],
                 control_bp=256 if i < 4 else 800, serotypes='', specificities='')
        for i in range(8)])
    ga = analyse_gel(bands, tray)
    assert ga.slope_b == DEFAULT_BP_SLOPE
    assert any('calibrat' in w for w in ga.warnings)
