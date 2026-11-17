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
    analyse_gel,
    cluster_lanes,
    estimate_bp,
)
from ssp.bands import Band
from ssp.trays import Tray, TrayLane

TESTDATA = Path(os.environ.get(
    'SSP_TESTDATA_DIR',
    str(Path(__file__).resolve().parents[2] / 'qupath-gelgenie' / 'testdata')))


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
