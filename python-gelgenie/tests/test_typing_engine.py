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
import os
from pathlib import Path

import pytest

from ssp.analysis import GelAnalysis, LaneCall, analyse_gel
from ssp.trays import Tray, TrayLane, load_tray_xlsx
from ssp.typing_engine import TypingResult, type_sample

TESTDATA = Path(os.environ.get(
    'SSP_TESTDATA_DIR',
    str(Path(__file__).resolve().parents[2] / 'testdata')))


def _lane(mix, positive, valid=True):
    return LaneCall(mix=mix, center_x=mix * 50.0, control_bp=256, valid=valid,
                    positive=positive, control_y=400.0 if valid else None, calls=[])


def _ga(positives, negatives, invalid=(), locus_size=None):
    lanes = ([_lane(m, True) for m in positives] +
             [_lane(m, False) for m in negatives] +
             [_lane(m, False, valid=False) for m in invalid])
    lanes.sort(key=lambda l: l.mix)
    return GelAnalysis(lanes=lanes, slope_b=81.0, panels=[(1, len(lanes))])


def _tray(specs, locus='A'):
    return Tray(name='t', lanes=[
        TrayLane(mix=m, locus=locus, product_bp=[500], control_bp=256,
                 serotypes='', specificities=s)
        for m, s in sorted(specs.items())
    ])


# ---------- synthetic pattern tests ----------

def test_heterozygous_exact_pair():
    tray = _tray({1: 'A*01:01', 2: 'A*02:01', 3: 'A*01:01, A*02:01', 4: 'A*03:01'})
    ga = _ga(positives=[1, 2, 3], negatives=[4])
    res = type_sample(ga, tray)
    a = res.loci['A']
    assert a.score == 0
    assert a.best_pairs == [('A*01:01', 'A*02:01')]
    assert a.positives == [1, 2, 3]
    assert a.negatives == [4]
    assert a.invalid == []


def test_homozygous_pair():
    tray = _tray({1: 'A*01:01', 2: 'A*02:01', 3: 'A*01:01, A*02:01', 4: 'A*03:01'})
    ga = _ga(positives=[1, 3], negatives=[2, 4])
    res = type_sample(ga, tray)
    a = res.loci['A']
    assert a.score == 0
    assert a.best_pairs == [('A*01:01', 'A*01:01')]


def test_indistinguishable_alleles_report_all_pairs():
    # A*01:01 and A*01:02 have identical lane coverage -> homozygous/hetero
    # combinations are all equally consistent
    tray = _tray({1: 'A*01:01/02', 2: 'A*02:01', 3: 'A*03:01'})
    ga = _ga(positives=[1], negatives=[2, 3])
    res = type_sample(ga, tray)
    a = res.loci['A']
    assert a.score == 0
    assert set(a.best_pairs) == {('A*01:01', 'A*01:01'),
                                 ('A*01:01', 'A*01:02'),
                                 ('A*01:02', 'A*01:02')}
    assert a.allele_options == [['A*01:01', 'A*01:02'], ['A*01:01', 'A*01:02']]


def test_invalid_lanes_skipped_and_reported():
    tray = _tray({1: 'A*01:01', 2: 'A*02:01', 3: 'A*01:01, A*02:01', 4: 'A*03:01'})
    ga = _ga(positives=[1, 3], negatives=[2], invalid=[4])
    res = type_sample(ga, tray)
    a = res.loci['A']
    assert a.invalid == [4]
    assert a.score == 0
    # lane 4 (the only A*03:01 lane) is invalid -> skipped in scoring, so both
    # A*01:01 homozygous and A*01:01+A*03:01 explain the observed pattern
    assert set(a.best_pairs) == {('A*01:01', 'A*01:01'), ('A*01:01', 'A*03:01')}
    # contrast: with lane 4 valid and negative, A*03:01 is excluded
    ga2 = _ga(positives=[1, 3], negatives=[2, 4])
    res2 = type_sample(ga2, tray)
    assert res2.loci['A'].best_pairs == [('A*01:01', 'A*01:01')]


def test_all_except_lane_semantics():
    # sample is A*01:01 + A*03:01: lane 4 ('All, except A*01:01') must be
    # predicted positive via A*03:01, and lane 2 stays negative
    tray = _tray({1: 'A*01:01', 2: 'A*02:01', 3: 'A*03:01',
                  4: 'All, except A*01:01'})
    ga = _ga(positives=[1, 3, 4], negatives=[2])
    res = type_sample(ga, tray)
    a = res.loci['A']
    assert a.score == 0
    assert a.best_pairs == [('A*01:01', 'A*03:01')]


def test_no_positives_warns():
    tray = _tray({1: 'A*01:01', 2: 'A*02:01'})
    ga = _ga(positives=[], negatives=[1, 2])
    res = type_sample(ga, tray)
    assert any('no positive' in w.lower() for w in res.loci['A'].warnings + res.warnings)


def test_all_invalid_locus_is_no_call():
    # every lane lacks a control -> explicit no-call, NOT a misleading score 0
    tray = _tray({1: 'A*01:01', 2: 'A*02:01', 3: 'A*01:01, A*02:01'})
    ga = _ga(positives=[], negatives=[], invalid=[1, 2, 3])
    res = type_sample(ga, tray)
    a = res.loci['A']
    assert a.score is None
    assert a.best_pairs == []
    assert a.invalid == [1, 2, 3]
    assert a.genotype_text() == 'no call'
    assert any('no valid lanes' in w for w in a.warnings)
    assert a.to_dict()['score'] is None  # JSON consumers see null, not 0


def test_candidates_exclude_other_locus_tokens():
    # C-section lane mentioning an A allele (like FLC mix 82): the A*26:04
    # token must not become a C-locus candidate
    tray = Tray(name='t', lanes=[
        TrayLane(mix=1, locus='C', product_bp=[500], control_bp=800,
                 serotypes='', specificities='C*07:03, A*26:04'),
        TrayLane(mix=2, locus='C', product_bp=[500], control_bp=800,
                 serotypes='', specificities='C*07:01'),
    ])
    ga = GelAnalysis(lanes=[
        LaneCall(mix=1, center_x=50.0, control_bp=800, valid=True, positive=True,
                 control_y=300.0),
        LaneCall(mix=2, center_x=100.0, control_bp=800, valid=True, positive=False,
                 control_y=300.0),
    ], slope_b=81.0, panels=[(1, 2)])
    res = type_sample(ga, tray)
    c = res.loci['C']
    assert 'A*26:04' not in c.candidates
    assert c.best_pairs == [('C*07:03', 'C*07:03')]


def test_result_to_dict_json_serializable():
    tray = _tray({1: 'A*01:01', 2: 'A*02:01', 3: 'A*01:01, A*02:01'})
    ga = _ga(positives=[1, 2, 3], negatives=[])
    res = type_sample(ga, tray)
    json.dumps(res.to_dict())
    assert isinstance(res.summary_text(), str) and 'A' in res.summary_text()


# ---------- integration: recorded G_26160 + FLC sheet ----------

FLC = TESTDATA / 'FLClasseI.xlsx'
G_BANDS = TESTDATA / 'ssp_bands_G26160.json'
requires_data = pytest.mark.skipif(not (FLC.exists() and G_BANDS.exists()),
                                   reason='G_26160 testdata not present')


@requires_data
def test_flc_typing_end_to_end():
    tray = load_tray_xlsx(FLC)
    with open(G_BANDS) as f:
        bands = json.load(f)['bands']
    ga = analyse_gel(bands, tray)
    res = type_sample(ga, tray)

    a, b, c = res.loci['A'], res.loci['B'], res.loci['C']

    # A*31 + A*33 (33 ambiguous between :01 and :03)
    assert a.score == 0
    assert a.positives == [19, 21, 23, 26]
    assert set(a.best_pairs) == {('A*31:01', 'A*33:01'), ('A*31:01', 'A*33:03')}

    # B*15-group + B*42:01; kit-level ambiguity across several B*15 alleles
    assert b.score == 0
    assert b.positives == [40, 41, 63, 70]
    assert b.invalid == [39, 49]
    b_second = {p[1] for p in b.best_pairs} | {p[0] for p in b.best_pairs}
    assert 'B*42:01' in b_second
    assert any(t.startswith('B*15:') for t in b_second)
    assert 'B*15:01' in b_second
    # B*42:02 excluded by negative lane 62
    assert not any('B*42:02' in p for p in b.best_pairs)

    # C*04/C*18 (unresolved: lane 96 invalid) + C*08
    assert c.score == 0
    assert c.positives == [76, 84]
    assert c.invalid == [90, 96]
    c_opts = {t for pair in c.best_pairs for t in pair}
    assert {'C*04:01', 'C*04:02', 'C*04:03', 'C*18:01'} <= c_opts
    assert {'C*08:01', 'C*08:02', 'C*08:03'} <= c_opts
