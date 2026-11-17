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

import os
from pathlib import Path

import pytest

from ssp.trays import load_tray_xlsx
from ssp.trays.specificities import SpecificitySet, family_of, parse_specificities

TESTDATA = Path(os.environ.get(
    'SSP_TESTDATA_DIR',
    str(Path(__file__).resolve().parents[2] / 'qupath-gelgenie' / 'testdata')))
FLC = TESTDATA / 'FLClasseI.xlsx'
requires_sheet = pytest.mark.skipif(not FLC.exists(), reason='FLC sheet not present')


def test_parse_simple_slash():
    s = parse_specificities('A*01:01/02')
    assert s.alleles == {'A*01:01', 'A*01:02'}


def test_parse_range():
    s = parse_specificities('A*02:01-19')
    assert len(s.alleles) == 19
    assert 'A*02:01' in s.alleles and 'A*02:19' in s.alleles
    assert 'A*02:20' not in s.alleles


def test_parse_mixed_and_null_suffix_stripped():
    s = parse_specificities('A*24:02-05/07/09-11N')
    assert s.alleles == {'A*24:02', 'A*24:03', 'A*24:04', 'A*24:05',
                         'A*24:07', 'A*24:09', 'A*24:10', 'A*24:11'}


def test_parse_multi_group_comma():
    s = parse_specificities('A*26:01/02/04/07/08, A*43:01')
    assert s.alleles == {'A*26:01', 'A*26:02', 'A*26:04', 'A*26:07',
                         'A*26:08', 'A*43:01'}


def test_parse_apostrophe_typo():
    s = parse_specificities("C*'08:01-03")
    assert s.alleles == {'C*08:01', 'C*08:02', 'C*08:03'}


def test_parse_missing_colon_typo():
    # mix 30 (cell E36) reads 'B*51:01-05/08, B*5201-02'; the typo means B*52:01/02
    s = parse_specificities('B*5201-02')
    assert s.alleles == {'B*52:01', 'B*52:02'}


def test_parse_family_only_tokens():
    s = parse_specificities('B*13, B*37')
    assert s.alleles == {'B*13', 'B*37'}


def test_parse_chained_range():
    s = parse_specificities('B*15:13-16-17/23/24')
    assert s.alleles == {f'B*15:{n:02d}' for n in (13, 14, 15, 16, 17, 23, 24)}


def test_parse_reversed_range_is_swapped():
    s = parse_specificities('B*15:32-25')
    assert s.alleles == {f'B*15:{n:02d}' for n in range(25, 33)}


def test_parse_all_except():
    s = parse_specificities('All, except A*02:01-19, A*23:01')
    assert s.alleles is None
    assert s.all_except is not None
    assert 'A*02:01' in s.all_except and 'A*02:19' in s.all_except
    assert 'A*23:01' in s.all_except
    assert len(s.all_except) == 20


def test_parse_class2_locus():
    s = parse_specificities('DRB1*03:01, DQB1*02:01')
    assert s.alleles == {'DRB1*03:01', 'DQB1*02:01'}


def test_parse_lowercase_normalized():
    s = parse_specificities('b*52:01')
    assert s.alleles == {'B*52:01'}


def test_parse_bare_all_covers_everything():
    s = parse_specificities('All')
    assert s.all_except == frozenset()
    assert s.contains('A*01:01')
    assert s.contains('C*07:01')


def test_parse_all_except_unparseable_raises():
    with pytest.raises(ValueError):
        parse_specificities('All, except something weird')


def test_family_of():
    assert family_of('A*31:01') == 'A*31'
    assert family_of('B*13') == 'B*13'


def test_contains_allele_and_family_semantics():
    s = parse_specificities('A*31:01')
    assert s.contains('A*31:01')
    assert not s.contains('A*31:02')
    assert s.contains('A*31')          # family candidate covered by allele lane
    fam = parse_specificities('B*13')
    assert fam.contains('B*13:01')     # allele candidate covered by family lane
    assert fam.contains('B*13:02')
    assert not fam.contains('B*14:01')


def test_contains_all_except():
    s = parse_specificities('All, except A*02:01-19, A*23:01')
    assert s.contains('A*31:01')
    assert not s.contains('A*02:05')
    assert not s.contains('A*23:01')
    assert not s.contains('A*02')      # family with excluded members -> conservative no


@requires_sheet
def test_flc_every_lane_parses_nonempty():
    tray = load_tray_xlsx(FLC)
    for lane in tray.lanes:
        s = parse_specificities(lane.specificities)
        assert (s.alleles or s.all_except is not None), f'mix {lane.mix}: {lane.specificities!r}'
        if s.alleles:
            assert len(s.alleles) > 0, f'mix {lane.mix}'


@requires_sheet
def test_flc_known_lane_semantics():
    tray = load_tray_xlsx(FLC)
    # mix 26 = 'All, except ...'
    s26 = parse_specificities(tray.lane(26).specificities)
    assert s26.all_except is not None
    assert s26.contains('A*31:01')
    assert not s26.contains('A*02:05')
    # Bw6 (mix 70) covers B*42 family; Bw4 (mix 69) does not
    s70 = parse_specificities(tray.lane(70).specificities)
    s69 = parse_specificities(tray.lane(69).specificities)
    assert s70.contains('B*42:01')
    assert not s69.contains('B*42:01')
    # typo mix 30 ('B*51:01-05/08, B*5201-02') parses as B*52:01/02 + B*51 range
    s30 = parse_specificities(tray.lane(30).specificities)
    assert s30.contains('B*52:01') and s30.contains('B*51:03')
    # prose cell (mix 92: 'C*15:05, other C*15 (weak band)') yields both the
    # allele and the family token
    s92 = parse_specificities(tray.lane(92).specificities)
    assert s92.contains('C*15:05')
    assert s92.contains('C*15:02')  # via family token 'C*15'
