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

from ssp.trays import Tray, TrayLane, load_tray_xlsx

TESTDATA = Path(os.environ.get(
    'SSP_TESTDATA_DIR',
    str(Path(__file__).resolve().parents[2] / 'qupath-gelgenie' / 'testdata')))
FLC = TESTDATA / 'FLClasseI.xlsx'

requires_sheet = pytest.mark.skipif(not FLC.exists(), reason=f'{FLC} not present')


# ---------- unit tests on a synthetic in-memory tray ----------

def _mini_tray():
    return Tray(name='mini', lanes=[
        TrayLane(mix=1, locus='A', product_bp=[574], control_bp=256,
                 serotypes='1', specificities='A*01:01/02', comment='Weak band'),
        TrayLane(mix=2, locus='A', product_bp=[], control_bp=256,
                 serotypes='36', specificities='A*36:01', comment=''),
    ])


def test_tray_lane_to_dict_roundtrip():
    t = _mini_tray()
    d = t.to_dict()
    s = json.dumps(d)  # must be serializable
    t2 = Tray.from_dict(json.loads(s))
    assert t2 == t


def test_tray_json_file_roundtrip(tmp_path):
    t = _mini_tray()
    p = tmp_path / 'tray.json'
    t.save_json(p)
    t2 = Tray.load_json(p)
    assert t2 == t


def test_tray_lookup_helpers():
    t = _mini_tray()
    assert t.lane(2).specificities == 'A*36:01'
    assert [l.mix for l in t.locus_lanes('A')] == [1, 2]
    assert t.locus_lanes('B') == []
    with pytest.raises(KeyError):
        t.lane(99)


# ---------- integration tests on the real FLC sheet ----------

@requires_sheet
def test_load_flc_lane_count_and_sections():
    t = load_tray_xlsx(FLC, name='FLClasseI')
    assert len(t.lanes) == 96
    assert [l.mix for l in t.lanes] == list(range(1, 97))
    loci = {l.mix: l.locus for l in t.lanes}
    assert all(loci[m] == 'A' for m in range(1, 29))
    assert all(loci[m] == 'B' for m in range(29, 71))
    assert all(loci[m] == 'C' for m in range(71, 97))


@requires_sheet
def test_load_flc_known_rows():
    t = load_tray_xlsx(FLC)
    l1 = t.lane(1)
    assert l1.product_bp == [574]
    assert l1.control_bp == 256
    assert l1.serotypes == '1'
    assert l1.specificities == 'A*01:01/02'
    assert 'Weak band' in l1.comment
    # multi-size cell
    assert t.lane(43).product_bp == [498, 508]
    # sheet quirk: lane 69 has no control-bp cell -> inferred 256 (B section)
    assert t.lane(69).control_bp == 256
    # sheet quirk: lane 64 has no product-size cell -> empty list
    assert t.lane(64).product_bp == []
    # C section control size
    assert t.lane(71).control_bp == 800
    assert t.lane(96).product_bp == [500]


@requires_sheet
def test_load_flc_control_bp_consistency():
    t = load_tray_xlsx(FLC)
    assert {l.control_bp for l in t.locus_lanes('A')} == {256}
    assert {l.control_bp for l in t.locus_lanes('B')} == {256}
    assert {l.control_bp for l in t.locus_lanes('C')} == {800}


@requires_sheet
def test_flc_json_roundtrip(tmp_path):
    t = load_tray_xlsx(FLC, name='FLClasseI')
    p = tmp_path / 'flc.json'
    t.save_json(p)
    assert Tray.load_json(p) == t


# ---------- hardening: malformed sheets and bad bp values ----------

def test_lane_rejects_nonpositive_bp():
    with pytest.raises(ValueError):
        TrayLane(mix=1, locus='A', product_bp=[500], control_bp=0,
                 serotypes='', specificities='A*01:01')
    with pytest.raises(ValueError):
        TrayLane(mix=1, locus='A', product_bp=[500, 0], control_bp=256,
                 serotypes='', specificities='A*01:01')


def _write_xlsx(path, rows):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    for r in rows:
        ws.append(r)
    wb.save(path)


def test_loader_unknown_section_raises(tmp_path):
    p = tmp_path / 'bad.xlsx'
    _write_xlsx(p, [
        ['HLA-A*'], ['Mix (lane)', 'size', 'ctrl', 'ser', 'spec', 'cmt'],
        [1, 500, 256, '1', 'A*01:01'],
        ['HLA-DRB1*'],
        [29, 500, 256, '1', 'DRB1*03:01'],
    ])
    with pytest.raises(ValueError, match='unsupported tray section'):
        load_tray_xlsx(p)


def test_loader_row_before_section_raises(tmp_path):
    p = tmp_path / 'bad.xlsx'
    _write_xlsx(p, [[1, 500, 256, '1', 'A*01:01']])
    with pytest.raises(ValueError, match='before any HLA'):
        load_tray_xlsx(p)


def test_loader_duplicate_mix_raises(tmp_path):
    p = tmp_path / 'bad.xlsx'
    _write_xlsx(p, [
        ['HLA-A*'],
        [1, 500, 256, '1', 'A*01:01'],
        [1, 600, 256, '2', 'A*02:01'],
    ])
    with pytest.raises(ValueError, match='duplicate'):
        load_tray_xlsx(p)


def test_loader_empty_sheet_raises(tmp_path):
    p = tmp_path / 'empty.xlsx'
    _write_xlsx(p, [['HLA-A*'], ['Mix (lane)']])
    with pytest.raises(ValueError, match='no tray rows'):
        load_tray_xlsx(p)


def test_loader_bad_bp_value_raises(tmp_path):
    p = tmp_path / 'bad.xlsx'
    _write_xlsx(p, [['HLA-A*'], [1, 500, 0, '1', 'A*01:01']])
    with pytest.raises(ValueError, match='control_bp'):
        load_tray_xlsx(p)
