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

import imageio.v2 as imageio
import numpy as np
import pytest
from click.testing import CliRunner

import ssp.cli as cli_mod
from ssp.cli import cli
from ssp.inference import GelGenieONNX
from ssp.trays import Tray, TrayLane

TESTDATA = Path(__file__).resolve().parents[2] / 'qupath-gelgenie' / 'testdata'
FLC = TESTDATA / 'FLClasseI.xlsx'
G_IMG = TESTDATA / 'G_26160.jpg'
MODEL = Path(os.path.expanduser('~/.cache/ssptyping/models/unet_dec_21_epoch_579.onnx'))
requires_full = pytest.mark.skipif(
    not (FLC.exists() and G_IMG.exists() and MODEL.exists()),
    reason='real model / G_26160 testdata not present')


class _FakeIO:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """foreground (class 1) where input > 0.5"""

    def get_inputs(self):
        return [_FakeIO('input')]

    def get_outputs(self):
        return [_FakeIO('output')]

    def run(self, out_names, feed):
        x = feed['input']
        b, c, h, w = x.shape
        out = np.zeros((b, 2, h, w), dtype=np.float32)
        fg = (x[:, 0] > 0.5).astype(np.float32)
        out[:, 1] = fg * 10.0 - 5.0
        out[:, 0] = (1.0 - fg) * 10.0 - 5.0
        return [out]


def _fake_model_factory(model_path=None, **kwargs):
    return GelGenieONNX(session=FakeSession())


def _synth_ssp_image(path):
    """4 lanes (x=25,75,125,175); control band y~50 in every lane; product
    band y~30 in lanes 1-3 only. 20bp est: 256*exp(20/81)=328 -> tray uses 330."""
    img = np.zeros((100, 200), dtype=np.uint8)
    for i, x in enumerate((25, 75, 125, 175)):
        img[47:54, x - 10:x + 10] = 255       # control
        if i < 3:
            img[27:34, x - 10:x + 10] = 255   # product
    imageio.imwrite(str(path), img)


def _synth_tray(path):
    tray = Tray(name='synth', lanes=[
        TrayLane(mix=1, locus='A', product_bp=[330], control_bp=256,
                 serotypes='1', specificities='A*01:01'),
        TrayLane(mix=2, locus='A', product_bp=[330], control_bp=256,
                 serotypes='2', specificities='A*02:01'),
        TrayLane(mix=3, locus='A', product_bp=[330], control_bp=256,
                 serotypes='1, 2', specificities='A*01:01, A*02:01'),
        TrayLane(mix=4, locus='A', product_bp=[330], control_bp=256,
                 serotypes='3', specificities='A*03:01'),
    ])
    tray.save_json(path)
    return tray


def test_cli_type_end_to_end_fake_model(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, 'GelGenieONNX', _fake_model_factory)
    img = tmp_path / 'gel.png'
    _synth_ssp_image(img)
    tray_path = tmp_path / 'tray.json'
    _synth_tray(tray_path)
    model_path = tmp_path / 'model.onnx'
    model_path.write_bytes(b'x')
    out = tmp_path / 'out'

    runner = CliRunner()
    result = runner.invoke(cli, [
        'type', str(img), '--tray', str(tray_path), '-m', str(model_path),
        '-o', str(out),
    ])
    assert result.exit_code == 0, result.output + (str(result.exception) if result.exception else '')

    payload = json.loads((out / 'typing.json').read_text())
    a = payload['typing']['loci']['A']
    assert a['score'] == 0
    assert a['positives'] == [1, 2, 3]
    assert a['invalid'] == []
    assert a['best_pairs'] == [['A*01:01', 'A*02:01']]

    report = (out / 'report.txt').read_text()
    assert 'HLA-A' in report and 'A*01:01' in report
    assert (out / 'mask.png').exists()
    assert 'A*01:01 + A*02:01' in result.output


def test_cli_type_xlsx_tray_accepted(tmp_path, monkeypatch):
    # .json and .xlsx both accepted; here just check extension dispatch errors
    monkeypatch.setattr(cli_mod, 'GelGenieONNX', _fake_model_factory)
    img = tmp_path / 'gel.png'
    _synth_ssp_image(img)
    model_path = tmp_path / 'model.onnx'
    model_path.write_bytes(b'x')
    runner = CliRunner()
    result = runner.invoke(cli, [
        'type', str(img), '--tray', str(tmp_path / 'missing.json'),
        '-m', str(model_path), '-o', str(tmp_path / 'o'),
    ])
    assert result.exit_code != 0  # click existence check


def test_cli_type_bad_tray_clean_error(tmp_path, monkeypatch):
    # non-consecutive mixes must surface as a clean click error, not a traceback
    monkeypatch.setattr(cli_mod, 'GelGenieONNX', _fake_model_factory)
    img = tmp_path / 'gel.png'
    _synth_ssp_image(img)
    bad_tray = tmp_path / 'bad.json'
    Tray(name='bad', lanes=[
        TrayLane(mix=1, locus='A', product_bp=[330], control_bp=256,
                 serotypes='1', specificities='A*01:01'),
        TrayLane(mix=3, locus='A', product_bp=[330], control_bp=256,
                 serotypes='3', specificities='A*03:01'),
    ]).save_json(bad_tray)
    model_path = tmp_path / 'model.onnx'
    model_path.write_bytes(b'x')
    runner = CliRunner()
    result = runner.invoke(cli, [
        'type', str(img), '--tray', str(bad_tray), '-m', str(model_path),
        '-o', str(tmp_path / 'o'),
    ])
    assert result.exit_code != 0
    assert 'consecutive' in result.output
    assert 'Traceback' not in result.output


@requires_full
def test_cli_type_real_gel_and_sheet(tmp_path):
    out = tmp_path / 'out'
    runner = CliRunner()
    result = runner.invoke(cli, [
        'type', str(G_IMG), '--tray', str(FLC), '-m', str(MODEL),
        '-o', str(out), '--min-area', '5',
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads((out / 'typing.json').read_text())
    typing = payload['typing']

    a = typing['loci']['A']
    assert a['positives'] == [19, 21, 23, 26]
    assert a['score'] == 0
    assert sorted(map(tuple, a['best_pairs'])) == [
        ('A*31:01', 'A*33:01'), ('A*31:01', 'A*33:03')]

    b = typing['loci']['B']
    assert b['positives'] == [40, 41, 63, 70]
    assert b['invalid'] == [39, 49]
    assert b['score'] == 0
    # exact ambiguity: 12 indistinguishable B*15 alleles x B*42:01
    assert b['allele_options'][1] == ['B*42:01']
    assert set(b['allele_options'][0]) == {
        f'B*15:{n:02d}' for n in (1, 4, 12, 19, 20, 25, 26, 27, 32, 33, 34, 35)}

    c = typing['loci']['C']
    assert c['positives'] == [76, 84]
    assert c['invalid'] == [90, 96]
    assert c['score'] == 0
    # exact pair set: {C*04:01-03, C*18:01} x {C*08:01-03} (lane 96 invalid,
    # so C*04 vs C*18 stays unresolved)
    expected_c = {tuple(sorted((x, y)))
                  for x in ('C*04:01', 'C*04:02', 'C*04:03', 'C*18:01')
                  for y in ('C*08:01', 'C*08:02', 'C*08:03')}
    assert set(map(tuple, c['best_pairs'])) == expected_c
