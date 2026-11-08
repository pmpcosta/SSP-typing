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

import imageio.v2 as imageio
import numpy as np
from click.testing import CliRunner

import ssp.cli as cli_mod
from ssp.cli import cli, segment_image
from ssp.inference import GelGenieONNX


class _FakeIO:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """foreground (class 1) wins where input channel > 0.5"""

    def get_inputs(self):
        return [_FakeIO("input")]

    def get_outputs(self):
        return [_FakeIO("output")]

    def run(self, out_names, feed):
        x = feed["input"]
        b, c, h, w = x.shape
        out = np.zeros((b, 2, h, w), dtype=np.float32)
        fg = (x[:, 0] > 0.5).astype(np.float32)
        out[:, 1] = fg * 10.0 - 5.0
        out[:, 0] = (1.0 - fg) * 10.0 - 5.0
        return [out]


def _fake_model_factory(model_path=None, **kwargs):
    return GelGenieONNX(session=FakeSession())


def _write_synth_gel(path, h=64, w=96):
    """Dark background with two bright 'bands' in different columns."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[20:30, 15:25] = 255   # band 1 (left)
    img[20:30, 70:80] = 255   # band 2 (right)
    imageio.imwrite(path, img)
    return img


def test_segment_image_helper_returns_mask_and_bands():
    raw = np.zeros((64, 96), dtype=np.uint8)
    raw[20:30, 15:25] = 255
    model = GelGenieONNX(session=FakeSession())
    mask, bands = segment_image(raw, model, invert=False, data_max_norm=True,
                                dtype_max=255, min_area=1)
    assert mask.shape == (64, 96)
    assert len(bands) == 1
    assert bands[0].area == 100


def test_segment_image_lane_edges():
    raw = np.zeros((64, 96), dtype=np.uint8)
    raw[20:30, 15:25] = 255
    raw[20:30, 70:80] = 255
    model = GelGenieONNX(session=FakeSession())
    _, bands = segment_image(raw, model, invert=False, data_max_norm=True,
                             dtype_max=255, min_area=1, lane_edges=[48.0])
    lanes = sorted(b.lane for b in bands)
    assert lanes == [0, 1]


def test_cli_segment_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "GelGenieONNX", _fake_model_factory)
    img_path = tmp_path / "gel.png"
    _write_synth_gel(str(img_path))
    # dummy model file (fake factory ignores its contents but CLI checks existence)
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"not-a-real-model")
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["segment", str(img_path), "--model", str(model_path), "-o", str(out_dir)],
    )
    assert result.exit_code == 0, result.output + (str(result.exception) if result.exception else "")

    mask_file = out_dir / "mask.png"
    bands_file = out_dir / "bands.json"
    assert mask_file.exists()
    assert bands_file.exists()

    data = json.loads(bands_file.read_text())
    assert data["image"] == "gel.png"
    assert data["num_bands"] == 2
    assert len(data["bands"]) == 2
    # mask png is 0/255
    m = imageio.imread(str(mask_file))
    assert set(np.unique(m)).issubset({0, 255})


def test_cli_segment_with_gap_lanes(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "GelGenieONNX", _fake_model_factory)
    img_path = tmp_path / "gel.png"
    _write_synth_gel(str(img_path))
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"x")
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["segment", str(img_path), "-m", str(model_path), "-o", str(out_dir),
         "--gap-threshold", "20"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads((out_dir / "bands.json").read_text())
    lanes = sorted(b["lane"] for b in data["bands"])
    assert lanes == [0, 1]


def test_cli_segment_missing_model_errors(tmp_path):
    img_path = tmp_path / "gel.png"
    _write_synth_gel(str(img_path))
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["segment", str(img_path), "-m", str(tmp_path / "nope.onnx"), "-o", str(tmp_path / "o")],
    )
    assert result.exit_code != 0


def test_cli_segment_malformed_lane_edges_is_clean_error(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "GelGenieONNX", _fake_model_factory)
    img_path = tmp_path / "gel.png"
    _write_synth_gel(str(img_path))
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"x")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["segment", str(img_path), "-m", str(model_path), "-o", str(tmp_path / "o"),
         "--lane-edges", "abc,48"],
    )
    assert result.exit_code != 0
    assert "lane-edges" in result.output
    # must be a clean click error, not a raw traceback
    assert "Traceback" not in result.output


def test_infer_dtype_max():
    from ssp.cli import _infer_dtype_max
    assert _infer_dtype_max(np.zeros((2, 2), dtype=np.uint8)) == 255.0
    assert _infer_dtype_max(np.zeros((2, 2), dtype=np.uint16)) == 65535.0
    assert _infer_dtype_max(np.zeros((2, 2), dtype=np.float32)) == 1.0
    assert _infer_dtype_max(np.zeros((2, 2), dtype=np.float64)) == 1.0
