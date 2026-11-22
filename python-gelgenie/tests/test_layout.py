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

Tests for ssp.layout: cutting raw gel photos into rows and joining them.

Synthetic gels encode each lane's identity in its band position, so a
mis-cut, a lane shift or a wrong row order shows up as a wrong band height:
band y (from the row top) = 20 + 50 * (mix % 2) + 10 * global_row_index.
Neighbouring lanes therefore differ by 50 px and rows by 10 px.
"""

import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from click.testing import CliRunner
from skimage.transform import AffineTransform, warp

from ssp.analysis import _panel_ranges
from ssp.bands import Band, assign_lanes_by_edges
from ssp.cli import cli
from ssp.layout import (
    Block,
    Layout,
    LayoutError,
    Manifest,
    Row,
    is_prestitched,
    preset_layout,
    rectify_block,
    render_overlay,
    render_stitched,
    stitch,
    stitched_layout,
)

PITCH = 20
ROW_H = 160
BG, WELL, BAND = 30.0, 120.0, 250.0


def _band_y(mix: int, global_row: int) -> int:
    return 20 + 50 * (mix % 2) + 10 * global_row


def _upright_block(n_rows, lanes, first_mix, first_row, empty_wells=0, band_ys=None):
    """Upright gel piece: wells on every slot, bands per used lane at
    ``band_ys(mix, global_row)`` (default: the identity-encoding height)."""
    band_ys = band_ys or (lambda m, r: [_band_y(m, r)])
    slots = lanes + empty_wells
    img = np.full((n_rows * ROW_H, slots * PITCH), BG, dtype=float)
    mix = first_mix
    for r in range(n_rows):
        top = r * ROW_H
        for s in range(slots):
            x0 = s * PITCH + 5
            img[top + 4:top + 9, x0:x0 + PITCH - 10] = WELL
            if s >= empty_wells:
                for y in band_ys(mix, first_row + r):
                    img[top + y - 1:top + y + 2, x0:x0 + PITCH - 10] = BAND
                mix += 1
    return img


def _place(canvas, upright, tform):
    """Paste ``upright`` into ``canvas`` through ``tform`` (upright -> canvas);
    returns the block outline (TL, TR, BR, BL) in canvas pixels."""
    h, w = upright.shape
    shape = canvas.shape
    warped = warp(upright, tform.inverse, output_shape=shape, order=1, preserve_range=True)
    inside = warp(np.ones_like(upright), tform.inverse, output_shape=shape, order=0) > 0.5
    canvas[inside] = warped[inside]
    corners = tform(np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float))
    return [tuple(map(float, p)) for p in corners]


def _gel_3x32(angle_deg=3.0):
    canvas = np.full((620, 760), BG)
    up = _upright_block(3, 32, first_mix=1, first_row=0)
    tform = AffineTransform(rotation=np.deg2rad(angle_deg), translation=(60, 40))
    quad = _place(canvas, up, tform)
    rows = [Row(top=i / 3, bottom=(i + 1) / 3, lanes=32) for i in range(3)]
    layout = Layout(blocks=[Block(quad=quad, rows=rows)], image_size=(760, 620))
    return canvas.astype(np.uint8), layout


def _gel_6x16(band_ys=None):
    """Two pieces: the right one sits lower, is tilted, and has 3 empty wells."""
    canvas = np.full((560, 800), BG)
    left = _upright_block(3, 16, first_mix=1, first_row=0, band_ys=band_ys)
    right = _upright_block(3, 16, first_mix=49, first_row=3, empty_wells=3,
                           band_ys=band_ys)
    q0 = _place(canvas, left, AffineTransform(translation=(20, 20)))
    q1 = _place(canvas, right, AffineTransform(rotation=np.deg2rad(-2), translation=(380, 45)))
    rows0 = [Row(top=i / 3, bottom=(i + 1) / 3, lanes=16) for i in range(3)]
    rows1 = [Row(top=i / 3, bottom=(i + 1) / 3, lanes=16, left=3 / 19, right=1.0)
             for i in range(3)]
    layout = Layout(blocks=[Block(quad=q0, rows=rows0), Block(quad=q1, rows=rows1)],
                    image_size=(800, 560))
    return canvas.astype(np.uint8), layout


def _check_lane_identity(stitched, manifest, rows_per_block):
    """Every lane's band must sit where its mix number says it should."""
    img = stitched.astype(float)
    lanes_per_row = len(manifest.lanes) // len(manifest.strips)
    for lane in manifest.lanes:
        strip = manifest.strips[(lane.mix - 1) // lanes_per_row]
        global_row = strip.block * rows_per_block + strip.row
        c = int(round(lane.center))
        profile = img[15:strip.height, c - 2:c + 3].mean(axis=1)
        y = 15 + int(np.argmax(profile))
        assert abs(y - _band_y(lane.mix, global_row)) <= 2, (
            f"mix {lane.mix}: band at {y}, expected {_band_y(lane.mix, global_row)}")
        assert profile.max() > 180, f"mix {lane.mix}: no band found"


# ---------------------------------------------------------------- presets

def test_presets_have_96_lanes_in_expected_blocks():
    a = preset_layout("3x32", (1600, 1400))
    assert (len(a.blocks), [len(b.rows) for b in a.blocks], a.n_lanes) == (1, [3], 96)
    b = preset_layout("6x16", (1600, 1400))
    assert (len(b.blocks), [len(b_.rows) for b_ in b.blocks], b.n_lanes) == (2, [3, 3], 96)
    assert a.validate(96) == [] and b.validate(96) == []


def test_preset_split_x_moves_block_boundary():
    lay = preset_layout("6x16", (1600, 1400), split_x=760)
    assert lay.blocks[0].quad[1][0] == 760 and lay.blocks[1].quad[0][0] == 760


def test_unknown_preset_rejected():
    with pytest.raises(LayoutError, match="unknown preset"):
        preset_layout("4x24", (100, 100))


def test_prestitched_detection_by_aspect():
    assert is_prestitched((4789, 580))
    assert not is_prestitched((1600, 1400))
    assert not is_prestitched((1944, 2592))


# ---------------------------------------------------------------- stitching

def test_stitch_rotated_3x32_keeps_lane_identity_and_order():
    gel, layout = _gel_3x32()
    stitched, man = stitch(gel, layout, expected_lanes=96)
    assert [l.mix for l in man.lanes] == list(range(1, 97))
    assert [(s.first_mix, s.last_mix) for s in man.strips] == [(1, 32), (33, 64), (65, 96)]
    _check_lane_identity(stitched, man, rows_per_block=3)


def test_stitch_6x16_offset_tilted_blocks_with_empty_wells():
    gel, layout = _gel_6x16()
    stitched, man = stitch(gel, layout, expected_lanes=96)
    assert [(s.block, s.row) for s in man.strips] == [(0, 0), (0, 1), (0, 2),
                                                      (1, 0), (1, 1), (1, 2)]
    _check_lane_identity(stitched, man, rows_per_block=3)


def test_default_gap_makes_each_strip_its_own_analysis_panel():
    gel, layout = _gel_6x16()
    _, man = stitch(gel, layout)
    panels = _panel_ranges([l.center for l in man.lanes])
    assert panels == [(s.first_mix - 1, s.last_mix - 1) for s in man.strips]


def test_explicit_gap_and_strip_positions():
    gel, layout = _gel_3x32(angle_deg=0)
    layout.gap = 7
    stitched, man = stitch(gel, layout)
    s0, s1, s2 = man.strips
    assert s1.x0 - s0.x1 == 7 and s2.x0 - s1.x1 == 7
    assert stitched.shape[1] == s2.x1 == man.image_size[0]


def test_lane_edges_drive_assign_lanes_by_edges():
    gel, layout = _gel_6x16()
    _, man = stitch(gel, layout)
    bands = [Band(id=l.mix, area=10, centroid_y=50, centroid_x=l.center, bbox=(0, 0, 1, 1))
             for l in man.lanes]
    assign_lanes_by_edges(bands, man.lane_edges)
    assert [b.lane for b in bands] == [l.mix - 1 for l in man.lanes]


def test_short_rows_are_padded_with_background():
    gel, layout = _gel_3x32(angle_deg=0)
    layout.blocks[0].rows[2].bottom = 0.9  # last row cut short
    stitched, man = stitch(gel, layout)
    s = man.strips[2]
    assert s.height < stitched.shape[0]
    assert np.all(stitched[s.height:, s.x0:s.x1] == BG)


def test_stitch_preserves_uint16_and_grayscale():
    gel, layout = _gel_3x32(angle_deg=0)
    gel16 = gel.astype(np.uint16) * 257
    stitched, _ = stitch(gel16, layout)
    assert stitched.dtype == np.uint16 and stitched.ndim == 2
    assert stitched.max() == 250 * 257


def test_stitch_rgb_keeps_channels():
    gel, layout = _gel_3x32(angle_deg=0)
    rgb = np.repeat(gel[..., None], 3, axis=-1)
    stitched, man = stitch(rgb, layout)
    assert stitched.shape[2] == 3 and stitched.dtype == np.uint8
    gray, _ = stitch(gel, layout)
    np.testing.assert_array_equal(stitched[..., 0], gray)


def test_stitched_layout_is_identity():
    img = np.random.default_rng(0).integers(0, 255, (60, 600), dtype=np.uint8)
    out, man = stitch(img, stitched_layout((600, 60)))
    assert out is img and man.stitched_input and man.lanes == []


def test_layout_rescaled_to_other_image_size():
    gel, layout = _gel_3x32()
    big = np.kron(gel, np.ones((2, 2), dtype=np.uint8))
    stitched, man = stitch(big, layout)  # layout made for the small image
    assert len(man.lanes) == 96
    small, _ = stitch(gel, layout)
    assert abs(stitched.shape[1] - 2 * small.shape[1]) <= 4


def test_rectify_axis_aligned_block_is_plain_crop():
    img = np.arange(100 * 80, dtype=np.uint16).reshape(100, 80)
    b = Block(quad=[(10, 20), (50, 20), (50, 70), (10, 70)])
    np.testing.assert_array_equal(rectify_block(img, b), img[20:70, 10:50])


# ---------------------------------------------------------------- validation

@pytest.mark.parametrize("quad, msg", [
    ([(0, 0), (10, 0), (10, 10)], "4 corners"),
    ([(0, 0), (0, 10), (10, 10), (10, 0)], "TL, TR, BR, BL"),       # counter-clockwise
    ([(0, 0), (10, 10), (10, 0), (0, 10)], "TL, TR, BR, BL"),       # self-crossing
    ([(0, 0), (500, 0), (500, 10), (0, 10)], "outside the image"),
])
def test_bad_outlines_rejected(quad, msg):
    lay = Layout(blocks=[Block(quad=quad, rows=[Row(0, 1, 96)])], image_size=(100, 100))
    with pytest.raises(LayoutError, match=msg):
        lay.validate()


@pytest.mark.parametrize("row, msg", [
    (Row(top=0.5, bottom=0.5, lanes=96), "top < bottom"),
    (Row(top=0.0, bottom=1.2, lanes=96), "top < bottom"),
    (Row(top=0, bottom=1, lanes=96, left=0.6, right=0.4), "left < right"),
    (Row(top=0, bottom=1, lanes=0), "lanes must be"),
])
def test_bad_rows_rejected(row, msg):
    lay = Layout(blocks=[Block(quad=[(0, 0), (100, 0), (100, 100), (0, 100)], rows=[row])],
                 image_size=(100, 100))
    with pytest.raises(LayoutError, match=msg):
        lay.validate()


def test_lane_count_mismatch_rejected():
    with pytest.raises(LayoutError, match="95 lanes, expected 96"):
        lay = preset_layout("3x32", (1000, 1000))
        lay.blocks[0].rows[0].lanes = 31
        lay.validate(96)


def test_overlap_and_order_warnings():
    lay = preset_layout("3x32", (1000, 1000))
    rows = lay.blocks[0].rows
    rows[0].bottom = 0.4  # overlaps row 2 (starts at 1/3)
    rows.reverse()
    w = lay.validate(96)
    assert any("overlap" in m for m in w) and any("top to bottom" in m for m in w)


def test_degenerate_outline_raises_layout_error():
    b = Block(quad=[(0, 0), (0, 0), (0, 0), (0, 0)])
    with pytest.raises(LayoutError, match="degenerate"):
        b.transform()


# ---------------------------------------------------------------- json

def test_json_roundtrip(tmp_path):
    _, layout = _gel_6x16()
    layout.gap = 12.5
    p = tmp_path / "layout.json"
    layout.save_json(str(p))
    back = Layout.load_json(str(p))
    assert back.to_dict() == layout.to_dict()
    assert json.loads(p.read_text())["order"] == "block-major"


@pytest.mark.parametrize("payload, msg", [
    ("[]", "JSON object"),
    ('{"version": 99, "image_size": [1, 1]}', "unsupported layout version"),
    ('{"blocks": []}', "image_size"),
    ('{"image_size": [10, 10], "blocks": [{"quad": "x"}]}', "invalid block"),
    ('{"image_size": [10, 10], "blocks": [{"quad": [[0,0],[1,0],[1,1],[0,1]], '
     '"rows": [{"top": 0}]}]}', "invalid row"),
    ("{not json", "not valid JSON"),
])
def test_bad_json_rejected(tmp_path, payload, msg):
    p = tmp_path / "bad.json"
    p.write_text(payload)
    with pytest.raises(LayoutError, match=msg):
        Layout.load_json(str(p))


# ---------------------------------------------------------------- rendering

def test_overlay_and_lane_check_render():
    gel, layout = _gel_6x16()
    ov = render_overlay(gel, layout)
    assert ov.shape == gel.shape + (3,) and ov.dtype == np.uint8
    assert (ov[..., 0] != ov[..., 2]).any()  # coloured annotations drawn
    stitched, man = stitch(gel, layout)
    chk = render_stitched(stitched, man)
    assert chk.shape == stitched.shape + (3,)


# ---------------------------------------------------------------- CLI

def test_cli_preprocess_with_layout(tmp_path):
    gel, layout = _gel_6x16()
    img, lay = tmp_path / "gel.png", tmp_path / "in.json"
    imageio.imwrite(img, gel)
    layout.save_json(str(lay))
    out = tmp_path / "out"
    r = CliRunner().invoke(cli, ["preprocess", str(img), "--layout", str(lay), "-o", str(out)])
    assert r.exit_code == 0, r.output
    for name in ("stitched.png", "layout.json", "lanes.json", "overlay.png",
                 "stitched_lanes.png"):
        assert (out / name).exists(), name
    lanes = json.loads((out / "lanes.json").read_text())
    assert [l["mix"] for l in lanes["lanes"]] == list(range(1, 97))
    assert imageio.imread(out / "stitched.png").shape[1] == lanes["image_size"][0]
    # the written layout re-applies to the same result
    r2 = CliRunner().invoke(cli, ["preprocess", str(img), "--layout",
                                  str(out / "layout.json"), "-o", str(tmp_path / "again")])
    assert r2.exit_code == 0, r2.output
    assert (tmp_path / "again" / "lanes.json").read_text() == (out / "lanes.json").read_text()


def test_cli_preprocess_preset_warns_to_check(tmp_path):
    gel, _ = _gel_3x32()
    img = tmp_path / "gel.png"
    imageio.imwrite(img, gel)
    r = CliRunner().invoke(cli, ["preprocess", str(img), "--preset", "3x32",
                                 "--no-detect", "-o", str(tmp_path / "o")])
    assert r.exit_code == 0, r.output
    assert "even split" in r.output and "check overlay.png" in r.output


def test_cli_preprocess_passes_through_joined_strip(tmp_path):
    strip = np.random.default_rng(1).integers(0, 255, (50, 400), dtype=np.uint8)
    img = tmp_path / "strip.png"
    imageio.imwrite(img, strip)
    r = CliRunner().invoke(cli, ["preprocess", str(img), "-o", str(tmp_path / "o")])
    assert r.exit_code == 0, r.output
    assert "already a joined strip" in r.output
    np.testing.assert_array_equal(imageio.imread(tmp_path / "o" / "stitched.png"), strip)
    assert not (tmp_path / "o" / "overlay.png").exists()


def test_cli_preprocess_raw_photo_needs_preset(tmp_path):
    gel, _ = _gel_3x32()
    img = tmp_path / "gel.png"
    imageio.imwrite(img, gel)
    r = CliRunner().invoke(cli, ["preprocess", str(img), "-o", str(tmp_path / "o")])
    assert r.exit_code == 2 and "--preset" in r.output


def test_cli_preprocess_rejects_preset_with_layout(tmp_path):
    gel, layout = _gel_3x32()
    img, lay = tmp_path / "gel.png", tmp_path / "l.json"
    imageio.imwrite(img, gel)
    layout.save_json(str(lay))
    r = CliRunner().invoke(cli, ["preprocess", str(img), "--preset", "3x32",
                                 "--layout", str(lay), "-o", str(tmp_path / "o")])
    assert r.exit_code == 2 and "not both" in r.output


def test_cli_preprocess_lane_count_mismatch_is_clean_error(tmp_path):
    gel, layout = _gel_3x32()
    layout.blocks[0].rows[0].lanes = 30
    img, lay = tmp_path / "gel.png", tmp_path / "l.json"
    imageio.imwrite(img, gel)
    layout.save_json(str(lay))
    r = CliRunner().invoke(cli, ["preprocess", str(img), "--layout", str(lay),
                                 "-o", str(tmp_path / "o")])
    assert r.exit_code == 1 and "94 lanes, expected 96" in r.output
    r = CliRunner().invoke(cli, ["preprocess", str(img), "--layout", str(lay),
                                 "--lanes", "0", "-o", str(tmp_path / "o")])
    assert r.exit_code == 0, r.output


# ---------------------------------------------------------------- manifest

def test_manifest_json_roundtrip_and_lane_map(tmp_path):
    gel, layout = _gel_6x16()
    _, man = stitch(gel, layout)
    p = tmp_path / "lanes.json"
    man.save_json(str(p))
    back = Manifest.load_json(str(p))
    lm = back.to_lane_map()
    assert len(lm.slots) == 96
    assert lm.panels == [(0, 15), (16, 31), (32, 47), (48, 63), (64, 79), (80, 95)]
    assert lm.centers == pytest.approx([l.center for l in man.lanes], abs=0.01)


def test_manifest_of_joined_input_has_no_lane_map():
    img = np.zeros((10, 100), dtype=np.uint8)
    _, man = stitch(img, stitched_layout((100, 10)))
    assert Manifest.from_dict(man.to_dict()).to_lane_map() is None


@pytest.mark.parametrize("payload, msg", [
    ({"version": 7, "image_size": [1, 1]}, "unsupported"),
    ({"lanes": []}, "invalid lanes.json"),
    ({"image_size": [10, 10], "lanes": [{"mix": 2, "block": 0, "row": 0, "x0": 0, "x1": 1}]},
     "1..N"),
])
def test_bad_manifest_rejected(payload, msg):
    with pytest.raises(LayoutError, match=msg):
        Manifest.from_dict(payload)


# ---------------------------------------------------------------- type with known layout

CTRL_Y, PROD_Y = 110, 55  # product 55 px above control: 256*exp(55/81) ~ 504 bp
EMPTY = 49  # first lane after the seam: undetectable from band gaps alone
POSITIVE = {m for m in range(1, 97) if m % 7 == 0} - {EMPTY}


def _typing_gel():
    def band_ys(mix, _row):
        if mix == EMPTY:
            return []
        return [CTRL_Y, PROD_Y] if mix in POSITIVE else [CTRL_Y]
    return _gel_6x16(band_ys=band_ys)


def _typing_tray(path):
    from ssp.trays import Tray, TrayLane
    tray = Tray(name="synth96", lanes=[
        TrayLane(mix=m, locus="A", product_bp=[500], control_bp=256,
                 serotypes="", specificities="A*01:01") for m in range(1, 97)])
    tray.save_json(str(path))


@pytest.fixture
def fake_model(monkeypatch, tmp_path):
    import ssp.cli as cli_mod
    from test_cli_type import _fake_model_factory
    monkeypatch.setattr(cli_mod, "GelGenieONNX", _fake_model_factory)
    m = tmp_path / "model.onnx"
    m.write_bytes(b"x")
    return m


def _check_typing(out):
    payload = json.loads((out / "typing.json").read_text())
    a = payload["analysis"]
    assert a["lane_assignment"] == "layout"
    assert a["invalid_lanes"] == [EMPTY]
    assert set(a["positive_lanes"]) == POSITIVE
    assert a["panels"] == [[1, 16], [17, 32], [33, 48], [49, 64], [65, 80], [81, 96]]
    return payload


def test_cli_type_raw_photo_with_layout(tmp_path, fake_model):
    gel, layout = _typing_gel()
    img, lay, tray = tmp_path / "raw.png", tmp_path / "layout.json", tmp_path / "tray.json"
    imageio.imwrite(img, gel)
    layout.save_json(str(lay))
    _typing_tray(tray)
    out = tmp_path / "out"
    r = CliRunner().invoke(cli, ["type", str(img), "-t", str(tray), "-m", str(fake_model),
                                 "--layout", str(lay), "-o", str(out)])
    assert r.exit_code == 0, r.output + repr(r.exception)
    _check_typing(out)
    stitched = imageio.imread(out / "stitched.png")
    assert imageio.imread(out / "mask.png").shape == stitched.shape[:2]
    assert json.loads((out / "lanes.json").read_text())["image_size"] == \
        [stitched.shape[1], stitched.shape[0]]
    # without the layout geometry the empty lane after the seam breaks numbering
    r = CliRunner().invoke(cli, ["type", str(out / "stitched.png"), "-t", str(tray),
                                 "-m", str(fake_model), "-o", str(tmp_path / "gaps")])
    assert r.exit_code == 1 and "expected 96" in r.output


def test_cli_preprocess_then_type_with_lanes(tmp_path, fake_model):
    gel, layout = _typing_gel()
    img, lay, tray = tmp_path / "raw.png", tmp_path / "layout.json", tmp_path / "tray.json"
    imageio.imwrite(img, gel)
    layout.save_json(str(lay))
    _typing_tray(tray)
    pre = tmp_path / "pre"
    r = CliRunner().invoke(cli, ["preprocess", str(img), "--layout", str(lay), "-o", str(pre)])
    assert r.exit_code == 0, r.output
    out = tmp_path / "out"
    r = CliRunner().invoke(cli, ["type", str(pre / "stitched.png"), "-t", str(tray),
                                 "-m", str(fake_model), "--lanes", str(pre / "lanes.json"),
                                 "-o", str(out)])
    assert r.exit_code == 0, r.output + repr(r.exception)
    _check_typing(out)
    assert not (out / "stitched.png").exists()


def test_cli_type_lanes_must_match_image(tmp_path, fake_model):
    gel, layout = _typing_gel()
    _, man = stitch(gel, layout)
    lanes, tray, img = tmp_path / "lanes.json", tmp_path / "tray.json", tmp_path / "x.png"
    man.save_json(str(lanes))
    _typing_tray(tray)
    imageio.imwrite(img, np.zeros((50, 500), dtype=np.uint8))
    r = CliRunner().invoke(cli, ["type", str(img), "-t", str(tray), "-m", str(fake_model),
                                 "--lanes", str(lanes), "-o", str(tmp_path / "o")])
    assert r.exit_code == 1 and "describes a" in r.output
    r = CliRunner().invoke(cli, ["type", str(img), "-t", str(tray), "-m", str(fake_model),
                                 "--lanes", str(lanes), "--layout", str(lanes),
                                 "-o", str(tmp_path / "o")])
    assert r.exit_code == 2 and "not both" in r.output


def test_cli_type_layout_lane_count_must_match_tray(tmp_path, fake_model):
    gel, layout = _typing_gel()
    img, lay, tray = tmp_path / "raw.png", tmp_path / "layout.json", tmp_path / "tray.json"
    imageio.imwrite(img, gel)
    layout.blocks[1].rows[2].lanes = 15
    layout.save_json(str(lay))
    _typing_tray(tray)
    r = CliRunner().invoke(cli, ["type", str(img), "-t", str(tray), "-m", str(fake_model),
                                 "--layout", str(lay), "-o", str(tmp_path / "o")])
    assert r.exit_code == 1 and "95 lanes, expected 96" in r.output


# ---------------------------------------------------------------- private real gels

TESTDATA = Path(os.environ.get(
    'SSP_TESTDATA_DIR',
    str(Path(__file__).resolve().parents[2] / 'testdata')))
PRE = TESTDATA / 'preprocessing'
MODEL = Path(os.path.expanduser('~/.cache/ssptyping/models/unet_dec_21_epoch_579.onnx'))
REAL_LAYOUTS = sorted(PRE.glob('*.layout.json')) if PRE.is_dir() else []


@pytest.mark.parametrize("layout_path", REAL_LAYOUTS, ids=lambda p: p.name)
def test_real_gel_layouts_stitch_96_lanes(layout_path):
    """Hand-fitted layouts for the private before/after pairs."""
    raw = np.asarray(imageio.imread(PRE / layout_path.name.replace('.layout.json', '_before.jpg')))
    layout = Layout.load_json(str(layout_path))
    stitched, man = stitch(raw, layout, expected_lanes=96)
    n_strips = sum(len(b.rows) for b in layout.blocks)
    assert len(man.strips) == n_strips and len(man.lanes) == 96
    assert stitched.shape[1] == man.image_size[0]


@pytest.mark.skipif(
    not ((PRE / 'G_26160.layout.json').exists() and (TESTDATA / 'G_26160.jpg').exists()
         and (TESTDATA / 'FLClasseI.xlsx').exists() and MODEL.exists()),
    reason='G_26160 raw photo + layout, tray sheet or model not present')
def test_g26160_raw_photo_types_like_hand_stitched(tmp_path):
    """Typing the raw photo (joined, and via --layout geometry) must match
    typing the hand-joined image."""
    raw = np.asarray(imageio.imread(PRE / 'G_26160_before.jpg'))
    stitched, _ = stitch(raw, Layout.load_json(str(PRE / 'G_26160.layout.json')), 96)
    mine = tmp_path / 'mine.png'
    imageio.imwrite(mine, stitched)
    results = {}
    runs = (('mine', mine, []), ('ref', TESTDATA / 'G_26160.jpg', []),
            ('layout', PRE / 'G_26160_before.jpg', ['--layout', str(PRE / 'G_26160.layout.json')]))
    for tag, img, extra in runs:
        out = tmp_path / tag
        r = CliRunner().invoke(cli, ['type', str(img), '-t', str(TESTDATA / 'FLClasseI.xlsx'),
                                     '-o', str(out)] + extra)
        assert r.exit_code == 0, r.output
        results[tag] = json.loads((out / 'typing.json').read_text())
    ref = results['ref']
    for tag in ('mine', 'layout'):
        a = results[tag]
        assert a['typing'] == ref['typing'], tag
        for k in ('positive_lanes', 'invalid_lanes', 'panels', 'warnings'):
            assert a['analysis'][k] == ref['analysis'][k], (tag, k)
    assert results['layout']['analysis']['lane_assignment'] == 'layout'
