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

Tests for ssp.layout_detect (automatic layout suggestion).

Synthetic gels mimic the real photos: wells are empty holes (darker than
the gel where bands are bright), every lane has a control band, some lanes a
product, plus blur and noise. Truth comes from the transform used to draw
them, so detected lane centres and row cuts are checked in image pixels.
"""

import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from click.testing import CliRunner
from scipy.ndimage import gaussian_filter
from skimage.transform import AffineTransform, warp

from ssp.cli import cli
from ssp.layout import Layout, LayoutError, stitch
from ssp.layout_detect import detect_layout, find_seam

PITCH = 30
ROW_H = 260
WELL_Y, CTRL_Y, PROD_Y, DIMER_Y = 20, 170, 120, 215


def _piece(n_rows, lanes, empty=0, positives=(), first_mix=1):
    """Upright gel piece in 'bands bright' polarity (dark background)."""
    slots = lanes + empty
    img = np.full((n_rows * ROW_H, slots * PITCH + 10), 0.25)
    mix = first_mix
    for r in range(n_rows):
        t = r * ROW_H
        for s in range(slots):
            x = 5 + s * PITCH
            img[t + WELL_Y - 3:t + WELL_Y + 4, x + 6:x + PITCH - 6] = 0.05   # well (hole)
            if s < empty:
                continue
            img[t + CTRL_Y - 3:t + CTRL_Y + 4, x + 6:x + PITCH - 6] = 0.95
            img[t + DIMER_Y - 6:t + DIMER_Y + 7, x + 5:x + PITCH - 5] += 0.15
            if mix in positives:
                img[t + PROD_Y - 3:t + PROD_Y + 4, x + 6:x + PITCH - 6] = 0.9
            mix += 1
    return img


def _render(pieces, size, light_background, seed=0):
    """pieces: [(upright, AffineTransform upright->image)] -> uint8 photo."""
    H, W = size
    canvas = np.full((H, W), 0.12)
    for up, tf in pieces:
        warped = warp(up, tf.inverse, output_shape=(H, W), order=1, cval=-1)
        canvas = np.where(warped >= 0, warped, canvas)
    canvas = gaussian_filter(canvas, 1.2)
    canvas += np.random.default_rng(seed).normal(0, 0.02, canvas.shape)
    if light_background:
        canvas = 1.0 - canvas
    return (np.clip(canvas, 0, 1) * 255).astype(np.uint8)


def _truth(tf, empty, lanes, n_rows):
    """Image coords of (lane centre at control line) and the well line per row."""
    out = []
    for r in range(n_rows):
        xs = 5 + (empty + np.arange(lanes) + 0.5) * PITCH
        ctrl = tf(np.c_[xs, np.full(lanes, r * ROW_H + CTRL_Y)])
        well = tf(np.c_[xs, np.full(lanes, r * ROW_H + WELL_Y)])
        out.append((ctrl, well))
    return out


def _lane_x_and_top(block, row, y_img, with_t=False):
    """Detected lane centres (image x) at image height y_img, and the row-top y
    under each lane centre (plus, with ``with_t``, y_img as a fraction of the
    row's height under each lane: 0 at its top cut, 1 at its bottom cut)."""
    w, h = block.size()
    tf = block.transform()
    step = (row.right - row.left) * w / row.lanes
    xs = row.left * w + (np.arange(row.lanes) + 0.5) * step
    top = tf(np.c_[xs, np.full(row.lanes, row.top * h)])
    bot = tf(np.c_[xs, np.full(row.lanes, row.bottom * h)])
    t = (y_img - top[:, 1]) / (bot[:, 1] - top[:, 1])
    x = top[:, 0] + (bot[:, 0] - top[:, 0]) * t
    return (x, top[:, 1], t) if with_t else (x, top[:, 1])


def _assert_matches(block, truth):
    for row, (ctrl, well) in zip(block.rows, truth):
        x_at_ctrl, row_top, t = _lane_x_and_top(block, row, ctrl[:, 1], with_t=True)
        # the control band must be inside this row's crop
        assert np.all((t >= 0) & (t < 1))
        assert np.abs(x_at_ctrl - ctrl[:, 0]).max() < 0.2 * PITCH
        # cut sits just above the wells: never below them, at most ~1 lane above
        assert np.all(row_top < well[:, 1])
        assert np.all(row_top > well[:, 1] - 1.2 * PITCH)


@pytest.mark.parametrize("light", [False, True], ids=["dark-bg", "light-bg"])
def test_detect_rotated_3x32(light):
    up = _piece(3, 32, positives={5, 40, 77})
    tf = AffineTransform(rotation=np.deg2rad(1.5), translation=(70, 50))
    img = _render([(up, tf)], (900, 1150), light)
    det = detect_layout(img, "3x32")
    (block,) = det.layout.blocks
    assert det.warnings == []
    _assert_matches(block, _truth(tf, 0, 32, 3))
    _, man = stitch(img, det.layout, expected_lanes=96)
    assert len(man.lanes) == 96


def test_detect_6x16_offset_tilted_piece_with_empty_wells():
    left = _piece(3, 16, positives={3, 20})
    right = _piece(3, 16, empty=3, positives={50, 70, 90}, first_mix=49)
    tf_l = AffineTransform(translation=(30, 20))
    tf_r = AffineTransform(rotation=np.deg2rad(-1.0), translation=(620, 60))
    img = _render([(left, tf_l), (right, tf_r)], (880, 1250), light_background=True)
    det = detect_layout(img, "6x16")
    assert len(det.layout.blocks) == 2
    _assert_matches(det.layout.blocks[0], _truth(tf_l, 0, 16, 3))
    _assert_matches(det.layout.blocks[1], _truth(tf_r, 3, 16, 3))


def test_wells_cut_off_at_photo_edge_are_inferred_with_warning():
    up = _piece(3, 32)
    tf = AffineTransform(translation=(40, -30))  # first row's wells above the photo
    img = _render([(up, tf)], (760, 1080), light_background=False)
    det = detect_layout(img, "3x32")
    assert any("row 1: wells not visible" in w for w in det.warnings)
    truth = _truth(tf, 0, 32, 3)
    block = det.layout.blocks[0]
    for row, (ctrl, well) in list(zip(block.rows, truth))[1:]:
        _, top = _lane_x_and_top(block, row, ctrl[:, 1])
        assert np.all((top < well[:, 1]) & (top > well[:, 1] - 1.2 * PITCH))
    assert block.rows[0].top == 0.0


def _lane_band_image(n=16, pitch=30, x0=100, faint_last=False, rows=(100, 300)):
    """Band-signal image with n occupied lanes starting at x0 (centre)."""
    img = np.zeros((420, x0 + (n + 4) * pitch))
    for yc in rows:
        for k in range(n):
            c = x0 + k * pitch
            img[yc - 3:yc + 4, c - 9:c + 10] = 0.15 if (faint_last and k == n - 1) else 1.0
    return img


@pytest.mark.parametrize("start, expected", [(0, 0), (-1, 1), (1, -1)])
def test_lane_shift_settled_by_occupancy(start, expected):
    from ssp.layout_detect import _settle_lane_shift
    band = _lane_band_image(faint_last=True)
    shift, occ = _settle_lane_shift(band, 100 + start * 30, 30.0, 16, (100, 300), 200.0)
    assert shift == expected
    assert occ[0] >= 2.0  # after settling, the first lane holds bands


def test_lane_shift_kept_when_ambiguous():
    from ssp.layout_detect import _settle_lane_shift
    band = _lane_band_image(n=18)  # bands beyond both ends: nothing to decide
    shift, _ = _settle_lane_shift(band, 130, 30.0, 16, (100, 300), 200.0)
    assert shift == 0


def test_cut_below_next_rows_bands_is_rejected(monkeypatch):
    """Increasing cuts are not enough: a cut pushed below the next row's
    control line would put two rows' bands in one strip."""
    import ssp.layout_detect as ld
    choose = ld._choose_wells

    def shifted(prof, windows, pitch):
        wells, inferred = choose(prof, windows, pitch)
        wells[1] += ROW_H  # second cut moves a whole row down
        return wells, list(range(len(wells)))  # use as-is (no walk-up)

    monkeypatch.setattr(ld, "_choose_wells", shifted)
    up = _piece(3, 32, positives={5, 40, 77})
    img = _render([(up, AffineTransform(translation=(70, 50)))], (900, 1150), False)
    with pytest.raises(LayoutError, match="row 2: its band row"):
        detect_layout(img, "3x32")


@pytest.mark.parametrize("width", range(1, 13))
def test_seam_search_on_tiny_images(width):
    """Never raises; a seam, if any, is centred in the middle third."""
    x = find_seam(np.random.default_rng(0).random((20, width)))
    if width <= 6:
        assert x is None
    else:
        assert x is None or width // 3 <= x < 2 * width // 3


def test_tiny_6x16_image_raises_layout_error():
    with pytest.raises(LayoutError):
        detect_layout(np.zeros((40, 8), dtype=np.uint8), "6x16")


def test_blank_image_raises_layout_error():
    with pytest.raises(LayoutError):
        detect_layout(np.full((400, 600), 128, dtype=np.uint8), "3x32")


def test_unknown_preset_rejected():
    with pytest.raises(LayoutError, match="supports"):
        detect_layout(np.zeros((10, 10), dtype=np.uint8), "stitched")


def test_cli_preprocess_detects_by_default(tmp_path):
    up = _piece(3, 32)
    tf = AffineTransform(rotation=np.deg2rad(1.0), translation=(60, 40))
    img_path = tmp_path / "raw.png"
    imageio.imwrite(img_path, _render([(up, tf)], (880, 1120), light_background=True))
    r = CliRunner().invoke(cli, ["preprocess", str(img_path), "--preset", "3x32",
                                 "-o", str(tmp_path / "o")])
    assert r.exit_code == 0, r.output
    assert "detected automatically" in r.output
    _assert_matches(Layout.load_json(str(tmp_path / "o" / "layout.json")).blocks[0],
                    _truth(tf, 0, 32, 3))


def test_cli_preprocess_falls_back_to_even_split(tmp_path):
    img_path = tmp_path / "blank.png"
    imageio.imwrite(img_path, np.full((400, 600), 128, dtype=np.uint8))
    r = CliRunner().invoke(cli, ["preprocess", str(img_path), "--preset", "3x32",
                                 "-o", str(tmp_path / "o")])
    assert r.exit_code == 0, r.output
    assert "automatic detection failed" in r.output and "even split" in r.output


# ---------------------------------------------------------------- private real gels

TESTDATA = Path(os.environ.get(
    'SSP_TESTDATA_DIR',
    str(Path(__file__).resolve().parents[2] / 'testdata')))
PRE = TESTDATA / 'preprocessing'
# preset per private gel photo (<name>_before.jpg)
REAL = {"G_26160": "3x32", "Gel_01": "3x32", "Gel_02": "6x16", "Gel_03": "3x32",
        "Gel_04": "3x32", "Gel_05": "6x16", "Gel_06": "6x16"}
present = [n for n in REAL if (PRE / f"{n}_before.jpg").exists()]


def _band_errors(raw, layout):
    """Signed x distance from each clearly segmented band to its nearest lane
    centre (at the band's height), plus the median lane pitch."""
    from scipy import ndimage as ndi
    from ssp.layout_detect import _band_signal, _bands_bright, _gray01
    band = _band_signal(_bands_bright(_gray01(raw), raw), 20)
    errs, pitches = [], []
    for b in layout.blocks:
        for r in b.rows:
            w, h = b.size()
            step = (r.right - r.left) * w / r.lanes
            xs = r.left * w + (np.arange(r.lanes) + 0.5) * step
            tf = b.transform()
            top = tf(np.c_[xs, np.full(r.lanes, r.top * h)])
            bot = tf(np.c_[xs, np.full(r.lanes, r.bottom * h)])
            pitch = float(np.mean(np.diff(top[:, 0])))
            pitches.append(pitch)
            y0, y1 = int(top[:, 1].min()), int(bot[:, 1].max())
            x0, x1 = max(0, int(top[0, 0] - pitch / 2)), int(top[-1, 0] + pitch / 2)
            sub = band[y0:y1, x0:x1]
            lab, _ = ndi.label(sub > np.percentile(sub, 97))
            for sl in ndi.find_objects(lab):
                hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
                if not (0.35 * pitch < ww < pitch and hh < 0.4 * pitch):
                    continue
                cy = y0 + (sl[0].start + sl[0].stop) / 2
                cx = x0 + (sl[1].start + sl[1].stop - 1) / 2
                t = (cy - top[:, 1]) / (bot[:, 1] - top[:, 1])
                xl = top[:, 0] + (bot[:, 0] - top[:, 0]) * t
                errs.append(cx - xl[np.argmin(np.abs(xl - cx))])
    return np.asarray(errs), float(np.median(pitches))


@pytest.mark.parametrize("name", present)
def test_real_gel_detection_puts_bands_in_their_lanes(name):
    """Every gel must yield a usable 96-lane layout whose lane centres sit on
    the bands actually present (independent of any hand-made layout); where
    reference layout exists (hand-fitted, or a detection whose edge lanes
    were checked by eye), no lane may be off by half a lane or more (which
    would renumber mixes)."""
    raw = np.asarray(imageio.imread(PRE / f"{name}_before.jpg"))
    det = detect_layout(raw, REAL[name])
    _, man = stitch(raw, det.layout, expected_lanes=96)
    assert len(man.lanes) == 96
    errs, pitch = _band_errors(raw, det.layout)
    assert len(errs) > 50, name
    assert np.median(np.abs(errs)) < 0.15 * pitch, name
    assert np.abs(errs).max() < 0.35 * pitch, name
    hand_path = PRE / f"{name}.layout.json"
    if hand_path.exists():
        hand = Layout.load_json(str(hand_path))
        for bd, bh in zip(det.layout.blocks, hand.blocks):
            for rd, rh in zip(bd.rows, bh.rows):
                wh, hh = bh.size()
                step = (rh.right - rh.left) * wh / rh.lanes
                xs = rh.left * wh + (np.arange(rh.lanes) + 0.5) * step
                mid = bh.transform()(np.c_[xs, np.full(rh.lanes, (rh.top + rh.bottom) / 2 * hh)])
                xd, _ = _lane_x_and_top(bd, rd, mid[:, 1])
                assert np.abs(xd - mid[:, 0]).max() < 0.5 * step, (name, "lane shift")


MODEL = Path(os.path.expanduser('~/.cache/ssptyping/models/unet_dec_21_epoch_579.onnx'))


@pytest.mark.skipif(
    not ((PRE / 'G_26160_before.jpg').exists() and (TESTDATA / 'G_26160.jpg').exists()
         and (TESTDATA / 'FLClasseI.xlsx').exists() and MODEL.exists()),
    reason='G_26160 photos, tray sheet or model not present')
def test_g26160_detected_layout_types_like_hand_stitched(tmp_path):
    """Fully automatic: detect -> join -> type must equal the hand-joined result."""
    pre = tmp_path / 'pre'
    r = CliRunner().invoke(cli, ['preprocess', str(PRE / 'G_26160_before.jpg'),
                                 '--preset', '3x32', '-o', str(pre)])
    assert r.exit_code == 0, r.output
    assert "detected automatically" in r.output, r.output  # not the even-split fallback
    results = {}
    runs = (('auto', PRE / 'G_26160_before.jpg', ['--layout', str(pre / 'layout.json')]),
            ('ref', TESTDATA / 'G_26160.jpg', []))
    for tag, img, extra in runs:
        out = tmp_path / tag
        r = CliRunner().invoke(cli, ['type', str(img), '-t', str(TESTDATA / 'FLClasseI.xlsx'),
                                     '-o', str(out)] + extra)
        assert r.exit_code == 0, r.output
        results[tag] = json.loads((out / 'typing.json').read_text())
    a, b = results['auto'], results['ref']
    assert a['typing'] == b['typing']
    for k in ('positive_lanes', 'invalid_lanes', 'panels', 'warnings'):
        assert a['analysis'][k] == b['analysis'][k], k


@pytest.mark.skipif(not present, reason="private real gel photos not available")
@pytest.mark.parametrize("name", present)
def test_real_gel_polarity_is_detected(name):
    """G_26160 is a UV photo (bright bands on dark); the others are negative
    exports. Judged both on the raw photo and on the joined strip."""
    from ssp.cli import detect_invert
    from ssp.layout import stitch
    raw = np.asarray(imageio.imread(PRE / f"{name}_before.jpg"))
    strip, _ = stitch(raw, Layout.load_json(str(PRE / f"{name}.layout.json")))
    expected = name != "G_26160"
    assert detect_invert(raw) is expected
    assert detect_invert(strip) is expected


def test_layout_polarity_matches_segmentation_after_the_stretch():
    """88% background at 180, 10% bands at 120, 2% labels at 255: the
    percentile stretch puts the background at ~0.44, below the 0.5 cut, but
    the model sees ~0.71 and inverts. Layout must flip too (CodeRabbit #3)."""
    from ssp.layout_detect import _bands_bright, _gray01
    from ssp.preprocessing import detect_invert, has_light_background
    img = np.full((100, 100), 180, dtype=np.uint8)
    img[:10] = 120
    img[10:12] = 255
    g = _gray01(img)
    assert not has_light_background(g) and detect_invert(img)  # the trap
    gb = _bands_bright(g, img)
    assert gb[5, 50] > gb[50, 50]  # bands brighter than the background
