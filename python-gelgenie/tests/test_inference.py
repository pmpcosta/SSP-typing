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

import numpy as np
import pytest

from ssp.inference import GelGenieONNX


class _FakeIO:
    def __init__(self, name):
        self.name = name
        self.shape = ["batch", 1, "height", "width"]
        self.type = "tensor(float)"


class FakeSession:
    """Mimics onnxruntime.InferenceSession: foreground (class 1) wins where the
    single input channel is > 0.5. Records the padded input it received."""

    def __init__(self, tie=False):
        self.last_input = None
        self.tie = tie

    def get_inputs(self):
        return [_FakeIO("input")]

    def get_outputs(self):
        return [_FakeIO("output")]

    def run(self, out_names, feed):
        x = feed["input"]
        self.last_input = x
        b, c, h, w = x.shape
        out = np.zeros((b, 2, h, w), dtype=np.float32)
        if self.tie:
            out[:, 0] = 1.0
            out[:, 1] = 1.0
        else:
            fg = (x[:, 0] > 0.5).astype(np.float32)
            out[:, 1] = fg * 10.0 - 5.0
            out[:, 0] = (1.0 - fg) * 10.0 - 5.0
        return [out]


def _bright_square(h=100, w=50):
    img = np.zeros((h, w), dtype=np.float32)
    img[20:40, 10:25] = 1.0  # a bright band region
    return img


def test_predict_mask_shape_and_values():
    sess = FakeSession()
    model = GelGenieONNX(session=sess)
    img = _bright_square()
    mask = model.predict_mask(img)
    assert mask.shape == (100, 50)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})
    # bright region is foreground, rest is background
    assert mask[30, 15] == 1
    assert mask[0, 0] == 0
    assert mask[20:40, 10:25].min() == 1


def test_padding_applied_then_removed():
    sess = FakeSession()
    model = GelGenieONNX(session=sess)
    img = _bright_square(100, 50)
    mask = model.predict_mask(img)
    # session saw the 32-divisible padded tensor (1,1,128,64)
    assert sess.last_input.shape == (1, 1, 128, 64)
    # mask is cropped back to the original size
    assert mask.shape == (100, 50)


def test_invert_flag_flips_foreground():
    sess = FakeSession()
    model = GelGenieONNX(session=sess)
    img = _bright_square()
    mask = model.predict_mask(img, invert=True)
    # after inversion the previously bright band is dark -> background,
    # and the previously dark surround is bright -> foreground
    assert mask[30, 15] == 0
    assert mask[0, 0] == 1


def test_tie_resolves_to_background():
    # GelSegmentationTranslator uses strict '>' so equal scores keep class 0.
    sess = FakeSession(tie=True)
    model = GelGenieONNX(session=sess)
    mask = model.predict_mask(_bright_square())
    assert mask.max() == 0


def test_rejects_non_2d_input():
    model = GelGenieONNX(session=FakeSession())
    with pytest.raises(ValueError):
        model.predict_mask(np.zeros((1, 1, 10, 10), dtype=np.float32))


def test_segment_normalizes_raw_uint8():
    sess = FakeSession()
    model = GelGenieONNX(session=sess)
    raw = np.zeros((64, 64), dtype=np.uint8)
    raw[10:20, 10:20] = 255
    mask = model.segment(raw, data_max_norm=True, dtype_max=255)
    assert mask.shape == (64, 64)
    assert mask[15, 15] == 1
    assert mask[0, 0] == 0


def test_invert_before_padding_keeps_pad_zero():
    """Deliberate deviation from runDJLModel (which inverted AFTER padding,
    making the pad region 1.0): ssp mirrors runOpenCVModel — invert first, so
    zero padding stays zero. This test pins that contract."""
    sess = FakeSession()
    model = GelGenieONNX(session=sess)
    img = _bright_square(100, 50)  # 1.0 at [20:40, 10:25], else 0.0
    model.predict_mask(img, invert=True)
    fed = sess.last_input[0, 0]  # (128, 64); pads: top=14, left=7
    # pad region stays 0.0 even though invert is on
    assert fed[0, 0] == 0.0
    assert fed[-1, -1] == 0.0
    assert fed[:14, :].max() == 0.0
    assert fed[:, :7].max() == 0.0
    # image region is inverted: bright band -> 0, dark surround -> 1
    assert fed[14 + 30, 7 + 15] == 0.0
    assert fed[14 + 0, 7 + 0] == 1.0


def test_infer_padding_is_removed_from_output():
    # output crop correctness for asymmetric padding (odd size)
    sess = FakeSession()
    model = GelGenieONNX(session=sess)
    img = np.zeros((101, 53), dtype=np.float32)
    img[50, 25] = 1.0
    mask = model.predict_mask(img)
    assert mask.shape == (101, 53)
    assert mask[50, 25] == 1
    assert mask.sum() == 1


def test_segment_detects_a_light_background_and_inverts():
    model = GelGenieONNX(session=FakeSession())
    positive = np.zeros((64, 64), dtype=np.uint8)
    positive[10:20, 10:20] = 255
    negative = 255 - positive          # the same gel, dark band on light
    np.testing.assert_array_equal(model.segment(negative), model.segment(positive))
    # an explicit choice still wins over detection
    forced = model.segment(negative, invert=False)
    assert forced[15, 15] == 0 and forced[0, 0] == 1
