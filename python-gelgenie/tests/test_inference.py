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
