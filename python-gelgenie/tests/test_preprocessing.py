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

from ssp import preprocessing as pp


def test_get_individual_padding_symmetric():
    # H=100 -> new_vert=32*(3+1)=128 -> top=14,bottom=14
    # W=50  -> new_horiz=32*(1+1)=64 -> left=7,right=7
    assert pp.get_individual_padding(100, 50) == ((14, 14), (7, 7))


def test_get_individual_padding_asymmetric():
    # H=101 -> new_vert=128 -> top=(27)//2=13, bottom=128-101-13=14
    (top, bottom), _ = pp.get_individual_padding(101, 50)
    assert (top, bottom) == (13, 14)
    assert top + bottom == 128 - 101


def test_get_individual_padding_always_adds_block_when_divisible():
    # Faithful to reference: even when divisible, H//32+1 adds a full block.
    (top, bottom), (left, right) = pp.get_individual_padding(64, 32)
    assert top + bottom == 32
    assert left + right == 32


def test_pad_to_divisible_shape_and_zeros():
    arr = np.ones((100, 50), dtype=np.float32)
    padded, pads = pp.pad_to_divisible(arr)
    assert padded.shape == (128, 64)
    assert pads == ((14, 14), (7, 7))
    # corners are zero padding
    assert padded[0, 0] == 0.0
    assert padded[-1, -1] == 0.0
    # interior preserved
    assert padded[14, 7] == 1.0


def test_pad_remove_roundtrip():
    arr = np.random.rand(101, 53).astype(np.float32)
    padded, pads = pp.pad_to_divisible(arr)
    recovered = pp.remove_padding(padded, pads)
    assert recovered.shape == arr.shape
    np.testing.assert_array_equal(recovered, arr)


def test_pad_to_divisible_batched_channels():
    # works on (..., H, W): only last two dims padded
    arr = np.ones((2, 3, 100, 50), dtype=np.float32)
    padded, pads = pp.pad_to_divisible(arr)
    assert padded.shape == (2, 3, 128, 64)
    assert pp.remove_padding(padded, pads).shape == (2, 3, 100, 50)


def test_invert():
    arr = np.array([[0.0, 0.25, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(pp.invert(arr), [[1.0, 0.75, 0.0]])


def test_squish_channels_mean():
    # (C, H, W) -> mean over C, keeping a single channel
    arr = np.stack([np.full((2, 2), 0.0), np.full((2, 2), 2.0), np.full((2, 2), 4.0)])
    out = pp.squish_channels(arr)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, np.full((2, 2), 2.0))


def test_normalize_data_max_norm_uint8():
    arr = np.array([[0, 128, 255]], dtype=np.uint8)
    out = pp.normalize_image(arr, data_max_norm=True, dtype_max=255)
    np.testing.assert_allclose(out, [[0.0, 128 / 255, 1.0]], rtol=1e-5)


def test_normalize_data_max_norm_uint16():
    arr = np.array([[0, 65535]], dtype=np.uint16)
    out = pp.normalize_image(arr, data_max_norm=True, dtype_max=65535)
    np.testing.assert_allclose(out, [[0.0, 1.0]], rtol=1e-5)


def test_normalize_percentile_maps_range_to_0_1():
    rng = np.random.default_rng(0)
    arr = rng.integers(1000, 5000, size=(64, 64)).astype(np.uint16)
    out = pp.normalize_image(arr, data_max_norm=False)
    assert out.dtype == np.float32
    # after 0.1/99.9 percentile stretch, bulk lies within [0,1]
    assert out.min() <= 0.0 + 1e-3
    assert out.max() >= 1.0 - 1e-3


def test_normalize_rgb_channels_averaged():
    # (H, W, 3) input averaged to single channel before normalizing
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[..., 0] = 30
    arr[..., 1] = 60
    arr[..., 2] = 90
    out = pp.normalize_image(arr, data_max_norm=True, dtype_max=255)
    assert out.ndim == 2
    np.testing.assert_allclose(out, np.full((4, 4), 60 / 255), rtol=1e-5)


def test_normalize_rgba_averages_first_three_channels():
    # alpha is dropped, RGB averaged (documented deviation is moot vs Java,
    # which never sees RGBA, but the behavior must be pinned)
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[..., 0] = 30
    arr[..., 1] = 60
    arr[..., 2] = 90
    arr[..., 3] = 255  # alpha ignored
    out = pp.normalize_image(arr, data_max_norm=True, dtype_max=255)
    assert out.ndim == 2
    np.testing.assert_allclose(out, np.full((4, 4), 60 / 255), rtol=1e-5)


def test_normalize_squeezes_single_channel_axis():
    # (H, W, 1) -> (H, W)
    arr = np.full((4, 4, 1), 128, dtype=np.uint8)
    out = pp.normalize_image(arr, data_max_norm=True, dtype_max=255)
    assert out.shape == (4, 4)
    np.testing.assert_allclose(out, np.full((4, 4), 128 / 255), rtol=1e-5)


def test_normalize_percentile_constant_image_gives_zeros():
    # hi == lo guard: constant image -> all zeros, no division by zero
    arr = np.full((8, 8), 1000, dtype=np.uint16)
    out = pp.normalize_image(arr, data_max_norm=False)
    np.testing.assert_array_equal(out, np.zeros((8, 8), dtype=np.float32))


def test_has_light_background_tells_polarity_from_the_background():
    dark = np.full((50, 80), 0.15, dtype=np.float32)
    dark[20:25, 10:70] = 0.9          # bright band on a dark gel
    assert pp.has_light_background(dark) is False
    assert pp.has_light_background(1.0 - dark) is True
    # bands covering a large share of the gel do not flip the call
    busy = np.full((50, 80), 0.2, dtype=np.float32)
    busy[:20] = 0.95
    assert pp.has_light_background(busy) is False
