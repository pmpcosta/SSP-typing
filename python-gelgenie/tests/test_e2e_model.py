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

"""End-to-end tests using the real GelGenie Universal ONNX model.

Skipped automatically when the model is not present locally. Download with:

    curl -sL -o ~/.cache/ssptyping/models/unet_dec_21_epoch_579.onnx \
      https://huggingface.co/mattaq/GelGenie-Universal-Dec-2023/resolve/main/onnx_checkpoints/unet_dec_21_epoch_579.onnx
"""

import os

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from ssp.bands import cluster_lanes_by_gap, mask_to_bands
from ssp.inference import GelGenieONNX

MODEL_PATH = os.path.expanduser("~/.cache/ssptyping/models/unet_dec_21_epoch_579.onnx")

requires_model = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason=f"real ONNX model not found at {MODEL_PATH}",
)

H, W = 420, 640
N_LANES = 8
BAND_ROWS = [60, 120, 180, 240, 300, 360]


def _synthetic_gel(seed=42):
    """Bright bands on dark background; returns (uint8 image, ground-truth list)."""
    rng = np.random.default_rng(seed)
    lane_w = W // N_LANES
    img = np.full((H, W), 25.0)
    img += rng.normal(0, 4.0, (H, W))
    img += np.linspace(0, 15, H)[:, None]

    gt = []
    for lane in range(N_LANES):
        x0 = lane * lane_w + lane_w // 5
        x1 = (lane + 1) * lane_w - lane_w // 5
        for row_i, y in enumerate(BAND_ROWS):
            if (lane + row_i) % 3 == 0:  # skip some bands to vary the pattern
                continue
            profile = np.exp(-0.5 * ((np.arange(H) - y) / 4.0) ** 2)
            img[:, x0:x1] += np.outer(profile, np.full(x1 - x0, rng.uniform(120, 220)))
            gt.append({"lane": lane, "y": float(y), "x0": x0, "x1": x1})

    img = gaussian_filter1d(img, sigma=1.5, axis=0)
    img = gaussian_filter1d(img, sigma=1.0, axis=1)
    return np.clip(img, 0, 255).astype(np.uint8), gt


@requires_model
def test_real_model_segments_synthetic_gel():
    img, gt = _synthetic_gel()
    model = GelGenieONNX(model_path=MODEL_PATH)
    mask = model.segment(img, data_max_norm=True, dtype_max=255)

    bands = mask_to_bands(mask, min_area=5)
    lanes = cluster_lanes_by_gap([b.centroid_x for b in bands], gap_threshold=30)
    for band, lane in zip(bands, lanes):
        band.lane = lane

    # every ground-truth band must have a detection nearby, and no spurious ones
    used = set()
    matched = 0
    for g in gt:
        gx = (g["x0"] + g["x1"]) / 2
        for i, b in enumerate(bands):
            if i in used:
                continue
            if abs(b.centroid_x - gx) + abs(b.centroid_y - g["y"]) < 25:
                used.add(i)
                matched += 1
                assert b.lane == g["lane"]
                break

    assert matched == len(gt), f"matched {matched}/{len(gt)}"
    assert len(bands) == matched, f"spurious detections: {len(bands) - matched}"


@requires_model
def test_real_model_inverted_polarity_finds_nothing():
    """This gel has bright bands on dark; inverting must not produce bands."""
    img, _ = _synthetic_gel(seed=7)
    model = GelGenieONNX(model_path=MODEL_PATH)
    mask = model.segment(img, invert=True, data_max_norm=True, dtype_max=255)
    assert mask.sum() < 0.01 * mask.size  # at most a speck of noise
