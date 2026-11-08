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

Image preprocessing for standalone segmentation.

Ports the transforms the QuPath extension applies before running a GelGenie
model, so results match bit-for-bit:

* ``get_individual_padding`` / ``pad_to_divisible`` mirror
  ``dataloaders.ImageMaskDataset.get_individual_padding`` (the Python original
  that ``ModelRunner.getPadding`` and ``DivisibleSizePad`` copied) - even,
  32-divisible zero padding.
* ``normalize_image`` mirrors ``ModelRunner.convertAndNormalizeNonStandardImage``
  (RGB channel averaging, then either datatype-max or 0.1/99.9 percentile
  normalization to [0, 1]).
* ``invert`` mirrors ``ImageInvert`` (1 - x).
* ``squish_channels`` mirrors ``ChannelSquisher`` (mean over the channel axis).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

PadWidth = Tuple[Tuple[int, int], Tuple[int, int]]


def get_individual_padding(height: int, width: int, multiplier: int = 32) -> PadWidth:
    """Return ``((top, bottom), (left, right))`` zero-padding to make an image
    divisible by ``multiplier``.

    Direct port of ``dataloaders.get_individual_padding``: the padded size is
    ``multiplier * (dim // multiplier + 1)``, so a full block is added even when
    the dimension is already divisible (matching the reference and the QuPath
    extension). Padding is split evenly, with any odd remainder going to
    bottom/right.
    """
    new_vert = multiplier * (height // multiplier + 1)
    new_horiz = multiplier * (width // multiplier + 1)

    top = (new_vert - height) // 2
    bottom = new_vert - height - top
    left = (new_horiz - width) // 2
    right = new_horiz - width - left
    return (top, bottom), (left, right)


def pad_to_divisible(arr: np.ndarray, multiplier: int = 32) -> Tuple[np.ndarray, PadWidth]:
    """Zero-pad the last two (H, W) dims of ``arr`` to be divisible by
    ``multiplier``. Works for ``(H, W)`` or ``(..., H, W)`` inputs.

    Returns ``(padded_array, pads)`` where ``pads`` can be passed to
    :func:`remove_padding` to recover the original region.
    """
    if arr.ndim < 2:
        raise ValueError(f"Expected at least 2D array, got shape {arr.shape}")
    height, width = arr.shape[-2], arr.shape[-1]
    (top, bottom), (left, right) = get_individual_padding(height, width, multiplier)
    pad_width = [(0, 0)] * (arr.ndim - 2) + [(top, bottom), (left, right)]
    padded = np.pad(arr, pad_width=pad_width, mode="constant", constant_values=0)
    return padded, ((top, bottom), (left, right))


def remove_padding(arr: np.ndarray, pads: PadWidth) -> np.ndarray:
    """Inverse of :func:`pad_to_divisible`: strip the recorded padding from the
    last two dims of ``arr``."""
    (top, bottom), (left, right) = pads
    height, width = arr.shape[-2], arr.shape[-1]
    bottom_idx = height - bottom if bottom else height
    right_idx = width - right if right else width
    return arr[..., top:bottom_idx, left:right_idx]


def invert(arr: np.ndarray) -> np.ndarray:
    """Mirror of ``ImageInvert``: return ``1 - arr`` (assumes [0, 1] float)."""
    return 1.0 - arr


def squish_channels(arr: np.ndarray) -> np.ndarray:
    """Mirror of ``ChannelSquisher``: average over the leading channel axis of a
    ``(C, H, W)`` array, returning a single ``(H, W)`` plane."""
    return arr.mean(axis=0)


def normalize_image(
    img: np.ndarray,
    data_max_norm: bool = True,
    dtype_max: float = 255.0,
) -> np.ndarray:
    """Normalize a raw image to float32 in [0, 1].

    Port of ``ModelRunner.convertAndNormalizeNonStandardImage``:
      * RGB (``(H, W, C)`` with C > 1) inputs are averaged across channels.
      * ``data_max_norm=True`` divides by ``dtype_max`` (255 for 8-bit, 65535
        for 16-bit) - the QuPath default.
      * ``data_max_norm=False`` applies a 0.1/99.9 percentile stretch and
        shifts to [0, 1] (used when datatype max would squash small values).
    """
    arr = np.asarray(img)

    # average RGB(A) channels into a single grayscale plane
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3].mean(axis=-1)

    arr = arr.astype(np.float32)

    if data_max_norm:
        arr = arr / np.float32(dtype_max)
    else:
        lo, hi = np.percentile(arr, [0.1, 99.9])
        scale = 1.0 / (hi - lo) if hi != lo else 0.0
        arr = (arr - lo) * scale

    return arr.astype(np.float32)
