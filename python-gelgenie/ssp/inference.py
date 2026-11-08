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

ONNX Runtime segmentation inference, standalone (no QuPath / no DJL).

Mirrors the QuPath extension's ``ModelRunner.runDJLModel`` preprocessing and
``GelSegmentationTranslator.processOutput`` post-processing:

    normalize -> [invert] -> add batch/channel dims -> 32-divisible zero pad
    -> model -> strip padding -> per-pixel argmax over the 2 class channels

The published GelGenie ONNX checkpoints have dynamic ``height``/``width`` axes,
so a single padded forward pass replaces the OpenCV 1024x1024 tiling path.

Model output is raw 2-class logits ``(B, 2, H, W)``; argmax over the class axis
gives the per-pixel label (0 = background, 1 = gel band). Ties keep class 0,
matching the strict ``>`` comparison in ``GelSegmentationTranslator``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ssp import preprocessing

try:  # onnxruntime is an optional runtime dependency
    import onnxruntime as ort
except ImportError:  # pragma: no cover - exercised only when ORT is absent
    ort = None


class GelGenieONNX:
    """Thin wrapper around an ONNX Runtime session for GelGenie band
    segmentation models.

    Parameters
    ----------
    model_path:
        Path to a ``.onnx`` checkpoint. Ignored when ``session`` is given.
    session:
        Pre-built session (or test double) exposing ``get_inputs``/
        ``get_outputs``/``run``. Injected mainly for testing.
    providers:
        ONNX Runtime execution providers. Defaults to CPU.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        session=None,
        providers: Optional[list] = None,
    ) -> None:
        if session is not None:
            self.session = session
        else:
            if ort is None:
                raise ImportError(
                    "onnxruntime is required to load a model. "
                    "Install it with `pip install onnxruntime`."
                )
            if model_path is None:
                raise ValueError("Provide either model_path or session.")
            self.session = ort.InferenceSession(
                model_path, providers=providers or ["CPUExecutionProvider"]
            )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict_mask(self, gray: np.ndarray, invert: bool = False) -> np.ndarray:
        """Segment a normalized grayscale image.

        Parameters
        ----------
        gray:
            ``(H, W)`` float array in [0, 1] (see
            :func:`ssp.preprocessing.normalize_image`).
        invert:
            When True, invert intensities (``1 - x``) before inference, matching
            the ``ImageInvert`` transform.

        Returns
        -------
        ``(H, W)`` uint8 mask with values in {0, 1}.
        """
        x = np.asarray(gray)
        if x.ndim != 2:
            raise ValueError(f"Expected a 2D (H, W) grayscale array, got shape {x.shape}")
        x = x.astype(np.float32)

        if invert:
            x = preprocessing.invert(x)

        x = x[None, None]  # (1, 1, H, W)
        padded, pads = preprocessing.pad_to_divisible(x)
        padded = np.ascontiguousarray(padded, dtype=np.float32)

        out = self.session.run([self.output_name], {self.input_name: padded})[0]
        out = preprocessing.remove_padding(out, pads)  # (1, 2, H, W)

        mask = np.argmax(out[0], axis=0).astype(np.uint8)  # (H, W)
        return mask

    def segment(
        self,
        raw_image: np.ndarray,
        invert: bool = False,
        data_max_norm: bool = True,
        dtype_max: float = 255.0,
    ) -> np.ndarray:
        """Convenience path from a raw (uint8/uint16/RGB) image to a binary
        mask: normalize, then :meth:`predict_mask`."""
        gray = preprocessing.normalize_image(
            raw_image, data_max_norm=data_max_norm, dtype_max=dtype_max
        )
        return self.predict_mask(gray, invert=invert)
