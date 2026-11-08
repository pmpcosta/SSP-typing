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

`ssp-typing` command line entry point.

Commands
--------
segment  Run a GelGenie ONNX model over a gel image and write a binary mask
         (PNG) plus a band table (JSON). This is the standalone replacement for
         the QuPath extension's interactive segmentation.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import rich_click as click

from ssp import __version__
from ssp.bands import (
    Band,
    assign_lanes_by_edges,
    cluster_lanes_by_gap,
    mask_to_bands,
)
from ssp.inference import GelGenieONNX

click.rich_click.TEXT_MARKUP = "rich"
click.rich_click.SHOW_ARGUMENTS = True


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "--version", "-V", prog_name="ssp-typing")
def cli() -> None:
    """Standalone SSP HLA typing toolkit (derived from GelGenie)."""


def segment_image(
    raw: np.ndarray,
    model: GelGenieONNX,
    invert: bool = False,
    data_max_norm: bool = True,
    dtype_max: float = 255.0,
    min_area: int = 1,
    lane_edges: Optional[List[float]] = None,
    gap_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, List[Band]]:
    """Core segmentation pipeline: raw image -> binary mask -> banded lanes.

    Kept separate from the CLI command so it can be reused (e.g. by the future
    web app) and unit-tested without file I/O.
    """
    mask = model.segment(raw, invert=invert, data_max_norm=data_max_norm, dtype_max=dtype_max)
    bands = mask_to_bands(mask, min_area=min_area)

    if lane_edges:
        assign_lanes_by_edges(bands, lane_edges)
    elif gap_threshold is not None:
        labels = cluster_lanes_by_gap([b.centroid_x for b in bands], gap_threshold)
        for band, lane in zip(bands, labels):
            band.lane = int(lane)

    return mask, bands


def _infer_dtype_max(raw: np.ndarray) -> float:
    if raw.dtype == np.uint16:
        return 65535.0
    return 255.0


@cli.command()
@click.argument("image", type=click.Path(exists=True, dir_okay=False))
@click.option("--model", "-m", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Path to a GelGenie .onnx segmentation model.")
@click.option("--output", "-o", required=True, type=click.Path(file_okay=False),
              help="Output directory for mask.png and bands.json.")
@click.option("--invert", is_flag=True, help="Invert intensities before inference (dark bands on bright background).")
@click.option("--percentile-norm", is_flag=True,
              help="Use 0.1/99.9 percentile normalization instead of datatype-max.")
@click.option("--dtype-max", type=float, default=None,
              help="Datatype max for normalization (default: inferred from image dtype).")
@click.option("--min-area", type=int, default=1, show_default=True,
              help="Discard bands with fewer pixels than this.")
@click.option("--lane-edges", type=str, default=None,
              help="Comma-separated x-boundaries defining lanes (e.g. '48,96').")
@click.option("--gap-threshold", type=float, default=None,
              help="Cluster bands into lanes by x-gap larger than this (alternative to --lane-edges).")
def segment(image, model, output, invert, percentile_norm, dtype_max, min_area,
            lane_edges, gap_threshold) -> None:
    """Segment a gel image and export a mask + band table."""
    raw = imageio.imread(image)

    if dtype_max is None:
        dtype_max = _infer_dtype_max(np.asarray(raw))

    edges: Optional[List[float]] = None
    if lane_edges:
        edges = [float(x) for x in lane_edges.split(",") if x.strip() != ""]

    model_obj = GelGenieONNX(model_path=model)
    mask, bands = segment_image(
        np.asarray(raw),
        model_obj,
        invert=invert,
        data_max_norm=not percentile_norm,
        dtype_max=dtype_max,
        min_area=min_area,
        lane_edges=edges,
        gap_threshold=gap_threshold,
    )

    os.makedirs(output, exist_ok=True)
    mask_path = os.path.join(output, "mask.png")
    imageio.imwrite(mask_path, (mask * 255).astype(np.uint8))

    result = {
        "image": os.path.basename(image),
        "num_bands": len(bands),
        "bands": [b.to_dict() for b in bands],
    }
    bands_path = os.path.join(output, "bands.json")
    with open(bands_path, "w") as f:
        json.dump(result, f, indent=2)

    click.echo(f"Segmented {os.path.basename(image)}: {len(bands)} band(s)")
    click.echo(f"  mask  -> {mask_path}")
    click.echo(f"  bands -> {bands_path}")


if __name__ == "__main__":
    cli()
