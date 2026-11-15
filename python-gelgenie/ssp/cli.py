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
type     Full pipeline: segment a gel image, match bands against a tray sheet
         (xlsx/json), and report HLA allele calls with ambiguity + QC flags.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import rich_click as click

from ssp import __version__
from ssp.analysis import analyse_gel
from ssp.bands import (
    Band,
    assign_lanes_by_edges,
    cluster_lanes_by_gap,
    mask_to_bands,
)
from ssp.inference import GelGenieONNX
from ssp.report import build_report
from ssp.trays import Tray, load_tray_xlsx
from ssp.typing_engine import type_sample

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
    if np.issubdtype(raw.dtype, np.floating):
        return 1.0  # QuPath pixel types report 1.0 as the float upper bound
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
        try:
            edges = [float(x) for x in lane_edges.split(",") if x.strip() != ""]
        except ValueError:
            raise click.BadParameter(
                f"expected comma-separated numbers, got {lane_edges!r}",
                param_hint="--lane-edges",
            )

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


def load_tray(path: str) -> Tray:
    """Load a tray definition from .json (Tray.save_json) or .xlsx (vendor sheet)."""
    if path.lower().endswith(".json"):
        return Tray.load_json(path)
    return load_tray_xlsx(path)


@cli.command(name="type")
@click.argument("image", type=click.Path(exists=True, dir_okay=False))
@click.option("--tray", "-t", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Tray definition (.xlsx vendor sheet or .json).")
@click.option("--model", "-m", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Path to a GelGenie .onnx segmentation model.")
@click.option("--output", "-o", required=True, type=click.Path(file_okay=False),
              help="Output directory for mask.png, typing.json and report.txt.")
@click.option("--invert", is_flag=True, help="Invert intensities before inference.")
@click.option("--percentile-norm", is_flag=True,
              help="Use 0.1/99.9 percentile normalization instead of datatype-max.")
@click.option("--dtype-max", type=float, default=None,
              help="Datatype max for normalization (default: inferred from image dtype).")
@click.option("--min-area", type=int, default=5, show_default=True,
              help="Discard bands with fewer pixels than this (default 5 for "
                   "typing to suppress speckle; `segment` defaults to 1 for raw fidelity).")
@click.option("--lane-gap", type=float, default=20.0, show_default=True,
              help="X-gap (px) separating lane clusters.")
@click.option("--slope-b", type=float, default=None,
              help="Override bp calibration slope (px per ln(bp)); default auto-calibrates.")
@click.option("--match-tol", type=float, default=0.25, show_default=True,
              help="Relative tolerance for matching a band to its lane's product size.")
def type_cmd(image, tray, model, output, invert, percentile_norm, dtype_max,
             min_area, lane_gap, slope_b, match_tol) -> None:
    """Segment a gel image and interpret it against an SSP tray definition.

    Writes mask.png, typing.json (machine-readable) and report.txt (human
    summary) into the output directory, and echoes the summary.
    """
    raw = np.asarray(imageio.imread(image))
    if dtype_max is None:
        dtype_max = _infer_dtype_max(raw)

    try:
        tray_obj = load_tray(tray)
    except ValueError as e:
        raise click.ClickException(str(e))
    model_obj = GelGenieONNX(model_path=model)
    mask, bands = segment_image(
        raw, model_obj, invert=invert,
        data_max_norm=not percentile_norm, dtype_max=dtype_max, min_area=min_area,
    )
    try:
        ga = analyse_gel(bands, tray_obj, lane_gap=lane_gap, slope_b=slope_b,
                         match_tol=match_tol)
    except ValueError as e:
        raise click.ClickException(str(e))
    result = type_sample(ga, tray_obj)

    os.makedirs(output, exist_ok=True)
    mask_path = os.path.join(output, "mask.png")
    imageio.imwrite(mask_path, (mask * 255).astype(np.uint8))

    payload = {
        "image": os.path.basename(image),
        "tray": tray_obj.name,
        "analysis": {
            "slope_b": round(ga.slope_b, 2),
            "panels": [list(p) for p in ga.panels],
            "warnings": ga.warnings,
            "invalid_lanes": [l.mix for l in ga.lanes if not l.valid],
            "positive_lanes": [l.mix for l in ga.lanes if l.positive],
        },
        "typing": result.to_dict(),
    }
    typing_path = os.path.join(output, "typing.json")
    with open(typing_path, "w") as f:
        json.dump(payload, f, indent=2)

    report_path = os.path.join(output, "report.txt")
    with open(report_path, "w") as f:
        f.write(build_report(os.path.basename(image), tray_obj.name, ga, result))

    click.echo(result.summary_text())
    click.echo(f"\n  mask   -> {mask_path}")
    click.echo(f"  typing -> {typing_path}")
    click.echo(f"  report -> {report_path}")


DEFAULT_MODEL_PATH = os.path.expanduser(
    "~/.cache/ssptyping/models/unet_dec_21_epoch_579.onnx")
DEFAULT_DATA_DIR = os.path.expanduser("~/.local/share/ssptyping")


def _run_server(config, host: str, port: int) -> None:
    """Blocking server run; isolated so tests can stub it out."""
    import uvicorn

    from ssp.webapp import create_app
    uvicorn.run(create_app(config), host=host, port=port, log_level="info")


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Interface to bind (local-only by default).")
@click.option("--port", "-p", default=8765, show_default=True, type=int)
@click.option("--model", "-m", default=None,
              help="ONNX model path (default: $SSP_MODEL, else the cached "
                   "GelGenie Universal model).")
@click.option("--data-dir", default=None,
              help=f"Session/tray store directory (default: {DEFAULT_DATA_DIR}).")
@click.option("--no-browser", is_flag=True, help="Do not auto-open a browser.")
def serve(host, port, model, data_dir, no_browser) -> None:
    """Run the local web app: upload gels + tray sheets, inspect and edit
    lane calls interactively, export typing results."""
    model_path = model or os.environ.get("SSP_MODEL") or DEFAULT_MODEL_PATH
    if not os.path.isfile(model_path):
        raise click.ClickException(
            f"model not found: {model_path}\n"
            "Pass --model or download the GelGenie Universal ONNX model:\n"
            "  mkdir -p ~/.cache/ssptyping/models && curl -sL -o "
            "~/.cache/ssptyping/models/unet_dec_21_epoch_579.onnx \\\n"
            "    https://huggingface.co/mattaq/GelGenie-Universal-Dec-2023/"
            "resolve/main/onnx_checkpoints/unet_dec_21_epoch_579.onnx")

    from ssp.webapp import AppConfig
    config = AppConfig(model_path=model_path,
                       data_dir=data_dir or DEFAULT_DATA_DIR)

    if not no_browser:
        import threading
        import webbrowser
        shown = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
        url = f"http://{shown}:{port}/"
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        click.echo(f"opening {url} ...")

    click.echo(f"serving on http://{host}:{port}  (data: {config.data_dir})")
    _run_server(config, host, port)


if __name__ == "__main__":
    cli()
