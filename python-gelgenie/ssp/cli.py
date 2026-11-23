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
preprocess
         Cut a raw gel photo (e.g. 3x32 or 6x16 lane blocks) into its lane
         rows and join them into the single 96-lane strip typing expects,
         with overlay images for visual confirmation.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import rich_click as click

from ssp import __version__
from ssp import models as model_zoo
from ssp.analysis import analyse_gel
from ssp.bands import (
    Band,
    assign_lanes_by_edges,
    cluster_lanes_by_gap,
    mask_to_bands,
)
from ssp.inference import GelGenieONNX
from ssp.layout import (
    PRESETS,
    STITCHED,
    Layout,
    LayoutError,
    Manifest,
    is_prestitched,
    preset_layout,
    render_overlay,
    render_stitched,
    stitch,
)
from ssp.layout_detect import detect_layout
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


def _resolve_model(explicit: Optional[str]):
    """Resolve --model/$SSP_MODEL/cache to an existing path or fail cleanly."""
    try:
        return model_zoo.resolve(explicit)
    except model_zoo.ModelFetchError as e:
        raise click.ClickException(str(e))


def _download_with_progress(name: str):
    """Fetch registry and download ``name`` into the cache with a progress bar."""
    reg = model_zoo.fetch_registry()
    if name not in reg:
        raise model_zoo.ModelFetchError(
            f"unknown model {name!r}; available: {', '.join(sorted(reg))}")
    info = reg[name]
    dest = info.local_path()
    if dest.is_file():
        return dest
    try:
        from rich.progress import (BarColumn, DownloadColumn, Progress,
                                   TimeRemainingColumn, TransferSpeedColumn)
        with Progress(BarColumn(), DownloadColumn(), TransferSpeedColumn(),
                      TimeRemainingColumn()) as prog:
            task = prog.add_task(f"downloading {info.abbrv}", total=None)

            def hook(done, total):
                if total:
                    prog.update(task, total=total, completed=done)
                else:
                    prog.update(task, completed=done)

            return model_zoo.download_model(info, progress=hook)
    except ImportError:      # rich missing: plain download
        return model_zoo.download_model(info)


@cli.group()
def models() -> None:
    """Manage GelGenie segmentation models (registry, cache, downloads)."""


@models.command("list")
def models_list() -> None:
    """List zoo models and whether each is cached locally."""
    try:
        reg = model_zoo.fetch_registry()
    except model_zoo.ModelFetchError as e:
        raise click.ClickException(str(e))
    for name in sorted(reg):
        info = reg[name]
        path = info.local_path()
        state = f"cached: {path}" if path.is_file() else "not downloaded"
        default = " (default)" if name == model_zoo.DEFAULT_MODEL_NAME else ""
        click.echo(f"{name}{default}\n    {info.abbrv} — {info.description}\n    {state}")


@models.command("download")
@click.argument("name", required=False, default=model_zoo.DEFAULT_MODEL_NAME)
def models_download(name) -> None:
    """Download a model into the local cache (default: the Universal model)."""
    try:
        dest = _download_with_progress(name)
    except model_zoo.ModelFetchError as e:
        raise click.ClickException(str(e))
    click.echo(f"saved: {dest}")


@models.command("path")
@click.argument("name", required=False, default=model_zoo.DEFAULT_MODEL_NAME)
def models_path(name) -> None:
    """Print the local cache path for a model (download it if missing)."""
    try:
        click.echo(_download_with_progress(name))
    except model_zoo.ModelFetchError as e:
        raise click.ClickException(str(e))


@cli.command()
@click.argument("image", type=click.Path(exists=True, dir_okay=False))
@click.option("--model", "-m", default=None,
              help="GelGenie .onnx model path (default: $SSP_MODEL or the cached "
                   "Universal model — see `ssp-typing models`).")
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

    model_obj = GelGenieONNX(model_path=str(_resolve_model(model)))
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
@click.option("--model", "-m", default=None,
              help="GelGenie .onnx model path (default: $SSP_MODEL or the cached "
                   "Universal model — see `ssp-typing models`).")
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
@click.option("--layout", "layout_path", type=click.Path(exists=True, dir_okay=False),
              default=None,
              help="IMAGE is a raw gel photo: join it with this layout.json (from "
                   "`preprocess`) first, and take lanes from the known geometry.")
@click.option("--lanes", "lanes_path", type=click.Path(exists=True, dir_okay=False),
              default=None,
              help="IMAGE is a joined strip from `preprocess`: take lanes from its "
                   "lanes.json instead of inferring them from band gaps.")
def type_cmd(image, tray, model, output, invert, percentile_norm, dtype_max,
             min_area, lane_gap, slope_b, match_tol, layout_path, lanes_path) -> None:
    """Segment a gel image and interpret it against an SSP tray definition.

    Writes mask.png, typing.json (machine-readable) and report.txt (human
    summary) into the output directory, and echoes the summary. With
    --layout, the joined strip (stitched.png) and lanes.json are written too,
    and mask.png refers to the joined strip.
    """
    if layout_path and lanes_path:
        raise click.UsageError("use either --layout or --lanes, not both")
    raw = np.asarray(imageio.imread(image))

    try:
        tray_obj = load_tray(tray)
    except ValueError as e:
        raise click.ClickException(str(e))

    manifest: Optional[Manifest] = None
    try:
        if layout_path:
            layout = Layout.load_json(layout_path)
            raw, manifest = stitch(raw, layout, expected_lanes=len(tray_obj.lanes))
        elif lanes_path:
            manifest = Manifest.load_json(lanes_path)
            size = (raw.shape[1], raw.shape[0])
            if not manifest.stitched_input and tuple(manifest.image_size) != size:
                raise LayoutError(
                    f"{os.path.basename(lanes_path)} describes a "
                    f"{manifest.image_size[0]}x{manifest.image_size[1]} image but "
                    f"{os.path.basename(image)} is {size[0]}x{size[1]}")
    except LayoutError as e:
        raise click.ClickException(f"layout: {e}")
    lane_map = manifest.to_lane_map() if manifest else None

    if dtype_max is None:
        dtype_max = _infer_dtype_max(raw)
    model_obj = GelGenieONNX(model_path=str(_resolve_model(model)))
    mask, bands = segment_image(
        raw, model_obj, invert=invert,
        data_max_norm=not percentile_norm, dtype_max=dtype_max, min_area=min_area,
    )
    try:
        ga = analyse_gel(bands, tray_obj, lane_gap=lane_gap, slope_b=slope_b,
                         match_tol=match_tol, lane_map=lane_map)
    except ValueError as e:
        raise click.ClickException(str(e))
    result = type_sample(ga, tray_obj)

    os.makedirs(output, exist_ok=True)
    mask_path = os.path.join(output, "mask.png")
    imageio.imwrite(mask_path, (mask * 255).astype(np.uint8))
    extra_paths = []
    if layout_path:
        stitched_path = os.path.join(output, "stitched.png")
        imageio.imwrite(stitched_path, raw)
        lanes_out = os.path.join(output, "lanes.json")
        manifest.save_json(lanes_out)
        extra_paths = [("joined", stitched_path), ("lanes", lanes_out)]

    payload = {
        "image": os.path.basename(image),
        "tray": tray_obj.name,
        "analysis": {
            "slope_b": round(ga.slope_b, 2),
            "lane_assignment": "layout" if lane_map else "band-gaps",
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
    for name, p in extra_paths:
        click.echo(f"  {name:6s} -> {p}")


@cli.command()
@click.argument("image", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", required=True, type=click.Path(file_okay=False),
              help="Output directory for stitched.png, layout.json, lanes.json "
                   "and the overlay check images.")
@click.option("--preset", type=click.Choice(list(PRESETS) + [STITCHED]), default=None,
              help="Gel format: 3x32 (one gel, 3 rows of 32), 6x16 (two gel pieces "
                   "side by side, 3 rows of 16 each) or stitched (already joined; "
                   "copied unchanged). Outlines and row cuts are detected "
                   "automatically; always confirm them on overlay.png.")
@click.option("--detect/--no-detect", default=True, show_default=True,
              help="With --preset: detect gel outlines and row cuts (default), or "
                   "start from an even split of the whole photo.")
@click.option("--layout", "layout_path", type=click.Path(exists=True, dir_okay=False),
              default=None, help="Layout JSON to apply (e.g. an adjusted layout.json "
                                 "from an earlier run). Rescaled if the image size differs.")
@click.option("--lanes", "expected_lanes", type=int, default=96, show_default=True,
              help="Expected total lane count; 0 skips the check.")
@click.option("--gap", type=float, default=None,
              help="Background gap between joined strips in px (default: 2 lane pitches).")
def preprocess(image, output, preset, layout_path, detect, expected_lanes, gap) -> None:
    """Rearrange a raw gel photo into a single joined lane strip.

    Writes stitched.png (input for `type`), layout.json (the cut geometry;
    edit it and pass it back with --layout), lanes.json (where each mix
    landed), overlay.png (the cuts drawn on the original) and
    stitched_lanes.png (the joined strip with mix numbers). Check both
    images before typing.
    """
    if preset and layout_path:
        raise click.UsageError("use either --preset or --layout, not both")
    raw = np.asarray(imageio.imread(image))
    size = (raw.shape[1], raw.shape[0])
    notes: List[str] = []
    source = "layout file" if layout_path else None

    try:
        if layout_path:
            layout = Layout.load_json(layout_path)
            if tuple(layout.image_size) != size:
                click.echo(f"note: layout was made for {layout.image_size[0]}x"
                           f"{layout.image_size[1]}; rescaled to {size[0]}x{size[1]}")
                layout = layout.scaled_to(size)
        elif preset and preset != STITCHED and detect:
            try:
                det = detect_layout(raw, preset)
                layout, source = det.layout, "detected"
                notes += det.warnings
            except LayoutError as e:
                layout, source = preset_layout(preset, size), "even split"
                notes.append(f"automatic detection failed ({e}); using an even split")
        elif preset:
            layout = preset_layout(preset, size)
            source = "even split"
        elif is_prestitched(size):
            layout = preset_layout(STITCHED, size)
        else:
            raise click.UsageError(
                f"{os.path.basename(image)} ({size[0]}x{size[1]}) looks like a raw gel "
                f"photo; choose --preset ({', '.join(PRESETS)}) or pass --layout")
        if gap is not None:
            layout.gap = gap
        warnings = notes + layout.validate(expected_lanes or None)
        stitched, manifest = stitch(raw, layout)
    except LayoutError as e:
        raise click.ClickException(f"layout: {e}")

    os.makedirs(output, exist_ok=True)
    paths = {
        "stitched": os.path.join(output, "stitched.png"),
        "layout": os.path.join(output, "layout.json"),
        "lanes": os.path.join(output, "lanes.json"),
        "overlay": os.path.join(output, "overlay.png"),
        "lane check": os.path.join(output, "stitched_lanes.png"),
    }
    imageio.imwrite(paths["stitched"], stitched)
    layout.save_json(paths["layout"])
    with open(paths["lanes"], "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    if layout.is_stitched:
        del paths["overlay"], paths["lane check"]
        click.echo(f"{os.path.basename(image)} is already a joined strip; copied unchanged.")
    else:
        imageio.imwrite(paths["overlay"], render_overlay(raw, layout))
        imageio.imwrite(paths["lane check"], render_stitched(stitched, manifest))
        click.echo(f"Joined {len(manifest.strips)} strip(s), {len(manifest.lanes)} lanes "
                   f"-> {stitched.shape[1]}x{stitched.shape[0]}")
        if source == "even split":
            click.echo("Layout is an even split of the photo: check overlay.png, "
                       "adjust layout.json and re-run with --layout.")
        elif source == "detected":
            click.echo("Layout was detected automatically: confirm the cuts and lane "
                       "numbers on overlay.png before typing; if anything is off, edit "
                       "layout.json and re-run with --layout.")
    for w in warnings:
        click.echo(f"warning: {w}")
    for name, p in paths.items():
        click.echo(f"  {name:10s} -> {p}")


DEFAULT_DATA_DIR = os.path.expanduser("~/.local/share/ssptyping")


def _run_server(config, host: str, port: int) -> None:
    """Blocking server run; isolated so tests can stub it out."""
    try:
        import uvicorn

        from ssp.webapp import create_app
    except ImportError as e:
        raise click.ClickException(
            f"web app dependency missing ({e.name}); install with: "
            "pip install 'GelGenie[webapp]'")
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
    try:
        resolved = model_zoo.resolve(model)
    except model_zoo.ModelFetchError as e:
        if model or os.environ.get("SSP_MODEL"):
            raise click.ClickException(str(e))  # explicit source broken: fail
        # first run: fetch the default model so `serve` just works
        click.echo("No model cached — downloading the GelGenie Universal "
                   "model (~57 MB)...")
        try:
            resolved = _download_with_progress(model_zoo.DEFAULT_MODEL_NAME)
        except model_zoo.ModelFetchError as e2:
            dest = model_zoo.cache_dir() / model_zoo.DEFAULT_MODEL_FILE
            raise click.ClickException(
                f"could not download model: {e2}\n\n"
                "Manual download:\n"
                f"  mkdir -p {dest.parent} && curl -sL -o {dest} \\\n"
                "    https://huggingface.co/mattaq/GelGenie-Universal-Dec-2023/"
                "resolve/main/onnx_checkpoints/unet_dec_21_epoch_579.onnx")
    model_path = str(resolved)

    from ssp.webapp import AppConfig
    config = AppConfig(model_path=model_path,
                       data_dir=data_dir or DEFAULT_DATA_DIR)

    if host == "0.0.0.0":
        click.echo("WARNING: binding on all interfaces. This app has no "
                   "authentication and results are advisory/research-use; "
                   "prefer the default 127.0.0.1.")

    if not no_browser:
        import threading
        import webbrowser
        shown = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
        url = f"http://{shown}:{port}/"
        timer = threading.Timer(1.0, lambda: webbrowser.open(url))
        timer.daemon = True  # never hold up exit if the bind fails
        timer.start()
        click.echo(f"opening {url} ...")

    click.echo(f"serving on http://{host}:{port}  (data: {config.data_dir})")
    _run_server(config, host, port)


if __name__ == "__main__":
    cli()
