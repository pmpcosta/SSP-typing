# SSP-typing

Main code repository for our SSP HLA typing app, Olerup style, derived from
[GelGenie](https://github.com/mattaq31/GelGenie). Starting with a geldoc image.

Standalone pipeline — no QuPath, no PyTorch at runtime:

    raw gel photo → layout (cut rows, join into one 96-lane strip)
              → ONNX segmentation → lanes/bands → control-normalized bp sizes
              → per-lane positivity → Olerup-style allele-pair typing

## Layout

- `python-gelgenie/ssp/` — the standalone app (this is where development happens)
  - `layout.py` — rearranges raw gel photos (3×32 single gel, or 6×16 = two
    gel pieces of 3×16) into the single joined strip typing expects:
    per-piece perspective rectification, row cuts, lane manifest, overlay
    images for visual confirmation
  - `layout_detect.py` — automatic first guess of that layout (gel pieces,
    lane grid, row cuts at the wells, tilt), with warnings where unsure
  - `preprocessing.py`, `inference.py` — pixel-exact port of the QuPath/DJL
    inference path (pad-to-divisible-by-32, normalize, ONNX Runtime, argmax)
  - `bands.py` — mask → connected-component bands → lanes
  - `trays/` — vendor tray-sheet loader (`.xlsx`) + allele specificity parser
    (ranges, typos, family tokens, "All, except …")
  - `analysis.py` — lane assignment (from a layout's lane map, else gap
    clustering with empty-lane recovery), panel-aware
    control-band fitting (smile correction), bp calibration, band classification
  - `typing_engine.py` — allele-pair scoring with full ambiguity reporting,
    invalid-lane QC
  - `webapp/` — local FastAPI server + canvas UI (layout step, interactive
    editing, audit log)
  - `models.py` — HuggingFace model-zoo registry, cache, downloads
  - `cli.py` — `ssp-typing` entry point
- `python-gelgenie/gelgenie/` — upstream GelGenie training pipeline (kept for
  fine-tuning; not used at runtime)

Upstream GelGenie documentation assets (notebooks, screenshots, presentations,
sample gel images, training logs), the original Java QuPath extension
(`qupath-gelgenie/`) and its extension-manager catalog were removed from this
repository **and its history** to keep it lean and free of lab data; the
original GelGenie repository retains them (https://github.com/mattaq31/GelGenie).
`ssp/` is a pixel-exact port of the extension's inference path. Branding
(`logo/`) is kept.

## Install

    cd python-gelgenie
    python3.12 -m venv .venv            # or: uv venv
    .venv/bin/pip install -e '.[ssp,webapp]'
    .venv/bin/ssp-typing models download        # ~57 MB, one time

## Use

CLI (batch):

    # 1. raw gel photo -> joined 96-lane strip (skip if already joined)
    ssp-typing preprocess raw.jpg --preset 3x32 -o pre/      # or --preset 6x16
    #    the layout is detected automatically (--no-detect: even split);
    #    ALWAYS confirm pre/overlay.png (cuts + mix numbers on the original,
    #    especially the lane numbering at each gel piece's edges) and
    #    pre/stitched_lanes.png; if anything is off, edit pre/layout.json and
    #    re-run with --layout pre/layout.json (layouts are reusable across
    #    photos from the same imager; they rescale to the image size)
    # 2. type the joined strip; --lanes takes lane positions from the layout
    #    instead of guessing them from band gaps (robust to empty lanes
    #    next to a strip boundary)
    ssp-typing type pre/stitched.png --lanes pre/lanes.json --tray FLClasseI.xlsx -o out/
    # or in one go, once a layout is confirmed:
    ssp-typing type raw.jpg --layout pre/layout.json --tray FLClasseI.xlsx -o out/
    # -> out/mask.png, out/typing.json, out/report.txt   (~5 s on CPU)

Web app (interactive):

    ssp-typing serve                      # opens http://127.0.0.1:8765
    # upload gel + tray. A raw gel photo (3x32 / 6x16) opens the layout step:
    # the detected outlines, row cuts and mix numbers are drawn on the photo;
    # drag corners / row bars / used-lane bounds to correct them (Detect,
    # Use last confirmed, Even split), check the joined-strip preview, then
    # "Confirm layout & type". The raw photo + layout are kept with the
    # session; "Re-cut layout" corrects them (as a new session). A joined
    # strip skips the layout step.
    # In the session: click bands to delete/restore, shift+click to add,
    # click lane headers to force +/−; every edit re-types instantly and is
    # recorded in an append-only audit log (edits_history.jsonl per session)

Model resolution order: `--model` → `$SSP_MODEL` → cached default
(`~/.cache/ssptyping/models`, override with `$SSP_MODEL_CACHE`).
`ssp-typing models list|download|path` manages the cache.

## Packaging

- Wheel: `pip install '.[ssp,webapp]'` (static UI included via package_data)
- Frozen app: PyInstaller one-dir bundle —
  `pyinstaller packaging/ssp-typing.spec --noconfirm --distpath dist --workpath build/pyi`
  → `dist/ssp-typing/ssp-typing` (Linux; rebuild per target OS)
- Docker: see `packaging/Dockerfile` (slim runtime, model cached in a volume)

## Validation status

- Segmentation matches the QuPath extension's annotations on real geldoc
  images (170/170 bands, zero misses/false positives on G_26160)
- Typing verified end-to-end on G_26160 + FLClasseI (class I, 96 lanes):
  CLI, web UI and tests agree; ambiguity groups hand-checked against the sheet
- Layout: typing the raw G_26160 photo through `preprocess` (hand-fitted
  layout) gives the same calls, positive/invalid lanes and panels as typing
  the hand-joined image, both from the joined strip and via `type --layout`
  lane geometry; synthetic rotated/offset gels cover lane identity
  and order for both 3×32 and 6×16
- Layout detection: on 7 real gels (3×32 and 6×16, dark and light
  backgrounds) lane centres sit on the segmented bands (median 0.5–5 px,
  worst 13.6 px, lane pitch ~42–48 px; G_26160's residual is lens
  distortion) and lane numbering at every piece edge was checked by eye;
  fully automatic detect → join → type on G_26160 gives the same calls as
  the hand-joined image
- Test suite: `cd python-gelgenie && pytest` (233 tests; real-gel tests
  skip unless fixtures are present). Validation fixtures (gel image, vendor
  tray sheet, expected bands) are NOT in this repo — put them in `testdata/`
  at the repo root (gitignored) or point `$SSP_TESTDATA_DIR` at a private
  directory holding `G_26160.jpg`, `FLClasseI.xlsx` and `ssp_bands_G26160.json`
  (plus `preprocessing/<name>_before.jpg` and optionally a reference
  `<name>.layout.json` for the layout/detection tests) to run them; model tests also need
  `ssp-typing models download` first

## License & redistribution posture

- **Code**: Apache-2.0, derived from GelGenie (University of Edinburgh
  headers retained in every module). Keep the headers in new files.
- **Model weights**: the GelGenie HuggingFace zoo (incl. the default
  Universal model) is published under Apache-2.0 — redistribution permitted
  with attribution.
- **Vendor tray sheets are NOT redistributable**: files like `FLClasseI.xlsx`
  are Olerup kit data, and gel images may be confidential lab data. They are
  NOT committed to this repository and were scrubbed from its git history;
  keep them outside the repo (see `$SSP_TESTDATA_DIR` above) and check
  redistribution rights with the vendor before shipping any such data.
- **Results are advisory / research use only.** This software is not IVD
  certified or clinically validated. Any use for real donor/patient typing
  decisions brings IVD software regulation into scope and must not happen on
  the strength of this tool alone. The UI and reports label results
  accordingly; keep it that way.
