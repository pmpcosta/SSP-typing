# SSP-typing

Main code repository for our SSP HLA typing app, Olerup style, derived from
[GelGenie](https://github.com/mattaq31/GelGenie). Starting with a geldoc image.

Standalone pipeline — no QuPath, no PyTorch at runtime:

    gel image → ONNX segmentation → lanes/bands → control-normalized bp sizes
              → per-lane positivity → Olerup-style allele-pair typing

## Layout

- `python-gelgenie/ssp/` — the standalone app (this is where development happens)
  - `preprocessing.py`, `inference.py` — pixel-exact port of the QuPath/DJL
    inference path (pad-to-divisible-by-32, normalize, ONNX Runtime, argmax)
  - `bands.py` — mask → connected-component bands → lanes
  - `trays/` — vendor tray-sheet loader (`.xlsx`) + allele specificity parser
    (ranges, typos, family tokens, "All, except …")
  - `analysis.py` — lane clustering with empty-lane recovery, panel-aware
    control-band fitting (smile correction), bp calibration, band classification
  - `typing_engine.py` — allele-pair scoring with full ambiguity reporting,
    invalid-lane QC
  - `webapp/` — local FastAPI server + canvas UI (interactive editing, audit log)
  - `models.py` — HuggingFace model-zoo registry, cache, downloads
  - `cli.py` — `ssp-typing` entry point
- `python-gelgenie/gelgenie/` — upstream GelGenie training pipeline (kept for
  fine-tuning; not used at runtime)
- `qupath-gelgenie/` — the original Java QuPath extension (reference only)

Upstream GelGenie documentation assets (notebooks, screenshots, presentations,
sample gel images, training logs) were removed from this repository **and its
history** to keep it lean and free of lab data; the original GelGenie repository
retains them. Branding (`logo/`) and functional code resources are kept.

## Install

    cd python-gelgenie
    python3.12 -m venv .venv            # or: uv venv
    .venv/bin/pip install -e '.[ssp,webapp]'
    .venv/bin/ssp-typing models download        # ~57 MB, one time

## Use

CLI (batch):

    ssp-typing type gel.jpg --tray FLClasseI.xlsx -o out/
    # -> out/mask.png, out/typing.json, out/report.txt   (~5 s on CPU)

Web app (interactive):

    ssp-typing serve                      # opens http://127.0.0.1:8765
    # upload gel + tray; click bands to delete/restore, shift+click to add,
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
- Test suite: `cd python-gelgenie && pytest` (149 tests; real-gel tests
  skip unless fixtures are present). Validation fixtures (gel image, vendor
  tray sheet, expected bands) are NOT in this repo — point `$SSP_TESTDATA_DIR`
  at a private directory holding `G_26160.jpg`, `FLClasseI.xlsx` and
  `ssp_bands_G26160.json` to run them; the ONNX model auto-downloads

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
