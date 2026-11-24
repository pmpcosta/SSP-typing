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

On-disk session store for the local web app.

A *session* is one gel image + one tray definition + the full pipeline result,
persisted under ``<data_dir>/sessions/<sid>/`` so every run is reproducible and
auditable (a requirement for a clinical-adjacent tool):

    image<ext>            the image that was segmented: the uploaded bytes, or
                          image.png = the joined strip when a layout was used
    original<ext>         (layout sessions) the raw gel photo as uploaded
    layout.json           (layout sessions) the confirmed cut geometry
    lanes.json            (layout sessions) where each mix sits in image.png;
                          analysis takes lanes from it instead of band gaps
    mask.png              segmentation mask
    bands.json            raw detected bands (pre-edit; the audit baseline)
    tray.json             normalized tray snapshot (immune to later store edits)
    meta.json             id, names, created, image size, params
    edits.json            current edit set (lane overrides, band add/delete)
    edits_history.jsonl   append-only audit log of every applied edit
    state.json            full SessionState (analysis + typing + all of above)
    typing.json           CLI-identical typing payload (download)
    report.txt            CLI-identical text report (download)

Edits never mutate the segmentation: they are re-applied to the raw bands and
the analysis/typing are recomputed from scratch, so results are deterministic
given (bands, tray, params, edits).

Raw gel photos go through a layout step first: the photo is *staged* under
``<data_dir>/uploads/<uid>/`` (detect / reuse / preview the layout there),
and the session is created from the staged photo plus the confirmed layout.
The last confirmed layout per preset is kept in ``<data_dir>/layouts/``.
Staged uploads are removed once used, and pruned after a day.
"""

from __future__ import annotations

import json
import math
import mimetypes
import os
import re
import shutil
import time
import uuid
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np

from ssp.analysis import MANUAL_KINDS, analyse_gel
from ssp.bands import Band
from ssp.cli import _infer_dtype_max, detect_invert, segment_image
from ssp.inference import GelGenieONNX
from ssp.layout import (
    PRESETS,
    Layout,
    LayoutError,
    Manifest,
    is_prestitched,
    preset_layout,
    render_stitched,
    stitch,
)
from ssp.layout_detect import detect_layout
from ssp.report import build_report
from ssp.trays import Tray, load_tray_xlsx
from ssp.typing_engine import type_sample

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_BAND_CORE = ("id", "area", "centroid_y", "centroid_x", "bbox", "lane")
# synthetic (user-added) bands get ids far above any real band id
SYNTHETIC_ID_BASE = 1_000_000


def _check_kind(kind) -> str:
    if kind not in MANUAL_KINDS:
        raise ValueError(f"band kind must be one of {list(MANUAL_KINDS)}: {kind!r}")
    return kind
UPLOAD_TTL_S = 24 * 3600
PREVIEW_MAX_W = 3000
LAYOUT_SOURCES = ("detect", "last", "even")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_stem(filename: str, fallback: str) -> str:
    return _SAFE.sub("_", Path(filename or "").stem) or fallback


def _atomic_write_text(path: Path, text: str) -> None:
    """Crash-safe write: readers never see a torn file (write + os.replace)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj, indent: Optional[int] = None) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=indent))


@dataclass
class AppConfig:
    model_path: str
    data_dir: str
    default_min_area: int = 5
    default_match_tol: float = 0.25
    default_lane_gap: float = 20.0

    @property
    def sessions_dir(self) -> Path:
        return Path(self.data_dir) / "sessions"

    @property
    def trays_dir(self) -> Path:
        return Path(self.data_dir) / "trays"

    @property
    def uploads_dir(self) -> Path:
        return Path(self.data_dir) / "uploads"

    @property
    def layouts_dir(self) -> Path:
        return Path(self.data_dir) / "layouts"


class SessionStore:
    """Creates, loads and edits typing sessions under ``config.data_dir``."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        for d in (config.sessions_dir, config.trays_dir, config.uploads_dir,
                  config.layouts_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---------- tray store ----------

    def save_tray(self, filename: str, data: bytes) -> Tuple[str, Tray]:
        """Parse an uploaded tray (.xlsx or .json) and store it normalized.

        Returns (store name, Tray). Raises ValueError on unparsable input;
        nothing is stored in that case.
        """
        stem = _safe_stem(filename, "tray")
        is_xlsx = (filename or "").lower().endswith(".xlsx")
        # openpyxl validates by file extension -> temp file needs the right one
        suffix = ".xlsx" if is_xlsx else ".json"
        tmp = self.config.trays_dir / f".upload-{uuid.uuid4().hex}{suffix}"
        tmp.write_bytes(data)
        try:
            tray = load_tray_xlsx(tmp, name=stem) if is_xlsx else Tray.load_json(tmp)
        except ValueError:
            raise
        except Exception as e:  # InvalidFileException, JSONDecodeError, ...
            raise ValueError(f"could not parse tray file {filename!r}: {e}")
        finally:
            tmp.unlink(missing_ok=True)
        # never silently overwrite a different tray stored under the same
        # name: identical content reuses the existing entry, different
        # content is stored under a suffixed name
        dest = self.config.trays_dir / f"{stem}.json"
        payload = json.dumps(tray.to_dict(), indent=1)
        if dest.exists() and dest.read_text() != payload:
            reuse = None
            for p in self.config.trays_dir.glob(f"{stem}-????.json"):
                try:
                    if p.read_text() == payload:
                        reuse = p
                        break
                except OSError:
                    continue
            if reuse is not None:
                dest, stem = reuse, reuse.stem
            else:
                stem = f"{stem}-{uuid.uuid4().hex[:4]}"
                dest = self.config.trays_dir / f"{stem}.json"
        _atomic_write_text(dest, payload)
        return stem, tray

    def list_trays(self) -> List[Dict]:
        out = []
        for p in sorted(self.config.trays_dir.glob("*.json")):
            out.append({
                "name": p.stem,
                "modified": datetime.fromtimestamp(
                    p.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
            })
        return out

    def _load_stored_tray(self, name: str) -> Tray:
        safe = _SAFE.sub("_", name)
        path = self.config.trays_dir / f"{safe}.json"
        if not path.exists():
            raise ValueError(f"unknown tray {name!r}; upload it first")
        return Tray.load_json(path)

    # ---------- staged uploads + layout step ----------

    def _prune_uploads(self) -> None:
        cutoff = time.time() - UPLOAD_TTL_S
        for d in self.config.uploads_dir.iterdir():
            try:
                if d.is_dir() and d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                continue

    def _udir(self, uid: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{12,32}", uid or ""):
            raise KeyError(uid)
        d = self.config.uploads_dir / uid
        if not (d / "meta.json").exists():
            raise KeyError(uid)
        return d

    def _upload_meta(self, uid: str) -> Dict:
        return json.loads((self._udir(uid) / "meta.json").read_text())

    def _read_upload(self, uid: str) -> np.ndarray:
        d = self._udir(uid)
        meta = json.loads((d / "meta.json").read_text())
        return np.asarray(imageio.imread(str(d / meta["image_file"])))

    def stage_upload(self, image_name: str, image_bytes: bytes) -> Dict:
        """Store a gel photo for the layout step; validate it is readable."""
        self._prune_uploads()
        uid = uuid.uuid4().hex[:16]
        d = self.config.uploads_dir / uid
        d.mkdir(parents=True)
        ext = _SAFE.sub("_", Path(image_name or "image.png").suffix or ".png")
        image_file = f"image{ext}"
        (d / image_file).write_bytes(image_bytes)
        try:
            raw = np.asarray(imageio.imread(str(d / image_file)))
        except Exception as e:
            shutil.rmtree(d, ignore_errors=True)
            raise ValueError(f"could not read image {image_name!r}: {e}")
        h, w = raw.shape[:2]
        meta = {"id": uid, "image_name": image_name, "image_file": image_file,
                "image_media_type": self._image_media_type(image_name),
                "image": {"width": int(w), "height": int(h)}, "created": _now()}
        _atomic_write_json(d / "meta.json", meta, indent=1)
        return self._upload_info(meta)

    def _upload_info(self, meta: Dict) -> Dict:
        w, h = meta["image"]["width"], meta["image"]["height"]
        return {"upload_id": meta["id"], "image_name": meta["image_name"],
                "image": meta["image"], "prestitched": is_prestitched((w, h)),
                "presets": list(PRESETS),
                "last_layouts": sorted(p for p in PRESETS
                                       if (self.config.layouts_dir / f"last-{p}.json").exists())}

    def upload_image(self, uid: str) -> Tuple[Path, str]:
        d = self._udir(uid)
        meta = json.loads((d / "meta.json").read_text())
        return d / meta["image_file"], meta.get("image_media_type", "application/octet-stream")

    def propose_layout(self, uid: str, preset: str, source: str = "detect") -> Dict:
        """Starting layout for a staged photo: detected, last confirmed, or
        an even split. Returns {layout, warnings, source}."""
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose one of {', '.join(PRESETS)}")
        if source not in LAYOUT_SOURCES:
            raise ValueError(f"unknown layout source {source!r}")
        raw = self._read_upload(uid)
        size = (raw.shape[1], raw.shape[0])
        warnings: List[str] = []
        if source == "detect":
            try:
                det = detect_layout(raw, preset)
                layout, warnings = det.layout, list(det.warnings)
            except LayoutError as e:
                layout, source = preset_layout(preset, size), "even"
                warnings.append(f"automatic detection failed ({e}); showing an even split")
        elif source == "last":
            path = self.config.layouts_dir / f"last-{preset}.json"
            if not path.exists():
                raise ValueError(f"no confirmed {preset} layout yet")
            layout = Layout.load_json(str(path)).scaled_to(size)
        else:
            layout = preset_layout(preset, size)
        return {"layout": layout.to_dict(), "warnings": warnings, "source": source}

    @staticmethod
    def _parse_layout(layout: Dict, size: Tuple[int, int]) -> Layout:
        try:
            lay = Layout.from_dict(layout)
        except LayoutError as e:
            raise ValueError(f"layout: {e}")
        if lay.is_stitched:
            raise ValueError("layout: no gel pieces defined")
        return lay.scaled_to(size) if tuple(lay.image_size) != size else lay

    def preview_layout(self, uid: str, layout: Dict) -> bytes:
        """PNG of the joined strip with mix numbers, for the layout editor."""
        raw = self._read_upload(uid)
        lay = self._parse_layout(layout, (raw.shape[1], raw.shape[0]))
        try:
            lay.validate()
            stitched, manifest = stitch(raw, lay)
        except LayoutError as e:
            raise ValueError(f"layout: {e}")
        img = render_stitched(stitched, manifest)
        if img.shape[1] > PREVIEW_MAX_W:
            from PIL import Image
            pil = Image.fromarray(img)
            pil.thumbnail((PREVIEW_MAX_W, PREVIEW_MAX_W))
            img = np.asarray(pil)
        buf = BytesIO()
        imageio.imwrite(buf, img, format="png")
        return buf.getvalue()

    def relayout(self, sid: str) -> Dict:
        """Stage a layout session's original photo again, with its layout,
        so the cuts can be corrected (the result is a new session)."""
        sdir = self._sdir(sid)
        meta = json.loads((sdir / "meta.json").read_text())
        if not meta.get("original_file"):
            raise ValueError("this session was not made from a raw gel photo")
        info = self.stage_upload(meta.get("original_name") or meta["image_name"],
                                 (sdir / meta["original_file"]).read_bytes())
        info["layout"] = json.loads((sdir / "layout.json").read_text())
        info["tray_store_name"] = meta.get("tray_store_name")
        # segmentation settings, so the corrected session is segmented alike;
        # a detected polarity is re-detected on the new cut rather than pinned
        # (sessions from before detection carry the old checkbox, so only an
        # explicit invert_auto=False counts as forced)
        params = meta.get("params", {})
        keys = ["min_area", "percentile_norm"]
        if params.get("invert_auto") is False:
            keys.append("invert")
        info["params"] = {k: params[k] for k in keys if k in params}
        return info

    # ---------- sessions ----------

    def _sdir(self, sid: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{6,32}", sid or ""):
            raise KeyError(sid)
        d = self.config.sessions_dir / sid
        if not (d / "meta.json").exists():
            raise KeyError(sid)
        return d

    def create_session(self, image_name: Optional[str] = None,
                       image_bytes: Optional[bytes] = None,
                       tray_filename: Optional[str] = None,
                       tray_bytes: Optional[bytes] = None,
                       tray_name: Optional[str] = None,
                       min_area: Optional[int] = None,
                       invert: Optional[bool] = None,
                       percentile_norm: bool = False,
                       upload_id: Optional[str] = None,
                       layout: Optional[Dict] = None) -> Dict:
        """Run the pipeline on an image (bytes, or a staged ``upload_id``).
        With ``layout``, the photo is first cut and joined, and lanes come
        from the layout's known geometry."""
        if upload_id:
            udir = self._udir(upload_id)
            umeta = json.loads((udir / "meta.json").read_text())
            image_name = umeta["image_name"]
            image_bytes = (udir / umeta["image_file"]).read_bytes()
        if image_bytes is None:
            raise ValueError("provide an image or an upload_id")
        # tray first: fail fast on a bad sheet before burning inference time
        if tray_bytes is not None:
            # save_tray returns the (possibly deduped) store name
            store_name, tray = self.save_tray(tray_filename or "tray.xlsx",
                                              tray_bytes)
        elif tray_name:
            tray = self._load_stored_tray(tray_name)
            store_name = _SAFE.sub("_", tray_name)
        else:
            raise ValueError("provide a tray file or tray_name")

        sid = uuid.uuid4().hex[:12]
        sdir = self.config.sessions_dir / sid
        sdir.mkdir(parents=True)
        ext = _SAFE.sub("_", Path(image_name or "image.png").suffix or ".png")
        image_file = f"image{ext}"
        try:
            (sdir / image_file).write_bytes(image_bytes)
            # read back from the stored file: same code path as the CLI, and
            # avoids format-sniffing issues with file-like objects (JPEG)
            try:
                raw = np.asarray(imageio.imread(str(sdir / image_file)))
            except Exception as e:  # unreadable/unsupported image
                raise ValueError(f"could not read image {image_name!r}: {e}")
            layout_obj: Optional[Layout] = None
            original = None
            if layout is not None:
                layout_obj = self._parse_layout(layout, (raw.shape[1], raw.shape[0]))
                try:
                    layout_obj.validate(expected_lanes=len(tray.lanes))
                    stitched, manifest = stitch(raw, layout_obj)
                except LayoutError as e:
                    raise ValueError(f"layout: {e}")
                original = {"file": f"original{ext}",
                            "image": {"width": int(raw.shape[1]), "height": int(raw.shape[0])}}
                (sdir / image_file).rename(sdir / original["file"])
                image_file = "image.png"
                raw = stitched
                imageio.imwrite(str(sdir / image_file), raw)
                layout_obj.save_json(str(sdir / "layout.json"))
                manifest.save_json(str(sdir / "lanes.json"))
            h, w = raw.shape[:2]

            dtype_max = _infer_dtype_max(raw)
            # polarity is detected unless the caller forces it; the detected
            # value is recorded so the run is reproducible
            invert_auto = invert is None
            if invert_auto:
                invert = detect_invert(raw, data_max_norm=not percentile_norm,
                                       dtype_max=dtype_max)
            params = {
                "min_area": int(min_area if min_area is not None
                                else self.config.default_min_area),
                "invert": bool(invert),
                "invert_auto": invert_auto,
                "percentile_norm": bool(percentile_norm),
                "match_tol": self.config.default_match_tol,
                "lane_gap": self.config.default_lane_gap,
            }
            model = GelGenieONNX(model_path=self.config.model_path)
            mask, bands = segment_image(
                raw, model, invert=params["invert"],
                data_max_norm=not params["percentile_norm"],
                dtype_max=dtype_max, min_area=params["min_area"],
            )
        except ValueError:
            shutil.rmtree(sdir, ignore_errors=True)
            raise
        except Exception as e:
            shutil.rmtree(sdir, ignore_errors=True)
            raise ValueError(f"session failed: {e}")
        imageio.imwrite(str(sdir / "mask.png"), (mask * 255).astype(np.uint8))
        _atomic_write_json(sdir / "bands.json", [b.to_dict() for b in bands])
        _atomic_write_text(sdir / "tray.json", json.dumps(tray.to_dict(), indent=1))

        meta = {
            "id": sid, "image_name": image_name, "image_file": image_file,
            # allowlist: never serve attacker-chosen types like text/html
            "image_media_type": ("image/png" if original
                                 else self._image_media_type(image_name)),
            "tray_name": tray.name, "tray_store_name": store_name,
            "created": _now(), "image": {"width": int(w), "height": int(h)},
            "params": params,
            "defaults": {"match_tol": self.config.default_match_tol,
                         "lane_gap": self.config.default_lane_gap},
        }
        if original:
            meta.update({"original_file": original["file"], "original_name": image_name,
                         "original_media_type": self._image_media_type(image_name),
                         "original_image": original["image"],
                         "layout_preset": layout_obj.preset})
            if layout_obj.preset in PRESETS:  # remember for "use last confirmed"
                layout_obj.save_json(str(self.config.layouts_dir /
                                         f"last-{layout_obj.preset}.json"))
        _atomic_write_json(sdir / "meta.json", meta, indent=1)
        self._write_edits(sdir, {"lane_overrides": [], "deleted_bands": [],
                                 "added_bands": [], "band_kinds": []})
        state = self._recompute(sid)
        if upload_id:
            shutil.rmtree(self.config.uploads_dir / upload_id, ignore_errors=True)
        return state

    @staticmethod
    def _image_media_type(name: str) -> str:
        mt = mimetypes.guess_type(name or "")[0] or ""
        return mt if mt.startswith("image/") else "application/octet-stream"

    def list_sessions(self) -> List[Dict]:
        out = []
        base = self.config.sessions_dir
        if base.is_dir():
            for d in base.iterdir():
                mp = d / "meta.json"
                if d.is_dir() and mp.exists():
                    try:
                        meta = json.loads(mp.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue
                    out.append({k: meta.get(k) for k in
                                ("id", "image_name", "tray_name", "created")})
        out.sort(key=lambda m: m.get("created") or "", reverse=True)
        return out

    def load_state(self, sid: str) -> Dict:
        sdir = self._sdir(sid)
        try:
            return json.loads((sdir / "state.json").read_text())
        except (OSError, ValueError):
            # missing or corrupt state.json: rebuild from on-disk inputs
            # (bands/tray/edits/meta are the source of truth)
            try:
                return self._recompute(sid)
            except Exception:
                raise KeyError(sid)

    def session_file(self, sid: str, name: str) -> Path:
        sdir = self._sdir(sid)
        path = (sdir / name).resolve()
        if path.parent != sdir.resolve() or not path.exists():
            raise KeyError(f"{sid}/{name}")
        return path

    def original_file(self, sid: str) -> Tuple[Path, str]:
        meta = json.loads((self._sdir(sid) / "meta.json").read_text())
        if not meta.get("original_file"):
            raise KeyError(f"{sid}/original")
        return (self.session_file(sid, meta["original_file"]),
                meta.get("original_media_type", "application/octet-stream"))

    def image_file(self, sid: str) -> Tuple[Path, str]:
        meta = json.loads((self._sdir(sid) / "meta.json").read_text())
        return (self.session_file(sid, meta["image_file"]),
                meta.get("image_media_type", "application/octet-stream"))

    def apply_edits(self, sid: str, body: Dict) -> Dict:
        sdir = self._sdir(sid)
        meta = json.loads((sdir / "meta.json").read_text())
        tray = Tray.load_json(sdir / "tray.json")
        edits = self._validate_edits(body, tray)

        try:
            old_edits = json.loads((sdir / "edits.json").read_text())
        except (OSError, ValueError):
            old_edits = {"lane_overrides": [], "deleted_bands": [], "added_bands": [], "band_kinds": []}
        old_params = dict(meta["params"])
        if body.get("match_tol") is not None:
            meta["params"]["match_tol"] = float(body["match_tol"])
        if body.get("lane_gap") is not None:
            meta["params"]["lane_gap"] = float(body["lane_gap"])
        self._write_edits(sdir, edits)
        _atomic_write_json(sdir / "meta.json", meta, indent=1)
        try:
            state = self._recompute(sid)
        except Exception:
            # roll back so a rejected edit never silently sticks
            self._write_edits(sdir, old_edits)
            meta["params"] = old_params
            _atomic_write_json(sdir / "meta.json", meta, indent=1)
            raise
        with open(sdir / "edits_history.jsonl", "a") as f:
            f.write(json.dumps({"ts": _now(), "edits": edits,
                                "params": meta["params"]}) + "\n")
        return state

    @staticmethod
    def _validate_edits(body: Dict, tray: Tray) -> Dict:
        if not isinstance(body, dict):
            raise ValueError("edits body must be a JSON object")
        mixes = {l.mix for l in tray.lanes}
        overrides = []
        for ov in body.get("lane_overrides") or ():
            if not isinstance(ov, dict) or "mix" not in ov:
                raise ValueError(f"lane override needs 'mix': {ov!r}")
            try:
                mix = int(ov["mix"])
            except (TypeError, ValueError):
                raise ValueError(f"lane override mix must be an integer: {ov!r}")
            if mix not in mixes:
                raise ValueError(f"lane override for unknown mix {mix}")
            entry: Dict = {"mix": mix}
            for key in ("positive", "valid"):
                if ov.get(key) is not None:
                    entry[key] = bool(ov[key])
            if len(entry) == 1:
                raise ValueError(f"lane override for mix {mix} sets nothing")
            overrides.append(entry)
        try:
            deleted = [int(b) for b in body.get("deleted_bands") or ()]
        except (TypeError, ValueError):
            raise ValueError("deleted_bands must be integers")
        # synthetic (user-added) bands are removed via added_bands, not here
        deleted = [b for b in deleted if b < SYNTHETIC_ID_BASE]
        added = []
        for ab in body.get("added_bands") or ():
            if not isinstance(ab, dict) or "x" not in ab or "y" not in ab:
                raise ValueError(f"added band needs x and y: {ab!r}")
            try:
                x, y = float(ab["x"]), float(ab["y"])
            except (TypeError, ValueError):
                raise ValueError(f"added band x/y must be numbers: {ab!r}")
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(f"added band x/y must be finite: {ab!r}")
            entry = {"x": x, "y": y}
            if ab.get("kind") is not None:
                entry["kind"] = _check_kind(ab["kind"])
            added.append(entry)
        # the user's band classifications, by detected band id (an added
        # band carries its own kind: its id shifts when another is removed)
        kinds: Dict[int, str] = {}
        for bk in body.get("band_kinds") or ():
            if not isinstance(bk, dict) or "id" not in bk or "kind" not in bk:
                raise ValueError(f"band kind needs id and kind: {bk!r}")
            try:
                bid = int(bk["id"])
            except (TypeError, ValueError):
                raise ValueError(f"band kind id must be an integer: {bk!r}")
            if bid >= SYNTHETIC_ID_BASE:
                raise ValueError("an added band's kind goes on its added_bands entry")
            kinds[bid] = _check_kind(bk["kind"])
        for key in ("match_tol", "lane_gap"):
            v = body.get(key)
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be a number")
            # NaN/Inf would poison meta.json and break every JSON response
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{key} must be finite and > 0")
        return {"lane_overrides": overrides, "deleted_bands": deleted,
                "added_bands": added,
                "band_kinds": [{"id": i, "kind": k} for i, k in kinds.items()]}

    @staticmethod
    def _write_edits(sdir: Path, edits: Dict) -> None:
        _atomic_write_json(sdir / "edits.json", edits, indent=1)

    # ---------- pipeline ----------

    def _recompute(self, sid: str) -> Dict:
        sdir = self.config.sessions_dir / sid
        meta = json.loads((sdir / "meta.json").read_text())
        tray = Tray.load_json(sdir / "tray.json")
        edits = json.loads((sdir / "edits.json").read_text())
        with open(sdir / "bands.json") as f:
            raw_bands = json.load(f)
        params = meta["params"]

        bands = self._apply_band_edits(raw_bands, edits)
        kinds = {int(bk["id"]): bk["kind"] for bk in edits.get("band_kinds") or ()}
        kinds.update({b.id: b.extra["kind"] for b in bands if b.extra.get("kind")})
        lane_map = None
        if (sdir / "lanes.json").exists():
            lane_map = Manifest.load_json(str(sdir / "lanes.json")).to_lane_map()
        ga = analyse_gel(bands, tray, lane_gap=params.get("lane_gap", 20.0),
                         match_tol=params.get("match_tol", 0.25), lane_map=lane_map,
                         band_kinds=kinds)
        self._apply_lane_overrides(ga, edits.get("lane_overrides"))
        result = type_sample(ga, tray)

        report = build_report(meta["image_name"], tray.name, ga, result)
        typing_payload = {
            "image": meta["image_name"], "tray": tray.name,
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
        _atomic_write_text(sdir / "report.txt", report)
        _atomic_write_json(sdir / "typing.json", typing_payload, indent=2)

        state = {
            "id": sid,
            "image_name": meta["image_name"],
            "tray_name": tray.name,
            "created": meta["created"],
            "image": meta["image"],
            "params": params,
            "defaults": meta.get("defaults", {
                "match_tol": self.config.default_match_tol,
                "lane_gap": self.config.default_lane_gap}),
            "bands": raw_bands,
            "tray": tray.to_dict(),
            "analysis": ga.to_dict(),
            "typing": result.to_dict(),
            "edits": edits,
            "lane_assignment": "layout" if lane_map else "band-gaps",
            "has_original": bool(meta.get("original_file")),
            "layout_preset": meta.get("layout_preset"),
        }
        _atomic_write_json(sdir / "state.json", state)
        return state

    @staticmethod
    def _apply_band_edits(raw_bands: List[Dict], edits: Dict) -> List[Band]:
        deleted = set(edits.get("deleted_bands") or ())
        out: List[Band] = []
        for d in raw_bands:
            if d["id"] in deleted:
                continue
            extra = {k: v for k, v in d.items() if k not in _BAND_CORE}
            out.append(Band(id=d["id"], area=d["area"],
                            centroid_y=d["centroid_y"], centroid_x=d["centroid_x"],
                            bbox=tuple(d["bbox"]), lane=d.get("lane"),
                            extra=extra))
        next_id = SYNTHETIC_ID_BASE
        for ab in edits.get("added_bands") or ():
            x, y = float(ab["x"]), float(ab["y"])
            extra = {"synthetic": True}
            if ab.get("kind"):
                extra["kind"] = ab["kind"]
            out.append(Band(id=next_id, area=1, centroid_y=y, centroid_x=x,
                            bbox=(int(y), int(x), int(y) + 1, int(x) + 1),
                            extra=extra))
            next_id += 1
        return out

    @staticmethod
    def _apply_lane_overrides(ga, overrides: Optional[List[Dict]]) -> None:
        by_mix = {l.mix: l for l in ga.lanes}
        for ov in overrides or ():
            lane = by_mix[int(ov["mix"])]
            if "positive" in ov:
                lane.positive = bool(ov["positive"])
            if "valid" in ov:
                lane.valid = bool(ov["valid"])
