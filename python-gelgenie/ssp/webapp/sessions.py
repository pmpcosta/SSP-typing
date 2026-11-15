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

    image<ext>            original uploaded image bytes
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
"""

from __future__ import annotations

import json
import mimetypes
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np

from ssp.analysis import analyse_gel
from ssp.bands import Band
from ssp.cli import _infer_dtype_max, segment_image
from ssp.inference import GelGenieONNX
from ssp.report import build_report
from ssp.trays import Tray, load_tray_xlsx
from ssp.typing_engine import type_sample

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_BAND_CORE = ("id", "area", "centroid_y", "centroid_x", "bbox", "lane")
# synthetic (user-added) bands get ids far above any real band id
SYNTHETIC_ID_BASE = 1_000_000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_stem(filename: str, fallback: str) -> str:
    return _SAFE.sub("_", Path(filename or "").stem) or fallback


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


class SessionStore:
    """Creates, loads and edits typing sessions under ``config.data_dir``."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        config.sessions_dir.mkdir(parents=True, exist_ok=True)
        config.trays_dir.mkdir(parents=True, exist_ok=True)

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
        tray.save_json(self.config.trays_dir / f"{stem}.json")
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

    # ---------- sessions ----------

    def _sdir(self, sid: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{6,32}", sid or ""):
            raise KeyError(sid)
        d = self.config.sessions_dir / sid
        if not (d / "meta.json").exists():
            raise KeyError(sid)
        return d

    def create_session(self, image_name: str, image_bytes: bytes,
                       tray_filename: Optional[str] = None,
                       tray_bytes: Optional[bytes] = None,
                       tray_name: Optional[str] = None,
                       min_area: Optional[int] = None,
                       invert: bool = False,
                       percentile_norm: bool = False) -> Dict:
        # tray first: fail fast on a bad sheet before burning inference time
        if tray_bytes is not None:
            _, tray = self.save_tray(tray_filename or "tray.xlsx", tray_bytes)
            store_name = _safe_stem(tray_filename or "tray", "tray")
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
            h, w = raw.shape[:2]

            params = {
                "min_area": int(min_area if min_area is not None
                                else self.config.default_min_area),
                "invert": bool(invert),
                "percentile_norm": bool(percentile_norm),
                "match_tol": self.config.default_match_tol,
                "lane_gap": self.config.default_lane_gap,
            }
            model = GelGenieONNX(model_path=self.config.model_path)
            mask, bands = segment_image(
                raw, model, invert=params["invert"],
                data_max_norm=not params["percentile_norm"],
                dtype_max=_infer_dtype_max(raw), min_area=params["min_area"],
            )
        except ValueError:
            shutil.rmtree(sdir, ignore_errors=True)
            raise
        except Exception as e:
            shutil.rmtree(sdir, ignore_errors=True)
            raise ValueError(f"session failed: {e}")
        imageio.imwrite(str(sdir / "mask.png"), (mask * 255).astype(np.uint8))
        with open(sdir / "bands.json", "w") as f:
            json.dump([b.to_dict() for b in bands], f)
        tray.save_json(sdir / "tray.json")

        meta = {
            "id": sid, "image_name": image_name, "image_file": image_file,
            "image_media_type": mimetypes.guess_type(image_name or "")[0]
                                or "application/octet-stream",
            "tray_name": tray.name, "tray_store_name": store_name,
            "created": _now(), "image": {"width": int(w), "height": int(h)},
            "params": params,
            "defaults": {"match_tol": self.config.default_match_tol,
                         "lane_gap": self.config.default_lane_gap},
        }
        with open(sdir / "meta.json", "w") as f:
            json.dump(meta, f, indent=1)
        self._write_edits(sdir, {"lane_overrides": [], "deleted_bands": [],
                                 "added_bands": []})
        return self._recompute(sid)

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
        state_path = sdir / "state.json"
        if not state_path.exists():
            raise KeyError(sid)
        return json.loads(state_path.read_text())

    def session_file(self, sid: str, name: str) -> Path:
        sdir = self._sdir(sid)
        path = (sdir / name).resolve()
        if path.parent != sdir.resolve() or not path.exists():
            raise KeyError(f"{sid}/{name}")
        return path

    def image_file(self, sid: str) -> Tuple[Path, str]:
        meta = json.loads((self._sdir(sid) / "meta.json").read_text())
        return (self.session_file(sid, meta["image_file"]),
                meta.get("image_media_type", "application/octet-stream"))

    def apply_edits(self, sid: str, body: Dict) -> Dict:
        sdir = self._sdir(sid)
        meta = json.loads((sdir / "meta.json").read_text())
        tray = Tray.load_json(sdir / "tray.json")
        edits = self._validate_edits(body, tray)

        old_edits = json.loads((sdir / "edits.json").read_text())
        old_params = dict(meta["params"])
        if body.get("match_tol") is not None:
            meta["params"]["match_tol"] = float(body["match_tol"])
        if body.get("lane_gap") is not None:
            meta["params"]["lane_gap"] = float(body["lane_gap"])
        self._write_edits(sdir, edits)
        with open(sdir / "meta.json", "w") as f:
            json.dump(meta, f, indent=1)
        try:
            state = self._recompute(sid)
        except Exception:
            # roll back so a rejected edit never silently sticks
            self._write_edits(sdir, old_edits)
            meta["params"] = old_params
            with open(sdir / "meta.json", "w") as f:
                json.dump(meta, f, indent=1)
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
            mix = int(ov["mix"])
            if mix not in mixes:
                raise ValueError(f"lane override for unknown mix {mix}")
            entry: Dict = {"mix": mix}
            for key in ("positive", "valid"):
                if ov.get(key) is not None:
                    entry[key] = bool(ov[key])
            if len(entry) == 1:
                raise ValueError(f"lane override for mix {mix} sets nothing")
            overrides.append(entry)
        deleted = [int(b) for b in body.get("deleted_bands") or ()]
        added = []
        for ab in body.get("added_bands") or ():
            if not isinstance(ab, dict) or "x" not in ab or "y" not in ab:
                raise ValueError(f"added band needs x and y: {ab!r}")
            added.append({"x": float(ab["x"]), "y": float(ab["y"])})
        for key in ("match_tol", "lane_gap"):
            if body.get(key) is not None and float(body[key]) <= 0:
                raise ValueError(f"{key} must be > 0")
        return {"lane_overrides": overrides, "deleted_bands": deleted,
                "added_bands": added}

    @staticmethod
    def _write_edits(sdir: Path, edits: Dict) -> None:
        with open(sdir / "edits.json", "w") as f:
            json.dump(edits, f, indent=1)

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
        ga = analyse_gel(bands, tray, lane_gap=params.get("lane_gap", 20.0),
                         match_tol=params.get("match_tol", 0.25))
        self._apply_lane_overrides(ga, edits.get("lane_overrides"))
        result = type_sample(ga, tray)

        report = build_report(meta["image_name"], tray.name, ga, result)
        typing_payload = {
            "image": meta["image_name"], "tray": tray.name,
            "analysis": {
                "slope_b": round(ga.slope_b, 2),
                "panels": [list(p) for p in ga.panels],
                "warnings": ga.warnings,
                "invalid_lanes": [l.mix for l in ga.lanes if not l.valid],
                "positive_lanes": [l.mix for l in ga.lanes if l.positive],
            },
            "typing": result.to_dict(),
        }
        (sdir / "report.txt").write_text(report)
        with open(sdir / "typing.json", "w") as f:
            json.dump(typing_payload, f, indent=2)

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
        }
        with open(sdir / "state.json", "w") as f:
            json.dump(state, f)
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
            out.append(Band(id=next_id, area=1, centroid_y=y, centroid_x=x,
                            bbox=(int(y), int(x), int(y) + 1, int(x) + 1),
                            extra={"synthetic": True}))
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
