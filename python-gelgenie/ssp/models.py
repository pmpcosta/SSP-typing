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

Model zoo bootstrap: registry lookup, local cache, and downloads.

Models come from the GelGenie HuggingFace zoo (Apache-2.0 weights). The
registry is a small JSON file listing repo id + onnx filename per model;
ONNX files are streamed into a local cache directory with an atomic
``.part`` rename so interrupted downloads never leave a truncated model
that later loads (or explodes) at inference time.

Resolution order used by the CLI: explicit ``--model`` path, then
``$SSP_MODEL``, then the cached default (Universal) model. The cache
location defaults to ``~/.cache/ssptyping/models`` and can be overridden
with ``$SSP_MODEL_CACHE`` (useful for portable installs and tests).
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

REGISTRY_URL = (
    "https://huggingface.co/datasets/mattaq/GelGenie-Model-Zoo/raw/main/registry.json"
)
DEFAULT_MODEL_NAME = "GelGenie-Universal-Dec-2023"
# filename of the default model in the cache (stable published artifact;
# kept as a constant so resolution never needs the network)
DEFAULT_MODEL_FILE = "unet_dec_21_epoch_579.onnx"


class ModelFetchError(RuntimeError):
    """Registry/download failure with a user-readable, actionable message."""


def cache_dir() -> Path:
    env = os.environ.get("SSP_MODEL_CACHE")
    if env:
        return Path(env).expanduser()
    return Path(os.path.expanduser("~/.cache/ssptyping/models"))


@dataclass(frozen=True)
class ModelInfo:
    name: str
    abbrv: str
    description: str
    repo_id: str
    revision: str
    onnx_file: str

    @property
    def url(self) -> str:
        return (f"https://huggingface.co/{self.repo_id}/resolve/"
                f"{self.revision}/onnx_checkpoints/{self.onnx_file}")

    def local_path(self, directory: Optional[Path] = None) -> Path:
        return (directory or cache_dir()) / self.onnx_file


def fetch_registry(timeout: float = 30.0) -> Dict[str, ModelInfo]:
    """Fetch and parse the model zoo registry. Network-free callers: don't."""
    try:
        with urllib.request.urlopen(REGISTRY_URL, timeout=timeout) as r:
            raw = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        raise ModelFetchError(f"could not fetch model registry: {e}") from e
    out: Dict[str, ModelInfo] = {}
    for name, m in (raw.get("models") or {}).items():
        if not m.get("onnx_file") or not m.get("hf_repo_id"):
            continue  # no ONNX export -> not usable by the standalone runtime
        out[name] = ModelInfo(
            name=name,
            abbrv=m.get("abbrvName") or name,
            description=m.get("description") or "",
            repo_id=m["hf_repo_id"],
            revision=m.get("hf_revision") or "main",
            onnx_file=m["onnx_file"],
        )
    if not out:
        raise ModelFetchError("model registry contained no usable ONNX models")
    return out


def download_model(
    info: ModelInfo,
    dest_dir: Optional[Path] = None,
    progress: Optional[Callable[[int, Optional[int]], None]] = None,
    timeout: float = 120.0,
) -> Path:
    """Stream ``info`` into ``dest_dir`` (default: cache). Atomic on success,
    cleans up the partial file on any failure."""
    dest_dir = Path(dest_dir) if dest_dir is not None else cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / info.onnx_file
    part = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(info.url, timeout=timeout) as r:
            try:
                total = int(r.headers.get("Content-Length") or 0) or None
            except ValueError:
                total = None
            done = 0
            with open(part, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, total)
    except Exception as e:
        part.unlink(missing_ok=True)
        raise ModelFetchError(f"download failed for {info.name!r}: {e}") from e
    os.replace(part, dest)
    return dest


def resolve(explicit: Optional[str] = None) -> Path:
    """Return the model path to use: explicit arg > $SSP_MODEL > cached default.

    Raises :class:`ModelFetchError` with an actionable message when the chosen
    source is missing. Never touches the network.
    """
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p
        raise ModelFetchError(f"model not found: {p}")
    env = os.environ.get("SSP_MODEL")
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
        raise ModelFetchError(f"$SSP_MODEL points to a missing file: {p}")
    p = cache_dir() / DEFAULT_MODEL_FILE
    if p.is_file():
        return p
    raise ModelFetchError(
        "no segmentation model found; download the default with:\n"
        "  ssp-typing models download\n"
        f"(expected at {p})")
