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
"""

import io
import json
import urllib.request

import pytest
from click.testing import CliRunner

import ssp.models as models
from ssp.cli import cli
from ssp.models import ModelFetchError, ModelInfo, download_model, fetch_registry, resolve

REGISTRY_FIXTURE = {
    "models": {
        "GelGenie-Universal-Dec-2023": {
            "abbrvName": "Universal Model",
            "modelName": "GelGenie-Universal-Dec-2023",
            "description": "SMPUNet trained on the whole dataset",
            "hf_repo_id": "mattaq/GelGenie-Universal-Dec-2023",
            "hf_revision": "main",
            "torchscript_file": "unet_dec_21_epoch_579.pt",
            "onnx_file": "unet_dec_21_epoch_579.onnx",
            "modelType": "Primary",
        },
        "No-Onnx-Model": {          # must be skipped: no onnx_file
            "hf_repo_id": "a/b",
            "torchscript_file": "y.pt",
        },
    }
}


class _FakeHTTP:
    """urlopen stand-in: serves bytes, reports Content-Length, can fail mid-stream."""

    def __init__(self, data: bytes, fail_after=None):
        self._buf = io.BytesIO(data)
        self.headers = {"Content-Length": str(len(data))}
        self._fail_after = fail_after
        self._sent = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        chunk = self._buf.read(n)
        if self._fail_after is not None:
            self._sent += len(chunk)
            if self._sent > self._fail_after:
                raise OSError("connection reset by peer")
        return chunk


@pytest.fixture()
def fake_registry(monkeypatch):
    data = json.dumps(REGISTRY_FIXTURE).encode()
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _FakeHTTP(data))


# ---------- registry ----------

def test_fetch_registry_parses_and_skips_non_onnx(fake_registry):
    reg = fetch_registry()
    assert set(reg) == {"GelGenie-Universal-Dec-2023"}
    info = reg["GelGenie-Universal-Dec-2023"]
    assert info.abbrv == "Universal Model"
    assert info.url == ("https://huggingface.co/mattaq/GelGenie-Universal-Dec-2023/"
                        "resolve/main/onnx_checkpoints/unet_dec_21_epoch_579.onnx")


def test_fetch_registry_network_error(monkeypatch):
    def boom(*a, **k):
        raise OSError("no route to host")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(ModelFetchError, match="could not fetch"):
        fetch_registry()


def test_fetch_registry_malformed_shapes(monkeypatch):
    """Garbage registries must raise ModelFetchError, never AttributeError."""
    bad_shapes = [
        [],                                          # top level not a dict
        {"models": []},                              # models not a dict
        {"models": {"x": "not-a-dict"}},             # entry not a dict
        {"models": {"x": {"onnx_file": "a.onnx"}}},  # missing hf_repo_id -> empty
    ]
    for bad in bad_shapes:
        data = json.dumps(bad).encode()
        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda *a, **k: _FakeHTTP(data))
        with pytest.raises(ModelFetchError):
            fetch_registry()


# ---------- download ----------

def test_download_model_atomic_and_progress(monkeypatch, tmp_path):
    payload = b"M" * (3 << 20)  # 3 MiB -> several chunks
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _FakeHTTP(payload))
    info = ModelInfo("n", "N", "d", "repo/id", "main", "model.onnx")
    seen = []
    out = download_model(info, dest_dir=tmp_path,
                         progress=lambda done, total: seen.append((done, total)))
    assert out == tmp_path / "model.onnx"
    assert out.read_bytes() == payload
    assert not list(tmp_path.glob("*.part"))
    assert seen and seen[-1] == (len(payload), len(payload))


def test_download_model_failure_cleans_partial(monkeypatch, tmp_path):
    payload = b"M" * (3 << 20)
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: _FakeHTTP(payload, fail_after=1 << 20))
    info = ModelInfo("n", "N", "d", "repo/id", "main", "model.onnx")
    with pytest.raises(ModelFetchError, match="download failed"):
        download_model(info, dest_dir=tmp_path)
    assert not list(tmp_path.iterdir())   # no .part, no truncated final


def test_download_model_truncated_stream_rejected(monkeypatch, tmp_path):
    """Clean EOF before Content-Length must not os.replace a short file."""
    resp = _FakeHTTP(b"short")
    resp.headers = {"Content-Length": "99999"}   # lies about the body size
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: resp)
    info = ModelInfo("n", "N", "d", "repo/id", "main", "model.onnx")
    with pytest.raises(ModelFetchError, match="truncated"):
        download_model(info, dest_dir=tmp_path)
    assert not list(tmp_path.iterdir())


# ---------- resolution ----------

def test_resolve_priority(monkeypatch, tmp_path):
    explicit = tmp_path / "e.onnx"; explicit.write_bytes(b"x")
    env = tmp_path / "env.onnx"; env.write_bytes(b"x")
    cache = tmp_path / "cache"; cache.mkdir()
    (cache / models.DEFAULT_MODEL_FILE).write_bytes(b"x")
    monkeypatch.setenv("SSP_MODEL_CACHE", str(cache))

    assert resolve(str(explicit)) == explicit
    monkeypatch.setenv("SSP_MODEL", str(env))
    assert resolve() == env                      # env beats cache
    assert resolve(str(explicit)) == explicit    # explicit beats env
    monkeypatch.delenv("SSP_MODEL")
    assert resolve() == cache / models.DEFAULT_MODEL_FILE


def test_resolve_errors_are_actionable(monkeypatch, tmp_path):
    monkeypatch.delenv("SSP_MODEL", raising=False)
    monkeypatch.setenv("SSP_MODEL_CACHE", str(tmp_path / "empty"))
    with pytest.raises(ModelFetchError, match="model not found"):
        resolve(str(tmp_path / "missing.onnx"))
    with pytest.raises(ModelFetchError, match="models download"):
        resolve()
    broken = tmp_path / "broken.onnx"  # env set but file absent
    monkeypatch.setenv("SSP_MODEL", str(broken))
    with pytest.raises(ModelFetchError, match="SSP_MODEL"):
        resolve()


# ---------- CLI ----------

def test_cli_models_list(fake_registry, tmp_path, monkeypatch):
    monkeypatch.setenv("SSP_MODEL_CACHE", str(tmp_path))
    r = CliRunner().invoke(cli, ["models", "list"])
    assert r.exit_code == 0, r.output
    assert "GelGenie-Universal-Dec-2023" in r.output
    assert "not downloaded" in r.output


def test_cli_models_download_and_list_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("SSP_MODEL_CACHE", str(tmp_path))
    payload = b"ONNXDATA"
    reg_data = json.dumps(REGISTRY_FIXTURE).encode()

    def fake_urlopen(url, *a, **k):
        return _FakeHTTP(reg_data if "registry.json" in str(url) else payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    r = CliRunner().invoke(cli, ["models", "download"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / models.DEFAULT_MODEL_FILE).read_bytes() == payload
    assert "saved" in r.output

    r = CliRunner().invoke(cli, ["models", "list"])
    assert f"cached: {tmp_path / models.DEFAULT_MODEL_FILE}" in r.output


def test_cli_models_download_unknown_name(fake_registry, tmp_path, monkeypatch):
    monkeypatch.setenv("SSP_MODEL_CACHE", str(tmp_path))
    r = CliRunner().invoke(cli, ["models", "download", "Nope-9000"])
    assert r.exit_code != 0
    assert "unknown model" in r.output and "GelGenie-Universal-Dec-2023" in r.output


def test_cli_segment_without_model_gives_hint(tmp_path, monkeypatch):
    monkeypatch.delenv("SSP_MODEL", raising=False)
    monkeypatch.setenv("SSP_MODEL_CACHE", str(tmp_path / "empty"))
    img = tmp_path / "gel.png"
    import numpy as np
    import imageio.v2 as imageio
    imageio.imwrite(str(img), np.zeros((64, 64), dtype=np.uint8))
    r = CliRunner().invoke(cli, ["segment", str(img), "-o", str(tmp_path / "o")])
    assert r.exit_code != 0
    assert "models download" in r.output
    assert "Traceback" not in r.output


def test_cli_serve_autodownloads_default(tmp_path, monkeypatch):
    import ssp.cli as cli_mod
    monkeypatch.setenv("SSP_MODEL_CACHE", str(tmp_path))
    monkeypatch.delenv("SSP_MODEL", raising=False)
    captured = {}
    monkeypatch.setattr(cli_mod, "_run_server",
                        lambda config, host, port: captured.update(config=config))
    payload = b"ONNXDATA"
    reg_data = json.dumps(REGISTRY_FIXTURE).encode()

    def fake_urlopen(url, *a, **k):
        return _FakeHTTP(reg_data if "registry.json" in str(url) else payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    r = CliRunner().invoke(cli, ["serve", "--no-browser",
                                 "--data-dir", str(tmp_path / "data")])
    assert r.exit_code == 0, r.output
    assert captured["config"].model_path == str(tmp_path / models.DEFAULT_MODEL_FILE)


def test_cli_serve_autodownload_failure_is_clean(tmp_path, monkeypatch):
    monkeypatch.setenv("SSP_MODEL_CACHE", str(tmp_path))
    monkeypatch.delenv("SSP_MODEL", raising=False)
    def boom(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    r = CliRunner().invoke(cli, ["serve", "--no-browser",
                                 "--data-dir", str(tmp_path / "data")])
    assert r.exit_code != 0
    assert "Traceback" not in r.output
    assert "model" in r.output.lower()
