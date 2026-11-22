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

Web app layout step: staged uploads, layout proposals, preview, sessions
made from a raw photo + confirmed layout, and re-cutting a session.
"""

import json
from io import BytesIO

import imageio.v2 as imageio
import numpy as np
import pytest
from fastapi.testclient import TestClient
from skimage.transform import AffineTransform

import ssp.webapp.sessions as sessions_mod
from ssp.webapp import AppConfig, create_app
from test_layout import EMPTY, POSITIVE, _typing_gel
from test_layout_detect import _piece, _render
from test_webapp import _fake_model_factory, _synth_image_bytes, _synth_tray_json


def _png(arr):
    buf = BytesIO()
    imageio.imwrite(buf, arr, format="png")
    return buf.getvalue()


def _tray96():
    from ssp.trays import Tray, TrayLane
    tray = Tray(name="synth96", lanes=[
        TrayLane(mix=m, locus="A", product_bp=[500], control_bp=256,
                 serotypes="", specificities="A*01:01") for m in range(1, 97)])
    return json.dumps(tray.to_dict()).encode()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(sessions_mod, "GelGenieONNX", _fake_model_factory)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"x")
    app = create_app(AppConfig(model_path=str(model), data_dir=str(tmp_path / "data")))
    with TestClient(app) as c:
        c.data_dir = tmp_path / "data"
        yield c


def _stage(client, arr, name="raw.png"):
    r = client.post("/api/uploads", files={"image": (name, _png(arr), "image/png")})
    assert r.status_code == 200, r.text
    return r.json()


def _typing_upload(client):
    gel, layout = _typing_gel()
    return _stage(client, gel), layout


def _create(client, upload_id, layout, tray=None):
    return client.post("/api/sessions", data={
        "upload_id": upload_id, "layout": json.dumps(layout.to_dict()
                                                     if hasattr(layout, "to_dict") else layout),
    }, files={"tray": ("tray.json", tray or _tray96(), "application/json")})


# ---------- staging ----------

def test_stage_upload_reports_size_and_kind(client):
    gel, _ = _typing_gel()
    info = _stage(client, gel)
    assert info["image"] == {"width": gel.shape[1], "height": gel.shape[0]}
    assert info["prestitched"] is False
    assert info["presets"] == ["3x32", "6x16"] and info["last_layouts"] == []
    r = client.get(f"/api/uploads/{info['upload_id']}/image")
    assert r.status_code == 200 and r.headers["x-content-type-options"] == "nosniff"
    wide = _stage(client, np.zeros((40, 400), dtype=np.uint8), "strip.png")
    assert wide["prestitched"] is True


def test_stage_rejects_garbage_and_unknown_ids(client):
    r = client.post("/api/uploads", files={"image": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400
    assert client.get("/api/uploads/zzz/image").status_code == 404
    assert client.get("/api/uploads/0123456789abcdef/image").status_code == 404
    r = client.post("/api/uploads/0123456789abcdef/layout", json={"preset": "3x32"})
    assert r.status_code == 404


# ---------- proposals + preview ----------

def test_propose_detect_and_even(client):
    tf = AffineTransform(rotation=np.deg2rad(1.0), translation=(60, 40))
    img = _render([(_piece(3, 32), tf)], (880, 1120), light_background=True)
    uid = _stage(client, img)["upload_id"]
    r = client.post(f"/api/uploads/{uid}/layout", json={"preset": "3x32"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["source"] == "detect"
    assert sum(row["lanes"] for b in d["layout"]["blocks"] for row in b["rows"]) == 96
    r = client.post(f"/api/uploads/{uid}/layout", json={"preset": "6x16", "source": "even"})
    assert r.json()["source"] == "even" and len(r.json()["layout"]["blocks"]) == 2


def test_propose_falls_back_and_validates(client):
    uid = _stage(client, np.full((300, 400), 128, dtype=np.uint8))["upload_id"]
    d = client.post(f"/api/uploads/{uid}/layout", json={"preset": "3x32"}).json()
    assert d["source"] == "even" and "detection failed" in d["warnings"][0]
    assert client.post(f"/api/uploads/{uid}/layout",
                       json={"preset": "4x24"}).status_code == 400
    assert client.post(f"/api/uploads/{uid}/layout",
                       json={"preset": "3x32", "source": "magic"}).status_code == 400
    r = client.post(f"/api/uploads/{uid}/layout", json={"preset": "3x32", "source": "last"})
    assert r.status_code == 400 and "no confirmed" in r.json()["detail"]


def test_preview_png_of_joined_strip(client):
    info, layout = _typing_upload(client)
    r = client.post(f"/api/uploads/{info['upload_id']}/preview",
                    json={"layout": layout.to_dict()})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    img = imageio.imread(BytesIO(r.content))
    assert img.ndim == 3 and img.shape[1] > img.shape[0] * 4  # a wide strip
    bad = layout.to_dict()
    bad["blocks"][0]["rows"][0]["top"] = 0.9
    r = client.post(f"/api/uploads/{info['upload_id']}/preview", json={"layout": bad})
    assert r.status_code == 400 and "top < bottom" in r.json()["detail"]


# ---------- sessions from a layout ----------

def test_session_from_layout_types_with_known_lanes(client):
    info, layout = _typing_upload(client)
    r = _create(client, info["upload_id"], layout)
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["lane_assignment"] == "layout" and s["has_original"] is True
    assert s["layout_preset"] is None  # the test layout carries no preset
    a = s["analysis"]
    assert a["invalid_lanes"] == [EMPTY]
    assert set(a["positive_lanes"]) == POSITIVE
    # the session image is the joined strip; the raw photo is kept as original
    img = imageio.imread(BytesIO(client.get(f"/api/sessions/{s['id']}/image").content))
    assert (img.shape[1], img.shape[0]) == (s["image"]["width"], s["image"]["height"])
    orig = imageio.imread(BytesIO(client.get(f"/api/sessions/{s['id']}/original").content))
    gel, _ = _typing_gel()
    np.testing.assert_array_equal(orig, gel)
    # staged upload consumed; edits keep using the layout's lanes
    assert client.get(f"/api/uploads/{info['upload_id']}/image").status_code == 404
    e = client.post(f"/api/sessions/{s['id']}/edits",
                    json={"lane_overrides": [{"mix": 7, "positive": False}]}).json()
    assert e["lane_assignment"] == "layout" and 7 not in e["analysis"]["positive_lanes"]
    typing = client.get(f"/api/sessions/{s['id']}/typing.json").json()
    assert typing["analysis"]["lane_assignment"] == "layout"


def test_confirmed_layout_is_remembered_per_preset(client):
    info, layout = _typing_upload(client)
    layout.preset = "6x16"
    assert _create(client, info["upload_id"], layout).status_code == 200
    gel, _ = _typing_gel()
    big = np.kron(gel, np.ones((2, 2), dtype=np.uint8))  # same imager, 2x size
    info2 = _stage(client, big)
    assert info2["last_layouts"] == ["6x16"]
    d = client.post(f"/api/uploads/{info2['upload_id']}/layout",
                    json={"preset": "6x16", "source": "last"}).json()
    q0 = np.array(d["layout"]["blocks"][0]["quad"])
    np.testing.assert_allclose(q0, 2 * np.array(layout.blocks[0].quad), atol=0.02)


def test_layout_errors_leave_upload_for_retry(client):
    info, layout = _typing_upload(client)
    layout.blocks[1].rows[2].lanes = 15
    r = _create(client, info["upload_id"], layout)
    assert r.status_code == 400 and "95 lanes, expected 96" in r.json()["detail"]
    assert client.get("/api/sessions").json() == []
    assert client.get(f"/api/uploads/{info['upload_id']}/image").status_code == 200


@pytest.mark.parametrize("data, msg", [
    ({"layout": "{}"}, "either an image or an upload_id"),
    ({"upload_id": "0123456789abcdef", "layout": "{nope"}, "not valid JSON"),
])
def test_session_request_validation(client, data, msg):
    r = client.post("/api/sessions", data=data,
                    files={"tray": ("tray.json", _tray96(), "application/json")})
    assert r.status_code == 400 and msg in r.json()["detail"]


def test_layout_requires_staged_upload(client):
    gel, layout = _typing_gel()
    r = client.post("/api/sessions", data={"layout": json.dumps(layout.to_dict())},
                    files={"image": ("raw.png", _png(gel), "image/png"),
                           "tray": ("tray.json", _tray96(), "application/json")})
    assert r.status_code == 400 and "upload_id" in r.json()["detail"]


def test_relayout_restages_original_with_its_layout(client):
    info, layout = _typing_upload(client)
    s = _create(client, info["upload_id"], layout).json()
    r = client.post(f"/api/sessions/{s['id']}/relayout")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["layout"]["blocks"] == layout.to_dict()["blocks"]
    assert d["tray_store_name"] == "tray"
    img = imageio.imread(BytesIO(client.get(f"/api/uploads/{d['upload_id']}/image").content))
    gel, _ = _typing_gel()
    np.testing.assert_array_equal(img, gel)


def test_relayout_needs_a_raw_photo_session(client):
    r = client.post("/api/sessions", files={
        "image": ("gel.png", _synth_image_bytes(), "image/png"),
        "tray": ("tray.json", _synth_tray_json(), "application/json")})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["has_original"] is False and s["lane_assignment"] == "band-gaps"
    r = client.post(f"/api/sessions/{s['id']}/relayout")
    assert r.status_code == 400 and "not made from a raw gel photo" in r.json()["detail"]
    assert client.get(f"/api/sessions/{s['id']}/original").status_code == 404
