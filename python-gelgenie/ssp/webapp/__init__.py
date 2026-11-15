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

Local web app for interactive SSP typing (Phase 3).

``create_app(AppConfig(...))`` returns a FastAPI application serving:

    GET  /                     -> the UI (static/) or an API pointer
    /ui/                       -> static UI assets (when present)
    POST /api/sessions         -> upload image + tray, run full pipeline
    GET  /api/sessions         -> session index
    GET  /api/sessions/{sid}   -> full session state (analysis + typing + edits)
    POST /api/sessions/{sid}/edits
                               -> apply lane/band edits + match_tol, re-type
    GET  /api/sessions/{sid}/image | /mask | /typing.json | /report.txt
    GET  /api/trays            -> stored tray list
    POST /api/trays            -> upload a tray for reuse across sessions

Design notes: no websockets — segmentation of a full gel takes a few seconds,
so plain request/response with a busy indicator in the UI is simpler and
sufficient. All state lives on disk (see sessions.py) so restarts and audits
are trivial. Expected user errors surface as HTTP 400 with a message; missing
sessions as 404.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ssp import __version__
from ssp.webapp.sessions import AppConfig, SessionStore

__all__ = ["create_app", "AppConfig", "SessionStore"]


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="SSP Typing", version=__version__)
    store = SessionStore(config)
    app.state.config = config
    app.state.store = store

    static_dir = Path(__file__).parent / "static"

    @app.get("/", include_in_schema=False)
    def root():
        if (static_dir / "index.html").exists():
            return RedirectResponse("/ui/")
        return {"app": "ssp-typing", "version": __version__,
                "ui": "not installed", "api_docs": "/docs"}

    if static_dir.is_dir():
        app.mount("/ui", StaticFiles(directory=str(static_dir), html=True),
                  name="ui")

    # ---------- sessions ----------

    @app.get("/api/sessions")
    def list_sessions():
        return store.list_sessions()

    @app.post("/api/sessions")
    async def create_session(
        image: UploadFile = File(...),
        tray: Optional[UploadFile] = File(None),
        tray_name: Optional[str] = Form(None),
        min_area: Optional[int] = Form(None),
        invert: bool = Form(False),
        percentile_norm: bool = Form(False),
    ):
        if tray is None and not tray_name:
            raise HTTPException(400, "provide a tray file or tray_name")
        image_bytes = await image.read()
        tray_bytes = await tray.read() if tray is not None else None
        try:
            return store.create_session(
                image.filename or "image.png", image_bytes,
                tray_filename=tray.filename if tray is not None else None,
                tray_bytes=tray_bytes, tray_name=tray_name,
                min_area=min_area, invert=invert,
                percentile_norm=percentile_norm)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except KeyError as e:
            raise HTTPException(404, str(e))

    @app.get("/api/sessions/{sid}")
    def get_session(sid: str):
        try:
            return store.load_state(sid)
        except KeyError:
            raise HTTPException(404, f"no such session: {sid}")

    @app.post("/api/sessions/{sid}/edits")
    async def apply_edits(sid: str, body: Dict = Body(...)):
        try:
            return store.apply_edits(sid, body)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except KeyError:
            raise HTTPException(404, f"no such session: {sid}")

    @app.get("/api/sessions/{sid}/image", include_in_schema=False)
    def get_image(sid: str):
        try:
            path, media_type = store.image_file(sid)
        except KeyError:
            raise HTTPException(404, f"no such session/file: {sid}")
        return FileResponse(path, media_type=media_type,
                            headers={"X-Content-Type-Options": "nosniff"})

    @app.get("/api/sessions/{sid}/mask", include_in_schema=False)
    def get_mask(sid: str):
        try:
            path = store.session_file(sid, "mask.png")
        except KeyError:
            raise HTTPException(404, f"no such session/file: {sid}")
        return FileResponse(path, media_type="image/png",
                            headers={"X-Content-Type-Options": "nosniff"})

    @app.get("/api/sessions/{sid}/typing.json")
    def download_typing(sid: str):
        try:
            path = store.session_file(sid, "typing.json")
        except KeyError:
            raise HTTPException(404, f"no such session/file: {sid}")
        return FileResponse(path, media_type="application/json",
                            filename=f"typing_{sid}.json")

    @app.get("/api/sessions/{sid}/report.txt")
    def download_report(sid: str):
        try:
            path = store.session_file(sid, "report.txt")
        except KeyError:
            raise HTTPException(404, f"no such session/file: {sid}")
        return FileResponse(path, media_type="text/plain",
                            filename=f"report_{sid}.txt")

    # ---------- tray store ----------

    @app.get("/api/trays")
    def list_trays():
        return store.list_trays()

    @app.post("/api/trays")
    async def upload_tray(tray: UploadFile = File(...)):
        data = await tray.read()
        try:
            stem, t = store.save_tray(tray.filename or "tray.xlsx", data)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return JSONResponse({"name": stem, "tray": t.name,
                             "lanes": len(t.lanes)})

    return app
