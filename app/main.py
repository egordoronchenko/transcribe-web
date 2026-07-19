"""FastAPI app: static page + JSON API + SSE."""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import media
from .jobs import HOST_ROOT, manager

app = FastAPI(title="transcribe-web")
STATIC = Path(__file__).parent / "static"


def resolve_rel(rel: str) -> Path:
    """Resolve a client-supplied relative path safely inside HOST_ROOT."""
    p = (HOST_ROOT / rel).resolve()
    root = HOST_ROOT.resolve()
    if p != root and root not in p.parents:
        raise HTTPException(400, "path outside root")
    return p


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    from .core import Config
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    return {
        "key_present": len(key) > 0,
        "model": os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3"),
        "language": os.environ.get("LANGUAGE", "ru"),
        "prompt_default": Config.from_env().prompt or "",
        "running": manager.running,
        "job": manager.state,
    }


@app.get("/api/tree")
def tree(path: str = ""):
    p = resolve_rel(path)
    if not p.is_dir():
        raise HTTPException(404, "not a directory")
    dirs = []
    try:
        for child in sorted(p.iterdir(), key=lambda c: c.name.lower()):
            if child.is_dir() and not child.name.startswith((".", "$")):
                dirs.append(child.name)
    except PermissionError:
        pass
    return {"path": path, "dirs": dirs}


@app.get("/api/scan")
def scan(path: str = ""):
    p = resolve_rel(path)
    if not p.is_dir():
        raise HTTPException(404, "not a directory")
    files = media.scan_media(p)
    return {
        "path": path,
        "files": files,
        "video": sum(1 for f in files if f["kind"] == "video"),
        "audio": sum(1 for f in files if f["kind"] == "audio"),
    }


class JobRequest(BaseModel):
    source: str
    output: str
    files: list[str]
    force: bool = False
    prompt: str | None = None


@app.post("/api/job")
async def start_job(req: JobRequest):
    if manager.running:
        raise HTTPException(409, "задание уже выполняется")
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise HTTPException(400, "OPENROUTER_API_KEY не задан (.env)")
    if not req.files:
        raise HTTPException(400, "не выбрано ни одного файла")
    src = resolve_rel(req.source)
    if not src.is_dir():
        raise HTTPException(400, "папка-источник не найдена")
    resolve_rel(req.output)
    for rp in req.files:
        if not (src / rp).is_file():
            raise HTTPException(400, f"файл не найден: {rp}")
    manager.start(req.source, req.output, req.files, req.force, prompt=req.prompt)
    return {"ok": True}


class CancelRequest(BaseModel):
    hard: bool = False


@app.post("/api/job/cancel")
async def cancel_job(req: CancelRequest):
    manager.cancel(req.hard)
    return {"ok": True}


@app.get("/api/events")
async def events():
    async def gen():
        async for event in manager.subscribe():
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
