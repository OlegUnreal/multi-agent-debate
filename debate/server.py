"""FastAPI web UI for the multi-agent debate.

Run:  python -m debate.server
Open: http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import SETTINGS
from .logging_config import get_logger, setup_logging
from .loop import run_debate, verdict_of

log = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="multi-agent-debate", version="0.3.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory job store: job_id -> {status, topic, rounds, verdict, error}
_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


class DebateRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=2000)
    max_rounds: int | None = Field(default=None, ge=1, le=20)


class DebateJob(BaseModel):
    id: str
    status: str
    topic: str
    rounds: list[dict[str, Any]] = []
    verdict: str | None = None
    error: str | None = None


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "has_key": SETTINGS.has_key, "model": SETTINGS.model}


def _run_job(job_id: str, topic: str, max_rounds: int | None) -> None:
    with _LOCK:
        _JOBS[job_id]["status"] = "running"
    try:
        if SETTINGS.has_key:
            from .llm import make_llm
            llm = make_llm()
        else:
            from .demo import stub_llm
            llm = stub_llm
        history = run_debate(topic, llm, max_rounds=max_rounds or SETTINGS.max_rounds)
        rounds = [
            {"n": r.n, "proposer": r.proposer, "critic": r.critic, "judge": r.judge}
            for r in history
        ]
        with _LOCK:
            _JOBS[job_id].update(
                status="done",
                rounds=rounds,
                verdict=verdict_of(history),
            )
        log.info("debate_done", extra={"job_id": job_id, "verdict": verdict_of(history), "rounds": len(history)})
    except Exception as exc:  # noqa: BLE001
        with _LOCK:
            _JOBS[job_id].update(status="error", error=f"{type(exc).__name__}: {exc}")
        log.exception("debate_error", extra={"job_id": job_id})


@app.post("/api/debate", response_model=DebateJob)
def start_debate(req: DebateRequest) -> DebateJob:
    import uuid
    job_id = uuid.uuid4().hex[:8]
    with _LOCK:
        _JOBS[job_id] = {"id": job_id, "status": "queued", "topic": req.topic,
                         "rounds": [], "verdict": None, "error": None}
    t = threading.Thread(target=_run_job, args=(job_id, req.topic, req.max_rounds), daemon=True)
    t.start()
    return DebateJob(**_JOBS[job_id])


@app.get("/api/debate/{job_id}", response_model=DebateJob)
def get_debate(job_id: str) -> DebateJob:
    with _LOCK:
        job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return DebateJob(**job)


@app.get("/api/debates")
def list_debates() -> list[DebateJob]:
    with _LOCK:
        return [DebateJob(**j) for j in _JOBS.values()]


def main() -> None:
    setup_logging()
    import uvicorn
    log.info("server_start", extra={"host": "127.0.0.1", "port": 8000, "has_key": SETTINGS.has_key})
    uvicorn.run(app, host="127.0.0.1", port=8000, log_config=None)


if __name__ == "__main__":
    main()
