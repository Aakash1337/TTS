"""FastAPI backend for the TTS Reader web app.

A thin shell over :mod:`ttscore.pipeline`: it accepts pasted text, a URL, or a
PDF upload, runs generation in a background thread (serialized so one job uses
the GPU/voice at a time), and serves the resulting MP3. The UI is a single static
page (``webapp/static/index.html``).

Paths resolve from this file, so it works no matter what directory it's launched
from — outputs always land under the project root (E:\\TTS\\output).
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ttscore.config import Config
from ttscore.pipeline import Source, preview_path_for, run

ROOT = Path(__file__).resolve().parents[1]          # E:\TTS
STATIC = Path(__file__).resolve().parent / "static"
OUTPUT_DIR = ROOT / "output"
UPLOAD_DIR = ROOT / ".uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="TTS Reader")

# In-memory job registry (fine for a local, single-user app).
JOBS: dict[str, dict] = {}
_GEN_LOCK = threading.Lock()   # one generation at a time (GPU + logging safety)
_MAX_JOBS = 40                 # prune finished registry entries beyond this


def _prune_jobs() -> None:
    """Drop the oldest finished jobs so a long-lived server doesn't grow forever
    (dicts preserve insertion order; audio files on disk are left alone)."""
    finished = [k for k, v in JOBS.items() if v["status"] in ("done", "failed")]
    for k in finished[: max(0, len(JOBS) - _MAX_JOBS)]:
        JOBS.pop(k, None)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/api/generate")
async def generate(
    input_type: str = Form(...),               # "text" | "url" | "pdf"
    text: str = Form(""),
    url: str = Form(""),
    voice: str = Form("en-US-AriaNeural"),
    engine: str = Form("edge"),                # "edge" | "chatterbox"
    speed: float = Form(1.0),
    mode: str = Form("plain"),                 # plain|article|academic|narrative|study|auto
    summarize: str = Form("false"),
    file: Optional[UploadFile] = File(None),
) -> JSONResponse:
    cfg = _base_cfg().merged_with({
        "engine": engine,
        "edge_voice": voice,
        "speed": max(0.5, min(2.0, float(speed))),
        "mode": mode,
        "summarize": str(summarize).lower() in ("true", "1", "on", "yes"),
    })

    _sweep_stale_previews()
    job_id = uuid.uuid4().hex[:12]
    out_path = OUTPUT_DIR / f"web_{job_id}.mp3"
    upload_dir: Optional[Path] = None   # per-job dir, removed after the run
    title_fallback = "Narration"

    if input_type == "pdf":   # the app's "file" tab: PDF or EPUB
        if file is None or not (file.filename or "").strip():
            raise HTTPException(400, "No file uploaded.")
        low = file.filename.lower()
        if not low.endswith((".pdf", ".epub")):
            raise HTTPException(400, "Please upload a .pdf or .epub file.")
        # Per-job subdir keeps the ORIGINAL stem (nicer title fallback) and makes
        # post-job cleanup a single rmtree.
        upload_dir = UPLOAD_DIR / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / _safe_name(file.filename)
        with open(dest, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
        src = Source("epub" if low.endswith(".epub") else "pdf", str(dest), str(out_path))
        title_fallback = dest.stem
    elif input_type == "url":
        if not url.strip():
            raise HTTPException(400, "No URL provided.")
        clean = url.strip()
        kind = "pdf" if clean.lower().split("?", 1)[0].rstrip("/").endswith(".pdf") else "url"
        src = Source(kind, clean, str(out_path))
        try:
            title_fallback = urlparse(clean).netloc or "Web page"
        except Exception:
            title_fallback = "Web page"
    else:  # text
        if not text.strip():
            raise HTTPException(400, "No text provided.")
        src = Source("text", text, str(out_path))
        title_fallback = "Pasted text"

    JOBS[job_id] = {"status": "queued", "done": 0, "total": 0,
                    "output": None, "out_path": str(out_path),
                    "error": None, "title": None, "audio_seconds": 0.0}
    _prune_jobs()
    threading.Thread(target=_run_job,
                     args=(job_id, src, cfg, upload_dir, title_fallback),
                     daemon=True).start()
    return JSONResponse({"job_id": job_id})


@app.post("/api/cancel/{job_id}")
def cancel(job_id: str) -> JSONResponse:
    """Flag a queued/running job for cancellation (it aborts at the next chunk)."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    job["cancel"] = True
    return JSONResponse({"ok": True})


@app.get("/api/status/{job_id}")
def status(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    public = {k: v for k, v in job.items() if k not in ("output", "out_path")}
    public["has_audio"] = bool(job.get("output"))
    public["has_timings"] = bool(job.get("output")) and _timings_path(job).is_file()
    public["has_preview"] = (job["status"] == "running"
                             and preview_path_for(Path(job["out_path"])).is_file())
    return JSONResponse(public)


@app.get("/api/preview/{job_id}")
def preview(job_id: str):
    """The first ~25s of a still-generating narration."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    p = preview_path_for(Path(job["out_path"]))
    if not p.is_file():
        raise HTTPException(404, "preview not ready")
    return FileResponse(str(p), media_type="audio/mpeg")


@app.get("/api/timings/{job_id}")
def timings(job_id: str):
    """Word-level timing data for the read-along view (edge voices only)."""
    job = JOBS.get(job_id)
    if job is None or not job.get("output"):
        raise HTTPException(404, "timings not available")
    p = _timings_path(job)
    if not p.is_file():
        raise HTTPException(404, "timings not available")
    return FileResponse(str(p), media_type="application/json")


def _timings_path(job: dict) -> Path:
    return Path(job["output"]).with_suffix(".words.json")


@app.get("/api/audio/{job_id}")
def audio(job_id: str):
    job = JOBS.get(job_id)
    if job is None or not job.get("output"):
        raise HTTPException(404, "audio not ready")
    p = Path(job["output"])
    if not p.is_file():
        raise HTTPException(404, "audio file missing")
    name = (job.get("title") or p.stem) + ".mp3"
    return FileResponse(str(p), media_type="audio/mpeg", filename=_safe_name(name))


# ── library: everything ever generated into output/ ─────────────────────────
_STEM_RE = re.compile(r"^[\w.\- ]{1,120}$")


@app.get("/api/library")
def library() -> JSONResponse:
    """All narrations in the output folder, newest first, from .meta.json
    sidecars (older files fall back to filename + mtime)."""
    items = []
    for mp3 in sorted(OUTPUT_DIR.glob("*.mp3"),
                      key=lambda p: p.stat().st_mtime, reverse=True)[:200]:
        meta: dict = {}
        mpath = mp3.with_suffix(".meta.json")
        if mpath.is_file():
            try:
                meta = json.loads(mpath.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        st = mp3.stat()
        items.append({
            "stem": mp3.stem,
            "title": meta.get("title") or mp3.stem,
            "seconds": meta.get("audio_seconds"),
            "voice": meta.get("voice"),
            "created": meta.get("created")
                       or datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "size": st.st_size,
            "has_timings": mp3.with_suffix(".words.json").is_file(),
        })
    return JSONResponse({"items": items})


def _library_path(stem: str, ext: str) -> Path:
    if not _STEM_RE.match(stem or ""):
        raise HTTPException(400, "bad name")
    p = OUTPUT_DIR / f"{stem}{ext}"
    if OUTPUT_DIR.resolve() != p.resolve().parent:   # no traversal, ever
        raise HTTPException(400, "bad name")
    if not p.is_file():
        raise HTTPException(404, "not found")
    return p


@app.get("/api/library/audio/{stem}")
def library_audio(stem: str):
    return FileResponse(str(_library_path(stem, ".mp3")),
                        media_type="audio/mpeg", filename=f"{stem}.mp3")


@app.get("/api/library/timings/{stem}")
def library_timings(stem: str):
    return FileResponse(str(_library_path(stem, ".words.json")),
                        media_type="application/json")


@app.delete("/api/library/{stem}")
def library_delete(stem: str) -> JSONResponse:
    _library_path(stem, ".mp3")   # validates the stem + existence
    for ext in (".mp3", ".words.json", ".meta.json", ".txt", ".wav"):
        try:
            (OUTPUT_DIR / f"{stem}{ext}").unlink(missing_ok=True)
        except OSError:
            pass
    return JSONResponse({"ok": True})


# ── helpers ───────────────────────────────────────────────────────────────────
def _base_cfg() -> Config:
    return Config(
        output_dir=str(OUTPUT_DIR),
        cache_dir=str(ROOT / ".tts_cache"),
        log_dir=str(ROOT / "logs"),
        keep_wav=False,          # web jobs: don't leave a 5-10x raw .wav per MP3
        write_transcript=False,  # nor a .txt the browser user never sees
        preview_seconds=25.0,    # start listening while long docs still generate
        ocr="auto",              # scanned/image-only PDF pages OCR automatically
    )


def _run_job(job_id: str, src: Source, cfg: Config,
             upload_dir: Optional[Path] = None, title_fallback: str = "") -> None:
    job = JOBS[job_id]
    with _GEN_LOCK:                       # serialize: one job on the GPU/voice at a time
        if job.get("cancel"):             # cancelled while still queued
            job["status"] = "failed"
            job["error"] = "Cancelled."
            _cleanup_job_dirs(job_id, upload_dir, cfg)
            return
        job["status"] = "running"

        def progress(done: int, total: int) -> None:
            job["done"], job["total"] = done, total
            if job.get("cancel"):
                # Raising here aborts between chunks; run() converts it into a
                # failed JobResult whose error we surface as "Cancelled."
                raise RuntimeError("Cancelled.")

        try:
            report = run([src], cfg, progress=progress)
            res = report.jobs[-1]
            if res.status == "failed":
                job["status"] = "failed"
                job["error"] = res.error or "generation failed"
            else:
                job["status"] = "done"
                job["output"] = res.output
                job["title"] = res.title or title_fallback or "Narration"
                job["audio_seconds"] = res.audio_seconds
        except Exception as exc:  # pragma: no cover
            job["status"] = "failed"
            job["error"] = str(exc)
        finally:
            _cleanup_job_dirs(job_id, upload_dir, cfg)


def _cleanup_job_dirs(job_id: str, upload_dir: Optional[Path], cfg: Config) -> None:
    """Web jobs are one-shot: the uploaded PDF and the per-job chunk cache
    (keyed by the unique web_<id> stem, so it can never be re-hit) are garbage
    once the run ends — remove both, plus the early-listen preview file (the
    client swaps to the final audio the moment the job reports done)."""
    for d in (upload_dir, Path(cfg.cache_dir) / f"web_{job_id}"):
        if d is not None:
            shutil.rmtree(d, ignore_errors=True)
    try:
        preview_path_for(OUTPUT_DIR / f"web_{job_id}.mp3").unlink(missing_ok=True)
    except OSError:
        pass  # may be mid-download on Windows; the stale sweep will get it


def _sweep_stale_previews() -> None:
    """Previews are transient; clear any left behind by crashes/locks."""
    import time
    cutoff = time.time() - 2 * 3600
    for p in OUTPUT_DIR.glob("*.preview.mp3"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


def _safe_name(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in (name or ""))
    return cleaned[:60] or "audio.mp3"
