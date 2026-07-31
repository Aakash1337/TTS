"""Speech-to-text: pull the transcript out of a video or audio file.

Runs **faster-whisper** locally (CTranslate2 Whisper — no cloud, no key).
The model downloads once into ``<project>/models/whisper`` and is cached per
process. CUDA is used when available, with an automatic CPU (int8) fallback —
on Windows, CTranslate2 finds cuDNN by borrowing torch's bundled DLLs.

``transcribe_media`` accepts any container ffmpeg/PyAV can read (mp4, mkv,
m4v, webm, mov, avi, mp3, wav, m4a, flac, ogg …). Long files report progress
via the ``on_progress`` callback and can be cancelled between segments.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Callable, Optional

from ..config import Config

log = logging.getLogger("ttscore")

MEDIA_EXTS = (".mp4", ".mkv", ".m4v", ".webm", ".mov", ".avi", ".wmv",
              ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus")

_model = None
_model_key: Optional[tuple] = None


class TranscribeCancelled(Exception):
    pass


def _whisper_dir() -> Path:
    d = Path(__file__).resolve().parents[2] / "models" / "whisper"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _add_torch_dlls() -> None:
    """Let CTranslate2 find cuDNN/cuBLAS from torch's bundled DLLs (Windows)."""
    try:
        torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
        if torch_lib.is_dir():
            os.add_dll_directory(str(torch_lib))
    except Exception:
        pass


def _get_model(cfg: Config):
    global _model, _model_key
    key = (cfg.stt_model, cfg.device)
    if _model is not None and _model_key == key:
        return _model

    _add_torch_dlls()
    from faster_whisper import WhisperModel

    root = str(_whisper_dir())
    if cfg.device == "cuda":
        try:
            log.info("Loading Whisper '%s' on CUDA ...", cfg.stt_model)
            _model = WhisperModel(cfg.stt_model, device="cuda",
                                  compute_type="float16", download_root=root)
            _model_key = key
            return _model
        except Exception as exc:
            log.warning("Whisper CUDA load failed (%s) — falling back to CPU.", exc)

    log.info("Loading Whisper '%s' on CPU (int8) ...", cfg.stt_model)
    _model = WhisperModel(cfg.stt_model, device="cpu",
                          compute_type="int8", download_root=root)
    _model_key = (cfg.stt_model, "cpu")
    return _model


def transcribe_media(source: str, cfg: Config, translate: bool = False,
                     on_progress: Optional[Callable[[float], None]] = None,
                     should_cancel: Optional[Callable[[], bool]] = None) -> dict:
    """Return ``{"text", "language", "duration", "segments"}`` for a media file.

    ``text`` is paragraph-formatted prose (a gap of >1.5s between segments
    starts a new paragraph). ``segments`` carry start/end seconds for SRT.
    """
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Media file not found: {source}")

    model = _get_model(cfg)
    segments_iter, info = model.transcribe(
        str(path),
        task="translate" if translate else "transcribe",
        vad_filter=True,           # skip long silences/music-only stretches
        beam_size=5,
    )
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    log.info("Transcribing %s: language=%s (p=%.2f), %.1fs of audio%s",
             path.name, info.language, info.language_probability, duration,
             " -> English" if translate else "")

    segments = []
    for seg in segments_iter:
        if should_cancel and should_cancel():
            raise TranscribeCancelled()
        text = seg.text.strip()
        if text:
            segments.append({"start": float(seg.start), "end": float(seg.end),
                             "text": text})
        if on_progress and duration > 0:
            on_progress(min(1.0, float(seg.end) / duration))

    if not segments:
        raise ValueError("No speech found in this file.")

    return {"text": _to_paragraphs(segments), "language": info.language,
            "duration": duration, "segments": segments}


def _to_paragraphs(segments: list[dict], gap: float = 1.5) -> str:
    """Join segments into flowing prose; a long pause starts a new paragraph."""
    paras: list[list[str]] = [[]]
    prev_end: Optional[float] = None
    for s in segments:
        if prev_end is not None and s["start"] - prev_end > gap and paras[-1]:
            paras.append([])
        paras[-1].append(s["text"])
        prev_end = s["end"]
    return "\n\n".join(" ".join(p) for p in paras if p)


def to_srt(segments: list[dict]) -> str:
    """Standard SRT — the same format the AI-Dubbing pipeline consumes."""
    def clock(t: float) -> str:
        ms = int(round(t * 1000))
        h, rem = divmod(ms, 3600000)
        m, rem = divmod(rem, 60000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, s in enumerate(segments, 1):
        lines.append(f"{i}\n{clock(s['start'])} --> {clock(s['end'])}\n{s['text']}\n")
    return "\n".join(lines)
