"""Microsoft Edge online neural voices via edge-tts (free, no API key).

Gives access to Microsoft's "Online (Natural)" voices — including
``en-US-AriaNeural`` = "Microsoft Aria Online (Natural) - English (United
States)". Requires an internet connection (Microsoft's servers do the synthesis).
Selected with ``engine: edge`` / ``--engine edge``.

edge-tts streams 24 kHz mono MP3; we decode each chunk to a float32 array and
the pipeline resamples it like any other engine.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile

import numpy as np

from ..config import Config
from .base import TTSEngine

log = logging.getLogger("ttscore")


def _speed_to_rate(speed: float) -> str:
    """Convert a speed multiplier (1.0 = normal) to an edge-tts rate string."""
    pct = round((speed - 1.0) * 100)
    return f"+{pct}%" if pct >= 0 else f"{pct}%"


class EdgeTTSEngine(TTSEngine):
    """Synthesize one chunk at a time through Microsoft's online neural voices."""

    handles_speed = True           # cfg.speed is applied natively via the voice rate
    supports_word_timings = True   # WordBoundary events — same data Edge's Read Aloud uses

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.voice = cfg.edge_voice
        self.rate = (_speed_to_rate(cfg.speed)
                     if abs(cfg.speed - 1.0) > 1e-6 else cfg.edge_rate)
        self.volume = cfg.edge_volume
        self.pitch = cfg.edge_pitch
        self.sample_rate = 24000                    # edge neural MP3 default rate
        self.model_id = f"edge-tts:{self.voice}"

        try:
            import edge_tts
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "The 'edge' engine needs the edge-tts package. Install it with:\n"
                "  E:\\TTS\\.venv\\Scripts\\python.exe -m pip install edge-tts"
            ) from exc
        self._edge = edge_tts

        # A dedicated, reused event loop. On Windows the Selector loop avoids the
        # noisy "Event loop is closed" teardown of the default Proactor loop when
        # aiohttp is driven once per chunk.
        if sys.platform.startswith("win"):
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            except Exception:
                pass
        self._loop = asyncio.new_event_loop()

    def synthesize(self, text: str, index: int = 0) -> np.ndarray:
        clip, _ = self.synthesize_with_words(text, index)
        return clip

    def synthesize_with_words(self, text: str, index: int = 0):
        """Return ``(clip, words)`` — words are ``{"t","s","e"}`` in seconds
        relative to the clip start, from the service's WordBoundary events."""
        text = text.strip()
        if not text:
            return np.zeros(int(0.05 * self.sample_rate), dtype=np.float32), []

        for attempt in (1, 2):  # one retry for transient network hiccups
            try:
                clip, sr, words = self._speak(text)
                if sr != self.sample_rate:
                    from .. import audio
                    clip = audio.resample(clip, sr, self.sample_rate)
                return clip.astype(np.float32, copy=False), words
            except Exception as exc:
                log.warning("edge-tts failed on chunk %d (attempt %d): %s",
                            index, attempt, exc)
        log.error("edge-tts produced no audio for chunk %d; inserting silence.", index)
        return np.zeros(int(0.3 * self.sample_rate), dtype=np.float32), []

    def _speak(self, text: str) -> tuple[np.ndarray, int, list[dict]]:
        fd, path = tempfile.mkstemp(suffix=".mp3", prefix="edge_")
        os.close(fd)
        try:
            words = self._loop.run_until_complete(self._stream_to(text, path))
            if os.path.getsize(path) == 0:
                raise RuntimeError("empty audio returned")
            clip, sr = _decode(path)
            return clip, sr, words
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    async def _stream_to(self, text: str, path: str) -> list[dict]:
        """Stream synthesis to ``path``; collect WordBoundary events (offsets
        arrive in 100-nanosecond ticks) as second-based word timings."""
        try:
            # edge-tts >= 7 defaults to SentenceBoundary; ask for per-word events.
            comm = self._edge.Communicate(text, self.voice, rate=self.rate,
                                          volume=self.volume, pitch=self.pitch,
                                          boundary="WordBoundary")
        except TypeError:  # older edge-tts: no boundary param, words come anyway
            comm = self._edge.Communicate(text, self.voice, rate=self.rate,
                                          volume=self.volume, pitch=self.pitch)
        words: list[dict] = []
        with open(path, "wb") as fh:
            async for msg in comm.stream():
                if msg["type"] == "audio":
                    fh.write(msg["data"])
                elif msg["type"] == "WordBoundary":
                    start = msg["offset"] / 1e7
                    words.append({"t": str(msg["text"]),
                                  "s": round(start, 3),
                                  "e": round(start + msg["duration"] / 1e7, 3)})
        return words

    def signature(self) -> dict:
        sig = super().signature()
        sig.update({"voice": self.voice, "rate": self.rate,
                    "volume": self.volume, "pitch": self.pitch})
        return sig

    def reset(self) -> None:
        pass


def _decode(path: str) -> tuple[np.ndarray, int]:
    """Decode an MP3 file to (mono float32, sample_rate)."""
    import soundfile as sf
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return np.ascontiguousarray(data, dtype=np.float32), int(sr)
    except Exception:
        import librosa  # robust fallback (uses ffmpeg via audioread)
        data, sr = librosa.load(path, sr=None, mono=True)
        return np.ascontiguousarray(data, dtype=np.float32), int(sr)
