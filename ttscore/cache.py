"""Per-chunk TTS clip cache for cheap resume.

A long document is hundreds of chunks; a crash (or OOM, or Ctrl-C) partway
through shouldn't throw away minutes of GPU work. Each generated clip is written
to disk keyed by a content hash of (voice/param signature + chunk index + text);
on the next run identical chunks load instantly and only the missing ones are
generated.

Clips are stored at the engine's NATIVE sample rate (resampling is deterministic
and cheap, and keeping native rate means changing the output sample_rate does
not invalidate the cache).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


class ChunkCache:
    """Disk cache for one document's generated chunk clips."""

    def __init__(self, root: Path, doc_key: str, signature: dict):
        # ``signature`` holds everything that affects generation but the text:
        # engine/model id, voice ref, exaggeration, cfg_weight, temperature, seed.
        self.dir = Path(root) / _safe(doc_key)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._sig = json.dumps(signature, sort_keys=True, ensure_ascii=False)

    def _hash(self, index: int, text: str) -> str:
        h = hashlib.sha1()
        h.update(self._sig.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(index).encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.hexdigest()[:16]

    def _path(self, index: int, text: str) -> Path:
        return self.dir / f"{index:05d}_{self._hash(index, text)}.wav"

    def get(self, index: int, text: str) -> Optional[tuple[np.ndarray, int]]:
        path = self._path(index, text)
        if not path.is_file():
            return None
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception:
            return None
        if data.ndim > 1:  # stored mono, but be defensive
            data = data.mean(axis=1)
        return data.astype(np.float32, copy=False), int(sr)

    def put(self, index: int, text: str, clip: np.ndarray, sr: int) -> Path:
        path = self._path(index, text)
        tmp = path.with_suffix(".wav.tmp")
        # Pass format/subtype explicitly: the ".tmp" extension defeats soundfile's
        # format-from-extension inference, and FLOAT keeps the model's native
        # float32 precision (the WAV default would quantize to 16-bit PCM).
        sf.write(str(tmp), np.asarray(clip, dtype=np.float32), sr,
                 format="WAV", subtype="FLOAT")
        tmp.replace(path)  # atomic-ish: never leave a half-written cache file
        return path

    # ── word-timing sidecars (engines that report WordBoundary events) ───────
    def get_words(self, index: int, text: str) -> Optional[list]:
        wpath = self._path(index, text).with_suffix(".words.json")
        if not wpath.is_file():
            return None
        try:
            with open(wpath, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    def put_words(self, index: int, text: str, words: list) -> None:
        wpath = self._path(index, text).with_suffix(".words.json")
        tmp = wpath.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(words, fh, ensure_ascii=False)
        tmp.replace(wpath)


def _safe(name: str) -> str:
    """Make a document key safe as a folder name."""
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return cleaned[:80] or "doc"
