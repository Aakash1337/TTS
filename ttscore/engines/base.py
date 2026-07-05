"""The TTS engine contract + a config-driven factory.

An engine turns a short chunk of text into a mono float32 waveform at its own
native sample rate. Resampling to the pipeline's working rate happens later in
:mod:`ttscore.audio`, so engines never worry about it.

``signature()`` returns everything that affects generation EXCEPT the text; it's
folded into the per-chunk cache key so changing voice/params/model invalidates
stale cached clips automatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import Config


class TTSEngine(ABC):
    """Contract every TTS backend implements."""

    sample_rate: int = 24000   # native output rate; set by the subclass on load
    model_id: str = "engine"   # stable id string; part of the cache signature
    handles_speed: bool = False  # True if the engine applies cfg.speed itself
    supports_word_timings: bool = False  # True if synthesize_with_words gives real timings

    @abstractmethod
    def synthesize(self, text: str, index: int = 0) -> np.ndarray:
        """Return mono float32 audio at :attr:`sample_rate` for one text chunk.

        ``index`` lets seed-based engines derive a per-chunk RNG so a given chunk
        is reproducible regardless of generation/cache order.
        """

    def synthesize_with_words(self, text: str, index: int = 0):
        """Like :meth:`synthesize`, but also return per-word timings.

        Returns ``(clip, words)`` where ``words`` is a list of
        ``{"t": word, "s": start_sec, "e": end_sec}`` relative to the clip
        start — or ``None`` when the engine can't provide timings (default).
        """
        return self.synthesize(text, index), None

    def signature(self) -> dict:
        """Voice/param fingerprint for the cache key. Subclasses extend this."""
        return {"model": self.model_id, "native_sr": self.sample_rate}

    def reset(self) -> None:
        """Free per-document resources (e.g. CUDA cache). No-op by default."""


def create_engine(cfg: Config) -> TTSEngine:
    """Instantiate the engine named by ``cfg.engine``."""
    name = cfg.engine
    if name == "chatterbox":
        from .chatterbox import ChatterboxEngine
        return ChatterboxEngine(cfg)
    if name in ("edge", "edgetts", "edge-tts"):
        from .edgetts import EdgeTTSEngine
        return EdgeTTSEngine(cfg)
    raise ValueError(
        f"Unknown TTS engine: {name!r}. Known engines: 'chatterbox', 'edge'. "
        "Add a new module under ttscore/engines/ and register it here."
    )


def reference_signature(reference_wav: Optional[str]) -> Optional[str]:
    """Cheap fingerprint of a reference voice file (name:size:mtime) for the
    cache key, so re-recording the same-named clip still invalidates clips."""
    if not reference_wav:
        return None
    p = Path(reference_wav)
    try:
        st = p.stat()
        return f"{p.name}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return reference_wav
