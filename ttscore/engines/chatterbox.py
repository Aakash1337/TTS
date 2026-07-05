"""Chatterbox TTS engine (Resemble AI, MIT).

Ported from the AI Dubbing pipeline's ``tts.py`` and wrapped in the
:class:`~ttscore.engines.base.TTSEngine` interface.

Verified against chatterbox-tts 0.1.7, whose English signature is::

    generate(text, repetition_penalty=1.2, min_p=0.05, top_p=1.0,
             audio_prompt_path=None, exaggeration=0.5, cfg_weight=0.5,
             temperature=0.8)

Two things that bite people, handled here:
  * ``audio_prompt_path`` is the 5th positional arg, so everything except
    ``text`` is passed by keyword.
  * the model has no ``seed`` argument — reproducibility is seeded externally,
    per chunk, so a clip is a pure function of (seed, index, text, params).

``generate()`` returns a CPU float32 tensor shaped ``(1, N)`` at ``model.sr``
(24000 Hz); we expose it as a 1-D float32 numpy array. Outputs carry Resemble's
inaudible Perth watermark by design.
"""

from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from .base import TTSEngine, reference_signature

log = logging.getLogger("ttscore")


def _default_model_cache() -> Path:
    """Project-local folder for downloaded model weights (``<root>/models``).
    Keeps the ~2 GB Chatterbox weights off the system drive."""
    return Path(__file__).resolve().parents[2] / "models"


def resolve_device(requested: str) -> str:
    """Honour the requested device, falling back to CPU if CUDA is absent."""
    import torch

    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        log.warning("device='cuda' requested but no CUDA GPU is available — using CPU (slow).")
        return "cpu"
    return "cpu"


class ChatterboxEngine(TTSEngine):
    """Loads Chatterbox once and synthesizes one chunk at a time."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._reference = cfg.reference_wav or None

        # Default the HuggingFace cache to a project-local folder (so weights
        # land next to the project, not on a possibly-full system drive) unless
        # the user has explicitly set HF_HOME. Must precede the chatterbox import.
        os.environ.setdefault("HF_HOME", str(_default_model_cache()))

        import torch
        from chatterbox.tts import ChatterboxTTS

        self._torch = torch
        self.device = resolve_device(cfg.device)

        # Cap VRAM so a co-resident model (e.g. an Ollama cleanup pass) or a
        # second worker can fit alongside. None = use all VRAM.
        if self.device == "cuda" and cfg.cuda_mem_fraction:
            try:
                torch.cuda.set_per_process_memory_fraction(float(cfg.cuda_mem_fraction))
            except Exception as exc:  # non-fatal: just means no cap
                log.warning("could not set CUDA mem fraction: %s", exc)

        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.sample_rate = int(getattr(self.model, "sr", 24000))
        self.model_id = f"ChatterboxTTS@{self.device}"

        # Only forward kwargs this installed version actually accepts — the
        # generate() signature has gained params across releases.
        try:
            self._accepted = set(inspect.signature(self.model.generate).parameters)
        except (ValueError, TypeError):
            self._accepted = set()

    def _generate_kwargs(self) -> dict[str, Any]:
        wanted = {
            "audio_prompt_path": self._reference,
            "exaggeration": self.cfg.exaggeration,
            "cfg_weight": self.cfg.cfg_weight,
            "temperature": self.cfg.temperature,
        }
        if not self._accepted:  # couldn't introspect; trust these stable names
            return {k: v for k, v in wanted.items() if v is not None}
        return {k: v for k, v in wanted.items()
                if k in self._accepted and v is not None}

    def synthesize(self, text: str, index: int = 0) -> np.ndarray:
        """Return mono float32 audio at :attr:`sample_rate` for one chunk."""
        text = text.strip()
        if not text:
            return np.zeros(int(0.05 * self.sample_rate), dtype=np.float32)

        if self.cfg.seed is not None:
            self._seed_for(index)
        wav = self.model.generate(text, **self._generate_kwargs())
        return _to_mono_f32(wav)

    def signature(self) -> dict:
        sig = super().signature()
        sig.update({
            "reference_wav": reference_signature(self._reference),
            "exaggeration": self.cfg.exaggeration,
            "cfg_weight": self.cfg.cfg_weight,
            "temperature": self.cfg.temperature,
            "seed": self.cfg.seed,
        })
        if self.cfg.seed is not None:
            sig["seed_scheme"] = "per_chunk_v1"
        return sig

    def _seed_for(self, index: int) -> None:
        """Seed all RNGs from (seed, index) so each chunk is reproducible
        regardless of generation order or cache state."""
        import random
        s = (int(self.cfg.seed) * 1000003 + int(index)) & 0x7FFFFFFF
        random.seed(s)
        np.random.seed(s)
        self._torch.manual_seed(s)
        if self.device == "cuda":
            self._torch.cuda.manual_seed_all(s)

    def reset(self) -> None:
        """Free cached VRAM between documents to avoid creep over a long run."""
        if self.device == "cuda":
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass


def _to_mono_f32(wav: Any) -> np.ndarray:
    """Coerce Chatterbox's (1, N) tensor (or any array) to a 1-D float32 array."""
    try:
        import torch
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().to("cpu").numpy()
    except Exception:
        pass
    arr = np.asarray(wav, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
    return np.ascontiguousarray(arr, dtype=np.float32)
