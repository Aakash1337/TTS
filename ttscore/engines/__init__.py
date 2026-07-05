"""Pluggable TTS engines.

Every backend implements :class:`~ttscore.engines.base.TTSEngine` (text in ->
mono float32 audio out). The rest of the pipeline only ever talks to that
interface, so swapping Chatterbox for a cloud voice later is a one-line config
change plus a new module here.
"""

from __future__ import annotations

from .base import TTSEngine, create_engine

__all__ = ["TTSEngine", "create_engine"]
