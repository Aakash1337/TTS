"""The arrange layer: raw text -> clean, speakable text.

    rules (always)  ->  optional local-LLM polish (if cfg.use_llm)

The active :class:`Mode` decides which rule toggles fire and which LLM prompt is
used. Returns the arranged text plus whether the LLM actually ran.
"""

from __future__ import annotations

from ..config import Config
from .modes import MODES, Mode, get_mode
from .rules import clean

__all__ = ["arrange", "MODES", "Mode", "get_mode", "clean"]


def arrange(raw_text: str, cfg: Config) -> tuple[str, bool]:
    """Return ``(arranged_text, used_llm)``."""
    mode = get_mode(cfg.mode)
    text = clean(raw_text, mode, cfg.expand_numbers)

    used_llm = False
    if cfg.use_llm and text.strip():
        from . import llm
        polished = llm.polish(text, mode, cfg)
        if polished and polished.strip():
            text = polished
            used_llm = True

    return text, used_llm
