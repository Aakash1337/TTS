"""Optional LLM polish via a LOCAL Ollama server (e.g. Gemma).

This is strictly opt-in (``use_llm: true``) and never a hard dependency: if the
server or model is unreachable, we log a warning and return the rules-only text
unchanged. Long inputs are polished paragraph-block by block so we stay within a
small model's comfortable context.

Uses the Ollama HTTP API directly (``/api/generate``) so the only dependency is
``requests`` — no model SDK required.
"""

from __future__ import annotations

import logging
import re

from ..config import Config
from .modes import Mode

log = logging.getLogger("ttscore")

_PARA = re.compile(r"\n\s*\n+")


def available(cfg: Config) -> bool:
    """True if an Ollama server answers at ``cfg.llm_host``."""
    try:
        import requests
        return requests.get(f"{cfg.llm_host}/api/tags", timeout=3).ok
    except Exception:
        return False


def polish(text: str, mode: Mode, cfg: Config) -> str:
    """Return LLM-cleaned text, or the original on any failure."""
    if not text.strip():
        return text
    if not available(cfg):
        log.warning("use_llm set but Ollama at %s is unreachable — keeping "
                    "rules-only text.", cfg.llm_host)
        return text

    blocks = _blocks(text, max_chars=4000)
    out: list[str] = []
    for i, block in enumerate(blocks):
        cleaned = _generate(block, mode.llm_prompt, cfg)
        out.append(cleaned or block)
        log.info("LLM polish: block %d/%d (%d chars)%s",
                 i + 1, len(blocks), len(block), "" if cleaned else "  [kept original]")
    return "\n\n".join(out).strip()


def _generate(block: str, prompt: str, cfg: Config) -> str:
    try:
        import requests
        resp = requests.post(
            f"{cfg.llm_host}/api/generate",
            json={
                "model": cfg.llm_model,
                "prompt": prompt + block,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=cfg.llm_timeout,
        )
        resp.raise_for_status()
        return (resp.json().get("response") or "").strip()
    except Exception as exc:
        log.warning("LLM polish failed on a block (%s) — keeping original.", exc)
        return ""


def _blocks(text: str, max_chars: int) -> list[str]:
    """Greedily pack whole paragraphs into <= max_chars blocks."""
    blocks: list[str] = []
    cur = ""
    for para in _PARA.split(text):
        para = para.strip()
        if not para:
            continue
        candidate = f"{cur}\n\n{para}" if cur else para
        if len(candidate) <= max_chars or not cur:
            cur = candidate
        else:
            blocks.append(cur)
            cur = para
    if cur:
        blocks.append(cur)
    return blocks
