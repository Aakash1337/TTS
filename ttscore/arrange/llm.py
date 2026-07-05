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


def classify(sample: str, cfg: Config) -> str | None:
    """Ask the local LLM which content mode fits ``sample``; None on failure."""
    from .modes import AUTO_CHOICES

    if not available(cfg):
        return None
    prompt = (
        "Classify the following text for a text-to-speech reader. Reply with "
        f"EXACTLY ONE word from this list: {', '.join(AUTO_CHOICES)}.\n"
        "- article: news/blog/magazine writing\n"
        "- academic: papers, textbooks, technical/scientific prose\n"
        "- narrative: fiction, novels, stories, dialogue-heavy prose\n"
        "- study: notes, tutorials, learning material\n"
        "- plain: anything else\n\nTEXT SAMPLE:\n" + sample[:1500]
    )
    reply = _generate("", prompt, cfg, temperature=0.0).strip().lower()
    for choice in AUTO_CHOICES:
        if choice in reply:
            log.info("auto mode: LLM classified the text as %r", choice)
            return choice
    log.warning("auto mode: unusable LLM reply %r — using 'plain'.", reply[:40])
    return None


def summarize(text: str, cfg: Config) -> str:
    """Condense ``text`` to roughly ``cfg.summary_words`` words for narration.

    Long inputs are summarized block-by-block, then the partial summaries are
    fused in a final pass (classic map-reduce). Raises if the LLM is
    unreachable — a user who asked for a summary should not silently get the
    full hour-long reading instead.
    """
    if not available(cfg):
        raise RuntimeError(
            f"Summary requested but the local LLM (Ollama at {cfg.llm_host}) is "
            "unreachable. Start Ollama, or turn off 'Summarize first'.")

    target = max(50, int(cfg.summary_words))
    blocks = _blocks(text, max_chars=6000)

    def _sum(block: str, words: int, part: str) -> str:
        prompt = (
            f"Summarize the following {part} in about {words} words, written as "
            "flowing prose to be READ ALOUD (no bullet points, no headings, no "
            "meta-commentary like 'this text discusses'). Keep the key facts, "
            "claims, and narrative beats.\n\nTEXT:\n"
        )
        return _generate(block, prompt, cfg, temperature=0.3)

    if len(blocks) == 1:
        out = _sum(blocks[0], target, "text")
        if not out:
            raise RuntimeError("The local LLM returned no summary (see logs).")
        return out

    per_block = max(60, target // len(blocks) + 40)
    partials = []
    for i, b in enumerate(blocks):
        log.info("summarize: block %d/%d", i + 1, len(blocks))
        partials.append(_sum(b, per_block, "text excerpt") or b[:800])
    fused = _sum("\n\n".join(partials), target, "set of partial summaries")
    if not fused:
        raise RuntimeError("The local LLM returned no summary (see logs).")
    return fused


def _generate(block: str, prompt: str, cfg: Config, temperature: float = 0.2) -> str:
    try:
        import requests
        resp = requests.post(
            f"{cfg.llm_host}/api/generate",
            json={
                "model": cfg.llm_model,
                "prompt": prompt + block,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=cfg.llm_timeout,
        )
        resp.raise_for_status()
        return (resp.json().get("response") or "").strip()
    except Exception as exc:
        log.warning("LLM call failed on a block (%s).", exc)
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
