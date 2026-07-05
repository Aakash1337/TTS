"""Content modes = named cleanup/narration profiles.

A "mode" is just a small bundle of {rule toggles + optional LLM prompt + voice/
pause overrides}. The arrange layer reads the active profile instead of having
behavior hardcoded, so adding a new mode later is *only* adding an entry to
:data:`MODES` — no changes to rules, the pipeline, or the engine.

MVP ships one mode: ``plain``. Commented stubs show how the planned academic /
article / narrative / study / auto modes slot in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("ttscore")


DEFAULT_LLM_PROMPT = (
    "You are preparing text to be read aloud by a text-to-speech voice. "
    "Rewrite the text below so it reads naturally as continuous speech:\n"
    "- Rejoin words broken across line breaks and merge wrapped lines into "
    "proper paragraphs.\n"
    "- Remove leftover web boilerplate: navigation, cookie/subscribe notices, "
    "share buttons, image captions/credits, 'related articles', and inline "
    "citation clutter like [12].\n"
    "- Keep ALL of the substantive content and its original order. Do NOT "
    "summarize, shorten, translate, or add any commentary or headings of your "
    "own.\n"
    "Return ONLY the cleaned text, with paragraphs separated by blank lines.\n\n"
    "TEXT:\n"
)


@dataclass(frozen=True)
class Mode:
    name: str
    description: str

    # ── rule toggles (consumed by ttscore.arrange.rules) ─────────────────────
    dehyphenate: bool = True          # rejoin "exam-\nple" -> "example"
    join_wrapped_lines: bool = True   # collapse single newlines within paragraphs
    strip_bracketed_refs: bool = False  # remove [12], [Smith 2020] citations
    drop_urls: bool = True            # remove bare URLs (they don't read aloud)
    replace_symbols: bool = True      # & -> "and", strip stray bullets, etc.
    expand_numbers: bool = True       # 1984 -> "nineteen eighty-four" (via cfg)

    # ── optional LLM polish prompt ───────────────────────────────────────────
    llm_prompt: str = DEFAULT_LLM_PROMPT

    # ── voice / pacing overrides (None = keep the Config value) ──────────────
    exaggeration: Optional[float] = None
    cfg_weight: Optional[float] = None
    pause_sentence_ms: Optional[int] = None
    pause_paragraph_ms: Optional[int] = None


MODES: dict[str, Mode] = {
    "plain": Mode(
        name="plain",
        description="Faithful cleanup: fix formatting, keep everything, minimal restructuring.",
    ),
    # ── Additive later — no core changes, just uncomment/extend ──────────────
    # "article":   Mode("article",   "News/blog: drop share cruft & citation noise.",
    #                   strip_bracketed_refs=True),
    # "academic":  Mode("academic",  "Papers: strip [12] refs & heading noise.",
    #                   strip_bracketed_refs=True),
    # "narrative": Mode("narrative", "Novels: more expressive voice, longer pauses.",
    #                   exaggeration=0.7, pause_paragraph_ms=1000),
    # "study":     Mode("study",     "Learning: measured pace, clearer pauses.",
    #                   pause_sentence_ms=450, pause_paragraph_ms=900),
    # "auto":      resolved by the LLM from a text sample -> one of the above.
}


def get_mode(name: str) -> Mode:
    """Look up a mode by name, falling back to 'plain' (with a warning) so an
    unknown value is never a hard error."""
    key = (name or "plain").strip().lower()
    mode = MODES.get(key)
    if mode is None:
        log.warning("Unknown mode %r; using 'plain'. Known modes: %s",
                    name, ", ".join(sorted(MODES)))
        return MODES["plain"]
    return mode
