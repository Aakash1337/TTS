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


_ACADEMIC_PROMPT = (
    "You are preparing an academic text to be read aloud by a text-to-speech "
    "voice. Rewrite it so it reads naturally as continuous speech:\n"
    "- Remove citation clutter: bracketed refs like [12], (Smith et al., 2020), "
    "figure/table pointers like '(see Fig. 3)', and footnote markers.\n"
    "- Drop reference lists, bibliographies, and acknowledgment boilerplate if "
    "present.\n"
    "- Spell out abbreviations that would be confusing aloud (e.g. 'e.g.' -> "
    "'for example').\n"
    "- Keep ALL substantive content, arguments, and results in their original "
    "order. Do NOT summarize or editorialize.\n"
    "Return ONLY the cleaned text, paragraphs separated by blank lines.\n\nTEXT:\n"
)

_NARRATIVE_PROMPT = (
    "You are preparing fiction to be read aloud. Clean it up while treating the "
    "prose as sacred:\n"
    "- Fix broken line-wraps and stray formatting artifacts only.\n"
    "- PRESERVE dialogue, punctuation, pacing, and every stylistic choice "
    "exactly — do not rephrase, modernize, or summarize anything.\n"
    "- Remove only non-story debris (page headers, translator notes, ads).\n"
    "Return ONLY the cleaned text, paragraphs separated by blank lines.\n\nTEXT:\n"
)

MODES: dict[str, Mode] = {
    "plain": Mode(
        name="plain",
        description="Faithful cleanup: fix formatting, keep everything, minimal restructuring.",
    ),
    "article": Mode(
        name="article",
        description="News/blog articles: also drop bracketed citation noise.",
        strip_bracketed_refs=True,
    ),
    "academic": Mode(
        name="academic",
        description="Papers & textbooks: strip [12]/(Smith 2020) refs; LLM pass "
                    "drops reference dumps; slightly clearer pacing.",
        strip_bracketed_refs=True,
        llm_prompt=_ACADEMIC_PROMPT,
        pause_sentence_ms=380,
    ),
    "narrative": Mode(
        name="narrative",
        description="Novels & stories: keep every word, expressive voice, longer "
                    "paragraph pauses; URLs/numbers left closer to the page.",
        drop_urls=False,
        expand_numbers=False,       # '1984' the novel should stay '1984'
        llm_prompt=_NARRATIVE_PROMPT,
        exaggeration=0.7,           # only affects engines with expressiveness (chatterbox)
        pause_paragraph_ms=1000,
    ),
    "study": Mode(
        name="study",
        description="Learning: measured pace with clearer pauses between ideas.",
        strip_bracketed_refs=True,
        pause_sentence_ms=450,
        pause_paragraph_ms=950,
    ),
    # "auto" is not a table entry: ttscore.arrange resolves it to one of the
    # above by showing a local LLM a sample of the text (falls back to plain).
}

AUTO_CHOICES = ("article", "academic", "narrative", "study", "plain")


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
