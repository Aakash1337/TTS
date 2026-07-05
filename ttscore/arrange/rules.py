"""Rule-based text cleanup: deterministic, offline, fast.

Turns pasted/scraped text into something that reads naturally aloud:
strip markdown/boilerplate, rejoin line-wrapped words and paragraphs, normalize
fancy punctuation, and (optionally) speak numbers/currency/percent as words.

This always runs; the optional LLM pass (``arrange.llm``) is layered on top for
messier input. Which toggles fire is decided by the active :class:`Mode`.
"""

from __future__ import annotations

import re
import unicodedata

from .modes import Mode

# ── patterns ─────────────────────────────────────────────────────────────────
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_BRACKET_REF = re.compile(
    r"\[\s*(?:\d+(?:\s*[,–-]\s*\d+)*"                 # [12]  [3-5]  [3, 4]
    r"|[A-Z][A-Za-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}[a-z]?)\s*\]"  # [Smith 2020]
)
_MULTISPACE = re.compile(r"[ \t]{2,}")
_MULTINEWLINE = re.compile(r"\n{3,}")
_PARA = re.compile(r"\n\s*\n+")

# markdown-ish artifacts (line-anchored ones must run before lines are joined)
_MD_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^()]|\([^()]*\))*\)")   # [text](url) -> text
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_MD_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", re.M)
_MD_BULLET = re.compile(r"^\s{0,3}[-*+•·]\s+", re.M)
_MD_ORDERED = re.compile(r"^\s{0,3}\d+[.)]\s+", re.M)
_DEHYPH = re.compile(r"(?<=[A-Za-z])-\n[ \t]*(?=[a-z])")

# number expansion. The body matches "5", "12345", and grouped "1,234,567" but
# NOT a trailing punctuation comma, so "5, 6" stays two numbers (a list comma is
# preserved) rather than being read as "56".
_NUMBODY = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_CURRENCY = re.compile(r"\$\s?(" + _NUMBODY + r")")
_PERCENT = re.compile(r"(" + _NUMBODY + r")\s?%")
_YEAR = re.compile(r"(?<![\w.])(1\d{3}|20\d{2})(?!\w)")   # 1000-1999, 2000-2099
_NUMBER = re.compile(r"(?<![\w.])" + _NUMBODY + r"(?!\w)")

_CHAR_MAP = {
    # Curly quotes, em/en dashes, and the ellipsis are LEFT INTACT — modern TTS
    # reads them correctly, and keeping them makes the transcript readable.
    "…": "...", " ": " ", "​": "",
}


def clean(text: str, mode: Mode, expand_numbers: bool) -> str:
    """Apply the rule pipeline to ``text`` using ``mode``'s toggles."""
    if not text or not text.strip():
        return ""

    text = _normalize_unicode(text)
    text = _strip_markdown(text)
    if mode.dehyphenate:
        text = _DEHYPH.sub("", text)
    if mode.join_wrapped_lines:
        text = _join_wrapped(text)
    if mode.strip_bracketed_refs:
        text = _BRACKET_REF.sub("", text)
    if mode.drop_urls:
        text = _URL.sub("", text)
    if mode.replace_symbols:
        text = _replace_symbols(text)
    if expand_numbers and mode.expand_numbers:
        text = _expand_numbers(text)

    return _tidy_whitespace(text)


# ── steps ────────────────────────────────────────────────────────────────────
def _normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for k, v in _CHAR_MAP.items():
        text = text.replace(k, v)
    return _CONTROL.sub("", text)


def _strip_markdown(text: str) -> str:
    text = _MD_IMG.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_BLOCKQUOTE.sub("", text)
    text = _MD_BULLET.sub("", text)
    text = _MD_ORDERED.sub("", text)
    return _strip_emphasis(text)


def _strip_emphasis(text: str) -> str:
    """Remove markdown emphasis markers only where they *wrap* text, so
    snake_case identifiers and a lone '*' (e.g. '5 * 3') survive untouched."""
    text = text.replace("`", "")                                    # code ticks: safe to drop
    text = re.sub(r"\*\*(\S(?:.*?\S)?)\*\*", r"\1", text)           # **bold**
    text = re.sub(r"(?<!\w)\*(\S(?:.*?\S)?)\*(?!\w)", r"\1", text)   # *italic*
    text = re.sub(r"(?<!\w)_(\S(?:.*?\S)?)_(?!\w)", r"\1", text)     # _italic_
    return text


def _join_wrapped(text: str) -> str:
    """Rejoin soft-wrapped lines into flowing paragraphs while KEEPING real
    paragraph breaks. Two source conventions exist:

      * blank-line-separated paragraphs (PDFs, most prose): a lone newline is a
        soft wrap to be joined; a blank line is a true paragraph break.
      * single-newline-separated paragraphs (many pasted web-novel / chat
        sources): every newline IS a paragraph break and must be preserved.

    We choose based on whether the text has any blank-line gap, so a single-
    newline story is no longer flattened into one giant block."""
    if _PARA.search(text):
        paras = [re.sub(r"[ \t]*\n[ \t]*", " ", p.strip()) for p in _PARA.split(text)]
    else:
        paras = [ln.strip() for ln in text.split("\n")]
    return "\n\n".join(p for p in paras if p)


def _replace_symbols(text: str) -> str:
    text = re.sub(r"\s*&\s*", " and ", text)
    text = text.replace("•", " ").replace("·", " ")
    return text


def _expand_numbers(text: str) -> str:
    try:
        from num2words import num2words
    except Exception:
        return text  # graceful: leave digits for the TTS engine to voice

    def _num(s: str):
        f = float(s)
        return int(f) if f.is_integer() else f

    def money(m: re.Match) -> str:
        # Explicit dollars/cents (num2words to='currency' misreads "$5" as 5 cents).
        raw = m.group(1).replace(",", "")
        try:
            if "." in raw:
                dollars, cents = raw.split(".", 1)
                cents = (cents + "00")[:2]
                out = f"{num2words(int(dollars or 0))} dollars"
                if int(cents):
                    out += f" and {num2words(int(cents))} cents"
                return out
            return f"{num2words(int(raw))} dollars"
        except Exception:
            return m.group(0)

    def pct(m: re.Match) -> str:
        try:
            return f"{num2words(_num(m.group(1).replace(',', '')))} percent"
        except Exception:
            return m.group(0)

    def plain(m: re.Match) -> str:
        try:
            return num2words(_num(m.group(0).replace(",", "")))
        except Exception:
            return m.group(0)

    def year(m: re.Match) -> str:
        try:
            return num2words(int(m.group(1)), to="year")  # 1984 -> "nineteen eighty-four"
        except Exception:
            return m.group(0)

    text = _CURRENCY.sub(money, text)
    text = _PERCENT.sub(pct, text)
    text = _YEAR.sub(year, text)
    return _NUMBER.sub(plain, text)


def _tidy_whitespace(text: str) -> str:
    text = _MULTISPACE.sub(" ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = _MULTINEWLINE.sub("\n\n", text)
    return text.strip()
