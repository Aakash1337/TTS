"""Split arranged text into TTS-sized, sentence-aligned chunks.

Chatterbox (and most neural TTS) degrades on long inputs, so we feed it ~1-2
sentences at a time. Rules:

  * Never split mid-sentence. Sentences come from pysbd, which correctly keeps
    "Dr.", "U.S.", "e.g." intact (a naive ``.split('.')`` would not).
  * Pack whole sentences greedily up to ``max_chunk_chars``.
  * A single sentence longer than the budget is split at clause punctuation
    (,;:— ) as a last resort, then on whitespace if it's *still* too long.
  * Paragraph boundaries (blank lines) are preserved as a flag on the last chunk
    of each paragraph, so the assembler can insert a longer pause there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import Config

# Paragraph break = one or more blank lines.
_PARA_SPLIT = re.compile(r"\n\s*\n+")
# Clause-level break points, tried in order when one sentence is over budget.
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:—])\s+|\s+[-–]\s+")


@dataclass
class Chunk:
    text: str
    paragraph_end: bool = False   # last chunk of its paragraph -> longer pause
    index: int = 0


def _segment_sentences(text: str) -> list[str]:
    """Sentence-split one paragraph. Uses pysbd; falls back to a regex."""
    text = text.strip()
    if not text:
        return []
    try:
        import pysbd
        seg = pysbd.Segmenter(language="en", clean=False)
        return [s.strip() for s in seg.segment(text) if s.strip()]
    except Exception:
        # Last-resort splitter: break after . ! ? followed by whitespace.
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]


def _split_oversized(sentence: str, max_chars: int) -> list[str]:
    """Break a single over-budget sentence into <= max_chars pieces, preferring
    clause punctuation, then whitespace, then a hard character cut."""
    sentence = sentence.strip()
    if len(sentence) <= max_chars:
        return [sentence]

    pieces: list[str] = []
    for clause in _pack(_CLAUSE_SPLIT.split(sentence), max_chars):
        if len(clause) <= max_chars:
            pieces.append(clause)
        else:
            pieces.extend(_pack(clause.split(" "), max_chars, hard=True))
    return [p for p in pieces if p]


def _pack(units: list[str], max_chars: int, hard: bool = False) -> list[str]:
    """Greedily join ``units`` (with single spaces) into <= max_chars strings.
    With ``hard=True``, a lone unit longer than the budget is character-sliced."""
    out: list[str] = []
    cur = ""
    for u in units:
        u = u.strip()
        if not u:
            continue
        if hard and len(u) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            for i in range(0, len(u), max_chars):
                out.append(u[i:i + max_chars])
            continue
        candidate = f"{cur} {u}".strip() if cur else u
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                out.append(cur)
            cur = u
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, cfg: Config) -> list[Chunk]:
    """Turn arranged text into ordered :class:`Chunk` objects."""
    max_chars = cfg.max_chunk_chars
    chunks: list[Chunk] = []

    paragraphs = [p for p in _PARA_SPLIT.split(text) if p.strip()]
    for para in paragraphs:
        sentences = _segment_sentences(para)
        # Expand any single over-budget sentence into sub-pieces up front, so the
        # greedy packer only ever sees units that individually fit.
        units: list[str] = []
        for s in sentences:
            units.extend(_split_oversized(s, max_chars) if len(s) > max_chars else [s])

        para_chunks = _pack(units, max_chars)
        for i, ctext in enumerate(para_chunks):
            chunks.append(Chunk(text=ctext, paragraph_end=(i == len(para_chunks) - 1)))

    for i, c in enumerate(chunks):
        c.index = i
    return chunks
