"""Pasted text and local text files — the trivial ingest paths."""

from __future__ import annotations

from pathlib import Path


def read_paste(text: str) -> str:
    """Pasted text passes straight through; real cleanup is the arrange layer."""
    return (text or "").strip()


def read_file(path: str | Path) -> str:
    """Read a local .txt/.md file as UTF-8 (tolerating stray bytes)."""
    p = Path(path)
    return p.read_text(encoding="utf-8", errors="replace").strip()
