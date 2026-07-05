"""The ingest layer: get raw text from wherever the user pasted or linked.

    read_paste(text)          -> raw text, unchanged
    read_file(path)           -> raw text from a .txt/.md file
    fetch_and_extract(url)    -> (title, main article text) via trafilatura

Cleanup happens later in :mod:`ttscore.arrange`; ingest only *obtains* text.
"""

from __future__ import annotations

from .epub import extract_epub
from .paste import read_file, read_paste
from .pdf import extract_pdf
from .web import fetch_and_extract

__all__ = ["read_paste", "read_file", "fetch_and_extract", "extract_pdf", "extract_epub"]
