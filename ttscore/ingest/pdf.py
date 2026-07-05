"""PDF ingest: extract a document's text, with an optional OCR fallback.

Most PDFs — papers, articles, ebooks, and even many "comic"/"book" sites that
are really just a PDF viewer (e.g. Blizzard's short-story pages) — carry a real
text layer, which we pull directly with PyMuPDF: fast, exact, no OCR. Pages that
are image-only (scanned documents, image-baked comics) yield little text; those
can be OCR'd on the GPU when ``cfg.ocr`` is 'auto'/'force' (needs easyocr).
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path

from ..config import Config

log = logging.getLogger("ttscore")

# A line that is only a page number, optionally dash-wrapped ("— 3 —", "- 3 -").
_PAGE_MARK = re.compile(r"^\s*[—\-–]{0,2}\s*\d{1,4}\s*[—\-–]{0,2}\s*$")

_ocr_reader = None  # lazily-built, reused easyocr.Reader (one per process)


def extract_pdf(source: str, cfg: Config) -> tuple[str, str]:
    """Return ``(title, text)`` for a PDF given a local path or an http(s) URL."""
    path, is_temp = _resolve(source, cfg)
    # Title fallback: the SOURCE's name, never the temp download's random stem.
    if source.lower().startswith(("http://", "https://")):
        from urllib.parse import urlparse
        base = Path(urlparse(source).path).stem
    else:
        base = Path(source).stem
    try:
        title, text = _extract(path, cfg)
        if not title or title == Path(path).stem:
            title = base or title
        return title, text
    finally:
        if is_temp:
            try:
                os.remove(path)
            except OSError:
                pass


def _resolve(source: str, cfg: Config) -> tuple[str, bool]:
    if source.lower().startswith(("http://", "https://")):
        return _download(source, cfg), True
    p = Path(source)
    if not p.is_file():
        raise FileNotFoundError(f"PDF not found: {source}")
    return str(p), False


def _download(url: str, cfg: Config) -> str:
    import requests
    fd, path = tempfile.mkstemp(suffix=".pdf", prefix="ttspdf_")
    os.close(fd)
    try:
        with requests.get(url, headers={"User-Agent": cfg.user_agent},
                          timeout=cfg.request_timeout, stream=True) as r:
            r.raise_for_status()
            with open(path, "wb") as fh:
                for chunk in r.iter_content(65536):
                    fh.write(chunk)
    except BaseException:
        try:
            os.remove(path)
        except OSError:
            pass
        raise
    return path


def _extract(path: str, cfg: Config) -> tuple[str, str]:
    import fitz  # pymupdf

    doc = fitz.open(path)
    title = ((doc.metadata or {}).get("title") or "").strip() or Path(path).stem
    parts: list[str] = []
    ocr_pages = 0
    for page in doc:
        text = _page_text(page)
        need_ocr = (cfg.ocr == "force"
                    or (cfg.ocr == "auto" and len(text) < cfg.ocr_min_chars
                        and bool(page.get_images())))
        if need_ocr:
            ocr_text = _ocr_page(page, cfg)
            if ocr_text:
                text = ocr_text
                ocr_pages += 1
        text = _clean(text)
        if text:
            parts.append(text)
    n = doc.page_count
    doc.close()

    if ocr_pages:
        log.info("OCR'd %d image-only page(s) of %d.", ocr_pages, n)
    return title, "\n\n".join(parts)


def _page_text(page) -> str:
    """Plain text of one page. Within a page, lines are wrapped with single
    newlines (the arrange step rejoins them into flowing prose); pages are joined
    with a blank line upstream, so each page break becomes a natural pause."""
    return page.get_text().strip()


def _clean(text: str) -> str:
    """Drop standalone page-number lines that would be read aloud awkwardly."""
    kept = [ln for ln in text.splitlines() if not _PAGE_MARK.match(ln)]
    return "\n".join(kept).strip()


def _ocr_page(page, cfg: Config) -> str:
    """Rasterize a page and OCR it with easyocr (GPU if device=cuda)."""
    global _ocr_reader
    try:
        import easyocr
        import numpy as np
    except Exception:
        log.warning("OCR requested but 'easyocr' isn't installed. Install it with "
                    "`E:\\TTS\\.venv\\Scripts\\python.exe -m pip install easyocr`. "
                    "Skipping OCR for this page.")
        return ""
    if _ocr_reader is None:
        langs = [s.strip() for s in cfg.ocr_language.split(",") if s.strip()] or ["en"]
        _ocr_reader = easyocr.Reader(langs, gpu=(cfg.device == "cuda"))
    pix = page.get_pixmap(dpi=cfg.ocr_dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:      # RGBA -> RGB
        img = img[:, :, :3]
    elif pix.n == 1:    # gray -> RGB
        img = np.repeat(img, 3, axis=2)
    lines = _ocr_reader.readtext(img, detail=0, paragraph=True)
    return "\n".join(str(x) for x in lines).strip()
