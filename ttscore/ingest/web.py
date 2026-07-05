"""URL -> clean main-article text.

Primary extractor is **trafilatura** (drops nav/ads/boilerplate/comments);
**readability-lxml** is the fallback. We fetch with our own headers/timeout so
we control the request. Note: both read *static* HTML — a JavaScript-rendered
SPA may yield little.

When a page has no readable text but embeds a PDF link (many "interactive
book"/comic viewers — e.g. Blizzard's story pages — are just a PDF viewer plus
a Download-PDF button), we raise :class:`PdfLinkFound` so the pipeline can
pivot to reading that PDF instead of failing.
"""

from __future__ import annotations

import html as _html
import logging
import re
from urllib.parse import urljoin

from ..config import Config

log = logging.getLogger("ttscore")

# Absolute .pdf URLs anywhere in the raw HTML (href, JS config, meta, ...).
_PDF_ABS = re.compile(r'https?://[^\s"\'<>()]+\.pdf(?:\?[^\s"\'<>()]*)?', re.I)
# Relative .pdf hrefs/srcs, resolved against the page URL.
_PDF_REL = re.compile(r'(?:href|src)\s*=\s*["\']([^"\':]+\.pdf(?:\?[^"\']*)?)["\']', re.I)


class PdfLinkFound(Exception):
    """The page itself isn't readable, but it embeds this PDF."""

    def __init__(self, pdf_url: str):
        super().__init__(pdf_url)
        self.pdf_url = pdf_url


def fetch_and_extract(url: str, cfg: Config) -> tuple[str, str]:
    """Return ``(title, text)`` for ``url``.

    Raises :class:`PdfLinkFound` when the page is unreadable but embeds a PDF
    (the caller should read that instead); plain errors otherwise."""
    page = _fetch(url, cfg)
    title, text = _extract(page, url)
    if not text or len(text.strip()) < 40:
        pdf = _find_pdf_link(page, url)
        if pdf:
            log.info("Page has no readable text but embeds a PDF — reading it: %s", pdf)
            raise PdfLinkFound(pdf)
        raise ValueError(
            f"Could not extract readable article text from {url}. The page may be "
            "JavaScript-rendered, paywalled, or blocking automated requests."
        )
    return title.strip(), text.strip()


def _find_pdf_link(page: str, base_url: str) -> str | None:
    """First embedded .pdf URL in the raw HTML, if any."""
    m = _PDF_ABS.search(page)
    if m:
        return m.group(0)
    m = _PDF_REL.search(page)
    if m:
        return urljoin(base_url, m.group(1))
    return None


def _fetch(url: str, cfg: Config) -> str:
    import requests
    resp = requests.get(url, headers={"User-Agent": cfg.user_agent},
                        timeout=cfg.request_timeout)
    resp.raise_for_status()
    resp.encoding = resp.encoding or "utf-8"
    return resp.text


def _extract(page: str, url: str) -> tuple[str, str]:
    title, text = "", ""
    try:
        import trafilatura
        text = trafilatura.extract(
            page, include_comments=False, include_tables=False,
            favor_precision=True, url=url,
        ) or ""
        try:
            md = trafilatura.extract_metadata(page)
            title = (getattr(md, "title", None) or "") if md else ""
        except Exception:
            pass
    except Exception as exc:
        log.warning("trafilatura extraction failed (%s); trying readability.", exc)

    if not text:
        try:
            from readability import Document
            doc = Document(page)
            title = title or (doc.short_title() or "")
            text = _strip_tags(doc.summary() or "")
        except Exception as exc:
            log.warning("readability fallback failed: %s", exc)

    return title, text


def _strip_tags(markup: str) -> str:
    markup = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", markup)
    markup = re.sub(r"(?i)</p>|<br\s*/?>", "\n\n", markup)
    markup = re.sub(r"<[^>]+>", " ", markup)
    markup = _html.unescape(markup)
    markup = re.sub(r"[ \t]{2,}", " ", markup)
    return re.sub(r"\n{3,}", "\n\n", markup).strip()
