"""EPUB ingest: chapters in reading order -> clean text.

An EPUB is a zip: ``META-INF/container.xml`` points at the OPF package file,
whose <spine> lists the reading order of the HTML chapter files in <manifest>.
We walk the spine, strip each chapter's markup (block tags become paragraph
breaks, so headings/paragraphs pause naturally), and join chapters with blank
lines. Pure stdlib — no extra dependency.
"""

from __future__ import annotations

import html as _html
import logging
import posixpath
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from ..config import Config

log = logging.getLogger("ttscore")

_NS_CONTAINER = "{urn:oasis:names:tc:opendocument:xmlns:container}"
_NS_OPF = "{http://www.idpf.org/2007/opf}"
_NS_DC = "{http://purl.org/dc/elements/1.1/}"

# Block-level closers/openers that should become paragraph breaks when stripped.
_BLOCK_BREAK = re.compile(
    r"(?i)</(?:p|div|h[1-6]|li|blockquote|section|article|tr)>|<br\s*/?>|<hr\s*/?>")
_SCRIPT_STYLE = re.compile(r"(?is)<(script|style)\b.*?</\1>")
_TAG = re.compile(r"<[^>]+>")
_MULTIBLANK = re.compile(r"\n{3,}")


def extract_epub(source: str, cfg: Config) -> tuple[str, str]:
    """Return ``(title, text)`` for a local .epub file."""
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"EPUB not found: {source}")

    with zipfile.ZipFile(path) as zf:
        opf_name = _find_opf(zf)
        opf_dir = posixpath.dirname(opf_name)
        title, chapter_files = _parse_opf(zf.read(opf_name), opf_dir)

        parts: list[str] = []
        for name in chapter_files:
            try:
                raw = zf.read(name).decode("utf-8", errors="replace")
            except KeyError:
                log.warning("EPUB spine references missing file: %s", name)
                continue
            text = _html_to_text(raw)
            if text:
                parts.append(text)

    if not parts:
        raise ValueError(f"No readable chapters found in {path.name}.")
    log.info("EPUB: %d chapter file(s) extracted.", len(parts))
    return (title or path.stem), "\n\n".join(parts)


def _find_opf(zf: zipfile.ZipFile) -> str:
    root = ET.fromstring(zf.read("META-INF/container.xml"))
    rootfile = root.find(f".//{_NS_CONTAINER}rootfile")
    if rootfile is None or not rootfile.get("full-path"):
        raise ValueError("Invalid EPUB: container.xml has no rootfile.")
    return rootfile.get("full-path")


def _parse_opf(opf_bytes: bytes, opf_dir: str) -> tuple[str, list[str]]:
    """Return (title, ordered chapter file names resolved against the zip root)."""
    root = ET.fromstring(opf_bytes)

    title = ""
    t = root.find(f".//{_NS_DC}title")
    if t is not None and t.text:
        title = t.text.strip()

    manifest: dict[str, str] = {}
    for item in root.iter(f"{_NS_OPF}item"):
        iid, href = item.get("id"), item.get("href")
        media = (item.get("media-type") or "").lower()
        if iid and href and ("html" in media or href.lower().endswith((".xhtml", ".html", ".htm"))):
            manifest[iid] = href

    ordered: list[str] = []
    for ref in root.iter(f"{_NS_OPF}itemref"):
        href = manifest.get(ref.get("idref") or "")
        if href:
            ordered.append(posixpath.normpath(posixpath.join(opf_dir, href)) if opf_dir else href)
    return title, ordered


def _html_to_text(markup: str) -> str:
    markup = _SCRIPT_STYLE.sub(" ", markup)
    markup = _BLOCK_BREAK.sub("\n\n", markup)
    markup = _TAG.sub(" ", markup)
    markup = _html.unescape(markup)
    markup = re.sub(r"[ \t]{2,}", " ", markup)
    markup = re.sub(r"[ \t]+\n", "\n", markup)
    return _MULTIBLANK.sub("\n\n", markup).strip()
