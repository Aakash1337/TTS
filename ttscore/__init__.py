"""ttscore — turn pasted text or a web link into cleaned, natural narration.

The whole system hangs off this one importable package; the CLI, the web app,
and any future desktop/hosted shell are thin front-ends over it.

Pipeline (each stage is swappable / optional):

    ingest  ->  arrange  ->  chunk  ->  synthesize  ->  assemble  ->  encode

Public surface is intentionally tiny; import the pieces you need:

    from ttscore import Config
    from ttscore.pipeline import synthesize   # (added in the pipeline module)
"""

from __future__ import annotations

from .config import Config

__all__ = ["Config"]
__version__ = "0.1.0"
