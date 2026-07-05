"""Orchestration: a source (paste / URL / file) -> one narration audio file.

Per source:
  1. ingest    obtain raw text (paste passthrough, file read, or URL extraction)
  2. arrange   rule-based cleanup (+ optional local-LLM polish)
  3. chunk     sentence-aligned pieces sized for the TTS engine
  4. synth     one clip per chunk, cached by content hash (crash-resumable)
  5. assemble  resample + concatenate with sentence/paragraph pauses
  6. write     loudnorm + encode to mp3/m4a (or raw wav) + optional transcript

Batch behaviour: idempotent (skip existing outputs unless --overwrite), one
try/except per source so a bad link can't sink the run, engine loaded once.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from . import audio, ffmpeg_utils
from .arrange import arrange, get_mode
from .cache import ChunkCache
from .chunk import chunk_text
from .config import Config
from .engines import TTSEngine, create_engine
from .ingest import extract_epub, extract_pdf, fetch_and_extract, read_file, read_paste
from .logging_setup import (JobResult, RunReport, get_logger, new_run_id,
                            setup_logging)

try:  # progress bars are nice-to-have, not required
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **_kw):  # type: ignore
        return it


@dataclass
class Source:
    kind: str                    # "text" | "url" | "file"
    value: str                   # the pasted text, the URL, or the file path
    out: Optional[str] = None    # explicit output path (else derived)

    @property
    def label(self) -> str:
        return "paste" if self.kind == "text" else self.value


# ── public entry points ──────────────────────────────────────────────────────
def run(sources: list[Source], cfg: Config, progress=None) -> RunReport:
    """Process every source with one loaded engine. Never raises on one bad source.

    ``progress`` is an optional callable ``progress(done, total)`` invoked as
    chunks are synthesized (used by the web UI for a live progress bar)."""
    cfg.validate()
    mode = get_mode(cfg.mode)
    cfg = _apply_mode_overrides(cfg, mode)

    run_id = new_run_id()
    log, log_file = setup_logging(cfg.log_path, run_id)
    report = RunReport(run_id=run_id,
                       started_at=datetime.now().isoformat(timespec="seconds"))
    log.info("Run %s | mode=%s engine=%s device=%s | log: %s",
             run_id, cfg.mode, cfg.engine, cfg.device, log_file)

    if cfg.audio_format != "wav" and not cfg.dry_run:
        ffmpeg_utils.ensure_tool()  # fail fast if we can't encode
    cfg.output_path.mkdir(parents=True, exist_ok=True)

    engine: Optional[TTSEngine] = None
    if not cfg.dry_run:
        log.info("Loading %s engine ...", cfg.engine)
        engine = create_engine(cfg)
        log.info("Engine ready: %s (native %d Hz).",
                 engine.model_id, engine.sample_rate)

    for src in sources:
        try:
            result = process_source(src, cfg, engine, progress)
        except Exception as exc:  # one bad source must not sink the run
            log.exception("FAIL  %s: %s", src.label[:80], exc)
            result = JobResult(source=src.label, status="failed",
                               mode=cfg.mode, error=str(exc))
        report.add(result)
        if engine is not None:
            engine.reset()

    summary_path = report.write(cfg.log_path)
    c = report.counts()
    log.info("Done. ok=%d skipped=%d failed=%d | summary: %s",
             c["ok"], c["skipped"], c["failed"], summary_path)
    return report


def synthesize(*, text: Optional[str] = None, url: Optional[str] = None,
               file: Optional[str] = None, cfg: Optional[Config] = None,
               out: Optional[str] = None) -> JobResult:
    """Convenience one-shot: give exactly one of text=/url=/file=, get a JobResult."""
    cfg = cfg or Config()
    if text is not None:
        src = Source("text", text, out)
    elif url is not None:
        src = Source("url", url, out)
    elif file is not None:
        src = Source("file", file, out)
    else:
        raise ValueError("Provide exactly one of text=, url=, or file=.")
    return run([src], cfg).jobs[-1]


# ── per-source pipeline ───────────────────────────────────────────────────────
def process_source(src: Source, cfg: Config, engine: Optional[TTSEngine],
                   progress=None) -> JobResult:
    log = get_logger()
    t0 = time.perf_counter()
    result = JobResult(source=src.label, status="ok", mode=cfg.mode)

    # 1. ingest
    raw, title = _ingest(src, cfg, log)
    result.title = title
    result.chars_in = len(raw)
    if not raw.strip():
        raise ValueError("No text to speak (empty input).")

    # 2. arrange
    text, used_llm = arrange(raw, cfg)
    result.used_llm = used_llm
    result.chars_spoken = len(text)
    if not text.strip():
        raise ValueError("Nothing left to speak after cleanup.")

    # 3. chunk
    chunks = chunk_text(text, cfg)
    result.n_chunks = len(chunks)
    if not chunks:
        raise ValueError("Nothing left to speak after chunking.")

    out_path = _out_path(cfg, src, title)
    result.output = str(out_path)

    # Dry run reports the analysis regardless of any existing output.
    if cfg.dry_run:
        log.info("DRY   %s: %d chars in -> %d spoken -> %d chunks (no audio)",
                 src.label[:60], result.chars_in, result.chars_spoken, len(chunks))
        return result

    # idempotent skip: a present, non-trivial output means "already done"
    if out_path.exists() and out_path.stat().st_size > 1024 and not cfg.overwrite:
        log.info("SKIP  %s (output exists; use --overwrite to redo)", out_path.name)
        result.status = "skipped"
        return result

    assert engine is not None  # guaranteed when not dry_run

    # 4. synthesize per chunk (cached)
    want_words = cfg.word_timings and getattr(engine, "supports_word_timings", False)
    cache = ChunkCache(cfg.cache_path, out_path.stem, engine.signature())
    clips: list = []
    para_ends: list[bool] = []
    chunk_words: list = []            # per chunk: list of {"t","s","e"} or None
    n_generated = 0
    total = len(chunks)
    preview_written = False
    accum_secs = 0.0
    if progress:
        progress(0, total)
    for i, ch in enumerate(tqdm(chunks, desc=out_path.stem[:28], unit="chunk", leave=False)):
        cached = cache.get(ch.index, ch.text)
        if cached is not None and want_words and not cache.get_words(ch.index, ch.text):
            cached = None   # clip cached without usable word timings -> regenerate
        if cached is not None:
            clip_native, native_sr = cached
            words = cache.get_words(ch.index, ch.text) if want_words else None
        else:
            if want_words:
                clip_native, words = engine.synthesize_with_words(ch.text, ch.index)
            else:
                clip_native, words = engine.synthesize(ch.text, ch.index), None
            native_sr = engine.sample_rate
            cache.put(ch.index, ch.text, clip_native, native_sr)
            if words is not None:
                cache.put_words(ch.index, ch.text, words)
            n_generated += 1
        clips.append(audio.resample(clip_native, native_sr, cfg.sample_rate))
        para_ends.append(ch.paragraph_end)
        chunk_words.append(words)
        accum_secs += clips[-1].shape[0] / cfg.sample_rate
        if progress:
            progress(i + 1, total)

        # Early preview: once enough audio exists (and more is coming), encode
        # the prefix so the app can start playback while the rest generates.
        # The prefix is bit-identical in timing to the final track's start, so
        # the player can swap to the full file without losing its position.
        if (not preview_written and cfg.preview_seconds
                and accum_secs >= cfg.preview_seconds and i + 1 < total
                # a post-encode speed change would desync preview vs final
                and (getattr(engine, "handles_speed", False)
                     or abs(cfg.speed - 1.0) < 1e-3)):
            try:
                _write_preview(clips, para_ends, out_path, cfg)
                preview_written = True
            except Exception as exc:   # a failed preview must never fail the job
                log.warning("preview encode failed (continuing): %s", exc)
                preview_written = True  # don't retry every chunk
    result.n_generated = n_generated

    # 5. assemble
    track, clip_offsets = audio.assemble(clips, para_ends, cfg.sample_rate,
                                         cfg.pause_sentence_ms, cfg.pause_paragraph_ms)
    track = audio.finalize(track)
    result.audio_seconds = audio.duration_seconds(track, cfg.sample_rate)

    # 6. write (speed already baked in by engines that handle it natively, e.g. edge)
    post_speed = 1.0 if getattr(engine, "handles_speed", False) else cfg.speed
    _write_output(track, text, out_path, cfg, post_speed)
    if want_words and any(chunk_words):
        _write_word_timings(out_path, chunks, chunk_words, clip_offsets)
    _write_meta(out_path, result, engine, src)

    result.elapsed_s = time.perf_counter() - t0
    log.info("OK    %s  (%d chunks, %d generated, %.1fs audio in %.1fs)",
             out_path.name, len(chunks), n_generated,
             result.audio_seconds, result.elapsed_s)
    return result


# ── helpers ───────────────────────────────────────────────────────────────────
def _ingest(src: Source, cfg: Config, log) -> tuple[str, str]:
    """Return (raw_text, title)."""
    if src.kind == "pdf":
        log.info("Reading PDF %s", src.value)
        title, text = extract_pdf(src.value, cfg)
        if title:
            log.info("  title: %s", title)
        return text, title
    if src.kind == "epub":
        log.info("Reading EPUB %s", src.value)
        title, text = extract_epub(src.value, cfg)
        if title:
            log.info("  title: %s", title)
        return text, title
    if src.kind == "url":
        from .ingest.web import PdfLinkFound
        log.info("Fetching %s", src.value)
        try:
            title, text = fetch_and_extract(src.value, cfg)
        except PdfLinkFound as found:
            # "Interactive book"/comic viewer pages are often a PDF viewer in
            # disguise — read the embedded PDF instead of giving up.
            title, text = extract_pdf(found.pdf_url, cfg)
        if title:
            log.info("  title: %s", title)
        return text, title
    if src.kind == "file":
        return read_file(src.value), Path(src.value).stem
    # Pasted text: title = its first few words, so the library/heading shows
    # something recognizable instead of a random job id.
    text = read_paste(src.value)
    return text, " ".join(text.split()[:8])[:60]


def _out_path(cfg: Config, src: Source, title: str) -> Path:
    if src.out:
        p = Path(src.out)
        # Ensure the output carries the audio format's extension, so ffmpeg can
        # pick a muxer and the transcript (.txt) can never collide with the audio.
        if p.suffix.lower().lstrip(".") != cfg.audio_format:
            p = p.with_suffix(f".{cfg.audio_format}")
        return p
    base = title or (_slug_from_url(src.value) if src.kind == "url" else "") or "narration"
    safe = re.sub(r"[^\w.-]+", "_", base).strip("_")[:60] or "narration"
    return cfg.output_path / f"{safe}.{cfg.audio_format}"


def _slug_from_url(url: str) -> str:
    try:
        p = urlparse(url)
        segs = [s for s in p.path.split("/") if s]
        base = segs[-1] if segs else p.netloc
        return re.sub(r"\.\w+$", "", base)
    except Exception:
        return ""


def _write_output(track, text: str, out_path: Path, cfg: Config,
                  post_speed: float = 1.0) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    change_speed = abs(post_speed - 1.0) > 1e-3

    if cfg.audio_format == "wav" and not change_speed:
        # Fast path: write the float wav directly (no ffmpeg needed).
        audio.write_wav(out_path, audio.to_channels(track, cfg.channels), cfg.sample_rate)
    else:
        # Write a mono working wav, then let ffmpeg apply speed + loudnorm +
        # channels + the target codec. Keep the wav beside mp3/m4a if requested.
        if cfg.audio_format != "wav" and cfg.keep_wav:
            wav_path = out_path.with_suffix(".wav")
            cleanup = False
        else:
            wav_path = out_path.with_name(out_path.stem + ".raw.wav")
            cleanup = True
        audio.write_wav(wav_path, track, cfg.sample_rate)
        ffmpeg_utils.encode(wav_path, out_path, cfg, atempo=post_speed)
        if cleanup:
            try:
                wav_path.unlink()
            except OSError:
                pass

    if cfg.write_transcript:
        txt_path = out_path.with_suffix(".txt")
        if txt_path != out_path:  # never clobber the audio with the transcript
            txt_path.write_text(text, encoding="utf-8")


def preview_path_for(out_path: Path) -> Path:
    return out_path.with_name(out_path.stem + ".preview.mp3")


def _write_preview(clips, para_ends, out_path: Path, cfg: Config) -> None:
    """Assemble + encode the current prefix to ``<out>.preview.mp3``."""
    track, _ = audio.assemble(clips, para_ends, cfg.sample_rate,
                              cfg.pause_sentence_ms, cfg.pause_paragraph_ms)
    track = audio.finalize(track)
    tmp_wav = out_path.with_name(out_path.stem + ".preview.raw.wav")
    audio.write_wav(tmp_wav, track, cfg.sample_rate)
    try:
        ffmpeg_utils.encode(tmp_wav, preview_path_for(out_path), cfg)
    finally:
        try:
            tmp_wav.unlink()
        except OSError:
            pass


def _write_meta(out_path: Path, result: JobResult, engine, src: Source) -> None:
    """Write ``<output>.meta.json`` — the library view (app 'recent narrations')
    is built from these sidecars, so titles/durations survive server restarts."""
    import json

    meta = {
        "title": result.title or out_path.stem,
        "source": src.label[:200],
        "kind": src.kind,
        "audio_seconds": round(result.audio_seconds, 1),
        "voice": getattr(engine, "model_id", "unknown"),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        out_path.with_suffix(".meta.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    except OSError:  # a failed sidecar must never fail the job
        pass


def _write_word_timings(out_path: Path, chunks, chunk_words, clip_offsets) -> None:
    """Write ``<output>.words.json``: each chunk's display text plus word spans
    as character ranges with global start/end seconds — the data the app's
    read-along view highlights from."""
    import json

    doc = {"version": 1, "chunks": []}
    for ch, words, offset in zip(chunks, chunk_words, clip_offsets):
        entry = {"text": ch.text, "para": bool(ch.paragraph_end), "words": []}
        if words:
            entry["words"] = _map_words_to_text(ch.text, words, offset)
        doc["chunks"].append(entry)

    wpath = out_path.with_suffix(".words.json")
    with open(wpath, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False)


def _map_words_to_text(text: str, words: list, offset: float) -> list:
    """Locate each spoken word in the chunk's display text (sequential,
    case-insensitive cursor scan) -> {"cs","ce","s","e"} char-range entries.
    Words that can't be located are skipped — the highlight just glides over."""
    out = []
    low = text.lower()
    cursor = 0
    for w in words:
        token = str(w.get("t", "")).strip()
        if not token:
            continue
        i = low.find(token.lower(), cursor)
        if i < 0:
            trimmed = re.sub(r"^\W+|\W+$", "", token)
            i = low.find(trimmed.lower(), cursor) if trimmed else -1
            if i >= 0:
                token = trimmed
        if i < 0:
            continue
        out.append({"cs": i, "ce": i + len(token),
                    "s": round(offset + float(w["s"]), 3),
                    "e": round(offset + float(w["e"]), 3)})
        cursor = i + len(token)
    return out


def _apply_mode_overrides(cfg: Config, mode) -> Config:
    """Fold a mode's non-None voice/pause overrides into the config."""
    overrides = {}
    for key in ("exaggeration", "cfg_weight", "pause_sentence_ms", "pause_paragraph_ms"):
        value = getattr(mode, key, None)
        if value is not None:
            overrides[key] = value
    return cfg.merged_with(overrides) if overrides else cfg
