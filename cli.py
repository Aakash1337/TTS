#!/usr/bin/env python
"""TTS Reader — command-line entry point.

Turn pasted text, a local text file, or a web link into a narrated audio file.

Examples:
    python cli.py --text "Hello world, this is a test." -o output/hello.mp3
    python cli.py --text-file notes.md
    python cli.py --url https://example.com/article
    Get-Content notes.txt | python cli.py --paste          # PowerShell
    python cli.py --url https://example.com/article --llm --mode plain

Config resolution: defaults < --config file.yaml < CLI flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ttscore.config import KNOWN_FORMATS, Config
from ttscore.pipeline import Source, run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli.py",
        description="Text or a web link -> cleaned narration -> speech.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_argument_group("input (choose one)")
    src.add_argument("--text", help="raw text to speak")
    src.add_argument("--text-file", help="path to a .txt/.md file to speak")
    src.add_argument("--url", help="web page URL to fetch, extract, and speak")
    src.add_argument("--paste", action="store_true",
                     help="read the text to speak from stdin")
    src.add_argument("--pdf", help="path or URL to a PDF to read (text layer; OCR optional)")

    out = p.add_argument_group("output")
    out.add_argument("-o", "--output", help="output file path (extension sets the format)")
    out.add_argument("--output-dir", help="directory for auto-named outputs")
    out.add_argument("--format", choices=KNOWN_FORMATS, help="audio format")

    arr = p.add_argument_group("arrange")
    arr.add_argument("--mode",
                     help="content profile: plain|article|academic|narrative|study|auto")
    arr.add_argument("--summarize", action="store_true", default=None,
                     help="speak a condensed summary instead of the full text (local LLM)")
    arr.add_argument("--summary-words", dest="summary_words", type=int,
                     help="approximate summary length in words")
    arr.add_argument("--llm", dest="use_llm", action="store_true", default=None,
                     help="polish text through the local Ollama LLM")
    arr.add_argument("--no-llm", dest="use_llm", action="store_false",
                     help="disable the LLM polish pass")
    arr.add_argument("--llm-model", help="Ollama model tag (e.g. gemma4:e4b)")
    arr.add_argument("--ocr", choices=("off", "auto", "force"),
                     help="OCR image-only PDF pages (needs the easyocr package)")

    tts = p.add_argument_group("voice / engine")
    tts.add_argument("--engine", help="TTS engine (MVP: 'chatterbox')")
    tts.add_argument("--voice", dest="reference_wav",
                     help="5-10s reference .wav to clone as the narrator voice")
    tts.add_argument("--device", choices=("cuda", "cpu"), help="compute device")
    tts.add_argument("--exaggeration", type=float, help="Chatterbox expressiveness")
    tts.add_argument("--cfg-weight", dest="cfg_weight", type=float, help="Chatterbox pacing")
    tts.add_argument("--seed", type=int, help="seed for reproducible audio")
    tts.add_argument("--max-chunk-chars", dest="max_chunk_chars", type=int,
                     help="max characters per TTS chunk")
    tts.add_argument("--edge-voice", dest="edge_voice",
                     help="Edge engine voice, e.g. en-US-AriaNeural (use with --engine edge)")
    tts.add_argument("--speed", type=float,
                     help="speech pace multiplier: 1.0 normal, 1.25 faster, 0.8 slower")

    rt = p.add_argument_group("runtime")
    rt.add_argument("--config", help="YAML config file to load first")
    rt.add_argument("--overwrite", action="store_true", default=None,
                    help="regenerate even if the output already exists")
    rt.add_argument("--dry-run", dest="dry_run", action="store_true", default=None,
                    help="ingest + arrange + chunk, but generate no audio")
    return p


def make_source(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Source:
    chosen = [args.text is not None, bool(args.text_file), bool(args.url),
              bool(args.paste), bool(args.pdf)]
    if sum(chosen) != 1:
        parser.error("provide exactly one of --text, --text-file, --url, --pdf, or --paste")

    if args.paste:
        data = sys.stdin.read()
        if not data.strip():
            parser.error("--paste given but stdin was empty")
        return Source("text", data, args.output)
    if args.text is not None:
        return Source("text", args.text, args.output)
    if args.pdf:
        return Source("pdf", args.pdf, args.output)
    if args.text_file:
        low = args.text_file.lower()
        kind = "pdf" if low.endswith(".pdf") else "epub" if low.endswith(".epub") else "file"
        return Source(kind, args.text_file, args.output)
    kind = "pdf" if args.url.lower().split("?", 1)[0].rstrip("/").endswith(".pdf") else "url"
    return Source(kind, args.url, args.output)


def make_config(args: argparse.Namespace) -> Config:
    base = Config.load_yaml(args.config) if args.config else Config()

    # Infer format from an explicit -o extension unless --format overrides it.
    fmt = args.format
    if fmt is None and args.output:
        ext = Path(args.output).suffix.lstrip(".").lower()
        if ext in KNOWN_FORMATS:
            fmt = ext

    overrides = {
        "mode": args.mode,
        "engine": args.engine,
        "device": args.device,
        "reference_wav": args.reference_wav,
        "audio_format": fmt,
        "output_dir": args.output_dir,
        "use_llm": args.use_llm,
        "llm_model": args.llm_model,
        "summarize": args.summarize,
        "summary_words": args.summary_words,
        "seed": args.seed,
        "exaggeration": args.exaggeration,
        "cfg_weight": args.cfg_weight,
        "max_chunk_chars": args.max_chunk_chars,
        "edge_voice": args.edge_voice,
        "speed": args.speed,
        "ocr": args.ocr,
        "overwrite": args.overwrite,
        "dry_run": args.dry_run,
    }
    return base.merged_with(overrides)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    source = make_source(args, parser)
    try:
        cfg = make_config(args)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    report = run([source], cfg)
    job = report.jobs[-1]

    if job.status == "failed":
        print(f"\nFAILED: {job.error}", file=sys.stderr)
        return 1
    if job.status == "skipped":
        print(f"\nSkipped (already exists): {job.output}")
        return 0
    if cfg.dry_run:
        print(f"\nDry run: {job.chars_in} chars in -> {job.chars_spoken} spoken "
              f"-> {job.n_chunks} chunks. Nothing generated.")
        return 0
    print(f"\nWrote {job.output}  ({job.audio_seconds:.1f}s audio, "
          f"{job.n_chunks} chunks, {job.elapsed_s:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
