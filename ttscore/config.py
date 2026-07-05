"""Central configuration for the TTS pipeline.

Every knob lives on :class:`Config`. Values resolve in three layers, lowest
precedence first::

    dataclass defaults  <  YAML file (--config)  <  CLI flags

Nothing here imports torch/chatterbox/ollama, so the config can be inspected
(``--help``, ``--dry-run``) without paying any model-import cost.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, MISSING
from pathlib import Path
from typing import Any, Optional


# Audio containers we know how to encode. wav needs no ffmpeg; mp3/m4a do.
KNOWN_FORMATS = ("mp3", "wav", "m4a")


@dataclass
class Config:
    """All pipeline settings. See ``config.example.yaml`` for an annotated copy."""

    # ── Output paths ─────────────────────────────────────────────────────────
    output_dir: str = "output"
    """Where finished audio (and optional .wav / transcript) is written."""

    cache_dir: str = ".tts_cache"
    """Per-chunk TTS clips are cached here so a crash mid-document resumes cheaply."""

    log_dir: str = "logs"
    """Per-run log + JSON summary land here."""

    # ── Arrange: cleanup / structuring ───────────────────────────────────────
    mode: str = "plain"
    """Content profile that shapes cleanup + narration. A "mode" is just a named
    bundle of rule toggles + (optional) LLM prompt + voice/pause params, defined
    in ``ttscore.arrange.modes``. MVP ships 'plain'; 'academic' / 'article' /
    'narrative' / 'study' / 'auto' are additive later. Unknown values fall back
    to 'plain' with a warning (never a hard error)."""

    expand_numbers: bool = True
    """Rules step: speak digits/currency/years as words (num2words)."""

    use_llm: bool = False
    """Optional polish pass through a LOCAL LLM (Ollama). Off by default; the
    rule-based cleanup always runs regardless. Never a hard dependency — if the
    server/model is unreachable we log and keep the rules-only text."""

    llm_model: str = "gemma4:e4b"
    """Ollama model tag for the optional polish pass."""

    llm_host: str = "http://localhost:11434"
    llm_timeout: float = 120.0

    summarize: bool = False
    """Speak a condensed summary instead of the full text (local LLM required;
    the job fails with a clear message if Ollama is unreachable)."""

    summary_words: int = 250
    """Approximate length of the spoken summary."""

    # ── Ingest: web fetch ────────────────────────────────────────────────────
    request_timeout: float = 20.0
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TTSReader/1.0 (+local)"
    )

    # ── Chunking ─────────────────────────────────────────────────────────────
    max_chunk_chars: int = 300
    """Soft upper bound per TTS call. Text is split on SENTENCE boundaries and
    never mid-sentence; a lone sentence longer than this is split at clause
    punctuation as a last resort. Chatterbox is happiest around 1-2 sentences."""

    # ── TTS engine (pluggable) ───────────────────────────────────────────────
    engine: str = "edge"
    """Which TTSEngine to use: 'edge' (Microsoft online neural voices incl. Aria —
    free, no key, needs internet; the default) or 'chatterbox' (local, offline,
    GPU, can clone a voice)."""

    reference_wav: Optional[str] = None
    """A single 5-10s clip cloned as the narrator voice. None -> built-in voice."""

    exaggeration: float = 0.5   # Chatterbox expressiveness
    cfg_weight: float = 0.5     # Chatterbox pacing
    temperature: float = 0.8
    seed: Optional[int] = None  # set for reproducible generations

    device: str = "cuda"        # "cuda" | "cpu"; auto-falls back to cpu if no GPU
    cuda_mem_fraction: Optional[float] = None
    """Cap THIS process's VRAM to a fraction (0-1). None = no cap."""

    # ── Edge TTS engine (engine == 'edge') ───────────────────────────────────
    edge_voice: str = "en-US-AriaNeural"
    """Microsoft online neural voice. 'en-US-AriaNeural' == "Microsoft Aria
    Online (Natural) - English (United States)". List all voices with:
    E:\\TTS\\.venv\\Scripts\\edge-tts --list-voices"""
    edge_rate: str = "+0%"      # speaking rate,  e.g. "+10%" faster / "-10%" slower
    edge_volume: str = "+0%"    # loudness,       e.g. "+10%" / "-10%"
    edge_pitch: str = "+0Hz"    # pitch,          e.g. "+20Hz" / "-20Hz"

    # ── PDF ingest + OCR ─────────────────────────────────────────────────────
    ocr: str = "off"
    """OCR mode for image-only PDF pages: 'off' (text layer only), 'auto' (OCR
    only pages with almost no extractable text), or 'force' (OCR every page).
    'auto'/'force' need the optional easyocr package installed."""
    ocr_min_chars: int = 16     # 'auto': a page with fewer chars is image-only
    ocr_language: str = "en"    # easyocr language code(s), comma-separated
    ocr_dpi: int = 200          # rasterization DPI for OCR

    # ── Assembly ─────────────────────────────────────────────────────────────
    sample_rate: int = 44100    # common working + output rate; clips resample to it
    channels: int = 1           # narration is mono by nature (1 = smaller, natural)
    pause_sentence_ms: int = 300   # silence inserted between chunks/sentences
    pause_paragraph_ms: int = 700  # longer silence at paragraph breaks
    speed: float = 1.0
    """Speech pace multiplier: 1.0 = normal, 1.25 = 25% faster, 0.8 = 20% slower.
    For the 'edge' engine this maps to the voice's native rate (best quality);
    for other engines it's applied as a pitch-preserving tempo change on encode."""

    # ── Loudness / encoding ──────────────────────────────────────────────────
    loudnorm_i: float = -16.0   # integrated loudness target (LUFS)
    loudnorm_tp: float = -1.5   # true-peak ceiling (dBTP)
    loudnorm_lra: float = 11.0  # loudness range
    audio_format: str = "mp3"   # "mp3" | "wav" | "m4a"
    audio_bitrate: str = "128k"
    mp3_codec: str = "libmp3lame"
    aac_codec: str = "aac"
    keep_wav: bool = True       # also keep the raw pre-encode .wav next to output
    write_transcript: bool = True  # write the exact spoken text as .txt
    word_timings: bool = True
    """Write ``<output>.words.json`` word-level timings when the engine reports
    them (edge voices do). Powers the read-along highlighting in the app."""

    preview_seconds: Optional[float] = None
    """When set (the web app uses ~25), encode ``<output>.preview.mp3`` as soon
    as this many seconds of audio exist, so listening can start while the rest
    of a long document is still generating. None = off (CLI default)."""

    # ── Runtime ──────────────────────────────────────────────────────────────
    overwrite: bool = False     # if False, skip when the output already exists
    dry_run: bool = False       # ingest + arrange + chunk, but generate no audio

    # ── Normalisation ────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        self.mode = str(self.mode).strip().lower()
        self.engine = str(self.engine).strip().lower()
        self.device = str(self.device).strip().lower()
        self.audio_format = str(self.audio_format).strip().lower().lstrip(".")
        self.ocr = str(self.ocr).strip().lower()

    # ── Derived helpers ──────────────────────────────────────────────────────
    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)

    @property
    def log_path(self) -> Path:
        return Path(self.log_dir)

    # ── Construction helpers ─────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Build a Config from a plain dict, rejecting unknown keys early."""
        known = {f.name for f in fields(cls)}
        clean: dict[str, Any] = {}
        for k, v in (data or {}).items():
            if k not in known:
                raise KeyError(f"Unknown config key: {k!r}")
            clean[k] = v
        return cls(**clean)

    @classmethod
    def load_yaml(cls, path: str | Path) -> "Config":
        import yaml  # local import so PyYAML isn't needed for pure-CLI use

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    def merged_with(self, overrides: dict[str, Any]) -> "Config":
        """Return a copy with ``overrides`` applied. None values are ignored so
        unset CLI flags don't clobber YAML/defaults."""
        current = {f.name: getattr(self, f.name) for f in fields(self)}
        for k, v in overrides.items():
            if v is None:
                continue
            current[k] = v
        return Config.from_dict(current)

    def validate(self) -> None:
        if self.device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got {self.device!r}")
        if self.audio_format not in KNOWN_FORMATS:
            raise ValueError(
                f"audio_format must be one of {KNOWN_FORMATS}, got {self.audio_format!r}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.channels not in (1, 2):
            raise ValueError(f"channels must be 1 or 2, got {self.channels}")
        if self.max_chunk_chars < 20:
            raise ValueError(
                f"max_chunk_chars must be >= 20, got {self.max_chunk_chars}")
        if self.pause_sentence_ms < 0 or self.pause_paragraph_ms < 0:
            raise ValueError("pause_*_ms must be >= 0")
        if not (0.5 <= self.speed <= 2.0):
            raise ValueError(f"speed must be in [0.5, 2.0], got {self.speed}")
        if self.ocr not in ("off", "auto", "force"):
            raise ValueError(f"ocr must be 'off', 'auto', or 'force', got {self.ocr!r}")
        if self.summary_words < 50:
            raise ValueError(f"summary_words must be >= 50, got {self.summary_words}")
        if self.cuda_mem_fraction is not None and not (0.0 < self.cuda_mem_fraction <= 1.0):
            raise ValueError(
                f"cuda_mem_fraction must be in (0, 1.0], got {self.cuda_mem_fraction}")
        if self.reference_wav and not Path(self.reference_wav).is_file():
            raise FileNotFoundError(f"reference_wav not found: {self.reference_wav}")


def default_field_help() -> dict[str, str]:
    """Map field name -> default repr, used by the CLI ``--help`` epilog."""
    out: dict[str, str] = {}
    for f in fields(Config):
        if f.default is not MISSING:
            out[f.name] = repr(f.default)
        elif f.default_factory is not MISSING:  # type: ignore[misc]
            out[f.name] = repr(f.default_factory())  # type: ignore[misc]
    return out
