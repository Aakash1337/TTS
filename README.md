# TTS Reader

Turn **pasted text** or a **web link** into a clean, natural-sounding **narration
file** (MP3). Paste an article, a paper, a chapter, or a wall of notes; the tool
extracts the real content, tidies it for the ear, and speaks it with
[Chatterbox TTS](https://github.com/resemble-ai/chatterbox) on your GPU.

> Built on the engine + install flow proven in the AI Dubbing project. Local and
> free by default — no API keys, no per-use cost.

---

## How it works

```
  paste text ─┐
              ├─▶  ingest  ─▶  arrange  ─▶  chunk  ─▶  synthesize  ─▶  assemble  ─▶  MP3
  a URL ──────┘              (rules +      (sentence   (Chatterbox    (concat +   (+ .wav,
                              optional      aligned)     TTS)           pauses,     + .txt
                              local LLM)                                loudnorm)   transcript)
```

1. **Ingest** — pasted text passes through; a URL is fetched and run through
   `trafilatura` to pull just the article body (nav/ads/comments stripped).
2. **Arrange** — rule-based cleanup always runs (fix line-wraps, de-hyphenate,
   strip markdown, normalize punctuation, speak numbers/currency/percent). An
   **optional** pass through a local **Ollama** model (e.g. Gemma) smooths messy
   input further — off by default, never a hard dependency.
3. **Chunk** — split into sentence-aligned pieces (`pysbd`) sized for the engine.
4. **Synthesize** — one clip per chunk, **cached** by content hash so a crash or
   re-run only regenerates what's missing.
5. **Assemble** — resample, concatenate with short pauses (longer at paragraph
   breaks), loudness-normalize (−16 LUFS), encode to MP3 (+ raw WAV + transcript).

---

## Install (Windows / PowerShell)

Requires **Python 3.10+**, **ffmpeg** on `PATH`, and (recommended) an NVIDIA GPU.

```powershell
cd E:\TTS
powershell -ExecutionPolicy Bypass -File .\setup.ps1        # CUDA build
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Cpu   # CPU-only
```

The first real run downloads the Chatterbox weights (~1–2 GB) into
`E:\TTS\models\` (a project-local HuggingFace cache, kept off the system drive).

---

## Desktop & web app

- **Desktop app (recommended):** double-click **`TTS Reader (Desktop).bat`** — or
  the **TTS Reader** shortcut on your Desktop. A native window opens (no browser,
  no terminal): paste text / a link / a PDF, pick a voice, set the speed, Generate.
- **Web app:** `\.venv\Scripts\python.exe serve.py` (or `Start TTS Reader.bat`)
  opens the same UI in your browser; it auto-picks a free port.

Both are thin shells over the same `ttscore` engine — features land in all of
them at once. Long jobs show a live progress bar and can be **cancelled**.

**Read along:** for Microsoft voices, the result includes a live transcript that
**highlights each word as it's spoken** (the same WordBoundary data Edge's Read
Aloud uses). Click any word to jump the audio there; the **playback speed**
slider adjusts live while it reads, pitch preserved. Word timings are also
written next to the file as `<name>.words.json` (disable with
`word_timings: false`).

**Library:** everything you've generated appears in the 📚 Library panel —
replay (with read-along), download, or delete past narrations. Long audio
**resumes where you left off**, and your voice/speed choices are remembered.

**Ebooks:** drop an **.epub** on the file tab (or `--text-file book.epub` in the
CLI) — chapters are read in order with proper paragraph pauses.

**Start listening immediately:** on long documents, playback begins with a
preview as soon as ~25 s of audio exists; when generation finishes, the player
swaps to the full file without losing your place.

**Contents:** long narrations get a ☰ Contents list (one entry per paragraph/
section with its timestamp) — click to jump the audio there.

## Usage

```powershell
$py = ".\.venv\Scripts\python.exe"

# Speak a bit of text:
& $py cli.py --text "Hello world, this is a test of the reader." -o output\hello.mp3

# Speak a local file:
& $py cli.py --text-file notes.md

# Speak a web article (auto-named from the page title):
& $py cli.py --url https://example.com/some-article

# Paste from the clipboard / a pipe:
Get-Content notes.txt | & $py cli.py --paste

# Turn on the local-LLM cleanup pass (needs Ollama running):
& $py cli.py --url https://example.com/article --llm --llm-model gemma4:e4b

# See what would happen without generating audio:
& $py cli.py --text-file notes.md --dry-run
```

Run `... cli.py --help` for the full flag list.

---

## Configuration

Defaults live in [`ttscore/config.py`](ttscore/config.py); copy
[`config.example.yaml`](config.example.yaml) to `config.yaml` and pass
`--config config.yaml`. Resolution order: **defaults → YAML → CLI flags**.

| Knob | Default | Meaning |
|---|---|---|
| `mode` | `plain` | content profile (cleanup + narration style) |
| `use_llm` | `false` | optional local-LLM polish (Ollama) |
| `llm_model` | `gemma4:e4b` | Ollama model tag for the polish pass |
| `engine` | `chatterbox` | TTS engine (pluggable) |
| `reference_wav` | `null` | 5–10 s clip to clone as the narrator voice |
| `max_chunk_chars` | `300` | sentence-aligned chunk size |
| `audio_format` | `mp3` | `mp3` \| `wav` \| `m4a` |
| `sample_rate` / `channels` | `44100` / `1` | output audio format |
| `device` | `cuda` | `cuda` \| `cpu` (auto-falls back) |

---

## Modes & the road ahead

A **mode** is a named profile = *rule toggles + optional LLM prompt + voice/pause
settings*, defined in [`ttscore/arrange/modes.py`](ttscore/arrange/modes.py).
MVP ships **`plain`**; the seam is built so these are purely additive later:

- `academic` — handle `[12]` citations, headings, skip reference dumps
- `article` — drop bylines / share cruft
- `narrative` — expressive voice, longer pauses for novels/stories
- `study` — measured pace, clearer pauses
- `auto` — a local LLM reads a sample and picks the profile

Because the pipeline is an ordered list of optional stages, planned features
(auto-summary, chapter splitting, translation) drop in without touching the core.
A **web app** and later a **desktop app** are thin front-ends over this same
`ttscore` package.

---

## Project layout

```
cli.py                  command-line entry point
config.example.yaml     annotated config
setup.ps1               venv + deps installer (Windows)
ttscore/
  config.py             Config dataclass (every knob) + YAML/CLI merge
  ingest/
    paste.py            pasted text / local file
    web.py              URL -> clean article text (trafilatura + readability)
  arrange/
    modes.py            mode/profile table (ship 'plain')
    rules.py            deterministic cleanup for natural speech
    llm.py              optional local-LLM polish (Ollama)
  chunk.py              sentence-aware splitting (pysbd)
  engines/
    base.py             TTSEngine interface + factory
    chatterbox.py       Chatterbox implementation
  audio.py              resample, concat with pauses, finalize
  ffmpeg_utils.py       loudnorm + WAV->MP3/M4A encode
  cache.py              per-chunk clip cache (resume)
  logging_setup.py      run logger + JSON summary
  pipeline.py           orchestration (run / synthesize)
```

---

## Notes

- **ffmpeg** is only needed for `mp3`/`m4a`; `audio_format: wav` skips it.
- **GPU**: Chatterbox auto-falls back to CPU (slow) if CUDA is unavailable.
- **LLM cleanup** is optional and local — if Ollama isn't running, the tool logs
  a warning and uses the rule-cleaned text.
- Chatterbox stamps an inaudible Perth watermark on its output by design.
