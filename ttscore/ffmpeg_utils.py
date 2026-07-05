"""Thin, well-logged wrappers around the system ``ffmpeg``.

Two jobs for the reader:
  * check ffmpeg is on PATH (only needed for mp3/m4a; wav is written directly),
  * encode the assembled WAV to the final format with single-pass ``loudnorm``
    (−16 LUFS) and the requested channels/sample-rate/bitrate.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .config import Config

FFMPEG = "ffmpeg"


class FFmpegError(RuntimeError):
    pass


def ensure_tool() -> None:
    """Raise early with a friendly message if ffmpeg isn't on PATH."""
    if shutil.which(FFMPEG) is None:
        raise FFmpegError(
            "ffmpeg not found on PATH. Install it (it bundles ffprobe) and re-run, "
            "or set audio_format: wav to skip encoding."
        )


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-12:])
        raise FFmpegError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")
    return proc


def atempo_chain(factor: float) -> str:
    """An ``atempo`` filter string for an arbitrary tempo factor (pitch
    preserved). Each instance is kept in atempo's well-behaved [0.5, 2.0] range
    and chained, so e.g. 2.5 -> 'atempo=1.581,atempo=1.581'. factor > 1 = faster."""
    if factor <= 0:
        raise ValueError("atempo factor must be > 0")
    parts: list[float] = []
    remaining = factor
    while remaining > 2.0:
        parts.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        parts.append(0.5)
        remaining /= 0.5
    parts.append(remaining)
    return ",".join(f"atempo={p:.6f}" for p in parts)


def encode(src_wav: Path, out: Path, cfg: Config, atempo: float = 1.0) -> None:
    """Encode ``src_wav`` -> ``out`` with an optional pitch-preserving speed
    change (``atempo``), single-pass ``loudnorm`` (mp3/m4a), and target format.

    Writes atomically (temp sibling + rename), so a present final file always
    means a complete encode — which the resume/skip guard depends on.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    filters: list[str] = []
    if abs(atempo - 1.0) > 1e-3:
        filters.append(atempo_chain(atempo))
    if cfg.audio_format != "wav":  # keep the wav path loudnorm-free (as before)
        filters.append(f"loudnorm=I={cfg.loudnorm_i}:TP={cfg.loudnorm_tp}"
                       f":LRA={cfg.loudnorm_lra}")

    if cfg.audio_format == "wav":
        codec = "pcm_s16le"
    elif cfg.audio_format == "mp3":
        codec = cfg.mp3_codec
    else:
        codec = cfg.aac_codec

    tmp = out.with_name(out.stem + ".part" + out.suffix)
    cmd = [FFMPEG, "-y", "-i", str(src_wav)]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-c:a", codec, "-ar", str(cfg.sample_rate), "-ac", str(cfg.channels)]
    if cfg.audio_format != "wav":
        cmd += ["-b:a", cfg.audio_bitrate]
    if cfg.audio_format == "m4a":
        cmd += ["-movflags", "+faststart"]   # web/stream-friendly moov atom
    cmd.append(str(tmp))
    try:
        _run(cmd)
        os.replace(str(tmp), str(out))
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
