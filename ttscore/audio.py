"""Audio DSP for assembling a narration track from per-chunk clips.

Unlike the dubbing pipeline (which placed clips at absolute timecodes on a
fixed-length canvas), narration is just clips played back-to-back with natural
pauses — shorter between sentences, longer between paragraphs. So this is a
straight resample + concatenate, which is simpler and can't drift.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


# ── resampling ───────────────────────────────────────────────────────────────
def resample(clip: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample a 1-D float clip from ``src_sr`` to ``dst_sr`` (high quality)."""
    clip = np.asarray(clip, dtype=np.float32).reshape(-1)
    if src_sr == dst_sr or clip.size == 0:
        return clip
    try:
        import torch
        import torchaudio.functional as AF

        t = torch.from_numpy(clip).unsqueeze(0)
        out = AF.resample(t, src_sr, dst_sr).squeeze(0).cpu().numpy()
        return out.astype(np.float32, copy=False)
    except Exception:
        # Last-resort linear interpolation if torchaudio is unavailable.
        n_out = int(round(clip.size * dst_sr / src_sr))
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        x_old = np.linspace(0.0, 1.0, num=clip.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, clip).astype(np.float32)


# ── assembly ─────────────────────────────────────────────────────────────────
def silence(ms: int, sr: int) -> np.ndarray:
    """A silent mono clip of ``ms`` milliseconds."""
    n = max(0, int(round(ms / 1000.0 * sr)))
    return np.zeros(n, dtype=np.float32)


def assemble(clips: list[np.ndarray], paragraph_ends: list[bool],
             sr: int, pause_sentence_ms: int, pause_paragraph_ms: int,
             ) -> tuple[np.ndarray, list[float]]:
    """Concatenate ``clips`` (already at ``sr``) with pauses between them.

    The pause BEFORE clip *i* is a paragraph pause when clip *i-1* ended a
    paragraph, otherwise a sentence pause. A short lead-in and tail are added so
    the track doesn't start/end abruptly.

    Returns ``(track, offsets)`` — ``offsets[i]`` is the second at which clip
    *i* starts in the final track (used to globalize per-clip word timings).
    """
    sent_gap = silence(pause_sentence_ms, sr)
    para_gap = silence(pause_paragraph_ms, sr)
    lead = silence(min(pause_sentence_ms, 200), sr)

    parts: list[np.ndarray] = [lead]
    offsets: list[float] = []
    pos = lead.size
    for i, clip in enumerate(clips):
        clip = np.asarray(clip, dtype=np.float32).reshape(-1)
        if i > 0:
            gap = para_gap if paragraph_ends[i - 1] else sent_gap
            parts.append(gap)
            pos += gap.size
        offsets.append(pos / sr if sr else 0.0)
        if clip.size:
            parts.append(clip)
            pos += clip.size
    parts.append(sent_gap)  # tail

    non_empty = [p for p in parts if p.size]
    if not non_empty:
        return np.zeros(0, dtype=np.float32), offsets
    return np.concatenate(non_empty).astype(np.float32, copy=False), offsets


def finalize(track: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    """Peak-guard before encode/loudnorm (cheap safety against any overshoot)."""
    if track.size == 0:
        return track
    peak = float(np.max(np.abs(track)))
    if peak > ceiling:
        track = track * (ceiling / peak)
    return track.astype(np.float32, copy=False)


def to_channels(track: np.ndarray, channels: int) -> np.ndarray:
    """Return a (N,) mono or (N, 2) stereo array as requested."""
    track = np.asarray(track, dtype=np.float32).reshape(-1)
    if channels == 2:
        return np.column_stack([track, track])
    return track


def write_wav(path: Path, track: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(track, dtype=np.float32), sr, subtype="FLOAT")


def duration_seconds(track: np.ndarray, sr: int) -> float:
    n = track.shape[0] if track.ndim else 0
    return n / sr if sr else 0.0
