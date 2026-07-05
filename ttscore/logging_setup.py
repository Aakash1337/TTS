"""Per-run logging: a console stream plus a timestamped file in ``log_dir``,
and :class:`RunReport`, a small accumulator for the end-of-run summary.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

LOGGER_NAME = "ttscore"


def setup_logging(log_dir: Path, run_id: str, verbose: bool = True) -> tuple[logging.Logger, Path]:
    """Configure the package logger with console + file handlers.

    Returns the logger and the path to this run's log file.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{run_id}.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    for h in logger.handlers:   # close before dropping — a cleared-but-open
        try:                    # FileHandler leaks a locked file on Windows
            h.close()
        except Exception:
            pass
    logger.handlers.clear()  # idempotent across repeated calls in one process
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    return logger, log_file


def new_run_id() -> str:
    """A filesystem-safe id, e.g. ``20260703-141502-12345`` (pid suffix keeps
    parallel processes from colliding on the same log file)."""
    return datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


@dataclass
class JobResult:
    """Outcome of turning one source (paste / URL / file) into one audio file."""
    source: str                      # "paste", a URL, or a file path
    status: str                      # "ok" | "skipped" | "failed"
    output: Optional[str] = None
    title: str = ""                  # human title (article/PDF title) when known
    mode: str = "plain"
    used_llm: bool = False
    summarized: bool = False
    chars_in: int = 0                # characters of raw ingested text
    chars_spoken: int = 0            # characters actually sent to the engine
    n_chunks: int = 0
    n_generated: int = 0             # chunks actually voiced (after cache/empties)
    audio_seconds: float = 0.0
    elapsed_s: float = 0.0
    error: Optional[str] = None


@dataclass
class RunReport:
    run_id: str
    started_at: str
    jobs: list[JobResult] = field(default_factory=list)

    def add(self, result: JobResult) -> None:
        self.jobs.append(result)

    def write(self, log_dir: Path) -> Path:
        """Write the JSON summary; return its path."""
        log_dir.mkdir(parents=True, exist_ok=True)
        summary_path = log_dir / f"run_{self.run_id}_summary.json"
        summary = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "counts": self.counts(),
            "jobs": [asdict(j) for j in self.jobs],
        }
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        return summary_path

    def counts(self) -> dict[str, int]:
        return {
            "total": len(self.jobs),
            "ok": sum(j.status == "ok" for j in self.jobs),
            "skipped": sum(j.status == "skipped" for j in self.jobs),
            "failed": sum(j.status == "failed" for j in self.jobs),
        }
