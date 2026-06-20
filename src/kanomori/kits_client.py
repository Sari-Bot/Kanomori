"""The subprocess boundary to KITS — the only place Kanomori touches KITS.

KITS is consumed strictly as a CLI subprocess (`uv run kits subtitle ...`), never imported:
it is AGPL and carries a GPU/torch stack we keep out of Kanomori's process. transcription
hard-requires a GPU (CUDA/MPS), so this module is the seam that the ingestion worker calls.

``transcribe`` takes an injectable ``runner`` (defaulting to ``subprocess.run``) so tests can
exercise the argv contract and error handling without a GPU or a real KITS invocation. The
argv is built as a **list** — never a shell string — so media filenames can't inject shell.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from kanomori.config import get_settings


class KitsError(RuntimeError):
    """Raised when the KITS subprocess fails or produces no usable SRT."""


def build_kits_argv(
    audio_path: Path,
    out_srt: Path,
    *,
    separate: bool = False,
    filter_game: Sequence[str] | None = None,
    language: str = "japanese",
) -> list[str]:
    """Construct the `uv run kits subtitle` argv list. No shell interpolation."""
    argv = [
        "uv", "run", "kits", "subtitle",
        "-i", str(audio_path),
        "-o", str(out_srt),
        "--language", language,
    ]
    if separate:
        argv.append("--separate")
    for game in filter_game or []:
        argv += ["--filter-game", game]
    return argv


# A runner has the shape of subprocess.run: (argv, **kwargs) -> CompletedProcess-like.
Runner = Callable[..., subprocess.CompletedProcess]


def transcribe(
    audio_path: Path,
    out_srt: Path,
    *,
    kits_dir: Path | None = None,
    separate: bool = False,
    filter_game: Sequence[str] | None = None,
    language: str = "japanese",
    timeout: float | None = None,
    runner: Runner | None = None,
) -> Path:
    """Run KITS to transcribe ``audio_path`` to ``out_srt``; return ``out_srt`` on success.

    Raises ``KitsError`` on a non-zero exit or if KITS produced no non-empty SRT. ``kits_dir``
    (cwd for the subprocess) and ``timeout`` default to settings. ``runner`` defaults to
    ``subprocess.run`` and is injectable for tests.
    """
    settings = get_settings()
    if kits_dir is None:
        kits_dir = settings.kits_dir
    if timeout is None:
        timeout = settings.kits_timeout
    if runner is None:
        runner = subprocess.run

    argv = build_kits_argv(
        audio_path, out_srt, separate=separate, filter_game=filter_game, language=language
    )
    out_srt.parent.mkdir(parents=True, exist_ok=True)

    result = runner(
        argv,
        cwd=kits_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    # Always persist the full, untruncated KITS output next to the SRT. KITS crashes deep in a
    # transformers/torch stack with long tracebacks; without this artifact the actual cause is
    # lost (the exception message is necessarily bounded). The log is the diagnosis record.
    log_path = out_srt.with_suffix(".kits.log")
    log_path.write_text(
        f"$ {' '.join(argv)}\n(cwd={kits_dir})\n"
        f"--- returncode: {result.returncode} ---\n"
        f"--- stdout ---\n{result.stdout or ''}\n"
        f"--- stderr ---\n{result.stderr or ''}\n",
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise KitsError(
            f"kits subtitle exited {result.returncode} "
            f"(full log: {log_path}): {(result.stderr or '').strip()[:500]}"
        )
    if not out_srt.is_file():
        raise KitsError(f"kits produced no output at {out_srt} (log: {log_path})")
    if not out_srt.read_text(encoding="utf-8").strip():
        raise KitsError(f"kits produced an empty SRT at {out_srt} (log: {log_path})")
    return out_srt
