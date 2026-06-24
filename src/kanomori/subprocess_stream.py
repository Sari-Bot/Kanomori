from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

LogOutput = Callable[[str, str], None]


def run_logged(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: float | None = None,
    log_output: LogOutput | None = None,
) -> subprocess.CompletedProcess:
    process = subprocess.Popen(
        list(argv),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdout: list[str] = []
    stderr: list[str] = []

    def pump(stream_name: str, pipe, sink: list[str]) -> None:
        for line in iter(pipe.readline, ""):
            sink.append(line)
            text = line.rstrip("\n")
            if text and log_output is not None:
                log_output(stream_name, text)
        pipe.close()

    threads = [
        threading.Thread(target=pump, args=("stdout", process.stdout, stdout), daemon=True),
        threading.Thread(target=pump, args=("stderr", process.stderr, stderr), daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise
    finally:
        for thread in threads:
            thread.join()
    return subprocess.CompletedProcess(list(argv), returncode, "".join(stdout), "".join(stderr))
