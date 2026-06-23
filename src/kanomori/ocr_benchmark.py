"""CLI for comparing OCR engines on labelled Kanomori frame screenshots."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from kanomori import ocr


@dataclass(frozen=True)
class OcrBenchmarkCase:
    id: str
    image: Path
    expected_terms: list[str]


@dataclass(frozen=True)
class OcrBenchmarkMetric:
    engine: str
    case_count: int
    term_recall: float
    empty_rate: float
    median_latency_ms: float
    p95_latency_ms: float


def load_cases(path: Path) -> list[OcrBenchmarkCase]:
    cases: list[OcrBenchmarkCase] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        image = _resolve_image_path(Path(payload["image"]), path.parent)
        cases.append(
            OcrBenchmarkCase(
                id=str(payload["id"]),
                image=image,
                expected_terms=[str(term) for term in payload["expected_terms"]],
            )
        )
        if not cases[-1].image.exists():
            raise FileNotFoundError(f"{path}:{line_no}: image not found: {cases[-1].image}")
    return cases


def _resolve_image_path(image: Path, case_dir: Path) -> Path:
    if image.is_absolute():
        return _resolve_glob(image) if _has_glob(image) else image
    if _has_glob(image):
        cwd_matches = _glob_matches(image)
        if cwd_matches:
            return _single_glob_match(image, cwd_matches)
        return _single_glob_match(case_dir / image, _glob_matches(case_dir / image))
    if image.exists():
        return image.resolve()
    return case_dir / image


def _has_glob(path: Path) -> bool:
    return any(char in str(path) for char in "*?[")


def _resolve_glob(pattern: Path) -> Path:
    return _single_glob_match(pattern, _glob_matches(pattern))


def _glob_matches(pattern: Path) -> list[Path]:
    return sorted(Path(match).resolve() for match in glob.glob(str(pattern)))


def _single_glob_match(pattern: Path, matches: Sequence[Path]) -> Path:
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"glob matched no files: {pattern}")
    raise ValueError(f"glob {pattern} matched {len(matches)} files; use an explicit path")


def evaluate_engine(
    engine: str,
    reader: ocr.OcrReader,
    cases: Sequence[OcrBenchmarkCase],
    *,
    timer: Callable[[], float] = time.perf_counter,
) -> OcrBenchmarkMetric:
    total_terms = 0
    matched_terms = 0
    empty_cases = 0
    latencies_ms: list[float] = []

    for case in cases:
        start = timer()
        results = reader.read_image(case.image)
        latencies_ms.append((timer() - start) * 1000)
        text = " ".join(result.text for result in results)
        print(f"{engine}: {case.id}: latency={latencies_ms[-1]:.2f}ms text={text!r}")
        if not text.strip():
            empty_cases += 1
        for term in case.expected_terms:
            total_terms += 1
            if term in text:
                matched_terms += 1
                print(f"{engine}: {case.id}: matched term: {term}")

    return OcrBenchmarkMetric(
        engine=engine,
        case_count=len(cases),
        term_recall=matched_terms / total_terms if total_terms else 0.0,
        empty_rate=empty_cases / len(cases) if cases else 0.0,
        median_latency_ms=statistics.median(latencies_ms) if latencies_ms else 0.0,
        p95_latency_ms=_percentile(latencies_ms, 0.95),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path, help="JSONL benchmark cases")
    parser.add_argument(
        "--engines",
        default=ocr.LEGACY_RAPIDOCR,
        help="Comma-separated OCR engines to benchmark",
    )
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    engines = [ocr.normalize_ocr_engine(engine) for engine in args.engines.split(",")]
    metrics = [
        evaluate_engine(engine, ocr.get_ocr_reader(engine), cases)
        for engine in engines
    ]
    print(json.dumps({"metrics": [asdict(metric) for metric in metrics]}, ensure_ascii=False))
    return 0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))
    return ordered[index]


if __name__ == "__main__":
    sys.exit(main())
