"""CLI for comparing OCR engines on labelled Kanomori frame screenshots."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
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
    model: str
    backend: str
    case_count: int
    term_recall: float
    empty_rate: float
    median_latency_ms: float
    p95_latency_ms: float
    throughput_images_per_sec: float


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


def evaluate_target(
    target: ocr.OcrConfig,
    reader: ocr.OcrReader,
    cases: Sequence[OcrBenchmarkCase],
    *,
    timer: Callable[[], float] = time.perf_counter,
    verbose: bool = False,
) -> OcrBenchmarkMetric:
    total_terms = 0
    matched_terms = 0
    empty_cases = 0
    latencies_ms: list[float] = []
    label = f"{target.model}+{target.backend}"

    for case in cases:
        start = timer()
        results = reader.read_image(case.image)
        latencies_ms.append((timer() - start) * 1000)
        text = " ".join(result.text for result in results)
        if verbose:
            print(
                f"{label}: {case.id}: latency={latencies_ms[-1]:.2f}ms text={text!r}",
                file=sys.stderr,
            )
        if not text.strip():
            empty_cases += 1
        for term in case.expected_terms:
            total_terms += 1
            if term in text:
                matched_terms += 1
                if verbose:
                    print(f"{label}: {case.id}: matched term: {term}", file=sys.stderr)

    total_latency_sec = sum(latencies_ms) / 1000

    return OcrBenchmarkMetric(
        model=target.model,
        backend=target.backend,
        case_count=len(cases),
        term_recall=matched_terms / total_terms if total_terms else 0.0,
        empty_rate=empty_cases / len(cases) if cases else 0.0,
        median_latency_ms=statistics.median(latencies_ms) if latencies_ms else 0.0,
        p95_latency_ms=_percentile(latencies_ms, 0.95),
        throughput_images_per_sec=len(cases) / total_latency_sec if total_latency_sec else 0.0,
    )


def evaluate_engine(
    engine: str,
    reader: ocr.OcrReader,
    cases: Sequence[OcrBenchmarkCase],
    *,
    timer: Callable[[], float] = time.perf_counter,
) -> OcrBenchmarkMetric:
    return evaluate_target(ocr.parse_ocr_engine_alias(engine), reader, cases, timer=timer)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path, help="JSONL benchmark cases")
    parser.add_argument(
        "--engines",
        help="Deprecated comma-separated OCR engine aliases; implies onnxruntime",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated OCR models to benchmark",
    )
    parser.add_argument(
        "--backends",
        help="Comma-separated OCR backends to benchmark",
    )
    parser.add_argument("--verbose", action="store_true", help="Write per-case OCR text to stderr")
    args = parser.parse_args(argv)

    try:
        cases = load_cases(args.cases)
        targets = _benchmark_targets(args.engines, args.models, args.backends)
        metrics = []
        for target in targets:
            with redirect_stdout(sys.stderr):
                reader = ocr.get_ocr_reader(
                    target.model,
                    target.backend,
                    allow_backend_fallback=False,
                )
                metric = evaluate_target(target, reader, cases, verbose=args.verbose)
            metrics.append(metric)
    except (FileNotFoundError, ValueError, ocr.OcrBackendUnavailable) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps({"metrics": [asdict(metric) for metric in metrics]}, ensure_ascii=False))
    return 0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _benchmark_targets(
    engines: str | None,
    models: str | None,
    backends: str | None,
) -> list[ocr.OcrConfig]:
    if engines and (models or backends):
        raise ValueError("--engines cannot be combined with --models or --backends")
    if engines:
        return [ocr.parse_ocr_engine_alias(engine) for engine in _csv_values(engines)]

    model_values = _csv_values(models or ocr.PPOCRV5_SERVER)
    backend_values = _csv_values(backends or ocr.OCR_BACKEND_ONNXRUNTIME)
    return [
        ocr.validate_ocr_config(ocr.OcrConfig(model, backend))
        for model in model_values
        for backend in backend_values
    ]


def _csv_values(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one comma-separated value")
    return values


if __name__ == "__main__":
    sys.exit(main())
