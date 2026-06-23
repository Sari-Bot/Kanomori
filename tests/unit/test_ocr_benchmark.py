from __future__ import annotations

import json

import pytest

from kanomori import ocr
from kanomori.ocr_benchmark import OcrBenchmarkCase, evaluate_target, load_cases, main


class FakeReader:
    def __init__(self, texts: list[str]):
        self.texts = texts

    def read_image(self, path):
        return [ocr.OcrResult(text=text, confidence=0.9, bbox={}) for text in self.texts]


def test_load_cases_reads_jsonl_cases(tmp_path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake")
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {"id": "frame-1", "image": str(image), "expected_terms": ["鹿乃", "歌枠"]},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_cases(cases_path)

    assert cases == [
        OcrBenchmarkCase(id="frame-1", image=image, expected_terms=["鹿乃", "歌枠"])
    ]


def test_load_cases_prefers_current_directory_relative_paths(monkeypatch, tmp_path) -> None:
    workdir = tmp_path / "work"
    case_dir = tmp_path / "eval"
    image = workdir / "media" / "frame.jpg"
    case_dir.mkdir()
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake")
    cases_path = case_dir / "cases.jsonl"
    cases_path.write_text(
        json.dumps({"id": "frame-1", "image": "media/frame.jpg", "expected_terms": ["鹿乃"]}),
        encoding="utf-8",
    )
    monkeypatch.chdir(workdir)

    cases = load_cases(cases_path)

    assert cases[0].image == image


def test_load_cases_resolves_single_glob_from_current_directory(monkeypatch, tmp_path) -> None:
    workdir = tmp_path / "work"
    case_dir = tmp_path / "eval"
    image = workdir / "media" / "hash-a" / "frames" / "frame_000000_000.jpg"
    case_dir.mkdir()
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake")
    cases_path = case_dir / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "frame-000",
                "image": "media/*/frames/frame_000000_000.jpg",
                "expected_terms": ["ww", "かわいい"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(workdir)

    cases = load_cases(cases_path)

    assert cases[0].image == image


def test_load_cases_rejects_ambiguous_glob(monkeypatch, tmp_path) -> None:
    workdir = tmp_path / "work"
    case_dir = tmp_path / "eval"
    first = workdir / "media" / "hash-a" / "frames" / "frame_000000_000.jpg"
    second = workdir / "media" / "hash-b" / "frames" / "frame_000000_000.jpg"
    case_dir.mkdir()
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"fake")
    second.write_bytes(b"fake")
    cases_path = case_dir / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "frame-000",
                "image": "media/*/frames/frame_000000_000.jpg",
                "expected_terms": ["ww"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(workdir)

    with pytest.raises(ValueError, match="matched 2 files"):
        load_cases(cases_path)


def test_evaluate_target_reports_recall_empty_rate_latency_and_throughput(tmp_path) -> None:
    present = tmp_path / "present.jpg"
    empty = tmp_path / "empty.jpg"
    present.write_bytes(b"fake")
    empty.write_bytes(b"fake")
    cases = [
        OcrBenchmarkCase(id="present", image=present, expected_terms=["鹿乃", "歌枠"]),
        OcrBenchmarkCase(id="empty", image=empty, expected_terms=["字幕"]),
    ]
    calls = iter([[ocr.OcrResult("鹿乃の歌枠", 0.9, {})], []])

    class Reader:
        def read_image(self, path):
            return next(calls)

    timer = iter([0.0, 0.010, 0.010, 0.040]).__next__

    metric = evaluate_target(
        ocr.OcrConfig(ocr.PPOCRV5_SERVER, ocr.OCR_BACKEND_ONNXRUNTIME),
        Reader(),
        cases,
        timer=timer,
    )

    assert metric.model == ocr.PPOCRV5_SERVER
    assert metric.backend == ocr.OCR_BACKEND_ONNXRUNTIME
    assert metric.case_count == 2
    assert metric.term_recall == pytest.approx(2 / 3)
    assert metric.empty_rate == pytest.approx(0.5)
    assert metric.median_latency_ms == pytest.approx(20.0)
    assert metric.p95_latency_ms == pytest.approx(30.0)
    assert metric.throughput_images_per_sec == pytest.approx(50.0)


def test_main_writes_model_backend_json_metrics(monkeypatch, tmp_path, capsys) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake")
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps({"id": "frame-1", "image": str(image), "expected_terms": ["鹿乃"]}),
        encoding="utf-8",
    )
    calls = []

    def get_reader(model, backend, *, allow_backend_fallback=True):
        calls.append((model, backend, allow_backend_fallback))
        return FakeReader(["鹿乃"])

    monkeypatch.setattr(ocr, "get_ocr_reader", get_reader)

    exit_code = main(
        [
            "--cases",
            str(cases_path),
            "--models",
            "ppocrv5_server",
            "--backends",
            "onnxruntime,cuda,tensorrt",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [(m["model"], m["backend"]) for m in payload["metrics"]] == [
        ("ppocrv5_server", "onnxruntime"),
        ("ppocrv5_server", "cuda"),
        ("ppocrv5_server", "tensorrt"),
    ]
    assert [m["term_recall"] for m in payload["metrics"]] == [1.0, 1.0, 1.0]
    assert calls == [
        ("ppocrv5_server", "onnxruntime", False),
        ("ppocrv5_server", "cuda", False),
        ("ppocrv5_server", "tensorrt", False),
    ]


def test_main_keeps_third_party_stdout_out_of_json(monkeypatch, tmp_path, capsys) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake")
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps({"id": "frame-1", "image": str(image), "expected_terms": ["鹿乃"]}),
        encoding="utf-8",
    )

    class NoisyReader:
        def read_image(self, path):
            print("third-party stdout")
            return [ocr.OcrResult(text="鹿乃", confidence=0.9, bbox={})]

    def get_reader(model, backend, *, allow_backend_fallback=True):
        print("reader construction stdout")
        return NoisyReader()

    monkeypatch.setattr(ocr, "get_ocr_reader", get_reader)

    exit_code = main(["--cases", str(cases_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out)["metrics"][0]["term_recall"] == 1.0
    assert "reader construction stdout" in captured.err
    assert "third-party stdout" in captured.err


def test_main_keeps_deprecated_engines_alias(monkeypatch, tmp_path, capsys) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake")
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps({"id": "frame-1", "image": str(image), "expected_terms": ["鹿乃"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ocr,
        "get_ocr_reader",
        lambda model, backend, *, allow_backend_fallback=True: FakeReader(["鹿乃"]),
    )

    exit_code = main(
        [
            "--cases",
            str(cases_path),
            "--engines",
            "legacy_rapidocr,rapidocr_ppocrv5_mobile",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [(m["model"], m["backend"]) for m in payload["metrics"]] == [
        ("legacy_rapidocr", "onnxruntime"),
        ("ppocrv5_mobile", "onnxruntime"),
    ]


def test_main_returns_nonzero_when_requested_backend_unavailable(
    monkeypatch, tmp_path, capsys
) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake")
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps({"id": "frame-1", "image": str(image), "expected_terms": ["鹿乃"]}),
        encoding="utf-8",
    )

    def get_reader(model, backend, *, allow_backend_fallback=True):
        raise ocr.OcrBackendUnavailable("TensorRT unavailable")

    monkeypatch.setattr(ocr, "get_ocr_reader", get_reader)

    exit_code = main(
        [
            "--cases",
            str(cases_path),
            "--models",
            "ppocrv5_server",
            "--backends",
            "tensorrt",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "TensorRT unavailable" in captured.err


def test_main_returns_nonzero_when_requested_cuda_backend_unavailable(
    monkeypatch, tmp_path, capsys
) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake")
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps({"id": "frame-1", "image": str(image), "expected_terms": ["鹿乃"]}),
        encoding="utf-8",
    )
    calls = []

    def get_reader(model, backend, *, allow_backend_fallback=True):
        calls.append((model, backend, allow_backend_fallback))
        raise ocr.OcrBackendUnavailable("CUDAExecutionProvider unavailable")

    monkeypatch.setattr(ocr, "get_ocr_reader", get_reader)

    exit_code = main(
        [
            "--cases",
            str(cases_path),
            "--models",
            "ppocrv5_server",
            "--backends",
            "cuda",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "CUDAExecutionProvider unavailable" in captured.err
    assert calls == [("ppocrv5_server", "cuda", False)]
