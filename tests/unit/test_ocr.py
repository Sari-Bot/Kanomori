from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from kanomori import ocr


def test_normalize_ocr_engine_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported OCR engine"):
        ocr.normalize_ocr_engine("missing")


def test_legacy_rapidocr_reader_normalizes_tuple_results(monkeypatch, tmp_path) -> None:
    class FakeRapidOCR:
        def __call__(self, path: str):
            assert Path(path).name == "frame.jpg"
            return [
                ([[0, 0], [1, 0], [1, 1], [0, 1]], "入口の看板", 0.91),
                ([[2, 2], [3, 2], [3, 3], [2, 3]], "", 0.5),
            ], (0.01, 0.02, 0.03)

    fake_module = types.SimpleNamespace(RapidOCR=FakeRapidOCR)
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", fake_module)
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake")

    results = ocr.LegacyRapidOcrReader().read_image(image)

    assert results == [
        ocr.OcrResult(
            text="入口の看板",
            confidence=pytest.approx(0.91),
            bbox=[[0, 0], [1, 0], [1, 1], [0, 1]],
        )
    ]


def test_rapidocr_v3_reader_configures_ppocrv5_mobile(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeEnumValue:
        def __init__(self, value: str):
            self.value = value

    class FakeRapidOCR:
        def __init__(self, params):
            captured.update(params)

        def __call__(self, path: str):
            assert Path(path).name == "frame.jpg"
            return types.SimpleNamespace(
                boxes=[[[0, 0], [4, 0], [4, 4], [0, 4]]],
                txts=["鹿乃"],
                scores=[0.88],
            )

    fake_module = types.SimpleNamespace(
        EngineType=types.SimpleNamespace(ONNXRUNTIME=FakeEnumValue("onnxruntime")),
        LangDet=types.SimpleNamespace(CH=FakeEnumValue("ch")),
        LangRec=types.SimpleNamespace(CH=FakeEnumValue("ch")),
        ModelType=types.SimpleNamespace(
            MOBILE=FakeEnumValue("mobile"),
            SERVER=FakeEnumValue("server"),
        ),
        OCRVersion=types.SimpleNamespace(PPOCRV5=FakeEnumValue("PP-OCRv5")),
        RapidOCR=FakeRapidOCR,
    )
    monkeypatch.setitem(sys.modules, "rapidocr", fake_module)
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake")

    results = ocr.RapidOcrPpOcrV5Reader(model_type="mobile").read_image(image)

    assert captured["Det.model_type"].value == "mobile"
    assert captured["Rec.model_type"].value == "mobile"
    assert captured["Det.ocr_version"].value == "PP-OCRv5"
    assert captured["Rec.ocr_version"].value == "PP-OCRv5"
    assert results == [
        ocr.OcrResult(
            text="鹿乃",
            confidence=pytest.approx(0.88),
            bbox=[[0, 0], [4, 0], [4, 4], [0, 4]],
        )
    ]


def test_normalize_rapidocr_v3_numpy_boxes_to_json_safe_lists() -> None:
    raw = types.SimpleNamespace(
        boxes=np.array([[[0, 0], [4, 0], [4, 4], [0, 4]]]),
        txts=["鹿乃"],
        scores=np.array([0.88]),
    )

    results = ocr.normalize_ocr_items(raw)

    assert results == [
        ocr.OcrResult(
            text="鹿乃",
            confidence=pytest.approx(0.88),
            bbox=[[0, 0], [4, 0], [4, 4], [0, 4]],
        )
    ]


def test_get_ocr_reader_uses_cached_configured_engine(monkeypatch) -> None:
    monkeypatch.setattr(ocr, "_READERS", {})
    monkeypatch.setattr(
        ocr,
        "_make_reader",
        lambda engine: ocr.LegacyRapidOcrReader() if engine == ocr.LEGACY_RAPIDOCR else None,
    )

    first = ocr.get_ocr_reader(ocr.LEGACY_RAPIDOCR)
    second = ocr.get_ocr_reader(" legacy_rapidocr ")

    assert first is second


def test_get_ocr_reader_uses_settings_engine(monkeypatch) -> None:
    from kanomori.config import get_settings

    class FakeReader:
        def read_image(self, path):
            return []

    created: list[str] = []
    monkeypatch.setattr(ocr, "_READERS", {})
    monkeypatch.setattr(
        ocr,
        "_make_reader",
        lambda engine: created.append(engine) or FakeReader(),
    )
    monkeypatch.setenv("KANOMORI_OCR_ENGINE", ocr.RAPIDOCR_PPOCRV5_MOBILE)
    get_settings.cache_clear()

    try:
        reader = ocr.get_ocr_reader()
    finally:
        get_settings.cache_clear()

    assert isinstance(reader, FakeReader)
    assert created == [ocr.RAPIDOCR_PPOCRV5_MOBILE]
