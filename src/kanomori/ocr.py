"""Shared OCR provider boundary for ingestion and screenshot query processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

LEGACY_RAPIDOCR = "legacy_rapidocr"
RAPIDOCR_PPOCRV5_MOBILE = "rapidocr_ppocrv5_mobile"
RAPIDOCR_PPOCRV5_SERVER = "rapidocr_ppocrv5_server"

OCR_ENGINES = {
    LEGACY_RAPIDOCR,
    RAPIDOCR_PPOCRV5_MOBILE,
    RAPIDOCR_PPOCRV5_SERVER,
}


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float | None = None
    bbox: dict | list | None = None


class OcrReader(Protocol):
    def read_image(self, path: Path) -> list[OcrResult]:
        """Extract OCR lines from an image path."""


class LegacyRapidOcrReader:
    """Compatibility wrapper around the current rapidocr-onnxruntime package."""

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR

        self._reader = RapidOCR()

    def read_image(self, path: Path) -> list[OcrResult]:
        raw, _elapsed = self._reader(str(path))
        return normalize_ocr_items(raw)


class RapidOcrPpOcrV5Reader:
    """RapidOCR v3 provider configured for PP-OCRv5 CJK recognition."""

    def __init__(self, *, model_type: str):
        try:
            from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "rapidocr PP-OCRv5 providers require `uv sync --group ocr-eval`"
            ) from exc

        model = ModelType.SERVER if model_type == "server" else ModelType.MOBILE
        params = {
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": model,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": LangRec.CH,
            "Rec.model_type": model,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
        }
        self._reader = RapidOCR(params=params)

    def read_image(self, path: Path) -> list[OcrResult]:
        return normalize_ocr_items(self._reader(str(path)))


_READERS: dict[str, OcrReader] = {}


def normalize_ocr_engine(engine: str | None) -> str:
    resolved = (engine or LEGACY_RAPIDOCR).strip().lower()
    if resolved not in OCR_ENGINES:
        valid = ", ".join(sorted(OCR_ENGINES))
        raise ValueError(f"Unsupported OCR engine {engine!r}; expected one of: {valid}")
    return resolved


def get_ocr_reader(engine: str | None = None) -> OcrReader:
    from kanomori.config import get_settings

    resolved = normalize_ocr_engine(engine or get_settings().ocr_engine)
    if resolved not in _READERS:
        _READERS[resolved] = _make_reader(resolved)
    return _READERS[resolved]


def read_image_ocr(path: Path, *, engine: str | None = None) -> list[OcrResult]:
    return get_ocr_reader(engine).read_image(path)


def _make_reader(engine: str) -> OcrReader:
    if engine == LEGACY_RAPIDOCR:
        return LegacyRapidOcrReader()
    if engine == RAPIDOCR_PPOCRV5_MOBILE:
        return RapidOcrPpOcrV5Reader(model_type="mobile")
    if engine == RAPIDOCR_PPOCRV5_SERVER:
        return RapidOcrPpOcrV5Reader(model_type="server")
    raise ValueError(f"Unsupported OCR engine {engine!r}")


def normalize_ocr_items(raw: Any) -> list[OcrResult]:
    if raw is None:
        return []
    if hasattr(raw, "boxes") and hasattr(raw, "txts"):
        return _normalize_box_text_score(raw.boxes, raw.txts, getattr(raw, "scores", []))
    if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], list):
        raw = raw[0]
    return [result for item in raw or [] if (result := _coerce_ocr_item(item)) is not None]


def _normalize_box_text_score(boxes: Any, texts: Any, scores: Any) -> list[OcrResult]:
    out: list[OcrResult] = []
    for i, text in enumerate(texts or []):
        if not text:
            continue
        out.append(
            OcrResult(
                text=str(text),
                confidence=_float_or_none(_item_at(scores, i)),
                bbox=_json_safe_bbox(_item_at(boxes, i)),
            )
        )
    return out


def _coerce_ocr_item(item: Any) -> OcrResult | None:
    if isinstance(item, OcrResult):
        return item if item.text else None
    if isinstance(item, dict):
        text = item.get("text") or item.get("txt") or item.get("rec_text")
        if not text:
            return None
        score = item.get("score", item.get("confidence"))
        bbox = item.get("bbox", item.get("box"))
        return OcrResult(
            text=str(text),
            confidence=_float_or_none(score),
            bbox=_json_safe_bbox(bbox),
        )
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        bbox, text, score = item[0], item[1], item[2]
        if not text:
            return None
        return OcrResult(
            text=str(text),
            confidence=_float_or_none(score),
            bbox=_json_safe_bbox(bbox),
        )
    return None


def _item_at(values: Any, index: int) -> Any:
    if values is None:
        return None
    try:
        return values[index]
    except (IndexError, TypeError):
        return None


def _json_safe_bbox(value: Any) -> dict | list | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
