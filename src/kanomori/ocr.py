"""Shared OCR provider boundary for ingestion and screenshot query processing."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

LEGACY_RAPIDOCR = "legacy_rapidocr"
RAPIDOCR_PPOCRV5_MOBILE = "rapidocr_ppocrv5_mobile"
RAPIDOCR_PPOCRV5_SERVER = "rapidocr_ppocrv5_server"

PPOCRV5_MOBILE = "ppocrv5_mobile"
PPOCRV5_SERVER = "ppocrv5_server"
OCR_BACKEND_ONNXRUNTIME = "onnxruntime"
OCR_BACKEND_TENSORRT = "tensorrt"

OCR_ENGINES = {
    LEGACY_RAPIDOCR,
    RAPIDOCR_PPOCRV5_MOBILE,
    RAPIDOCR_PPOCRV5_SERVER,
}
OCR_MODELS = {
    LEGACY_RAPIDOCR,
    PPOCRV5_MOBILE,
    PPOCRV5_SERVER,
}
OCR_BACKENDS = {
    OCR_BACKEND_ONNXRUNTIME,
    OCR_BACKEND_TENSORRT,
}
OCR_SCOPES = {"ingest", "query"}

_LEGACY_ENGINE_CONFIGS: dict[str, OcrConfig]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float | None = None
    bbox: dict | list | None = None


@dataclass(frozen=True)
class OcrConfig:
    model: str
    backend: str


class OcrReader(Protocol):
    def read_image(self, path: Path) -> list[OcrResult]:
        """Extract OCR lines from an image path."""


class OcrBackendUnavailable(RuntimeError):
    """Raised when an OCR backend cannot be initialized in this environment."""


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

    def __init__(self, *, model_type: str, backend: str):
        try:
            from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "rapidocr PP-OCRv5 providers require `uv sync --group ocr-eval`"
            ) from exc

        model = ModelType.SERVER if model_type == "server" else ModelType.MOBILE
        engine = (
            EngineType.TENSORRT
            if backend == OCR_BACKEND_TENSORRT
            else EngineType.ONNXRUNTIME
        )
        params = {
            "Det.engine_type": engine,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": model,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Rec.engine_type": engine,
            "Rec.lang_type": LangRec.CH,
            "Rec.model_type": model,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
        }
        self._reader = RapidOCR(params=params)

    def read_image(self, path: Path) -> list[OcrResult]:
        return normalize_ocr_items(self._reader(str(path)))


_LEGACY_ENGINE_CONFIGS = {
    LEGACY_RAPIDOCR: OcrConfig(LEGACY_RAPIDOCR, OCR_BACKEND_ONNXRUNTIME),
    RAPIDOCR_PPOCRV5_MOBILE: OcrConfig(PPOCRV5_MOBILE, OCR_BACKEND_ONNXRUNTIME),
    RAPIDOCR_PPOCRV5_SERVER: OcrConfig(PPOCRV5_SERVER, OCR_BACKEND_ONNXRUNTIME),
}
_READERS: dict[tuple[OcrConfig, bool], OcrReader] = {}


def normalize_ocr_engine(engine: str | None) -> str:
    resolved = (engine or LEGACY_RAPIDOCR).strip().lower()
    if resolved not in OCR_ENGINES:
        valid = ", ".join(sorted(OCR_ENGINES))
        raise ValueError(f"Unsupported OCR engine {engine!r}; expected one of: {valid}")
    return resolved


def normalize_ocr_model(model: str | None) -> str:
    resolved = (model or PPOCRV5_SERVER).strip().lower()
    if resolved in _LEGACY_ENGINE_CONFIGS:
        return _LEGACY_ENGINE_CONFIGS[resolved].model
    if resolved not in OCR_MODELS:
        valid = ", ".join(sorted(OCR_MODELS))
        raise ValueError(f"Unsupported OCR model {model!r}; expected one of: {valid}")
    return resolved


def normalize_ocr_backend(backend: str | None) -> str:
    resolved = (backend or OCR_BACKEND_ONNXRUNTIME).strip().lower()
    if resolved not in OCR_BACKENDS:
        valid = ", ".join(sorted(OCR_BACKENDS))
        raise ValueError(f"Unsupported OCR backend {backend!r}; expected one of: {valid}")
    return resolved


def parse_ocr_engine_alias(engine: str | None) -> OcrConfig:
    return _LEGACY_ENGINE_CONFIGS[normalize_ocr_engine(engine)]


def resolve_ocr_config(
    model: str | None = None,
    backend: str | None = None,
    *,
    scope: str = "ingest",
    settings: Any | None = None,
) -> OcrConfig:
    if model is None and backend is None:
        return _settings_ocr_config(scope=scope, settings=settings)
    if model is not None and backend is None:
        maybe_engine = model.strip().lower()
        if maybe_engine in _LEGACY_ENGINE_CONFIGS:
            return validate_ocr_config(_LEGACY_ENGINE_CONFIGS[maybe_engine])

    base = _settings_ocr_config(scope=scope, settings=settings)
    config = OcrConfig(
        model=normalize_ocr_model(model) if model is not None else base.model,
        backend=normalize_ocr_backend(backend) if backend is not None else base.backend,
    )
    return validate_ocr_config(config)


def validate_ocr_config(config: OcrConfig) -> OcrConfig:
    config = OcrConfig(
        model=normalize_ocr_model(config.model),
        backend=normalize_ocr_backend(config.backend),
    )
    if config.model == LEGACY_RAPIDOCR and config.backend != OCR_BACKEND_ONNXRUNTIME:
        raise ValueError("legacy_rapidocr supports only the onnxruntime OCR backend")
    return config


def get_ocr_reader(
    model: str | None = None,
    backend: str | None = None,
    *,
    scope: str = "ingest",
    allow_backend_fallback: bool = True,
) -> OcrReader:
    config = resolve_ocr_config(model, backend, scope=scope)
    cache_key = (config, allow_backend_fallback)
    if cache_key not in _READERS:
        _READERS[cache_key] = _make_reader_with_fallback(
            config,
            allow_backend_fallback=allow_backend_fallback,
        )
    return _READERS[cache_key]


def read_image_ocr(
    path: Path,
    *,
    model: str | None = None,
    backend: str | None = None,
    scope: str = "ingest",
    allow_backend_fallback: bool = True,
) -> list[OcrResult]:
    return get_ocr_reader(
        model,
        backend,
        scope=scope,
        allow_backend_fallback=allow_backend_fallback,
    ).read_image(path)


def _settings_ocr_config(*, scope: str, settings: Any | None = None) -> OcrConfig:
    from kanomori.config import get_settings

    if scope not in OCR_SCOPES:
        valid = ", ".join(sorted(OCR_SCOPES))
        raise ValueError(f"Unsupported OCR scope {scope!r}; expected one of: {valid}")

    settings = settings or get_settings()
    model_field = f"{scope}_ocr_model"
    backend_field = f"{scope}_ocr_backend"
    fields_set = getattr(settings, "model_fields_set", set())
    has_scope_override = model_field in fields_set or backend_field in fields_set
    if getattr(settings, "ocr_engine", None) and not has_scope_override:
        return parse_ocr_engine_alias(settings.ocr_engine)

    return validate_ocr_config(
        OcrConfig(
            model=getattr(settings, model_field),
            backend=getattr(settings, backend_field),
        )
    )


def _make_reader_with_fallback(
    config: OcrConfig,
    *,
    allow_backend_fallback: bool,
) -> OcrReader:
    try:
        return _make_reader(config)
    except OcrBackendUnavailable:
        if config.backend != OCR_BACKEND_TENSORRT or not allow_backend_fallback:
            raise
        fallback = OcrConfig(config.model, OCR_BACKEND_ONNXRUNTIME)
        logger.warning(
            "OCR backend %s unavailable for model %s; falling back to %s",
            config.backend,
            config.model,
            fallback.backend,
        )
        return _make_reader(fallback)


def _make_reader(config: OcrConfig) -> OcrReader:
    config = validate_ocr_config(config)
    if config.model == LEGACY_RAPIDOCR:
        return LegacyRapidOcrReader()
    if config.backend == OCR_BACKEND_TENSORRT:
        _ensure_tensorrt_available()
    if config.model == PPOCRV5_MOBILE:
        return RapidOcrPpOcrV5Reader(model_type="mobile", backend=config.backend)
    if config.model == PPOCRV5_SERVER:
        return RapidOcrPpOcrV5Reader(model_type="server", backend=config.backend)
    raise ValueError(f"Unsupported OCR model {config.model!r}")


def _ensure_tensorrt_available() -> None:
    missing: list[str] = []
    for module_name in ("tensorrt", "cuda.bindings.runtime"):
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        modules = ", ".join(missing)
        raise OcrBackendUnavailable(f"TensorRT OCR backend unavailable; missing {modules}")


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
