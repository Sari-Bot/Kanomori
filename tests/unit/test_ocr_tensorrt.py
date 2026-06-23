from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from kanomori import ocr
from kanomori.ocr_tensorrt import (
    ensure_cuda_ep_available,
    ensure_tensorrt_ep_available,
    onnxruntime_cuda_provider_chain,
    onnxruntime_tensorrt_provider_chain,
    rapidocr_onnxruntime_ep_adapter,
    rapidocr_tensorrt_ep_adapter,
)


class FakeEnumValue:
    def __init__(self, value: str):
        self.value = value


def fake_rapidocr_module(rapidocr_cls):
    module = types.ModuleType("rapidocr")
    module.__path__ = []
    module.EngineType = types.SimpleNamespace(
        ONNXRUNTIME=FakeEnumValue("onnxruntime"),
        TENSORRT=FakeEnumValue("tensorrt"),
    )
    module.LangDet = types.SimpleNamespace(CH=FakeEnumValue("ch"))
    module.LangRec = types.SimpleNamespace(CH=FakeEnumValue("ch"))
    module.ModelType = types.SimpleNamespace(
        MOBILE=FakeEnumValue("mobile"),
        SERVER=FakeEnumValue("server"),
    )
    module.OCRVersion = types.SimpleNamespace(PPOCRV5=FakeEnumValue("PP-OCRv5"))
    module.RapidOCR = rapidocr_cls
    return module


def install_fake_onnxruntime(monkeypatch, providers: list[str]) -> None:
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        types.SimpleNamespace(get_available_providers=lambda: providers),
    )


def install_fake_rapidocr_provider(monkeypatch, provider_cls) -> None:
    rapidocr_parent = types.ModuleType("rapidocr.inference_engine")
    rapidocr_parent.__path__ = []
    onnxruntime_parent = types.ModuleType("rapidocr.inference_engine.onnxruntime")
    onnxruntime_parent.__path__ = []
    main = types.ModuleType("rapidocr.inference_engine.onnxruntime.main")
    provider_config = types.ModuleType(
        "rapidocr.inference_engine.onnxruntime.provider_config"
    )
    main.ProviderConfig = provider_cls
    provider_config.ProviderConfig = provider_cls

    monkeypatch.setitem(sys.modules, "rapidocr.inference_engine", rapidocr_parent)
    monkeypatch.setitem(
        sys.modules, "rapidocr.inference_engine.onnxruntime", onnxruntime_parent
    )
    monkeypatch.setitem(sys.modules, main.__name__, main)
    monkeypatch.setitem(sys.modules, provider_config.__name__, provider_config)


def test_rapidocr_v3_reader_maps_tensorrt_to_onnxruntime_ep_chain(
    monkeypatch, tmp_path
) -> None:
    captured = {}

    class BaseProviderConfig:
        def __init__(self, engine_cfg):
            self.engine_cfg = engine_cfg

        def get_ep_list(self):
            return [("CPUExecutionProvider", {})]

    class FakeRapidOCR:
        def __init__(self, params):
            captured.update(params)
            provider_cls = sys.modules[
                "rapidocr.inference_engine.onnxruntime.main"
            ].ProviderConfig
            captured["providers"] = provider_cls(types.SimpleNamespace()).get_ep_list()

        def __call__(self, path: str):
            assert Path(path).name == "frame.jpg"
            return types.SimpleNamespace(
                boxes=[[[0, 0], [4, 0], [4, 4], [0, 4]]],
                txts=["鹿乃"],
                scores=[0.88],
            )

    monkeypatch.setitem(sys.modules, "rapidocr", fake_rapidocr_module(FakeRapidOCR))
    install_fake_onnxruntime(
        monkeypatch,
        ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    install_fake_rapidocr_provider(monkeypatch, BaseProviderConfig)
    cache_dir = tmp_path / "trt-cache"
    monkeypatch.setenv("KANOMORI_OCR_TENSORRT_CACHE_DIR", str(cache_dir))
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake")

    results = ocr.RapidOcrPpOcrV5Reader(
        model_type="mobile",
        backend=ocr.OCR_BACKEND_TENSORRT,
    ).read_image(image)

    assert captured["Det.engine_type"].value == "onnxruntime"
    assert captured["Rec.engine_type"].value == "onnxruntime"
    assert captured["Det.model_type"].value == "mobile"
    assert captured["Rec.model_type"].value == "mobile"
    assert captured["Det.ocr_version"].value == "PP-OCRv5"
    assert captured["Rec.ocr_version"].value == "PP-OCRv5"
    assert [provider for provider, _options in captured["providers"]] == [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert captured["providers"][0][1]["device_id"] == 0
    assert captured["providers"][0][1]["trt_fp16_enable"] is True
    assert captured["providers"][0][1]["trt_engine_cache_enable"] is True
    assert captured["providers"][0][1]["trt_engine_cache_path"] == str(cache_dir)
    assert results == [
        ocr.OcrResult(
            text="鹿乃",
            confidence=pytest.approx(0.88),
            bbox=[[0, 0], [4, 0], [4, 4], [0, 4]],
        )
    ]


def test_rapidocr_v3_reader_maps_cuda_to_onnxruntime_ep_chain(
    monkeypatch, tmp_path
) -> None:
    captured = {}

    class BaseProviderConfig:
        def __init__(self, engine_cfg):
            self.engine_cfg = engine_cfg

        def get_ep_list(self):
            return [("CPUExecutionProvider", {})]

    class FakeRapidOCR:
        def __init__(self, params):
            captured.update(params)
            provider_cls = sys.modules[
                "rapidocr.inference_engine.onnxruntime.main"
            ].ProviderConfig
            captured["providers"] = provider_cls(types.SimpleNamespace()).get_ep_list()

    monkeypatch.setitem(sys.modules, "rapidocr", fake_rapidocr_module(FakeRapidOCR))
    install_fake_onnxruntime(
        monkeypatch,
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    install_fake_rapidocr_provider(monkeypatch, BaseProviderConfig)

    ocr.RapidOcrPpOcrV5Reader(
        model_type="server",
        backend=ocr.OCR_BACKEND_CUDA,
    )

    assert captured["Det.engine_type"].value == "onnxruntime"
    assert captured["Rec.engine_type"].value == "onnxruntime"
    assert captured["providers"] == [
        ("CUDAExecutionProvider", {"device_id": 0}),
        ("CPUExecutionProvider", {}),
    ]


def test_tensorrt_provider_chain_creates_configured_cache_dir(
    monkeypatch, tmp_path
) -> None:
    cache_dir = tmp_path / "trt-cache"
    monkeypatch.setenv("KANOMORI_OCR_TENSORRT_CACHE_DIR", str(cache_dir))

    providers = onnxruntime_tensorrt_provider_chain()

    assert cache_dir.is_dir()
    assert providers[0][1]["trt_engine_cache_path"] == str(cache_dir)
    assert providers[0][1]["trt_timing_cache_path"] == str(cache_dir)


def test_ensure_tensorrt_ep_available_preloads_onnxruntime_cuda(monkeypatch) -> None:
    calls = []
    fake_onnxruntime = types.SimpleNamespace(
        get_available_providers=lambda: ["TensorrtExecutionProvider"],
        preload_dlls=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)

    ensure_tensorrt_ep_available(ocr.OcrBackendUnavailable)

    assert calls == [{"directory": ""}]


def test_tensorrt_provider_chain_defaults_to_user_cache_dir(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("KANOMORI_OCR_TENSORRT_CACHE_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    providers = onnxruntime_tensorrt_provider_chain()

    cache_dir = tmp_path / ".cache" / "kanomori" / "onnxruntime-tensorrt"
    assert cache_dir.is_dir()
    assert providers[0][1]["trt_engine_cache_path"] == str(cache_dir)


def test_tensorrt_provider_chain_uses_rapidocr_model_profiles(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("KANOMORI_OCR_TENSORRT_CACHE_DIR", str(tmp_path / "trt-cache"))

    det_options = onnxruntime_tensorrt_provider_chain(
        "ch_PP-OCRv5_det_server.onnx"
    )[0][1]
    rec_options = onnxruntime_tensorrt_provider_chain(
        "ch_PP-OCRv5_rec_server.onnx"
    )[0][1]
    cls_options = onnxruntime_tensorrt_provider_chain(
        "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
    )[0][1]

    assert det_options["trt_profile_min_shapes"] == "x:1x3x32x32"
    assert det_options["trt_profile_opt_shapes"] == "x:1x3x736x736"
    assert det_options["trt_profile_max_shapes"] == "x:1x3x2048x2048"
    assert rec_options["trt_profile_min_shapes"] == "x:1x3x48x32"
    assert rec_options["trt_profile_opt_shapes"] == "x:6x3x48x320"
    assert rec_options["trt_profile_max_shapes"] == "x:6x3x48x2048"
    assert cls_options["trt_profile_min_shapes"] == "x:1x3x48x32"
    assert cls_options["trt_profile_opt_shapes"] == "x:6x3x48x192"
    assert cls_options["trt_profile_max_shapes"] == "x:6x3x48x192"


def test_cuda_provider_chain_uses_cuda_then_cpu() -> None:
    assert onnxruntime_cuda_provider_chain() == [
        ("CUDAExecutionProvider", {"device_id": 0}),
        ("CPUExecutionProvider", {}),
    ]


def test_tensorrt_adapter_overrides_session_providers_and_disables_fallback(
    monkeypatch, tmp_path
) -> None:
    captured = {}

    class BaseProviderConfig:
        def __init__(self, engine_cfg):
            self.engine_cfg = engine_cfg

        def get_ep_list(self):
            return [("CPUExecutionProvider", {})]

    class FakeInferenceSession:
        def __init__(self, model_path, **kwargs):
            captured["model_path"] = model_path
            captured["providers"] = kwargs["providers"]

        def disable_fallback(self):
            captured["fallback_disabled"] = True

    install_fake_rapidocr_provider(monkeypatch, BaseProviderConfig)
    sys.modules[
        "rapidocr.inference_engine.onnxruntime.main"
    ].InferenceSession = FakeInferenceSession
    monkeypatch.setenv("KANOMORI_OCR_TENSORRT_CACHE_DIR", str(tmp_path / "trt-cache"))

    with rapidocr_tensorrt_ep_adapter(ocr.OcrBackendUnavailable):
        sys.modules["rapidocr.inference_engine.onnxruntime.main"].InferenceSession(
            "ch_PP-OCRv5_rec_server.onnx",
            providers=[("CPUExecutionProvider", {})],
        )

    trt_options = captured["providers"][0][1]
    assert captured["model_path"] == "ch_PP-OCRv5_rec_server.onnx"
    assert captured["fallback_disabled"] is True
    assert trt_options["trt_profile_min_shapes"] == "x:1x3x48x32"


def test_cuda_adapter_overrides_session_providers_and_disables_fallback(
    monkeypatch,
) -> None:
    captured = {}

    class BaseProviderConfig:
        def __init__(self, engine_cfg):
            self.engine_cfg = engine_cfg

        def get_ep_list(self):
            return [("CPUExecutionProvider", {})]

    class FakeInferenceSession:
        def __init__(self, model_path, **kwargs):
            captured["model_path"] = model_path
            captured["providers"] = kwargs["providers"]

        def disable_fallback(self):
            captured["fallback_disabled"] = True

    install_fake_rapidocr_provider(monkeypatch, BaseProviderConfig)
    sys.modules["rapidocr.inference_engine.onnxruntime.main"].InferenceSession = (
        FakeInferenceSession
    )

    with rapidocr_onnxruntime_ep_adapter(
        "CUDA",
        ["CUDAExecutionProvider"],
        onnxruntime_cuda_provider_chain,
        ocr.OcrBackendUnavailable,
    ):
        sys.modules["rapidocr.inference_engine.onnxruntime.main"].InferenceSession(
            "ch_PP-OCRv5_rec_server.onnx",
            providers=[("CPUExecutionProvider", {})],
        )

    assert captured["model_path"] == "ch_PP-OCRv5_rec_server.onnx"
    assert captured["fallback_disabled"] is True
    assert captured["providers"] == [
        ("CUDAExecutionProvider", {"device_id": 0}),
        ("CPUExecutionProvider", {}),
    ]


def test_rapidocr_v3_reader_keeps_normal_onnxruntime_provider_config(monkeypatch) -> None:
    captured = {}

    class BaseProviderConfig:
        def __init__(self, engine_cfg):
            self.engine_cfg = engine_cfg

        def get_ep_list(self):
            return [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]

    class FakeRapidOCR:
        def __init__(self, params):
            provider_cls = sys.modules[
                "rapidocr.inference_engine.onnxruntime.main"
            ].ProviderConfig
            captured["providers"] = provider_cls(types.SimpleNamespace()).get_ep_list()
            captured.update(params)

    monkeypatch.setitem(sys.modules, "rapidocr", fake_rapidocr_module(FakeRapidOCR))
    install_fake_rapidocr_provider(monkeypatch, BaseProviderConfig)

    ocr.RapidOcrPpOcrV5Reader(
        model_type="server",
        backend=ocr.OCR_BACKEND_ONNXRUNTIME,
    )

    assert captured["Det.engine_type"].value == "onnxruntime"
    assert captured["Rec.engine_type"].value == "onnxruntime"
    assert captured["providers"] == [
        ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})
    ]


def test_rapidocr_v3_reader_rejects_tensorrt_when_ort_provider_missing(monkeypatch) -> None:
    class FakeRapidOCR:
        def __init__(self, params):
            raise AssertionError("RapidOCR should not be constructed")

    monkeypatch.setitem(sys.modules, "rapidocr", fake_rapidocr_module(FakeRapidOCR))
    install_fake_onnxruntime(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    with pytest.raises(ocr.OcrBackendUnavailable, match="TensorrtExecutionProvider"):
        ocr.RapidOcrPpOcrV5Reader(
            model_type="server",
            backend=ocr.OCR_BACKEND_TENSORRT,
        )


def test_rapidocr_v3_reader_rejects_cuda_when_ort_provider_missing(monkeypatch) -> None:
    class FakeRapidOCR:
        def __init__(self, params):
            raise AssertionError("RapidOCR should not be constructed")

    monkeypatch.setitem(sys.modules, "rapidocr", fake_rapidocr_module(FakeRapidOCR))
    install_fake_onnxruntime(monkeypatch, ["CPUExecutionProvider"])

    with pytest.raises(ocr.OcrBackendUnavailable, match="CUDAExecutionProvider"):
        ocr.RapidOcrPpOcrV5Reader(
            model_type="server",
            backend=ocr.OCR_BACKEND_CUDA,
        )


def test_ensure_cuda_ep_available_preloads_nvidia_wheels_before_importing_ort(
    monkeypatch,
) -> None:
    calls = []
    fake_onnxruntime = types.SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    def fake_import_module(name):
        assert name == "onnxruntime"
        assert calls == ["preload"]
        calls.append("import")
        return fake_onnxruntime

    monkeypatch.setattr(
        "kanomori.ocr_tensorrt._preload_nvidia_wheel_libraries",
        lambda: calls.append("preload"),
    )
    monkeypatch.setattr(
        "kanomori.ocr_tensorrt.importlib.import_module",
        fake_import_module,
    )

    ensure_cuda_ep_available(ocr.OcrBackendUnavailable)

    assert calls == ["preload", "import"]


def test_ensure_cuda_ep_available_rejects_onnxruntime_without_provider_api(
    monkeypatch,
) -> None:
    fake_onnxruntime = types.SimpleNamespace(__file__="/tmp/shadowed/onnxruntime.py")
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)

    with pytest.raises(ocr.OcrBackendUnavailable, match="get_available_providers"):
        ensure_cuda_ep_available(ocr.OcrBackendUnavailable)


def test_ensure_cuda_ep_available_does_not_preload_tensorrt(monkeypatch) -> None:
    fake_onnxruntime = types.SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        preload_dlls=lambda **kwargs: pytest.fail("CUDA backend must not preload DLLs"),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)

    ensure_cuda_ep_available(ocr.OcrBackendUnavailable)


def test_rapidocr_v3_reader_wraps_tensorrt_initialization_errors(monkeypatch) -> None:
    class FakeRapidOCR:
        def __init__(self, params):
            raise AttributeError("NetworkDefinitionCreationFlag has no attribute EXPLICIT_BATCH")

    monkeypatch.setitem(sys.modules, "rapidocr", fake_rapidocr_module(FakeRapidOCR))
    install_fake_onnxruntime(
        monkeypatch,
        ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    with pytest.raises(ocr.OcrBackendUnavailable, match="TensorRT initialization failed"):
        ocr.RapidOcrPpOcrV5Reader(model_type="server", backend=ocr.OCR_BACKEND_TENSORRT)
