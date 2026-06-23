"""ONNX Runtime GPU provider adapters for RapidOCR."""

from __future__ import annotations

import ctypes
import importlib
import os
import site
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

TENSORRT_EP = "TensorrtExecutionProvider"
CUDA_EP = "CUDAExecutionProvider"
CPU_EP = "CPUExecutionProvider"
TENSORRT_CACHE_SUBDIR = Path("kanomori") / "onnxruntime-tensorrt"
_DET_PROFILE = ("x:1x3x32x32", "x:1x3x736x736", "x:1x3x2048x2048")
_REC_PROFILE = ("x:1x3x48x32", "x:6x3x48x320", "x:6x3x48x2048")
_CLS_PROFILE = ("x:1x3x48x32", "x:6x3x48x192", "x:6x3x48x192")


def ensure_cuda_ep_available(unavailable_error: type[Exception]) -> None:
    _preload_nvidia_wheel_libraries()
    try:
        onnxruntime = importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise unavailable_error(
            "CUDA OCR backend unavailable; onnxruntime is not installed"
        ) from exc

    _ensure_provider_available(
        onnxruntime,
        CUDA_EP,
        "CUDA",
        unavailable_error,
    )


def ensure_tensorrt_ep_available(unavailable_error: type[Exception]) -> None:
    _preload_nvidia_wheel_libraries()
    try:
        onnxruntime = importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise unavailable_error(
            "TensorRT OCR backend unavailable; onnxruntime is not installed"
        ) from exc

    _preload_onnxruntime_gpu_libraries(onnxruntime, unavailable_error)
    _preload_tensorrt_libraries(unavailable_error)
    _ensure_provider_available(
        onnxruntime,
        TENSORRT_EP,
        "TensorRT",
        unavailable_error,
    )


def _ensure_provider_available(
    onnxruntime: Any,
    provider: str,
    backend_label: str,
    unavailable_error: type[Exception],
) -> None:
    get_available_providers = getattr(onnxruntime, "get_available_providers", None)
    if not callable(get_available_providers):
        module_path = getattr(onnxruntime, "__file__", "<unknown>")
        raise unavailable_error(
            f"{backend_label} OCR backend unavailable; imported onnxruntime module "
            f"at {module_path} does not expose get_available_providers()"
        )

    providers = list(get_available_providers())
    if provider not in providers:
        raise unavailable_error(
            f"{backend_label} OCR backend unavailable; {provider} not in "
            f"ONNX Runtime providers: {providers}"
        )


def _preload_nvidia_wheel_libraries() -> None:
    for lib_path in _nvidia_wheel_library_paths():
        try:
            ctypes.CDLL(str(lib_path), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        except OSError:
            continue


def _nvidia_wheel_library_paths() -> list[Path]:
    return sorted(
        {
            lib_path.resolve()
            for lib_dir in _nvidia_wheel_library_dirs()
            for lib_path in lib_dir.glob("*.so*")
            if lib_path.is_file()
        }
    )


def _nvidia_wheel_library_dirs() -> list[Path]:
    roots = [Path(path) for path in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(Path(user_site))
    return [
        lib_dir
        for root in roots
        for lib_dir in sorted((root / "nvidia").glob("*/lib"))
        if lib_dir.is_dir()
    ]


@contextmanager
def rapidocr_onnxruntime_ep_adapter(
    backend_label: str,
    required_providers: Sequence[str],
    provider_chain_factory: Callable[
        [str | Path | None],
        list[tuple[str, dict[str, Any]]],
    ],
    unavailable_error: type[Exception],
) -> Iterator[None]:
    main = importlib.import_module("rapidocr.inference_engine.onnxruntime.main")
    provider_config = importlib.import_module(
        "rapidocr.inference_engine.onnxruntime.provider_config"
    )
    original_main_provider = main.ProviderConfig
    original_provider_config = provider_config.ProviderConfig
    original_inference_session = getattr(main, "InferenceSession", None)

    class GpuProviderConfig(original_main_provider):
        def get_ep_list(self):
            return provider_chain_factory(None)

        def verify_providers(self, session_providers: Sequence[str]) -> None:
            missing = [
                provider for provider in required_providers if provider not in session_providers
            ]
            if missing:
                raise unavailable_error(
                    f"{backend_label} OCR backend unavailable; session providers are "
                    f"{list(session_providers)}"
                )

    def gpu_inference_session(*args, **kwargs):
        model_path = args[0] if args else kwargs.get("path_or_bytes")
        kwargs["providers"] = provider_chain_factory(model_path)
        session = original_inference_session(*args, **kwargs)
        disable_fallback = getattr(session, "disable_fallback", None)
        if callable(disable_fallback):
            disable_fallback()
        return session

    main.ProviderConfig = GpuProviderConfig
    provider_config.ProviderConfig = GpuProviderConfig
    if original_inference_session is not None:
        main.InferenceSession = gpu_inference_session
    try:
        yield
    finally:
        main.ProviderConfig = original_main_provider
        provider_config.ProviderConfig = original_provider_config
        if original_inference_session is not None:
            main.InferenceSession = original_inference_session


@contextmanager
def rapidocr_tensorrt_ep_adapter(
    unavailable_error: type[Exception],
) -> Iterator[None]:
    with rapidocr_onnxruntime_ep_adapter(
        "TensorRT",
        [TENSORRT_EP],
        onnxruntime_tensorrt_provider_chain,
        unavailable_error,
    ):
        yield


def onnxruntime_cuda_provider_chain(
    model_path: str | Path | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    return [
        (CUDA_EP, {"device_id": 0}),
        (CPU_EP, {}),
    ]


def onnxruntime_tensorrt_provider_chain(
    model_path: str | Path | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    cache_dir = _tensorrt_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = str(cache_dir)
    trt_options: dict[str, Any] = {
        "device_id": 0,
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache_path,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": cache_path,
    }
    if profile := _profile_for_model(model_path):
        trt_options.update(
            {
                "trt_profile_min_shapes": profile[0],
                "trt_profile_opt_shapes": profile[1],
                "trt_profile_max_shapes": profile[2],
            }
        )
    return [
        (TENSORRT_EP, trt_options),
        (CUDA_EP, {"device_id": 0}),
        (CPU_EP, {}),
    ]


def _tensorrt_cache_dir() -> Path:
    default_cache_dir = Path.home() / ".cache" / TENSORRT_CACHE_SUBDIR
    return Path(os.environ.get("KANOMORI_OCR_TENSORRT_CACHE_DIR", str(default_cache_dir)))


def _profile_for_model(model_path: str | Path | None) -> tuple[str, str, str] | None:
    if model_path is None:
        return None
    name = Path(str(model_path)).name.lower()
    if "_det_" in name:
        return _DET_PROFILE
    if "_rec_" in name:
        return _REC_PROFILE
    if "_cls_" in name:
        return _CLS_PROFILE
    return None


def _preload_onnxruntime_gpu_libraries(
    onnxruntime: Any,
    unavailable_error: type[Exception],
) -> None:
    preload_dlls = getattr(onnxruntime, "preload_dlls", None)
    if not callable(preload_dlls):
        return
    try:
        preload_dlls(directory="")
    except Exception as exc:
        raise unavailable_error(
            f"TensorRT OCR backend unavailable; failed to preload ONNX Runtime GPU "
            f"libraries: {exc}"
        ) from exc


def _preload_tensorrt_libraries(unavailable_error: type[Exception]) -> None:
    for lib_dir in _tensorrt_library_dirs():
        for lib_name in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
            lib_path = lib_dir / lib_name
            if not lib_path.exists():
                continue
            try:
                ctypes.CDLL(str(lib_path), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            except OSError as exc:
                raise unavailable_error(
                    f"TensorRT OCR backend unavailable; failed to load {lib_path}: {exc}"
                ) from exc


def _tensorrt_library_dirs() -> list[Path]:
    roots = [Path(path) for path in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(Path(user_site))
    return [root / "tensorrt_libs" for root in roots if (root / "tensorrt_libs").is_dir()]
