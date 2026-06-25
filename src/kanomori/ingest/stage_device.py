"""Resolve worker-configured CPU/GPU stage affinity for offline ingestion stages."""

from __future__ import annotations

from typing import Literal

from kanomori.config import Settings, get_settings

StageName = Literal["parse_transcript", "ocr", "classify", "image_embed"]
StageDevice = Literal["cpu", "gpu"]

_STAGE_DEVICE_FIELDS: dict[StageName, str] = {
    "parse_transcript": "stage_parse_transcript_device",
    "ocr": "stage_ocr_device",
    "classify": "stage_classify_device",
    "image_embed": "stage_image_embed_device",
}


def device_for_stage(stage: StageName, settings: Settings | None = None) -> StageDevice:
    settings = settings or get_settings()
    try:
        field_name = _STAGE_DEVICE_FIELDS[stage]
    except KeyError as exc:
        valid = ", ".join(sorted(_STAGE_DEVICE_FIELDS))
        raise ValueError(f"Unsupported stage {stage!r}; expected one of: {valid}") from exc
    return getattr(settings, field_name)


def torch_device_for(device: StageDevice, *, stage_name: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(f"Stage {stage_name} requires torch to resolve its device") from exc

    if device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Stage {stage_name} requires GPU execution but CUDA is unavailable"
            )
        return torch.device("cuda")
    return torch.device("cpu")


def torch_device_for_stage(stage: StageName, settings: Settings | None = None):
    return torch_device_for(device_for_stage(stage, settings=settings), stage_name=stage)
