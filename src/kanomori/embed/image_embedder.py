"""Lazy image models for visual ingestion and screenshot query processing."""

from __future__ import annotations

import numpy as np

from kanomori.config import get_settings

IMAGE_EMBED_DIM = 768


class DINOv2Embedder:
    """Lazily-loaded DINOv2 ViT-B/14 embedder returning normalized 768-d vectors."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or get_settings().image_model
        self._processor = None
        self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoImageProcessor, AutoModel

            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
        return self._processor, self._model

    def embed_image(self, image) -> np.ndarray:
        import torch

        processor, model = self._load()
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        vec = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()[0].astype(np.float32)
        vec /= np.linalg.norm(vec) or 1.0
        return vec

    def embed_image_path(self, path: str) -> np.ndarray:
        from PIL import Image

        with Image.open(path) as image:
            return self.embed_image(image.convert("RGB"))

    def embed_image_bytes(self, data: bytes) -> np.ndarray:
        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(data)) as image:
            return self.embed_image(image.convert("RGB"))


class SigLIPClassifier:
    """Zero-shot scene classifier using SigLIP image/text similarities."""

    def __init__(self, model_name: str | None = None, labels: dict[str, list[str]] | None = None):
        self.model_name = model_name or get_settings().scene_model
        self.labels = labels or {}
        self._processor = None
        self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModel, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
        return self._processor, self._model

    def classify_image(self, image) -> dict[str, float]:
        import torch

        processor, model = self._load()
        label_names = list(self.labels)
        prompts = [self.labels[label][0] for label in label_names]
        inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
        return {label: float(score) for label, score in zip(label_names, scores, strict=True)}
