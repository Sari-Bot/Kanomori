from __future__ import annotations

import sys
import types

import numpy as np

from kanomori.embed.image_embedder import DINOv2Embedder, SigLIPClassifier


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)
        self.devices: list[str] = []

    def to(self, device):
        self.devices.append(str(device))
        return self

    def __getitem__(self, key):
        return FakeTensor(self.value[key])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def softmax(self, dim):
        shifted = self.value - np.max(self.value, axis=dim, keepdims=True)
        exp = np.exp(shifted)
        return FakeTensor(exp / exp.sum(axis=dim, keepdims=True))


class FakeModel:
    def __init__(self, outputs):
        self.outputs = outputs
        self.to_calls: list[str] = []
        self.eval_called = False

    def to(self, device):
        self.to_calls.append(str(device))
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, **inputs):
        self.inputs = inputs
        return self.outputs


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _install_fake_torch(monkeypatch, *, cuda_available: bool = True) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            no_grad=lambda: _NoGrad(),
            device=lambda name: name,
            cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
        ),
    )


def test_dinov2_embedder_moves_model_and_inputs_to_device(monkeypatch) -> None:
    pixel_values = FakeTensor([[[1.0, 2.0, 3.0]]])
    model = FakeModel(types.SimpleNamespace(last_hidden_state=FakeTensor([[[3.0, 4.0, 0.0]]])))

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, _name):
            return cls()

        def __call__(self, *, images, return_tensors):
            assert return_tensors == "pt"
            assert images is not None
            return {"pixel_values": pixel_values}

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, _name):
            return model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoImageProcessor=FakeProcessor, AutoModel=FakeAutoModel),
    )
    _install_fake_torch(monkeypatch)

    embedder = DINOv2Embedder(model_name="fake-dino", device="cpu")
    vec = embedder.embed_image(object())

    assert model.to_calls == ["cpu"]
    assert model.eval_called is True
    assert pixel_values.devices == ["cpu"]
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_siglip_classifier_moves_model_and_inputs_to_device(monkeypatch) -> None:
    pixel_values = FakeTensor([[[1.0, 2.0, 3.0]]])
    input_ids = FakeTensor([[1.0, 2.0]])
    model = FakeModel(types.SimpleNamespace(logits_per_image=FakeTensor([[1.0, 3.0]])))

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, _name):
            return cls()

        def __call__(self, *, text, images, return_tensors, padding):
            assert text == ["chatting prompt", "singing prompt"]
            assert return_tensors == "pt"
            assert padding is True
            assert images is not None
            return {"pixel_values": pixel_values, "input_ids": input_ids}

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, _name):
            return model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoProcessor=FakeProcessor, AutoModel=FakeAutoModel),
    )
    _install_fake_torch(monkeypatch)

    classifier = SigLIPClassifier(
        model_name="fake-siglip",
        device="cpu",
        labels={
            "chatting": ["chatting prompt"],
            "singing": ["singing prompt"],
        },
    )
    scores = classifier.classify_image(object())

    assert model.to_calls == ["cpu"]
    assert model.eval_called is True
    assert pixel_values.devices == ["cpu"]
    assert input_ids.devices == ["cpu"]
    assert set(scores) == {"chatting", "singing"}
    assert np.isclose(sum(scores.values()), 1.0)


def test_image_embed_stage_builds_device_scoped_embedder(monkeypatch) -> None:
    from kanomori.config import get_settings
    from kanomori.embed import image_embedder
    from kanomori.ingest.stages import image_embed

    created: list[str | None] = []

    class FakeEmbedder:
        def __init__(self, model_name=None, device=None):
            created.append(device)

    image_embed._EMBEDDER = None
    monkeypatch.setattr(image_embedder, "DINOv2Embedder", FakeEmbedder)
    monkeypatch.setenv("KANOMORI_STAGE_IMAGE_EMBED_DEVICE", "gpu")
    get_settings.cache_clear()

    try:
        image_embed._embedder()
    finally:
        image_embed._EMBEDDER = None
        get_settings.cache_clear()

    assert created == ["gpu"]


def test_classify_stage_builds_device_scoped_classifier(monkeypatch) -> None:
    from kanomori.config import get_settings
    from kanomori.embed import image_embedder
    from kanomori.ingest.stages import classify

    created: list[str | None] = []

    class FakeClassifier:
        def __init__(self, model_name=None, labels=None, device=None):
            created.append(device)

    classify._CLASSIFIER = None
    monkeypatch.setattr(image_embedder, "SigLIPClassifier", FakeClassifier)
    monkeypatch.setenv("KANOMORI_STAGE_CLASSIFY_DEVICE", "gpu")
    get_settings.cache_clear()

    try:
        classify._classifier()
    finally:
        classify._CLASSIFIER = None
        get_settings.cache_clear()

    assert created == ["gpu"]
