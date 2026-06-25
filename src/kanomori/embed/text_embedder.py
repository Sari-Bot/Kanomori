"""BGE-M3 text embedder (dense), used for both offline ingestion and online queries.

BGE-M3 is multilingual and long-context — a good fit for Japanese content queried in JP/ZH/EN.
We use its **dense** 1024-d output; the dimension is pinned to the SQL schema
(``transcript_segments.embedding vector(1024)``). The model and its torch backing are imported
lazily inside ``_load`` so importing this module costs nothing until an embedder is actually
constructed and used — keeping the value out of base imports and the CPU query path's cold
start cheap (the model is loaded once and reused).
"""

from __future__ import annotations

import numpy as np

from kanomori.config import get_settings
from kanomori.ingest.stage_device import torch_device_for

# BGE-M3 dense dimension. Authoritative copy lives in migrations/0001_init.sql (vector(1024)).
EMBED_DIM = 1024


class BGEEmbedder:
    """Lazily-loaded BGE-M3 dense embedder. Construct once, reuse for many embed calls."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or get_settings().text_model
        self.device = device
        self._model = None

    def _resolved_device(self) -> str | None:
        if self.device is None:
            return None
        return str(torch_device_for(self.device, stage_name="parse_transcript"))

    def _load(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            kwargs = {"use_fp16": False}
            if (device := self._resolved_device()) is not None:
                kwargs["devices"] = device
            self._model = BGEM3FlagModel(self.model_name, **kwargs)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of texts into dense 1024-d vectors (float32)."""
        if not texts:
            return []
        model = self._load()
        out = model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        return [row for row in dense]

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string into a dense 1024-d vector (float32)."""
        return self.embed_texts([text])[0]
