"""Wire contract for ingestion stage results + a compact float32 vector codec.

The ingestion pipeline is split into a GPU-tolerant **compute** side (a worker) and a CPU
**persist** side (the coordinator). Stage outputs cross that boundary as JSON, so this module
defines the shape of what travels: one ``StageResult`` per stage that writes DB rows, plus a
manifest of binary artifacts (frame JPEGs, the SRT) that ride alongside as multipart files.

Two deliberate properties:

* **Natural keys only.** Rows carry ``content_hash`` / ``ts_sec`` / ``seq`` — never database
  ids. The coordinator resolves ``video_id`` (by content_hash) and ``frame_id`` (by ts_sec) at
  persist time. A test guards this invariant.
* **Embeddings as base64 float32.** A 2 h stream yields thousands of 1024-d transcript rows and
  768-d frame rows; JSON arrays of floats are bulky and lossy-formatted. ``encode_vector`` /
  ``decode_vector`` pack them little-endian float32 and base64-encode — compact and exact to
  float32 precision. The model field is the base64 ``str``; ``build(...)`` accepts numpy/lists
  and ``to_vectors()`` hands numpy back, so compute and persist sides never touch base64 by hand.

The full-text ``tsv`` columns are intentionally absent: they are computed coordinator-side with
``to_tsvector`` from the carried ``text`` / ``text_norm``, mirroring the stages' INSERTs.
"""

from __future__ import annotations

import base64
from typing import Literal

import numpy as np
from pydantic import BaseModel

# --- vector codec -----------------------------------------------------------------------


def encode_vector(vec) -> str:
    """Pack a list[float] or numpy array as little-endian float32 and base64-encode to ascii."""
    arr = np.asarray(vec, dtype="<f4")
    return base64.b64encode(arr.tobytes()).decode("ascii")


def decode_vector(s: str) -> list[float]:
    """Inverse of :func:`encode_vector`. Returns a plain ``list[float]`` (JSON-friendly).

    Floats are native Python ``float`` (64-bit) but only carry float32 worth of precision,
    since that is all the encoded form preserved.
    """
    raw = base64.b64decode(s.encode("ascii"))
    return np.frombuffer(raw, dtype="<f4").astype(float).tolist()


# --- artifact manifest ------------------------------------------------------------------

ArtifactKind = Literal["frame", "srt"]


class ArtifactRef(BaseModel):
    """One binary artifact accompanying a stage result as a multipart part.

    ``name`` is the relative filename the worker attaches under (e.g. the deterministic
    ``frame_000008_000.jpg`` or ``transcript.srt``); the bytes themselves never live in JSON.
    """

    name: str
    kind: ArtifactKind


# --- row models -------------------------------------------------------------------------


class TranscriptSegmentRow(BaseModel):
    """A transcript_segments row (embedding as base64; tsv derived coordinator-side)."""

    seq: int
    start_sec: float
    end_sec: float
    text: str
    text_norm: str
    embedding: str  # base64 float32, via the codec

    @classmethod
    def build(
        cls,
        *,
        seq: int,
        start_sec: float,
        end_sec: float,
        text: str,
        text_norm: str,
        embedding,
    ) -> TranscriptSegmentRow:
        """Construct from a numpy array / list embedding (encoded to base64 here)."""
        return cls(
            seq=seq,
            start_sec=start_sec,
            end_sec=end_sec,
            text=text,
            text_norm=text_norm,
            embedding=encode_vector(embedding),
        )

    def vector(self) -> np.ndarray:
        """Decode this row's embedding back to a float32 numpy array."""
        return np.asarray(decode_vector(self.embedding), dtype="<f4")


class FrameRow(BaseModel):
    """A frames row, keyed by ``ts_sec``. ``artifact`` names the JPEG riding as a multipart part."""

    ts_sec: float
    artifact: str


class OcrRow(BaseModel):
    """An ocr_segments row, keyed to its frame by ``ts_sec`` (tsv derived coordinator-side)."""

    ts_sec: float
    text: str
    confidence: float | None = None
    bbox: dict | list | None = None


class SceneSegmentRow(BaseModel):
    """A scene_segments row (span + label)."""

    start_sec: float
    end_sec: float
    scene_type: str
    confidence: float


class ImageEmbedRow(BaseModel):
    """A frames UPDATE payload, keyed by ``ts_sec``: perceptual hash + DINOv2 embedding."""

    ts_sec: float
    phash: int
    embedding: str  # base64 float32, via the codec

    @classmethod
    def build(cls, *, ts_sec: float, phash: int, embedding) -> ImageEmbedRow:
        """Construct from a numpy array / list embedding (encoded to base64 here)."""
        return cls(ts_sec=ts_sec, phash=phash, embedding=encode_vector(embedding))

    def vector(self) -> np.ndarray:
        """Decode this row's embedding back to a float32 numpy array."""
        return np.asarray(decode_vector(self.embedding), dtype="<f4")


# --- per-stage results ------------------------------------------------------------------


class RegisterResult(BaseModel):
    """videos identity + metadata. ``content_hash`` is the idempotency key the coordinator
    upserts on. ``media_path`` is omitted: it is a worker-local source path the coordinator
    does not need (it stores ``source_url`` as the public link instead)."""

    stage: Literal["register"] = "register"
    content_hash: str
    source_url: str | None = None
    source_platform: str | None = None
    title: str | None = None
    stream_type: str | None = None


class TranscribeResult(BaseModel):
    """No DB rows; declares the SRT artifact so the coordinator persists it for resume."""

    stage: Literal["transcribe"] = "transcribe"
    artifact: str = "transcript.srt"

    def artifacts(self) -> list[ArtifactRef]:
        return [ArtifactRef(name=self.artifact, kind="srt")]


class ParseTranscriptResult(BaseModel):
    """transcript_segments rows for one video."""

    stage: Literal["parse_transcript"] = "parse_transcript"
    segments: list[TranscriptSegmentRow]

    def to_vectors(self) -> list[np.ndarray]:
        """Decode every segment embedding back to float32 numpy arrays (persist-side helper)."""
        return [s.vector() for s in self.segments]


class FramesResult(BaseModel):
    """frames rows + the scene-timestamp recipe + a manifest of the JPEG artifacts.

    ``scene_timestamps`` is the ``detect_scene_timestamps`` output carried so a resumed /
    re-derived run can reproduce the sampling plan without re-probing the source video.
    """

    stage: Literal["frames"] = "frames"
    frames: list[FrameRow]
    scene_timestamps: list[float]

    def artifacts(self) -> list[ArtifactRef]:
        return [ArtifactRef(name=f.artifact, kind="frame") for f in self.frames]


class OcrResult(BaseModel):
    """ocr_segments rows for one video."""

    stage: Literal["ocr"] = "ocr"
    rows: list[OcrRow]


class ClassifyResult(BaseModel):
    """scene_segments rows + the dominant ``stream_type`` the coordinator writes onto videos."""

    stage: Literal["classify"] = "classify"
    segments: list[SceneSegmentRow]
    stream_type: str


class ImageEmbedResult(BaseModel):
    """Per-frame phash + embedding updates for one video."""

    stage: Literal["image_embed"] = "image_embed"
    rows: list[ImageEmbedRow]

    def to_vectors(self) -> list[np.ndarray]:
        """Decode every row embedding back to float32 numpy arrays (persist-side helper)."""
        return [r.vector() for r in self.rows]
