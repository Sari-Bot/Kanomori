from __future__ import annotations

from pathlib import Path

from kanomori import ocr
from kanomori.retrieval import screenshot


def test_upload_ocr_reader_uses_query_scoped_ocr_reader(monkeypatch) -> None:
    calls: list[str | None] = []

    class FakeReader:
        def read_image(self, path: Path):
            assert path.exists()
            return [ocr.OcrResult("query text", 0.9, {})]

    def get_reader(*, scope=None):
        calls.append(scope)
        return FakeReader()

    monkeypatch.setattr(ocr, "get_ocr_reader", get_reader)

    reader = screenshot.UploadOcrReader()

    assert reader.text_from_image_bytes(b"fake") == "query text"
    assert calls == ["query"]
