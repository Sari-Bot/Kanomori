# Sample Corpus Layout

[简体中文](README.zh-CN.md) | English

This directory is the local mirror of the source store used by distributed ingestion.

The media files here are raw ingest inputs, not derived artifacts. Generated frames, transcript
logs, OCR output, and other rebuildable artifacts belong under `../media/`, not here.

## Layout

```text
samples/
  manifest.jsonl
  <title>_<date>/
    video.mp4
```

`manifest.jsonl` is authoritative. Each line is one job record consumed by `POST /ingest/batch`
and later by the worker.

Example:

```json
{"path":"鹿乃的2月18日歌回直播_2024-02-18/video.mp4","title":"鹿乃的2月18日歌回直播","streamed_at":"2024-02-18","source_platform":"bilibili","source_url":"https://...","separate":true}
```

Important fields:

- `path`: canonical source key used by the worker
- `title`: human-readable title
- `streamed_at`: stream date
- `source_platform`: origin platform label
- `source_url`: original public source link
- `separate`: explicit singing-stream override for KITS vocal separation

## Notes

- Folder names are for humans; `path` is the real key
- This layout is shared by local and WebDAV source-store modes
- Rebuildable derived outputs should not be committed here
