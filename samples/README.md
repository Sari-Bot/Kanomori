# samples/ — source clips you ingest manually

Raw input streams the ingestion path reads. **Not committed** (gitignored): large and
copyright-sensitive. Only this README and `manifest.jsonl` are tracked.

These are *inputs*. Derived output (frame thumbnails, KITS SRT/log artifacts) goes to
`media/`, never here, so you can wipe `media/` and re-ingest without touching source clips.

## Layout

This directory is the **local mirror** of the distributed source store (WebDAV in prod).
The layout is identical in both, so a batch validated here moves to WebDAV by copying the
tree and pointing `KANOMORI_MEDIA_SOURCE=webdav` at it.

```
samples/                         # = MEDIA_SOURCE_ROOT when KANOMORI_MEDIA_SOURCE=local
  manifest.jsonl                 # batch input — one job spec per line (the source of truth)
  <title>_<date>/
    video.mp4
```

`manifest.jsonl` is authoritative: each line's `path` is the natural key the worker
fetches and the coordinator records. Folder names are human-readable convenience only.
A line carries the metadata that lands in the job request:

```json
{"path":"鹿乃的2月18日歌回直播_2024-02-18/video.mp4","title":"...","streamed_at":"2024-02-18","source_platform":"bilibili","source_url":"https://...","separate":true}
```

`separate` is set `true` on singing streams (歌回 / 演唱会): the `locate_media` karaoke
heuristic only matches Japanese keywords (歌枠 …), so Chinese-titled singing streams need
the explicit flag to trigger KITS vocal isolation.

## Current clips

| Folder | Scene | Notes |
|--------|-------|-------|
| `2024_talk_cut/` | chatting | clear JP speech, minimal BGM; smallest — best for dry-run L0/L1 |
| `kano元気_2025-08-04/` | talk/announcement | YouTube source |
| `鹿乃的2月18日歌回直播_2024-02-18/` | singing | `separate:true` |
| `鹿乃演唱会50w粉丝纪念_2020-02-22/` | singing | concert, `separate:true` |
| `鹿乃特别直播演唱会いつかの約束を君に_2019-10-19/` | singing | concert, `separate:true` |

Metadata above is deduced from the original filenames (dates, Bilibili AV ids); refine
`manifest.jsonl` if you have better ground truth.
