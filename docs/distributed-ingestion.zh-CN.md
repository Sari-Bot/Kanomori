# 分布式摄取

简体中文 | [English](distributed-ingestion.md)

本文描述的是 Kanomori 当前已经实现的 coordinator/worker 分布式摄取路径。

如果你需要先完成本地安装，请先看 [快速上手](getting-started.zh-CN.md)。如果你想理解系统边界，
请看 [架构说明](ARCHITECTURE.zh-CN.md)。

## 系统角色

- Coordinator：FastAPI 应用，负责 `jobs` 表、阶段持久化与最终产物
- Worker：`kanomori-worker` 进程，负责 claim 任务、本地执行阶段并回传结果
- Source store：只读输入树，可以是本地镜像，也可以是 WebDAV 树

在分布式模式下，worker 不会直接写 PostgreSQL。数据库写入统一由 coordinator 负责。

## Coordinator 接口

创建任务：

- `POST /ingest`
- `POST /ingest/batch`
- `GET /ingest/{job_id}`

远程 worker 控制：

- `POST /jobs/claim`
- `POST /jobs/{job_id}/heartbeat`
- `POST /jobs/{job_id}/stage/{stage_name}`
- `POST /jobs/{job_id}/complete`
- `POST /jobs/{job_id}/fail`

## 共享前置依赖

- `uv`
- `ffmpeg`
- PostgreSQL + pgvector
- `KANOMORI_KITS_DIR` 指向的 KITS

Worker 安装目标：

```bash
uv sync --group worker-cpu
uv sync --group worker-cuda
```

应使用完整 worker 依赖组，而不是只选择某个叶子 OCR 依赖组。

## Coordinator 配置

常用变量：

```bash
KANOMORI_DATABASE_URL=postgresql://kanomori:kanomori@localhost:5433/kanomori
KANOMORI_MEDIA_ROOT=./media
KANOMORI_COORDINATOR_TOKEN=replace-this-shared-secret
```

说明：

- `KANOMORI_COORDINATOR_TOKEN` 用于保护 `/jobs/*`
- 未设置时，`/jobs/*` 会 fail closed
- 每个 worker 都必须使用同一个 token

## Worker 配置

典型本地 worker 变量：

```bash
KANOMORI_COORDINATOR_URL=http://localhost:8000
KANOMORI_COORDINATOR_TOKEN=replace-this-shared-secret
KANOMORI_MEDIA_SOURCE=local
KANOMORI_MEDIA_SOURCE_ROOT=./samples
KANOMORI_KITS_DIR=/path/to/KITS
KANOMORI_STAGE_PARSE_TRANSCRIPT_DEVICE=cpu
KANOMORI_STAGE_OCR_DEVICE=cpu
KANOMORI_STAGE_CLASSIFY_DEVICE=cpu
KANOMORI_STAGE_IMAGE_EMBED_DEVICE=cpu
```

WebDAV source store：

```bash
KANOMORI_MEDIA_SOURCE=webdav
KANOMORI_MEDIA_SOURCE_URL=https://dav.example.com/store
KANOMORI_MEDIA_SOURCE_USER=...
KANOMORI_MEDIA_SOURCE_PASSWORD=...
```

阶段设备说明：

- 一个 worker 会 claim 整个任务，并在本地完成所有阶段
- `cpu` 表示该阶段强制走 CPU
- `gpu` 表示该阶段必须走 GPU，不可用时直接失败
- `KANOMORI_STAGE_OCR_DEVICE=gpu` 需要 `KANOMORI_INGEST_OCR_BACKEND=cuda` 或 `tensorrt`

## Source store 目录结构

分布式路径要求的目录结构与 [../samples/README.zh-CN.md](../samples/README.zh-CN.md) 一致：

```text
<root>/
  manifest.jsonl
  <title>_<date>/
    video.mp4
```

Manifest 示例：

```json
{"path":"鹿乃的2月18日歌回直播_2024-02-18/video.mp4","title":"鹿乃的2月18日歌回直播","streamed_at":"2024-02-18","source_platform":"bilibili","source_url":"https://...","separate":true}
```

其中 `path` 是 worker 识别源文件的规范键。

如果需要把 WebDAV 根目录下的散视频整理成 source store 结构，可以使用
`kanomori-organize-source`。干跑、执行和跳过规则见
[样本语料目录结构](../samples/README.zh-CN.md#整理-webdav-source-store)。

## 启动顺序

先启动 coordinator 侧：

```bash
docker compose up -d
uv run kanomori-migrate
uv run uvicorn kanomori.api.app:app --host 0.0.0.0 --port 8000
```

启动 worker：

```bash
uv run kanomori-worker --worker-id gpu-worker-01
```

单次执行：

```bash
uv run kanomori-worker --once --worker-id gpu-worker-01
```

不接 coordinator 的干跑：

```bash
uv run kanomori-worker --compute-only --source local --manifest-index 0
```

## 提交任务

单条任务：

```bash
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"media_path":"2024_talk_cut/video.mp4","source_url":"https://example.invalid/video"}'
```

Manifest 批量任务：

```bash
curl -X POST http://localhost:8000/ingest/batch \
  -H 'Content-Type: application/json' \
  -d '{"manifest_path":"manifest.jsonl"}'
```

## 监控与恢复

观察 worker 日志中的 claim、阶段开始、阶段完成、heartbeat 延长与失败信息。如果 worker 在部分
阶段完成后退出，coordinator 可以重新租出该任务，后续 worker 会从最后一个已持久化阶段继续，
而不是从头跑完整条流水线。

## WSL CUDA OCR 说明

在 WSL GPU worker 上，如果 ONNX Runtime 只报告 `AzureExecutionProvider` 和
`CPUExecutionProvider`，通常意味着环境不一致，而不是应用代码逻辑错误。建议恢复流程：

```bash
uv sync --group worker-cuda
uv pip uninstall onnxruntime
uv pip install --reinstall onnxruntime-gpu==1.27.0
```

然后在启动 worker 前，确保 `LD_LIBRARY_PATH` 同时包含 `/usr/lib/wsl/lib` 与虚拟环境里的
NVIDIA 运行库路径。

## 相关代码

- `src/kanomori/api/app.py`
- `src/kanomori/api/jobs.py`
- `src/kanomori/ingest/worker.py`
- `src/kanomori/ingest/coordinator_client.py`
- `src/kanomori/ingest/lease.py`
- `src/kanomori/media_source.py`
- `src/kanomori/config.py`
