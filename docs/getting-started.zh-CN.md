# 快速上手

简体中文 | [English](getting-started.md)

本指南的目标是让你在本地把 Kanomori 跑起来，并完成一次摄取任务与一次检索查询。

## 前置依赖

- `uv`
- Docker 与 Docker Compose
- `ffmpeg`
- 通过 `uv` 提供的 Python 3.12
- 一个同级目录下的 KITS 仓库，用于真实转写

## 安装依赖组

按任务选择最小依赖集合：

```bash
uv sync
uv sync --group ingest
uv sync --group embed
uv sync --group worker-cpu
uv sync --group worker-cuda
```

说明：

- `uv sync` 足够支撑核心 API 开发与大部分纯逻辑测试
- `ingest` 增加抽帧、OCR 与日文分词依赖
- `embed` 增加文本与图像模型依赖
- `worker-cpu` 是完整 CPU worker 安装目标
- `worker-cuda` 是完整 CUDA worker 安装目标

## 配置环境

```bash
cp .env.example .env
```

关键变量：

- `KANOMORI_DATABASE_URL`：PostgreSQL + pgvector 连接串
- `KANOMORI_KITS_DIR`：同级 KITS 仓库路径
- `KANOMORI_MEDIA_ROOT`：派生产物输出目录
- `KANOMORI_MEDIA_SOURCE_ROOT`：source-store 镜像，本地开发默认 `./samples`

分布式 worker 还需要：

- `KANOMORI_COORDINATOR_TOKEN`
- `KANOMORI_COORDINATOR_URL`

## 启动数据库

```bash
docker compose up -d
uv run kanomori-migrate
```

默认 Compose 栈会把 PostgreSQL 暴露在 `localhost:5433`。

## 启动 API

```bash
uv run uvicorn kanomori.api.app:app --reload
```

常用路由：

- `POST /search/transcript`
- `POST /search/screenshot`
- `POST /ingest`
- `POST /ingest/batch`
- `GET /ingest/{job_id}`
- `/`：基于 Jinja2 + htmx 的轻量演示 UI

## 启动 worker

本地持续轮询 worker：

```bash
uv run kanomori-worker
```

单次执行：

```bash
uv run kanomori-worker --once --worker-id local-smoke
```

仅计算干跑：

```bash
uv run kanomori-worker --compute-only --source local --manifest-index 0
```

## 第一次摄取

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

轮询任务状态：

```bash
curl http://localhost:8000/ingest/<job_id>
```

## 第一次查询

台词检索：

```bash
curl -X POST http://localhost:8000/search/transcript \
  -H 'Content-Type: application/json' \
  -d '{"query":"歌枠","k":5}'
```

截图检索是 multipart 请求，通常更适合通过演示 UI 或一个小的 HTTP 客户端脚本来验证。

## 下一步

- [架构说明](ARCHITECTURE.zh-CN.md)
- [分布式摄取](distributed-ingestion.zh-CN.md)
- [CUDA Worker Docker 部署](cuda-worker-docker.zh-CN.md)
- [样本语料目录结构](../samples/README.zh-CN.md)
