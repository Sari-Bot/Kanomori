# CUDA Worker Docker 部署

简体中文 | [English](cuda-worker-docker.md)

这种部署方式面向仅运行 worker 的 NVIDIA 主机。FastAPI coordinator 与 PostgreSQL 继续留在
coordinator 机器上；CUDA 容器通过 `/jobs/*` claim 任务、处理任务，并把结果上传回去。

## 为什么这样部署

- 远程 worker 不直接写 PostgreSQL
- coordinator 继续作为任务状态的唯一拥有者
- GPU 专属依赖被隔离在 worker 主机上
- KITS 仍然保持为同级仓库，并以子进程方式调用

## 前置依赖

- worker 主机已安装 NVIDIA 驱动
- Docker Engine 与 Docker Compose
- NVIDIA Container Toolkit
- 一个可挂载进容器的 KITS 仓库

先确认 Docker 能看到 GPU：

```bash
docker run --rm --gpus all nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04 nvidia-smi
```

## 准备 coordinator

在 coordinator 主机上：

```bash
docker compose up -d
uv run kanomori-migrate
KANOMORI_COORDINATOR_TOKEN=replace-this-shared-secret \
uv run uvicorn kanomori.api.app:app --host 0.0.0.0 --port 8000
```

## 准备 worker 环境文件

```bash
cp .env.cuda-worker.example .env.cuda-worker
```

至少设置以下变量：

```bash
KANOMORI_COORDINATOR_URL=http://coordinator-host:8000
KANOMORI_COORDINATOR_TOKEN=replace-this-shared-secret
KANOMORI_KITS_DIR_HOST=/absolute/path/to/KITS
KANOMORI_MEDIA_SOURCE_ROOT_HOST=/absolute/path/to/source-store
```

## 构建并预热

```bash
docker compose -f docker-compose.cuda-worker.yml --env-file .env.cuda-worker build
docker compose -f docker-compose.cuda-worker.yml --env-file .env.cuda-worker run --rm \
  cuda-worker sh -lc 'cd /opt/kits && uv sync'
```

## 启动 worker

```bash
docker compose -f docker-compose.cuda-worker.yml --env-file .env.cuda-worker up -d
```

查看日志：

```bash
docker compose -f docker-compose.cuda-worker.yml --env-file .env.cuda-worker logs -f cuda-worker
```

## 冒烟检查

在容器内探测 CUDA provider：

```bash
docker compose -f docker-compose.cuda-worker.yml --env-file .env.cuda-worker run --rm \
  cuda-worker uv run --no-sync python - <<'PY'
from kanomori.ocr import OcrBackendUnavailable
from kanomori.ocr_tensorrt import ensure_cuda_ep_available
import onnxruntime as ort

ensure_cuda_ep_available(OcrBackendUnavailable)
print(ort.get_available_providers())
PY
```

运行一次 compute-only 样例：

```bash
docker compose -f docker-compose.cuda-worker.yml --env-file .env.cuda-worker run --rm \
  cuda-worker uv run --no-sync kanomori-worker --compute-only --source local --manifest-index 0
```

运行一次分布式 claim：

```bash
docker compose -f docker-compose.cuda-worker.yml --env-file .env.cuda-worker run --rm \
  cuda-worker uv run --no-sync kanomori-worker --once --worker-id cuda-worker-smoke
```

## 说明

- `KANOMORI_STAGE_*_DEVICE=gpu` 是 fail-fast 的
- GPU OCR 阶段要求 `KANOMORI_INGEST_OCR_BACKEND=cuda`
- 如果 source media 走 WebDAV，应使用对应 WebDAV 环境变量，而不是本地 source 挂载
