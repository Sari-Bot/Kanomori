# Kanomori

简体中文 | [English](README.md)

<p align="center">
  <img src="imgs/logo.png" alt="Kanomori logo" width="360">
</p>

Kanomori 是一个面向 VTuber 直播档案的多模态、时刻级检索系统。
当你只记得一句台词、一张截图、一句歌词，或者某个模糊的印象时，
它的目标是还原出准确的原始直播与时间戳。

当前代码库是一个 Phase-1 MVP，重点已经落在真实可运行的摄取与检索基础设施上：
分布式离线摄取、台词检索、截图检索，以及场景感知重排。

## 已实现能力

- `POST /search/transcript`：基于转写片段的检索
- `POST /search/screenshot`：结合 OCR 与图像向量的截图检索
- `POST /ingest` 与 `POST /ingest/batch`：提交离线摄取任务
- `GET /ingest/{job_id}`：查询任务状态
- `POST /jobs/*`：远程 worker 使用的 coordinator 接口
- 一个基于 Jinja2 + htmx 的轻量演示 UI

## 系统结构

Kanomori 分成两个松耦合部分：

- 离线摄取：注册媒体、提取音频与帧、通过 KITS 转写、执行 OCR、场景分类、
  构建向量，并持久化派生产物
- 在线查询：针对 PostgreSQL 中的索引执行低延迟台词检索与截图检索，并将候选
  重排为带时间戳的结果

查询路径以 CPU 为主，重型模型计算应放在离线 worker 路径中。

## 快速开始

前置依赖：

- `uv`
- Docker 与 Docker Compose
- `ffmpeg`
- 一个同级目录下的 [KITS](https://github.com/kanbereina/KITS) 仓库，用于真实转写

按需安装依赖：

```bash
uv sync
uv sync --group ingest
uv sync --group embed
uv sync --group worker-cpu
uv sync --group worker-cuda
cp .env.example .env
docker compose up -d
uv run kanomori-migrate
```

启动主要服务：

```bash
uv run uvicorn kanomori.api.app:app --reload
uv run kanomori-worker
```

然后：

1. 调用 `POST /ingest` 或 `POST /ingest/batch`
2. 轮询 `GET /ingest/{job_id}`
3. 使用 `POST /search/transcript` 或 `POST /search/screenshot` 查询

逐步上手说明见 [docs/getting-started.zh-CN.md](docs/getting-started.zh-CN.md)。

## 文档导航

- [快速上手](docs/getting-started.zh-CN.md)
- [文档索引](docs/README.zh-CN.md)
- [架构说明](docs/ARCHITECTURE.zh-CN.md)
- [分布式摄取](docs/distributed-ingestion.zh-CN.md)
- [CUDA Worker Docker 部署](docs/cuda-worker-docker.zh-CN.md)
- [样本语料目录结构](samples/README.zh-CN.md)

## 当前状态

当前已实现：

- 台词检索
- 截图检索
- 可恢复的摄取流水线
- 分布式 coordinator/worker 摄取
- OCR 后端切换与 worker 分阶段 CPU/GPU 指定

后续规划：

- 音频片段检索
- 歌回与剪辑反查
- 模糊记忆与证据式问答工作流

## 仓库结构

```text
src/kanomori/            应用代码
docs/                    维护者与运维文档
samples/                 本地 source-store 镜像与 manifest 示例
migrations/              前向 SQL 迁移
tests/                   单元与集成测试
imgs/                    项目资源
```

## 相关文件

- `kanomori_project_white_paper.md`：原始产品愿景
- `HANDOFF_TO_GPT.md`：内部持续开发交接记录

本 README 体系中的公开文档以当前代码实现为准。

## 致谢

Kanomori 最初是一个个人项目，灵感来自鹿乃（Kano）的直播档案与创作作品。

项目的视觉方向受到绘本《こまったましろ》的启发，该作品由鹿乃创作、水玉子绘制。

原始角色、插画及相关作品的一切权利均归其各自的创作者与出版方所有。

Kanomori 是一个独立的粉丝创作项目，与原作者无隶属关系，也未获得其官方背书。

## 许可证

Kanomori 自身源码使用 MIT 许可证。

KITS 是独立的 AGPL-3.0 项目，Kanomori 仅以外部子进程方式调用。档案内容与源直播的
相关权利仍归原始权利方所有。
