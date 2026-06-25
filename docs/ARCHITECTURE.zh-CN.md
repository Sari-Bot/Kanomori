# 架构说明

简体中文 | [English](ARCHITECTURE.md)

Kanomori 是一个面向 VTuber 直播档案的多模态检索系统。它的实际目标不是找到视频标题，
而是从不完整的记忆中还原准确的直播与时间戳。

## 核心拆分

系统刻意分为两部分：

- 离线摄取：可使用 GPU、面向批处理、负责写入派生产物与索引
- 在线查询：以 CPU 为主、关注低延迟、负责读取索引并完成候选重排

这两部分不共享长期运行的进程。摄取侧负责准备数据，API 侧则基于 PostgreSQL 中的索引
提供查询能力。

## 主要摄取路径

当前摄取流水线为：

1. 注册媒体与任务元数据
2. 定位源媒体
3. 提取或准备音频
4. 通过外部 KITS 子进程完成转写
5. 解析转写片段
6. 抽取视频帧
7. 执行 OCR
8. 场景分类并构建图像向量

worker 会逐阶段持久化进度，因此中断后的任务可以恢复。

## 主要查询路径

当前已实现的查询入口：

- 台词检索：对 transcript segment 执行词法检索与向量检索
- 截图检索：OCR + 图像向量 + 合并重排

候选生成按模态分别进行，最终统一合并为绑定原始直播的带时间戳结果。

## 关键设计选择

### KITS 保持在应用进程外

Kanomori 通过 `KANOMORI_KITS_DIR` 下的 `uv run kits subtitle ...` 调用 KITS，而不会把
KITS 当作 Python 库导入。这样可以把 ASR 依赖栈与主 API/查询环境隔离开，并保持离线转写
与在线检索之间的清晰边界。

### PostgreSQL 是中心事实源

当前实现使用 PostgreSQL + pgvector 保存任务状态、元数据、全文检索数据与向量索引。
在分布式模式下，数据库写入统一由 coordinator 负责。

### Worker 通过 HTTP 与 coordinator 通信

远程 worker 不会直接写 PostgreSQL。它们通过带认证的 `/jobs/*` 接口来 claim 任务、
发送 heartbeat、上报阶段结果并标记完成。

### 查询路径比摄取路径更轻

重型计算应留在 worker 路径中。在线查询 API 的设计目标之一，就是避免依赖完整的离线模型
与工具链。

## 当前模块边界

- `src/kanomori/api/`：公开 API 与服务端渲染演示 UI
- `src/kanomori/ingest/`：worker 循环、各阶段、租约逻辑与 coordinator client
- `src/kanomori/retrieval/`：台词与截图检索流程
- `src/kanomori/embed/`：文本与图像向量模块
- `src/kanomori/models.py`：请求与响应数据结构

## Roadmap 边界

当前仓库已经实现的是摄取、台词检索与截图检索。音频片段检索、歌回反查、剪辑反查，以及更广义
的记忆型助理流程仍属后续工作，不应被文档表述为已上线能力。
