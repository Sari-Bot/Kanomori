# 文档索引

简体中文 | [English](README.md)

本目录保存当前 Kanomori 实现对应的公开维护与运维文档。

## 从这里开始

- [快速上手](getting-started.zh-CN.md)：本地安装、首次启动、首次摄取与查询
- [架构说明](ARCHITECTURE.zh-CN.md)：系统边界与核心设计选择
- [分布式摄取](distributed-ingestion.zh-CN.md)：coordinator/worker 运行手册
- [CUDA Worker Docker 部署](cuda-worker-docker.zh-CN.md)：远程 NVIDIA worker 部署
- [样本语料目录结构](../samples/README.zh-CN.md)：source-store 镜像与 manifest 结构

## 如何选择阅读路径

- 新贡献者：先读 [快速上手](getting-started.zh-CN.md)
- 想理解系统模型：读 [架构说明](ARCHITECTURE.zh-CN.md)
- 需要跑多个 worker：读 [分布式摄取](distributed-ingestion.zh-CN.md)
- 需要用 Docker 部署 NVIDIA 机器：读 [CUDA Worker Docker 部署](cuda-worker-docker.zh-CN.md)
- 需要准备本地或 WebDAV 语料：读 [样本语料目录结构](../samples/README.zh-CN.md)

## 范围说明

这些文档描述的是当前已经实现的代码行为。产品愿景与长期设想仍可参考
`../kanomori_project_white_paper.md`，但当前行为以代码和本目录文档为准。
