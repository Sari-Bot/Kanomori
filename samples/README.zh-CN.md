# 样本语料目录结构

简体中文 | [English](README.md)

该目录是分布式摄取所使用 source store 的本地镜像。

这里的媒体文件属于原始摄取输入，不是派生产物。生成出的帧、转写日志、OCR 输出和其他可重建产物
应放在 `../media/` 下，而不是这里。

## 目录结构

```text
samples/
  manifest.jsonl
  <title>_<date>/
    video.mp4
```

`manifest.jsonl` 是权威输入。每一行都是一条任务记录，先被 `POST /ingest/batch` 使用，
之后再由 worker 消费。

示例：

```json
{"path":"鹿乃的2月18日歌回直播_2024-02-18/video.mp4","title":"鹿乃的2月18日歌回直播","streamed_at":"2024-02-18","source_platform":"bilibili","source_url":"https://...","separate":true}
```

关键字段：

- `path`：worker 使用的规范源键
- `title`：面向人的标题
- `streamed_at`：直播日期
- `source_platform`：来源平台标识
- `source_url`：原始公开链接
- `separate`：为歌回显式开启 KITS 人声分离的覆盖开关

## 说明

- 目录名主要给人看，真正的键是 `path`
- 本地 source store 与 WebDAV source store 共用同一套布局
- 可重建的派生产物不应提交到这里
