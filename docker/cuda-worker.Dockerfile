# syntax=docker/dockerfile:1.7

ARG CUDA_IMAGE=nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.19
ARG APT_MIRROR_HOST=mirrors.tuna.tsinghua.edu.cn

FROM ${UV_IMAGE} AS uv-bin

FROM ${CUDA_IMAGE} AS runtime

ARG APT_MIRROR_HOST

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
        sed -ri "s|https?://archive.ubuntu.com/ubuntu/?|https://${APT_MIRROR_HOST}/ubuntu/|g; s|https?://security.ubuntu.com/ubuntu/?|https://${APT_MIRROR_HOST}/ubuntu/|g; s|https?://ports.ubuntu.com/ubuntu-ports/?|https://${APT_MIRROR_HOST}/ubuntu-ports/|g" /etc/apt/sources.list.d/ubuntu.sources; \
    fi \
    && if [ -f /etc/apt/sources.list ]; then \
        sed -ri "s|https?://archive.ubuntu.com/ubuntu/?|https://${APT_MIRROR_HOST}/ubuntu/|g; s|https?://security.ubuntu.com/ubuntu/?|https://${APT_MIRROR_HOST}/ubuntu/|g; s|https?://ports.ubuntu.com/ubuntu-ports/?|https://${APT_MIRROR_HOST}/ubuntu-ports/|g" /etc/apt/sources.list; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        git \
        python3.12 \
        python3.12-venv \
        tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv-bin /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group worker-cuda --python 3.12

ENTRYPOINT ["tini", "--"]
CMD ["uv", "run", "--no-sync", "kanomori-worker"]
