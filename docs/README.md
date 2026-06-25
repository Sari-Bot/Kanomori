# Documentation

[简体中文](README.zh-CN.md) | English

This directory holds the public maintainer and operator documentation for the current Kanomori
implementation.

## Start here

- [Getting Started](getting-started.md): local setup, first run, and first ingest/search cycle
- [Architecture](ARCHITECTURE.md): system boundaries and design choices
- [Distributed Ingestion](distributed-ingestion.md): coordinator/worker runbook
- [CUDA Worker Docker Deployment](cuda-worker-docker.md): remote NVIDIA worker deployment
- [Sample Corpus Layout](../samples/README.md): source-store mirror and manifest structure

## Which document to read

- New contributor: start with [Getting Started](getting-started.md)
- Want the system model: read [Architecture](ARCHITECTURE.md)
- Running multiple workers: read [Distributed Ingestion](distributed-ingestion.md)
- Deploying an NVIDIA box in Docker: read [CUDA Worker Docker Deployment](cuda-worker-docker.md)
- Preparing a local or WebDAV corpus: read [Sample Corpus Layout](../samples/README.md)

## Scope

These docs describe the codebase as currently implemented. Product vision and longer-term ideas
remain in `../kanomori_project_white_paper.md`, but the code and the docs in this directory are
the source of truth for present behavior.
