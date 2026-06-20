"""Embedding models for Kanomori.

The text embedder (BGE-M3) is used both offline (embedding transcript segments during
ingestion) and online (embedding the query). Image embedders arrive in a later step. All heavy
model stacks (torch / FlagEmbedding) are imported lazily so importing this package never pulls
them in — only constructing an embedder does.
"""
