"""Japanese text normalization and tokenization for full-text search.

Two responsibilities, deliberately split by dependency weight:

- ``normalize`` — pure stdlib (Unicode NFKC + whitespace collapse). Used everywhere, including
  the CPU query path. NFKC folds full-width ASCII, half-width katakana, and other compatibility
  variants so ASR output and user queries compare on equal footing.
- ``tokenize_for_fts`` — segments Japanese into space-separated surface tokens via fugashi
  (MeCab). Stock PostgreSQL cannot segment Japanese (no spaces ⇒ one token), so we tokenize
  application-side and index/query with the ``simple`` text-search config. fugashi is an
  ``ingest``-group dependency and is imported lazily, so importing this module never requires
  it — only callers that actually tokenize do.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFKC-normalize and collapse whitespace. Pure stdlib; safe on the query path."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip()


@lru_cache(maxsize=1)
def _tagger():
    """Build and cache a fugashi Tagger. Lazy import keeps fugashi out of base imports."""
    import fugashi

    return fugashi.Tagger()


def tokenize_for_fts(text: str) -> str:
    """Normalize then segment into space-separated tokens for `to_tsvector('simple', …)`.

    The query path must call this with the same logic so lexical matching is symmetric.
    Empty / whitespace-only input yields an empty string (no tokens).
    """
    norm = normalize(text)
    if not norm:
        return ""
    return " ".join(word.surface for word in _tagger()(norm))
