"""Per-modality candidate generation for retrieval.

Each searcher returns a rank-ordered list of ``Candidate`` objects; the merge layer fuses
across modalities. Transcript and screenshot searchers live here. Keep query paths CPU-only:
the only model touched online is the (lazily-loaded) embedder for the text query and uploaded
screenshots.
"""
