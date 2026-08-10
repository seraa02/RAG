"""
Chunking of a NormalizedDocument.

Purpose:
    Split a NormalizedDocument's text into retrieval-sized chunks, each
    with a stable, unique chunk_id. This module is source-agnostic -- it
    only ever sees NormalizedDocument, never SEC-specific structure
    (see src/ingest/normalize.py for the boundary).

    The same chunk_id is used by both the vector pipeline (pgvector) and
    the graph pipeline (as source_chunk_id on extracted relationships),
    which is what makes citations/provenance possible later. This is a
    core invariant of the whole system: never generate separate,
    unrelated IDs for vector storage vs. graph storage.

TODO (Phase 1, after inspecting real documents):
    - Decide chunk size and overlap based on document structure, token
      count, and retrieval quality -- not an arbitrary default.
    - Assign a stable, human-legible chunk_id, e.g. NVDA_2026_10K_00123
      (source + fiscal year + filing type + sequence).
    - Attach metadata: chunk_id, document_id, company, filing date,
      section path, text, source position.
"""
