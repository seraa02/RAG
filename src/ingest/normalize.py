"""
NormalizedDocument: the shared, source-agnostic document representation.

Purpose:
    Define the common internal shape that every source adapter
    (src/ingest/sources/*.py) must produce, and the source-agnostic
    helpers that operate on it (content hashing, validation).

    This module is the normalization boundary described in README.md:
    everything before it may be SEC-specific (or PDF-specific, wiki-
    specific, etc.); everything after it (chunk.py, embed.py, extract.py,
    resolve.py, graph_writer.py) operates only on NormalizedDocument /
    chunks and must not contain source-specific logic.

Conceptual shape (finalized as a Pydantic model in src/schema.py once
Phase 0 data is in hand):

    document_id     stable identifier for this document
    source          e.g. "sec_edgar"
    document_type   e.g. "10-K"
    title
    date
    metadata        source-specific extras (company, ticker, CIK, ...)
    sections        ordered list of (section_path, text)
    text            full cleaned text
    source_uri      where this came from (URL or local path)
    content_hash    SHA-256 of the raw source content -- identifies the
                    document by content, not filename; used for
                    idempotent ingestion, duplicate detection, and cache
                    invalidation (see extract.py / embed.py).

TODO (Phase 0):
    - Define NormalizedDocument as a Pydantic model in src/schema.py.
    - Implement content-hashing and validation helpers here.
    - Each source adapter constructs a NormalizedDocument; this module
      never reaches back into source-specific parsing.
"""
