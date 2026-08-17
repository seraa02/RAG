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

Design decisions (made after inspecting the real corpus):
    - Chunk *within* a section, never across section boundaries. Sections
      here are SEC 10-K Items (e.g. "Item 1A. Risk Factors"), which range
      from tens of chars ("Item 4.") to 100K+ chars (NVDA Risk Factors).
      Keeping a chunk inside one section means every chunk carries one
      accurate, single section_path -- crossing sections would blend an
      Item 1 sentence with an unrelated Item 1A sentence in the same chunk.
    - Target ~800 tokens per chunk with ~100 token overlap. Small enough
      for precise citation, large enough that Claude has enough context to
      extract a relationship correctly (a sentence split across chunks
      loses its subject).
    - tiktoken (cl100k_base) is used only as a fast, consistent LOCAL
      sizing heuristic for splitting -- not as a stand-in for Claude's
      actual tokenizer or for billing. Actual extraction cost is measured
      separately via the Anthropic count_tokens endpoint before running
      extraction (see src/ingest/extract.py).
    - chunk_id = f"{document_id}_{chunk_index:05d}", e.g.
      "NVDA_2026_10K_00123" -- human-legible, stable across re-runs as
      long as the chunking parameters don't change.
"""

from __future__ import annotations

import tiktoken

from src.schema import Chunk, NormalizedDocument

_ENCODING = tiktoken.get_encoding("cl100k_base")

TARGET_TOKENS = 800
OVERLAP_TOKENS = 100


def _split_section_text(text: str, target: int, overlap: int) -> list[str]:
    """Split one section's text into overlapping token windows."""
    tokens = _ENCODING.encode(text)
    if not tokens:
        return []

    pieces: list[str] = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + target, n)
        pieces.append(_ENCODING.decode(tokens[start:end]))
        if end == n:
            break
        start = end - overlap  # step forward, keeping `overlap` tokens of context
    return pieces


def chunk_document(
    document: NormalizedDocument,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    """
    Split a NormalizedDocument into Chunks, one section at a time.

    Empty or whitespace-only sections produce no chunks. chunk_index is
    assigned sequentially across the whole document (not per-section), so
    chunk order reflects document order.
    """
    chunks: list[Chunk] = []
    index = 0

    for section in document.sections:
        text = section.text.strip()
        if not text:
            continue

        for piece in _split_section_text(text, target_tokens, overlap_tokens):
            piece = piece.strip()
            if not piece:
                continue
            chunk_id = f"{document.document_id}_{index:05d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    chunk_index=index,
                    section_path=section.section_path,
                    text=piece,
                    metadata={
                        "company": document.metadata.get("company"),
                        "ticker": document.metadata.get("ticker"),
                        "document_type": document.document_type,
                        "date": document.date.isoformat(),
                    },
                )
            )
            index += 1

    return chunks
