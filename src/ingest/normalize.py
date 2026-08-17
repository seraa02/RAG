"""
NormalizedDocument: the shared, source-agnostic document representation.

Purpose:
    This module is the normalization boundary described in README.md:
    everything before it may be SEC-specific (or PDF-specific, wiki-
    specific, etc.); everything after it (chunk.py, embed.py, extract.py,
    resolve.py, graph_writer.py) operates only on NormalizedDocument /
    Chunk objects and must never contain source-specific logic.

    NormalizedDocument itself is defined in src/schema.py (it's a data
    contract, so it lives with the other Pydantic models). This module
    holds the one function every source adapter calls to cross the
    boundary, so construction/validation happens in exactly one place
    regardless of which adapter produced the data.
"""

from __future__ import annotations

from typing import Any

from src.schema import NormalizedDocument


def build_normalized_document(**fields: Any) -> NormalizedDocument:
    """
    Construct and validate a NormalizedDocument. Source adapters
    (src/ingest/sources/*.py) call this as the last step of their
    pipeline instead of instantiating NormalizedDocument directly --
    if a cross-cutting invariant needs to be added later (e.g. logging
    every document that crosses the boundary), this is the one place
    to add it.
    """
    return NormalizedDocument(**fields)
