"""
Pydantic schemas shared across the ingestion and retrieval pipeline.

Purpose:
    Define the data contracts that keep the graph pipeline and the vector
    pipeline in sync (e.g. chunk_id, document_id) and that validate
    Claude's structured extraction output before anything is written to
    Neo4j.

    entity_type / relationship_type are plain `str` for now. Once the
    ontology is defined in Phase 1 (config/ontology.yaml), extraction code
    validates these against the finalized ontology lists rather than the
    schema hardcoding an enum -- keeping the ontology config-driven, not
    duplicated in Python (see README's Design Principles).
"""

from __future__ import annotations

import hashlib
from datetime import date as date_type
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DocumentSection(BaseModel):
    """One labeled section of a source document, e.g. SEC 'Item 1A. Risk Factors'."""

    section_path: str
    text: str


class NormalizedDocument(BaseModel):
    """
    The source-agnostic document shape every ingestion adapter
    (src/ingest/sources/*.py) must emit. Nothing downstream of this model
    (chunk.py, embed.py, extract.py, resolve.py, graph_writer.py) may know
    or care whether the source was SEC EDGAR, a PDF, or a wiki -- that is
    the normalization boundary described in README.md and normalize.py.
    """

    document_id: str
    source: str  # e.g. "sec_edgar"
    document_type: str  # e.g. "10-K"
    title: str
    date: date_type
    metadata: dict[str, Any] = Field(default_factory=dict)
    sections: list[DocumentSection]
    text: str
    source_uri: str
    content_hash: str  # SHA-256 of the raw source content, not the filename

    @field_validator("content_hash")
    @classmethod
    def _validate_hash_format(cls, v: str) -> str:
        v = v.lower()
        if len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError("content_hash must be a 64-character hex SHA-256 digest")
        return v

    @staticmethod
    def compute_content_hash(raw_content: bytes) -> str:
        return hashlib.sha256(raw_content).hexdigest()


class Chunk(BaseModel):
    """
    A retrieval-sized slice of a NormalizedDocument's text.

    chunk_id is the shared key used identically by the vector pipeline
    (pgvector row key) and the graph pipeline (ExtractedRelationship /
    ExtractedEntity.source_chunk_id). This is the core invariant that
    makes citations and provenance possible later -- see chunk.py.
    """

    chunk_id: str
    document_id: str
    chunk_index: int = Field(ge=0)
    section_path: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedEntity(BaseModel):
    """
    One entity extracted by Claude from a single chunk, before entity
    resolution has run. canonical_name is filled in later by
    src/ingest/resolve.py -- at extraction time we only have the raw
    surface form Claude saw in the text (e.g. "Advanced Micro Devices,
    Inc." vs. the eventual canonical "AMD").
    """

    entity_type: str
    raw_name: str
    canonical_name: str | None = None
    source_chunk_id: str
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractedRelationship(BaseModel):
    """
    One relationship extracted by Claude from a single chunk, before
    entity resolution. source_entity / target_entity hold raw surface
    forms; resolve.py maps them onto canonical entity nodes before the
    graph_writer MERGEs anything into Neo4j.
    """

    relationship_type: str
    source_entity: str
    target_entity: str
    source_chunk_id: str
    confidence: float = Field(ge=0.0, le=1.0)


class FilingMetadata(BaseModel):
    """
    One record in the SEC download manifest -- SEC-specific, so it is
    used only inside src/ingest/sources/sec.py, never downstream of the
    NormalizedDocument boundary.
    """

    company: str
    ticker: str
    cik: str
    accession_number: str
    filing_type: str
    filing_date: date_type
    local_path: str
    downloaded_at: str  # ISO 8601 timestamp
    sha256: str
