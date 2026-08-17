"""
Tests for src/schema.py -- the Pydantic data contracts shared by the
graph and vector pipelines.

What these tests prove:
    - Valid data constructs the models without error.
    - Invalid data (bad hash format, out-of-range confidence) is rejected
      at the model boundary, which is the whole point of using Pydantic
      instead of plain dicts (Rule #9 -- structured outputs must be
      validated, not trusted as-is).
"""

import pytest
from pydantic import ValidationError

from src.schema import (
    Chunk,
    DocumentSection,
    ExtractedEntity,
    ExtractedRelationship,
    NormalizedDocument,
)


def _make_document(**overrides):
    defaults = dict(
        document_id="NVDA_2026_10K",
        source="sec_edgar",
        document_type="10-K",
        title="NVIDIA Corporation 10-K",
        date="2026-01-15",
        metadata={"company": "NVIDIA", "ticker": "NVDA"},
        sections=[DocumentSection(section_path="Item 1. Business", text="NVIDIA designs GPUs.")],
        text="NVIDIA designs GPUs.",
        source_uri="https://www.sec.gov/example",
        content_hash=NormalizedDocument.compute_content_hash(b"NVIDIA designs GPUs."),
    )
    defaults.update(overrides)
    return NormalizedDocument(**defaults)


def test_normalized_document_accepts_valid_data():
    doc = _make_document()
    assert doc.document_id == "NVDA_2026_10K"
    assert len(doc.content_hash) == 64


def test_content_hash_is_deterministic():
    h1 = NormalizedDocument.compute_content_hash(b"same content")
    h2 = NormalizedDocument.compute_content_hash(b"same content")
    h3 = NormalizedDocument.compute_content_hash(b"different content")
    assert h1 == h2
    assert h1 != h3


def test_normalized_document_rejects_malformed_hash():
    with pytest.raises(ValidationError):
        _make_document(content_hash="not-a-real-sha256-hash")


def test_chunk_requires_non_negative_index():
    chunk = Chunk(
        chunk_id="NVDA_2026_10K_00001",
        document_id="NVDA_2026_10K",
        chunk_index=0,
        section_path="Item 1. Business",
        text="NVIDIA designs GPUs.",
    )
    assert chunk.chunk_index == 0

    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="NVDA_2026_10K_00001",
            document_id="NVDA_2026_10K",
            chunk_index=-1,
            section_path="Item 1. Business",
            text="NVIDIA designs GPUs.",
        )


def test_extracted_entity_rejects_out_of_range_confidence():
    ExtractedEntity(
        entity_type="Company",
        raw_name="Advanced Micro Devices, Inc.",
        source_chunk_id="NVDA_2026_10K_00001",
        confidence=0.92,
    )
    with pytest.raises(ValidationError):
        ExtractedEntity(
            entity_type="Company",
            raw_name="Advanced Micro Devices, Inc.",
            source_chunk_id="NVDA_2026_10K_00001",
            confidence=1.5,
        )


def test_extracted_relationship_rejects_out_of_range_confidence():
    ExtractedRelationship(
        relationship_type="COMPETES_WITH",
        source_entity="NVIDIA",
        target_entity="AMD",
        source_chunk_id="NVDA_2026_10K_00001",
        confidence=0.8,
    )
    with pytest.raises(ValidationError):
        ExtractedRelationship(
            relationship_type="COMPETES_WITH",
            source_entity="NVIDIA",
            target_entity="AMD",
            source_chunk_id="NVDA_2026_10K_00001",
            confidence=-0.1,
        )
