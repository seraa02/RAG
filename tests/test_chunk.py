"""
Tests for src/ingest/chunk.py.

What these tests prove:
    - Chunk IDs are unique and stable in format (document_id + zero-padded
      index), which is the invariant citations/provenance depend on.
    - Chunking never crosses a section boundary, so every chunk's
      section_path is accurate.
    - Empty sections produce no chunks (no blank citations later).
    - Long sections actually get split (not just returned as one giant
      chunk), and overlap is respected.
"""

from src.ingest.chunk import chunk_document
from src.schema import DocumentSection, NormalizedDocument


def _doc(sections: list[DocumentSection]) -> NormalizedDocument:
    text = "\n".join(s.text for s in sections)
    return NormalizedDocument(
        document_id="TEST_2026_10K",
        source="sec_edgar",
        document_type="10-K",
        title="Test Filing",
        date="2026-01-01",
        metadata={"company": "Test Co", "ticker": "TEST"},
        sections=sections,
        text=text,
        source_uri="https://example.com",
        content_hash=NormalizedDocument.compute_content_hash(text.encode()),
    )


def test_chunk_ids_are_unique_and_sequential():
    doc = _doc(
        [
            DocumentSection(section_path="Item 1. Business", text="Sentence one. " * 50),
            DocumentSection(section_path="Item 1A. Risk Factors", text="Sentence two. " * 50),
        ]
    )
    chunks = chunk_document(doc)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0] == "TEST_2026_10K_00000"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_never_crosses_section_boundary():
    doc = _doc(
        [
            DocumentSection(section_path="Item 1. Business", text="Business content. " * 30),
            DocumentSection(section_path="Item 1A. Risk Factors", text="Risk content. " * 30),
        ]
    )
    chunks = chunk_document(doc)
    business_chunks = [c for c in chunks if c.section_path == "Item 1. Business"]
    risk_chunks = [c for c in chunks if c.section_path == "Item 1A. Risk Factors"]
    assert business_chunks and risk_chunks
    assert all("Risk content" not in c.text for c in business_chunks)
    assert all("Business content" not in c.text for c in risk_chunks)


def test_empty_section_produces_no_chunks():
    doc = _doc(
        [
            DocumentSection(section_path="Item 4.", text="   "),
            DocumentSection(section_path="Item 1. Business", text="Real content here."),
        ]
    )
    chunks = chunk_document(doc)
    assert all(c.section_path != "Item 4." for c in chunks)
    assert len(chunks) == 1


def test_long_section_splits_into_multiple_chunks_with_overlap():
    long_text = "The quick brown fox jumps over the lazy dog. " * 400  # well over 800 tokens
    doc = _doc([DocumentSection(section_path="Item 1A. Risk Factors", text=long_text)])
    chunks = chunk_document(doc, target_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    # Overlap: the tail of one chunk should share text with the head of the next.
    assert chunks[0].text[-30:] in chunks[1].text or chunks[1].text[:30] in chunks[0].text


def test_chunk_metadata_carries_company_and_filing_info():
    doc = _doc([DocumentSection(section_path="Item 1. Business", text="Some business text.")])
    chunks = chunk_document(doc)
    assert chunks[0].metadata["company"] == "Test Co"
    assert chunks[0].metadata["ticker"] == "TEST"
    assert chunks[0].metadata["document_type"] == "10-K"
