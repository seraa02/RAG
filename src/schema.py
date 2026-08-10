"""
Pydantic schemas shared across the ingestion and retrieval pipeline.

Purpose:
    Define the data contracts that keep the graph pipeline and the vector
    pipeline in sync (e.g. chunk_id, document_id) and that validate
    Claude's structured extraction output before anything is written to
    Neo4j.

TODO:
    - FilingMetadata (company, ticker, CIK, accession number, filing type,
      filing date, local path, download timestamp, SHA-256)
    - NormalizedDocument -- the source-agnostic shape every source adapter
      (src/ingest/sources/*.py) must produce: document_id, source,
      document_type, title, date, metadata, sections, text, source_uri,
      content_hash. This is the boundary past which nothing downstream
      may contain source-specific (e.g. SEC-specific) logic
      (see src/ingest/normalize.py).
    - Chunk (chunk_id, document_id, company, filing date, section path,
      text, source position)
    - ExtractedEntity (entity_type, canonical_name, source_chunk_id,
      confidence) -- fields depend on the ontology defined in Phase 1.
    - ExtractedRelationship (relationship_type, source_entity, target_entity,
      source_chunk_id, confidence)
    - Validation must reject/trigger-retry on any extraction output that
      does not conform to these schemas (Rule #7).
"""
