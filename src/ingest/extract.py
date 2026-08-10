"""
Claude-based entity and relationship extraction.

Purpose:
    Send each chunk to Claude with a schema-constrained prompt (based on
    the ontology defined in config/ontology.yaml) and extract entities and
    relationships, each tagged with source_chunk_id and a confidence score.

TODO (Phase 1, after ontology is defined):
    - Build extraction prompt constrained to the finalized ontology.
    - Validate every extraction response against the Pydantic schema in
      src/schema.py; retry on schema violation (Rule #7).
    - Cache extraction results in data/cache/, keyed by a VERSIONED cache
      identity, not just the document's SHA-256 hash:
          document_hash + chunk_id + model_version + prompt_version + schema_version
      e.g. cache key = (abc123, chunk_0042, model_v1, extraction_v2, schema_v1)
      This means changing the extraction prompt or schema automatically
      invalidates only the affected cache entries -- no manual cache
      wipe required (Rule #10, spec 23.6).
    - Track token usage / cost per extraction call (Rule #11).
"""
