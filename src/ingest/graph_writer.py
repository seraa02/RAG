"""
Neo4j graph writer.

Purpose:
    Write resolved entities and relationships into Neo4j.

TODO (Phase 1):
    - Use MERGE (never CREATE) for both nodes and relationships, so
      re-running ingestion does not create duplicates (Rule #13).
    - Every relationship written must carry a source_chunk_id property,
      preserving provenance back to the exact chunk that produced it.
    - Batch writes for efficiency once volume grows beyond the initial
      six-document corpus.
"""
