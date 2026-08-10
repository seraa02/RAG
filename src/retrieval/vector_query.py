"""
pgvector similarity search.

Purpose:
    Given a question embedding, retrieve the top-k most similar chunks
    from PostgreSQL/pgvector.

TODO (Phase 2/3):
    - Embed the incoming question with the same model used for ingestion.
    - Run an HNSW-indexed similarity search with a tuned ef_search.
    - Return chunks with their chunk_id and metadata for downstream
      citation.
"""
