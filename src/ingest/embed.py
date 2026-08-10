"""
Chunk embedding for vector retrieval.

Purpose:
    Generate vector embeddings for each chunk and write them, alongside
    chunk metadata, into PostgreSQL/pgvector.

TODO (Phase 2):
    - Choose embedding provider: Voyage AI vs. a local open-source model.
      This decision will be presented explicitly (quality, cost, latency,
      setup complexity, privacy, reproducibility, CPU/GPU needs) before
      implementation -- not chosen silently.
    - Embed chunks and upsert into the pgvector table with chunk_id as key.
    - Store embedding_model + embedding_model_version alongside each
      vector, so switching models later never silently mixes incompatible
      embeddings in the same similarity search (spec 23.21).
    - Cache/skip re-embedding when chunk content and embedding model
      version are unchanged.
    - Build an HNSW index; tune ef_search based on measured recall@k.
"""
