"""
Chunk embedding into pgvector.

Purpose:
    Embed the same chunks that feed the graph pipeline (chunk.py) into
    PostgreSQL/pgvector, with metadata (document id, section path, date,
    company/ticker), keyed by the same chunk_id used everywhere else.
    That shared key is what makes it possible to jump from a graph path
    back to source text, and from a retrieved passage into the graph
    neighborhood around it (see README's "Why Chunk IDs Matter").

Design decisions:
    - Local embedding model (BAAI/bge-large-en-v1.5, 1024 dims), per
      EMBEDDING_PROVIDER=local / EMBEDDING_MODEL in .env -- no per-call
      API cost, runs on CPU (slower but fine at this corpus size).
    - BGE models are trained with an asymmetric instruction convention:
      the query gets a "Represent this sentence for searching relevant
      passages: " prefix at *query* time; passages are embedded as-is,
      with no prefix. Getting this backwards silently degrades recall
      without erroring, so the query-side prefix lives in
      src/retrieval/vector_query.py, not here -- passages never get it.
    - HNSW index with cosine distance (vector_cosine_ops), since BGE
      embeddings are typically compared by cosine similarity.
    - Writes are idempotent: `ON CONFLICT (chunk_id) DO UPDATE`, so
      re-running ingestion after a chunking change re-embeds and
      overwrites rather than duplicating rows.
"""

from __future__ import annotations

from functools import lru_cache

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from src.config import Settings, get_settings
from src.schema import Chunk

EMBEDDING_DIM = 1024  # BAAI/bge-large-en-v1.5


@lru_cache
def _get_model():
    """Lazy-load the sentence-transformers model once per process."""
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    if settings.embedding_provider != "local":
        raise RuntimeError(
            f"embed.py only implements the 'local' provider; "
            f"EMBEDDING_PROVIDER={settings.embedding_provider!r} is not supported."
        )
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of passage texts (no query instruction prefix -- see module docstring)."""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL,
    section_path    TEXT NOT NULL,
    text            TEXT NOT NULL,
    company         TEXT,
    ticker          TEXT,
    document_type   TEXT,
    filing_date     DATE,
    embedding       vector({EMBEDDING_DIM}) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# HNSW build requires data present to be efficient; created separately so
# callers can build the table first, bulk-load, then index (see run_embedding).
INDEX_SQL = """
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def get_connection(settings: Settings | None = None) -> psycopg.Connection:
    """
    Connect and ensure the pgvector extension + schema exist before
    registering the `vector` type adapter -- register_vector() looks up
    the type's OID in pg_type, which doesn't exist until `CREATE
    EXTENSION vector` has run at least once on this database.
    """
    settings = settings or get_settings()
    conn = psycopg.connect(settings.postgres_dsn)
    ensure_schema(conn)
    register_vector(conn)
    return conn


def ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()


def build_index(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(INDEX_SQL)
    conn.commit()


def write_chunks(conn: psycopg.Connection, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Upsert chunks + their embeddings. chunks and embeddings must be aligned by index."""
    assert len(chunks) == len(embeddings), "chunks/embeddings length mismatch"
    with conn.cursor() as cur:
        for chunk, vector in zip(chunks, embeddings):
            cur.execute(
                """
                INSERT INTO chunks
                    (chunk_id, document_id, section_path, text, company, ticker,
                     document_type, filing_date, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    section_path = EXCLUDED.section_path,
                    text = EXCLUDED.text,
                    company = EXCLUDED.company,
                    ticker = EXCLUDED.ticker,
                    document_type = EXCLUDED.document_type,
                    filing_date = EXCLUDED.filing_date,
                    embedding = EXCLUDED.embedding
                """,
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.section_path,
                    chunk.text,
                    chunk.metadata.get("company"),
                    chunk.metadata.get("ticker"),
                    chunk.metadata.get("document_type"),
                    chunk.metadata.get("date"),
                    Vector(vector),
                ),
            )
    conn.commit()


def run_embedding(batch_size: int = 32) -> None:
    """
    Embed every chunk of every document in data/processed/*.json into
    pgvector, then build the HNSW index. Run after extraction (or in
    parallel with it -- embedding and extraction are independent branches
    of the same chunk stream, see README's architecture diagram).
    """
    import glob
    import json

    from src.ingest.chunk import chunk_document
    from src.schema import NormalizedDocument

    settings = get_settings()
    conn = get_connection(settings)

    total_chunks = 0
    for path in sorted(glob.glob("data/processed/*.json")):
        doc = NormalizedDocument(**json.loads(open(path).read()))
        chunks = chunk_document(doc)

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectors = embed_texts([c.text for c in batch])
            write_chunks(conn, batch, vectors)

        total_chunks += len(chunks)
        print(f"[embedded] {doc.document_id}: {len(chunks)} chunks")

    print("Building HNSW index...")
    build_index(conn)
    conn.close()
    print(f"=== Embedding complete: {total_chunks} chunks indexed ===")


if __name__ == "__main__":
    run_embedding()
