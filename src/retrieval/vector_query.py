"""
Vector (pgvector) retrieval.

Purpose:
    Answer definition, policy-lookup, and single-fact questions via
    semantic similarity search over the chunks embedded by
    src/ingest/embed.py.

Design decisions:
    - BGE's asymmetric instruction convention (see embed.py's docstring):
      passages were embedded WITHOUT a prefix; the query gets
      "Represent this sentence for searching relevant passages: "
      prepended before embedding. Getting this backwards doesn't error --
      it just silently degrades recall, so the prefix is centralized here
      as the one place a query is ever embedded for retrieval.
    - Cosine distance via pgvector's <=> operator, matching the HNSW
      index built with vector_cosine_ops in embed.py. Similarity score
      returned to callers is 1 - distance (so higher = more similar,
      matching how recall@k and citation-confidence thresholds are
      normally reasoned about).
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg
from pgvector import Vector

from src.ingest.embed import _get_model, get_connection

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@dataclass
class VectorResult:
    chunk_id: str
    document_id: str
    section_path: str
    text: str
    company: str | None
    similarity: float


def embed_query(query_text: str) -> list[float]:
    """Embed a query with the BGE query-side instruction prefix (see module docstring)."""
    model = _get_model()
    vector = model.encode(QUERY_INSTRUCTION + query_text, normalize_embeddings=True, show_progress_bar=False)
    return vector.tolist()


def vector_search(
    conn: psycopg.Connection,
    query_text: str,
    k: int = 5,
    min_similarity: float = 0.0,
) -> list[VectorResult]:
    """Top-k chunks by cosine similarity to query_text. min_similarity filters weak matches."""
    # Wrapped explicitly in pgvector.Vector -- register_vector(conn) only
    # auto-adapts numpy.ndarray/Vector to the `vector` type, not a plain
    # Python list (see embed.py's write_chunks, which relies on Postgres's
    # implicit array->vector cast in an INSERT's column-typed context;
    # that context doesn't exist for a bare `<=>` comparison, so it must
    # be explicit here).
    query_vector = Vector(embed_query(query_text))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_id, document_id, section_path, text, company,
                   1 - (embedding <=> %s) AS similarity
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_vector, query_vector, k),
        )
        rows = cur.fetchall()

    return [
        VectorResult(
            chunk_id=row[0],
            document_id=row[1],
            section_path=row[2],
            text=row[3],
            company=row[4],
            similarity=row[5],
        )
        for row in rows
        if row[5] >= min_similarity
    ]


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What does NVIDIA's Data Center segment include?"
    conn = get_connection()
    for r in vector_search(conn, question, k=5):
        print(f"[{r.similarity:.3f}] {r.chunk_id} ({r.section_path}, {r.company})")
        print(f"   {r.text[:160]}...")
    conn.close()
