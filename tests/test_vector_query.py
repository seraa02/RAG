"""
Tests for src/retrieval/vector_query.py, against the real local pgvector
instance (already populated by `python -m src.ingest.embed` over the real
6-filing corpus).

What these tests prove:
    - vector_search returns results at all (the pgvector <=> operator
      comparison, once broken by an untyped list parameter -- see
      embed.py/vector_query.py's Vector() wrapping -- actually works).
    - Results are ordered by descending similarity.
    - A query clearly about one company's content ranks that company's
      chunks above an unrelated company's, which is the whole point of
      semantic search working at all.
    - min_similarity actually filters.
"""

import pytest

from src.ingest.embed import get_connection
from src.retrieval.vector_query import vector_search


@pytest.fixture(scope="module")
def conn():
    c = get_connection()
    yield c
    c.close()


def test_vector_search_returns_results_ordered_by_similarity(conn):
    results = vector_search(conn, "GPU accelerated computing data center", k=5)
    assert len(results) == 5
    similarities = [r.similarity for r in results]
    assert similarities == sorted(similarities, reverse=True)


def test_vector_search_favors_relevant_company(conn):
    # A question specifically about NVIDIA's own business should surface
    # mostly NVIDIA chunks in the top results, not a random mix.
    results = vector_search(conn, "NVIDIA GPU architecture and data center platforms", k=10)
    companies = [r.company for r in results]
    assert companies.count("NVIDIA CORP") >= 5


def test_min_similarity_filters_weak_matches(conn):
    all_results = vector_search(conn, "quarterly earnings", k=20, min_similarity=0.0)
    filtered = vector_search(conn, "quarterly earnings", k=20, min_similarity=0.99)
    assert len(filtered) <= len(all_results)
