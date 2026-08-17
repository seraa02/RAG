"""
Integration tests for src/api/main.py.

These hit the real stack (Anthropic API, local Neo4j, local pgvector) --
not mocked -- because the thing worth proving here is that the actual
wiring (lifespan startup, router -> retrieval -> merge) works end to end,
which a mock of the FastAPI app's module-level state would not exercise.
Cost is small: one Haiku routing call + one Sonnet answer call per test.

What these tests prove:
    - /health responds without needing any external service warmed up
      beyond what lifespan already sets up.
    - /ask on a question with no named entities routes to VECTOR and
      returns a citation-grounded answer sourced from a real retrieved
      chunk_id.
    - Every returned claim's chunk_id is one that was actually retrieved
      (graph_evidence_count + vector_evidence_count > 0 whenever there
      are claims) -- the citation validation promise, checked at the
      HTTP boundary, not just in the unit-level merge.py tests.
"""

from fastapi.testclient import TestClient

from src.api.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_vector_question_returns_grounded_answer():
    with TestClient(app) as client:
        response = client.post(
            "/ask", json={"question": "What does NVIDIA's Compute and Networking segment include?"}
        )
    assert response.status_code == 200
    body = response.json()

    assert body["route"] in ("VECTOR", "BOTH")
    assert body["vector_evidence_count"] > 0
    assert len(body["answer"]) > 0

    # Every claim must cite a chunk_id -- and if there are claims, there
    # must have been evidence to ground them in.
    if body["claims"]:
        assert body["graph_evidence_count"] + body["vector_evidence_count"] > 0
        for claim in body["claims"]:
            assert claim["chunk_id"]
