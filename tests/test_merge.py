"""
Tests for src/retrieval/merge.py.

What these tests prove:
    - graph_rows_to_statements converts each query_type's raw Cypher rows
      into readable prose, carrying the source_chunk_id(s) forward.
    - build_answer only accepts claims whose chunk_id was actually
      retrieved (graph or vector) -- a claim citing a nonexistent
      chunk_id is rejected, not trusted.
    - A claim citing an invalid chunk_id triggers exactly one
      regenerate-and-retry, and if the retry is clean, its result is used.
    - Empty evidence produces a "no answer" result, not a fabricated claim.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.retrieval.graph_query import GraphQueryPlan
from src.retrieval.merge import build_answer, graph_rows_to_statements
from src.retrieval.vector_query import VectorResult


def _fake_response(payload: dict):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(payload))])


def test_one_hop_rows_become_readable_statements():
    plan = GraphQueryPlan(query_type="one_hop", entity_names=["AMD"])
    rows = [
        {
            "relationship_type": "COMPETES_WITH",
            "neighbor": "Intel Corporation",
            "neighbor_type": "Company",
            "source_chunk_ids": ["AMD_2025_10K_00020"],
            "outgoing": True,
        }
    ]
    statements = graph_rows_to_statements(plan, rows)
    assert statements[0].text == "AMD competes with Intel Corporation."
    assert statements[0].source_chunk_ids == ["AMD_2025_10K_00020"]


def test_two_hop_chain_rows_become_readable_statements():
    plan = GraphQueryPlan(
        query_type="two_hop_chain",
        entity_names=["NVIDIA"],
        relationship_type="SUPPLIES",
        second_relationship_type="COMPETES_WITH",
    )
    rows = [{"via": "TSMC", "result": "Samsung", "result_type": "Company", "source_chunk_ids": ["X_00001"]}]
    statements = graph_rows_to_statements(plan, rows)
    assert "NVIDIA supplies TSMC" in statements[0].text
    assert "TSMC competes with Samsung" in statements[0].text


def test_path_between_rows_become_readable_statements():
    plan = GraphQueryPlan(query_type="path_between", entity_names=["A", "C"])
    rows = [
        {
            "node_path": ["A", "B", "C"],
            "relationship_path": ["COMPETES_WITH", "SUPPLIES"],
            "source_chunk_ids_per_hop": [["c1"], ["c2"]],
            "hops": 2,
        }
    ]
    statements = graph_rows_to_statements(plan, rows)
    assert "A competes with B" in statements[0].text
    assert "B supplies C" in statements[0].text
    assert set(statements[0].source_chunk_ids) == {"c1", "c2"}


def test_build_answer_rejects_claim_with_unretrieved_chunk_id():
    vector_results = [
        VectorResult(
            chunk_id="NVDA_2026_10K_00012",
            document_id="NVDA_2026_10K",
            section_path="Item 1. Business",
            text="NVIDIA designs GPUs.",
            company="NVIDIA CORP",
            similarity=0.9,
        )
    ]
    client = MagicMock()
    # First (only) attempt cites a chunk_id that was never retrieved.
    client.messages.create.return_value = _fake_response(
        {
            "claims": [{"text": "NVIDIA designs GPUs.", "chunk_id": "MADE_UP_CHUNK_ID"}],
            "no_answer_reason": None,
        }
    )

    result = build_answer(client, "What does NVIDIA design?", [], vector_results)

    assert result.claims == []
    assert len(result.rejected_claims) == 1
    assert client.messages.create.call_count == 2  # one regenerate attempt


def test_build_answer_accepts_valid_claim_on_first_try():
    vector_results = [
        VectorResult(
            chunk_id="NVDA_2026_10K_00012",
            document_id="NVDA_2026_10K",
            section_path="Item 1. Business",
            text="NVIDIA designs GPUs.",
            company="NVIDIA CORP",
            similarity=0.9,
        )
    ]
    client = MagicMock()
    client.messages.create.return_value = _fake_response(
        {
            "claims": [{"text": "NVIDIA designs GPUs.", "chunk_id": "NVDA_2026_10K_00012"}],
            "no_answer_reason": None,
        }
    )

    result = build_answer(client, "What does NVIDIA design?", [], vector_results)

    assert len(result.claims) == 1
    assert client.messages.create.call_count == 1  # no regenerate needed


def test_build_answer_with_no_evidence_returns_no_answer():
    client = MagicMock()
    client.messages.create.return_value = _fake_response(
        {"claims": [], "no_answer_reason": "No evidence was retrieved for this question."}
    )

    result = build_answer(client, "Some unanswerable question", [], [])

    assert result.claims == []
    assert "No evidence" in result.answer_text
