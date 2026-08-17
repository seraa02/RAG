"""
Tests for src/retrieval/router.py.

What these tests prove:
    - A low-confidence classification is forced to BOTH rather than
      trusted as-is (the guide's explicit "low confidence fallback that
      runs both paths and merges").
    - A high-confidence classification is NOT overridden.
    - Every call appends one line to the routing log -- required for
      Phase 5 to reconstruct router accuracy after the benchmark run.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.retrieval import router


def _fake_response(payload: dict):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(payload))])


def _base_payload(**overrides):
    payload = {
        "route": "VECTOR",
        "confidence": 0.9,
        "entity_names": ["NVIDIA"],
        "query_type": None,
        "relationship_type": None,
        "second_relationship_type": None,
        "max_hops": None,
    }
    payload.update(overrides)
    return payload


def test_low_confidence_forces_both(tmp_path, monkeypatch):
    monkeypatch.setattr(router, "ROUTING_LOG_PATH", tmp_path / "routing_log.jsonl")
    client = MagicMock()
    client.messages.create.return_value = _fake_response(_base_payload(route="VECTOR", confidence=0.4))

    decision = router.classify_route(client, "some ambiguous question")

    assert decision.route == "BOTH"
    assert decision.forced_both_low_confidence is True


def test_high_confidence_is_not_overridden(tmp_path, monkeypatch):
    monkeypatch.setattr(router, "ROUTING_LOG_PATH", tmp_path / "routing_log.jsonl")
    client = MagicMock()
    client.messages.create.return_value = _fake_response(_base_payload(route="GRAPH", confidence=0.95))

    decision = router.classify_route(client, "who competes with NVIDIA?")

    assert decision.route == "GRAPH"
    assert decision.forced_both_low_confidence is False


def test_every_call_is_logged(tmp_path, monkeypatch):
    log_path = tmp_path / "routing_log.jsonl"
    monkeypatch.setattr(router, "ROUTING_LOG_PATH", log_path)
    client = MagicMock()
    client.messages.create.return_value = _fake_response(_base_payload())

    router.classify_route(client, "question one")
    router.classify_route(client, "question two")

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    logged = [json.loads(line) for line in lines]
    assert logged[0]["question"] == "question one"
    assert logged[1]["question"] == "question two"
    assert "latency_ms" in logged[0]
