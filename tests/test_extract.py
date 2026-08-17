"""
Tests for src/ingest/extract.py.

What these tests prove:
    - Ontology.load() correctly parses config/ontology.yaml into the
      prompt block and JSON schema Claude is constrained to.
    - extract_chunk() doesn't trust the model's source_chunk_id -- it's
      always set from the chunk being processed.
    - A dangling relationship (referencing an entity the model didn't
      also emit) is dropped rather than written as a broken edge.
    - Regression: a response truncated at max_tokens (stop_reason ==
      "max_tokens") is retried with a doubled token budget instead of
      being retried unchanged (which would fail identically) -- this is
      the exact bug that broke the first full-corpus extraction run
      (AMD_2025_10K_00013, "Unterminated string" from a truncated JSON
      response on an entity-dense chunk).
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.ingest.extract import Ontology, extract_chunk
from src.schema import Chunk


def _chunk(text: str = "AMD competes with Intel.") -> Chunk:
    return Chunk(
        chunk_id="TEST_2026_10K_00001",
        document_id="TEST_2026_10K",
        chunk_index=1,
        section_path="Item 1. Business",
        text=text,
        metadata={"company": "TEST CORP", "ticker": "TEST", "document_type": "10-K"},
    )


def _fake_response(payload: dict, stop_reason: str = "end_turn", in_tok: int = 100, out_tok: int = 50):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
    )


def test_ontology_loads_real_config():
    ontology = Ontology.load()
    assert "Company" in ontology.entity_types
    assert "COMPETES_WITH" in ontology.relationship_types
    schema = ontology.json_schema()
    assert schema["properties"]["entities"]["items"]["properties"]["entity_type"]["enum"] == ontology.entity_types


def test_extract_chunk_sets_source_chunk_id_from_chunk_not_model():
    chunk = _chunk()
    ontology = Ontology.load()
    payload = {
        "entities": [{"entity_type": "Company", "raw_name": "TEST CORP", "confidence": 0.9}],
        "relationships": [],
    }
    client = MagicMock()
    client.messages.create.return_value = _fake_response(payload)

    result = extract_chunk(client, chunk, ontology, use_cache=False)
    assert result.entities[0].source_chunk_id == chunk.chunk_id


def test_extract_chunk_drops_dangling_relationship():
    chunk = _chunk()
    ontology = Ontology.load()
    payload = {
        "entities": [{"entity_type": "Company", "raw_name": "TEST CORP", "confidence": 0.9}],
        "relationships": [
            {
                "relationship_type": "COMPETES_WITH",
                "source_entity": "TEST CORP",
                "target_entity": "Ghost Company",  # never emitted as an entity
                "confidence": 0.8,
            }
        ],
    }
    client = MagicMock()
    client.messages.create.return_value = _fake_response(payload)

    result = extract_chunk(client, chunk, ontology, use_cache=False)
    assert result.relationships == []


def test_extract_chunk_retries_with_doubled_budget_on_truncation():
    chunk = _chunk()
    ontology = Ontology.load()
    good_payload = {
        "entities": [{"entity_type": "Company", "raw_name": "TEST CORP", "confidence": 0.9}],
        "relationships": [],
    }
    client = MagicMock()
    # First call: truncated (stop_reason max_tokens, unparseable partial JSON).
    # Second call: succeeds.
    client.messages.create.side_effect = [
        SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"entities": [{"entity_type": "Compan')],
            stop_reason="max_tokens",
            usage=SimpleNamespace(input_tokens=100, output_tokens=8192),
        ),
        _fake_response(good_payload),
    ]

    result = extract_chunk(client, chunk, ontology, use_cache=False)

    assert client.messages.create.call_count == 2
    first_call_tokens = client.messages.create.call_args_list[0].kwargs["max_tokens"]
    second_call_tokens = client.messages.create.call_args_list[1].kwargs["max_tokens"]
    assert second_call_tokens == first_call_tokens * 2
    assert result.entities[0].raw_name == "TEST CORP"
