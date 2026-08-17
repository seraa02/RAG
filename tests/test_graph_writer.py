"""
Tests for src/ingest/graph_writer.py.

These run against the real local Neo4j instance (docker-compose's
rag_neo4j) rather than a mock -- MERGE idempotency is a database-level
guarantee we actually want to verify against the real query engine, not
assume. A dedicated "TestEntity" label (not part of the real ontology)
keeps this test's writes isolated from real corpus data, and the fixture
tears down everything it created.

What these tests prove:
    - Writing the same entity twice (write_entities called twice with the
      same canonical_name) produces exactly one node, not two -- this is
      the guide's explicit "same company exists as four disconnected
      nodes" failure mode, prevented at the database level via the
      uniqueness constraint + MERGE.
    - Writing the same relationship from two different source chunks
      merges into ONE edge whose source_chunk_ids list accumulates both
      chunk ids, rather than creating two edges for one real-world fact.
"""

import pytest

from src.ingest.graph_writer import (
    ensure_constraints,
    get_driver,
    write_entities,
    write_relationships,
)
from src.ingest.resolve import ResolvedEntity
from src.schema import ExtractedRelationship

_TEST_LABEL = "TestEntity"


@pytest.fixture
def driver():
    d = get_driver()
    yield d
    with d.session() as session:
        session.run(f"MATCH (n:{_TEST_LABEL}) DETACH DELETE n")
    d.close()


def test_write_entities_is_idempotent(driver):
    ensure_constraints(driver, [_TEST_LABEL])
    entity = ResolvedEntity(canonical_name="Acme Corporation", entity_type=_TEST_LABEL, aliases=["ACME"])

    write_entities(driver, [entity])
    write_entities(driver, [entity])  # re-run: must not duplicate

    with driver.session() as session:
        count = session.run(
            f"MATCH (n:{_TEST_LABEL} {{canonical_name: 'Acme Corporation'}}) RETURN count(n) AS c"
        ).single()["c"]
    assert count == 1


def test_write_relationships_merges_and_accumulates_source_chunks(driver):
    ensure_constraints(driver, [_TEST_LABEL])
    source = ResolvedEntity(canonical_name="Acme Corporation", entity_type=_TEST_LABEL, aliases=[])
    target = ResolvedEntity(canonical_name="Widget Co", entity_type=_TEST_LABEL, aliases=[])
    write_entities(driver, [source, target])

    raw_to_canonical = {
        (_TEST_LABEL, "Acme Corporation"): "Acme Corporation",
        (_TEST_LABEL, "Widget Co"): "Widget Co",
    }
    entity_type_by_raw = {"Acme Corporation": _TEST_LABEL, "Widget Co": _TEST_LABEL}

    rel_from_chunk_1 = ExtractedRelationship(
        relationship_type="COMPETES_WITH",
        source_entity="Acme Corporation",
        target_entity="Widget Co",
        source_chunk_id="TEST_2026_10K_00001",
        confidence=0.9,
    )
    rel_from_chunk_2 = ExtractedRelationship(
        relationship_type="COMPETES_WITH",
        source_entity="Acme Corporation",
        target_entity="Widget Co",
        source_chunk_id="TEST_2026_10K_00002",
        confidence=0.8,
    )

    write_relationships(driver, [rel_from_chunk_1], raw_to_canonical, entity_type_by_raw)
    write_relationships(driver, [rel_from_chunk_2], raw_to_canonical, entity_type_by_raw)

    with driver.session() as session:
        record = session.run(
            f"MATCH (:{_TEST_LABEL} {{canonical_name: 'Acme Corporation'}})"
            f"-[r:COMPETES_WITH]->(:{_TEST_LABEL} {{canonical_name: 'Widget Co'}}) "
            f"RETURN count(r) AS edge_count, r.source_chunk_ids AS chunk_ids, r.confidence AS confidence"
        ).single()

    assert record["edge_count"] == 1
    assert set(record["chunk_ids"]) == {"TEST_2026_10K_00001", "TEST_2026_10K_00002"}
    assert record["confidence"] == 0.9  # max() of the two confidences


def test_write_relationships_skips_unresolved_endpoints(driver):
    ensure_constraints(driver, [_TEST_LABEL])
    rel = ExtractedRelationship(
        relationship_type="COMPETES_WITH",
        source_entity="Nobody",
        target_entity="Nothing",
        source_chunk_id="TEST_2026_10K_00003",
        confidence=0.5,
    )
    written = write_relationships(driver, [rel], raw_to_canonical={}, entity_type_by_raw={})
    assert written == 0
