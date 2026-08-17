"""
Tests for src/retrieval/graph_query.py, against the real local Neo4j.

What these tests prove:
    - resolve_entity_name matches on canonical_name, on a stored alias,
      and on a normalized (suffix-stripped) form -- the three lookup
      paths a user's question text can hit.
    - one_hop, two_hop_chain, path_between, and aggregate_count each
      return the expected shape from a small hand-built graph.
    - An entity that doesn't resolve returns [] rather than raising.
"""

import pytest

from src.ingest.graph_writer import ensure_constraints, get_driver, write_entities, write_relationships
from src.ingest.resolve import ResolvedEntity
from src.retrieval.graph_query import GraphQueryPlan, execute_graph_query, resolve_entity_name
from src.schema import ExtractedRelationship

_LABEL = "TestQueryEntity"


@pytest.fixture
def graph(driver=None):
    """Build a small fixed graph: A COMPETES_WITH B, B SUPPLIES C, A -- (no direct edge) -- C."""
    d = get_driver()
    ensure_constraints(d, [_LABEL])

    a = ResolvedEntity(canonical_name="Alpha Corporation", entity_type=_LABEL, aliases=["Alpha", "ALPH"])
    b = ResolvedEntity(canonical_name="Beta Corporation", entity_type=_LABEL, aliases=["Beta"])
    c = ResolvedEntity(canonical_name="Gamma Corporation", entity_type=_LABEL, aliases=["Gamma"])
    write_entities(d, [a, b, c])

    raw_to_canonical = {
        (_LABEL, "Alpha Corporation"): "Alpha Corporation",
        (_LABEL, "Beta Corporation"): "Beta Corporation",
        (_LABEL, "Gamma Corporation"): "Gamma Corporation",
    }
    entity_type_by_raw = {k[1]: _LABEL for k in raw_to_canonical}

    rels = [
        ExtractedRelationship(
            relationship_type="COMPETES_WITH",
            source_entity="Alpha Corporation",
            target_entity="Beta Corporation",
            source_chunk_id="TEST_00001",
            confidence=0.9,
        ),
        ExtractedRelationship(
            relationship_type="SUPPLIES",
            source_entity="Beta Corporation",
            target_entity="Gamma Corporation",
            source_chunk_id="TEST_00002",
            confidence=0.9,
        ),
    ]
    write_relationships(d, rels, raw_to_canonical, entity_type_by_raw)

    yield d

    with d.session() as session:
        session.run(f"MATCH (n:{_LABEL}) DETACH DELETE n")
    d.close()


def test_resolve_entity_name_matches_canonical_name(graph):
    assert resolve_entity_name(graph, "Alpha Corporation") == "Alpha Corporation"


def test_resolve_entity_name_matches_alias_case_insensitive(graph):
    assert resolve_entity_name(graph, "alph") == "Alpha Corporation"


def test_resolve_entity_name_matches_normalized_suffix_variant(graph):
    # "Alpha Corp" isn't stored anywhere, but normalizes to the same form
    # as the stored alias "Alpha" once suffixes are stripped from both.
    assert resolve_entity_name(graph, "Alpha Corp") == "Alpha Corporation"


def test_resolve_entity_name_returns_none_for_unknown(graph):
    assert resolve_entity_name(graph, "Nonexistent Inc") is None


def test_one_hop_returns_direct_neighbor(graph):
    plan = GraphQueryPlan(query_type="one_hop", entity_names=["Alpha Corporation"])
    rows = execute_graph_query(graph, plan)
    neighbors = {r["neighbor"] for r in rows}
    assert neighbors == {"Beta Corporation"}


def test_one_hop_filters_by_relationship_type(graph):
    plan = GraphQueryPlan(
        query_type="one_hop", entity_names=["Beta Corporation"], relationship_type="SUPPLIES"
    )
    rows = execute_graph_query(graph, plan)
    assert {r["neighbor"] for r in rows} == {"Gamma Corporation"}


def test_two_hop_chain_reaches_result_via_intermediate(graph):
    plan = GraphQueryPlan(
        query_type="two_hop_chain",
        entity_names=["Alpha Corporation"],
        relationship_type="COMPETES_WITH",
        second_relationship_type="SUPPLIES",
    )
    rows = execute_graph_query(graph, plan)
    assert len(rows) == 1
    assert rows[0]["via"] == "Beta Corporation"
    assert rows[0]["result"] == "Gamma Corporation"


def test_path_between_finds_two_hop_path(graph):
    plan = GraphQueryPlan(query_type="path_between", entity_names=["Alpha Corporation", "Gamma Corporation"])
    rows = execute_graph_query(graph, plan)
    assert len(rows) == 1
    assert rows[0]["hops"] == 2
    assert rows[0]["node_path"] == ["Alpha Corporation", "Beta Corporation", "Gamma Corporation"]


def test_aggregate_count(graph):
    plan = GraphQueryPlan(
        query_type="aggregate_count", entity_names=["Alpha Corporation"], relationship_type="COMPETES_WITH"
    )
    rows = execute_graph_query(graph, plan)
    assert rows[0]["count"] == 1


def test_unresolvable_entity_returns_empty_list_not_error(graph):
    plan = GraphQueryPlan(query_type="one_hop", entity_names=["Totally Unknown Company"])
    assert execute_graph_query(graph, plan) == []
