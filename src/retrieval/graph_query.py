"""
Graph query templates and execution.

Purpose:
    Answer questions that require traversing the knowledge graph --
    connections, multi-hop chains, comparisons across entities, and
    aggregations over relationships.

Security / reproducibility (Rule from the build guide):
    The LLM NEVER emits raw Cypher against the database. Claude only
    picks a query_type from a fixed enum and fills typed parameters
    (entity names, an optional relationship_type, max_hops) into one of
    the parameterized templates below. This is a deliberate security
    boundary (no Cypher injection surface) and it also makes results
    reproducible -- the same query_type + parameters always produces the
    same Cypher, so results can be re-run and compared, unlike freeform
    model-generated queries.

Template coverage (mapped to the Phase 5 benchmark's hop-count strata):
    - one_hop            -> single-hop questions ("who competes with NVIDIA?")
    - two_hop_chain       -> two-hop questions via a specific relationship
                             chain ("who competes with NVIDIA's suppliers?")
    - path_between        -> open-ended multi-hop connection questions
                             ("how are Google and Intel connected?"),
                             bounded by max_hops (supports 2-hop and 3-hop)
    - aggregate_count      -> aggregation questions ("how many companies
                             does NVIDIA compete with?")

Entity resolution at query time:
    A question references entities by whatever surface form the user
    typed ("Nvidia", "NVIDIA Corp", "the company"). resolve_entity_name()
    looks the name up against canonical_name and the stored aliases list
    first (fast, exact/case-insensitive), then falls back to the same
    normalize_name() used at ingestion time (strips corporate suffixes).
    This intentionally does NOT do an embedding-similarity fallback at
    query time (unlike ingestion-time resolution in resolve.py) --
    keeping query-time resolution fast and deterministic matters more
    here than catching every possible paraphrase; see README stretch
    goals for a documented future improvement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from neo4j import Driver

from src.ingest.resolve import normalize_name

QueryType = Literal["one_hop", "two_hop_chain", "path_between", "aggregate_count"]

QUERY_TYPES: list[str] = ["one_hop", "two_hop_chain", "path_between", "aggregate_count"]


@dataclass
class GraphQueryPlan:
    query_type: QueryType
    entity_names: list[str]  # 1 entity for one_hop/aggregate_count/two_hop_chain; 2 for path_between
    relationship_type: str | None = None  # required for one_hop filter, two_hop_chain (first hop), aggregate_count
    second_relationship_type: str | None = None  # two_hop_chain only (second hop)
    max_hops: int = 3  # path_between only


def resolve_entity_name(driver: Driver, surface_form: str) -> str | None:
    """
    Resolve a user-typed entity name to a canonical_name already in the
    graph. Returns None if nothing matches (the caller should then treat
    the question as out-of-scope for the graph path rather than guessing).
    """
    with driver.session() as session:
        exact = session.run(
            "MATCH (n) WHERE toLower(n.canonical_name) = toLower($name) "
            "OR any(alias IN coalesce(n.aliases, []) WHERE toLower(alias) = toLower($name)) "
            "RETURN n.canonical_name AS canonical_name LIMIT 1",
            name=surface_form,
        ).single()
        if exact:
            return exact["canonical_name"]

        # Fallback: compare normalized forms (strips corporate suffixes),
        # since a node's canonical_name/aliases might all carry a suffix
        # the user's phrasing dropped (e.g. user says "Intel", node aliases
        # are all "Intel Corporation" / "Intel Corp").
        target_norm = normalize_name(surface_form)
        candidates = session.run(
            "MATCH (n) RETURN n.canonical_name AS canonical_name, coalesce(n.aliases, []) AS aliases"
        )
        for record in candidates:
            names = [record["canonical_name"], *record["aliases"]]
            if any(normalize_name(name) == target_norm for name in names):
                return record["canonical_name"]
    return None


# --- Parameterized Cypher templates -----------------------------------------
# Every template takes only bound parameters ($name, $rel_type, $hops) --
# never string-interpolated user or model input. relationship_type/
# second_relationship_type ARE interpolated into the type pattern, but only
# because they are validated against config/ontology.yaml's fixed enum
# before this function is ever called (see router.py's structured output
# schema) -- never pass an unvalidated string here.

_ONE_HOP_ALL = """
MATCH (source {canonical_name: $name})-[r]-(neighbor)
RETURN type(r) AS relationship_type, neighbor.canonical_name AS neighbor,
       labels(neighbor)[0] AS neighbor_type,
       [x IN r.source_chunk_ids] AS source_chunk_ids,
       startNode(r).canonical_name = $name AS outgoing
"""

_ONE_HOP_TYPED = """
MATCH (source {{canonical_name: $name}})-[r:{rel_type}]-(neighbor)
RETURN type(r) AS relationship_type, neighbor.canonical_name AS neighbor,
       labels(neighbor)[0] AS neighbor_type,
       [x IN r.source_chunk_ids] AS source_chunk_ids,
       startNode(r).canonical_name = $name AS outgoing
"""

_TWO_HOP_CHAIN = """
MATCH (source {{canonical_name: $name}})-[r1:{rel_type_1}]-(mid)-[r2:{rel_type_2}]-(result)
WHERE result.canonical_name <> $name
RETURN DISTINCT mid.canonical_name AS via, result.canonical_name AS result,
       labels(result)[0] AS result_type,
       [x IN r1.source_chunk_ids] + [x IN r2.source_chunk_ids] AS source_chunk_ids
"""

_PATH_BETWEEN = """
MATCH path = shortestPath((a {{canonical_name: $name_a}})-[*..{max_hops}]-(b {{canonical_name: $name_b}}))
RETURN [n IN nodes(path) | n.canonical_name] AS node_path,
       [r IN relationships(path) | type(r)] AS relationship_path,
       [r IN relationships(path) | [x IN r.source_chunk_ids]] AS source_chunk_ids_per_hop,
       length(path) AS hops
"""

_AGGREGATE_COUNT = """
MATCH (source {{canonical_name: $name}})-[r:{rel_type}]-(neighbor)
RETURN count(DISTINCT neighbor) AS count, collect(DISTINCT neighbor.canonical_name) AS neighbors
"""


def execute_graph_query(driver: Driver, plan: GraphQueryPlan) -> list[dict]:
    """
    Resolve entity_names to canonical graph names, then run the template
    matching plan.query_type. Returns [] (not an error) if an entity
    can't be resolved -- an unresolvable entity means this question is
    out-of-scope for the graph, which the router/merge layer should treat
    as "no graph evidence" rather than a crash.
    """
    resolved = [resolve_entity_name(driver, name) for name in plan.entity_names]
    if any(r is None for r in resolved):
        return []

    with driver.session() as session:
        if plan.query_type == "one_hop":
            query = (
                _ONE_HOP_TYPED.format(rel_type=plan.relationship_type)
                if plan.relationship_type
                else _ONE_HOP_ALL
            )
            result = session.run(query, name=resolved[0])

        elif plan.query_type == "two_hop_chain":
            if not (plan.relationship_type and plan.second_relationship_type):
                raise ValueError("two_hop_chain requires relationship_type and second_relationship_type")
            query = _TWO_HOP_CHAIN.format(
                rel_type_1=plan.relationship_type, rel_type_2=plan.second_relationship_type
            )
            result = session.run(query, name=resolved[0])

        elif plan.query_type == "path_between":
            if len(resolved) != 2:
                raise ValueError("path_between requires exactly 2 entity_names")
            query = _PATH_BETWEEN.format(max_hops=plan.max_hops)
            result = session.run(query, name_a=resolved[0], name_b=resolved[1])

        elif plan.query_type == "aggregate_count":
            if not plan.relationship_type:
                raise ValueError("aggregate_count requires relationship_type")
            query = _AGGREGATE_COUNT.format(rel_type=plan.relationship_type)
            result = session.run(query, name=resolved[0])

        else:
            raise ValueError(f"Unknown query_type: {plan.query_type}")

        return [record.data() for record in result]
