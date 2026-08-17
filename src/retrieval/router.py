"""
Query router: classify a question as GRAPH / VECTOR / BOTH.

Purpose:
    Route connection questions, multi-hop chains, comparisons, and
    aggregations over relationships to the graph. Route definitions,
    policy lookups, and single-fact questions to vector search.

Design decisions:
    - A cheap model call (Haiku), not Sonnet/Opus -- routing is a
      classification task, not a reasoning task, and it runs on every
      single question. Few-shot examples in the prompt, structured
      output constrained to the fixed enum (GRAPH/VECTOR/BOTH) plus the
      graph-path fields (query_type, entity_names, relationship_type(s),
      max_hops) needed to build a GraphQueryPlan without a second model
      call.
    - Low-confidence fallback: below CONFIDENCE_THRESHOLD, force route to
      BOTH rather than trusting a shaky classification -- cheaper to run
      both paths and merge than to silently answer from the wrong one.
    - Every routing decision is logged (question, route, entities, query
      plan, confidence, latency) to data/cache/routing_log.jsonl. This is
      required, not optional -- the build guide is explicit that Phase 5
      needs this data to reconstruct router accuracy after the fact, and
      there's no other way to recover "what did the router decide for
      question N" once the benchmark run is over.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import anthropic

from src.retrieval.graph_query import QUERY_TYPES

Route = Literal["GRAPH", "VECTOR", "BOTH"]

CONFIDENCE_THRESHOLD = 0.6
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ROUTING_LOG_PATH = _PROJECT_ROOT / "data" / "cache" / "routing_log.jsonl"

_RELATIONSHIP_TYPES = [
    "COMPETES_WITH",
    "ACQUIRED",
    "SUBSIDIARY_OF",
    "SUPPLIES",
    "CUSTOMER_OF",
    "PARTNERS_WITH",
    "DEPENDS_ON",
    "REGULATED_BY",
    "OFFERS_PRODUCT",
    "PART_OF_SEGMENT",
    "HAS_SEGMENT",
    "OFFICER_OF",
]

SYSTEM_PROMPT = f"""You route a user's question about SEC 10-K filings to the retrieval path that can answer it.

ROUTES:
- VECTOR: definitions, policy/fact lookups, "what does X say about Y" questions answerable from a single passage of text.
- GRAPH: connection questions, comparisons across named entities, multi-hop chains ("who competes with X's suppliers"), and aggregations ("how many companies does X compete with").
- BOTH: the question could benefit from both, or you are not confident which single path is correct.

If routing to GRAPH or BOTH, also emit a graph query plan:
- query_type: one of {QUERY_TYPES}
  - "one_hop": direct relationships of ONE entity, optionally filtered by relationship_type
  - "two_hop_chain": ONE entity -> relationship_type -> intermediate -> second_relationship_type -> result
  - "path_between": how TWO named entities are connected, within max_hops
  - "aggregate_count": count of relationships of a given relationship_type from ONE entity
- entity_names: the company/entity names mentioned in the question, exactly as the user wrote them (1 name for one_hop/two_hop_chain/aggregate_count, 2 for path_between)
- relationship_type / second_relationship_type: one of {_RELATIONSHIP_TYPES}, or null if not applicable
- max_hops: integer, only meaningful for path_between (default 3)

If the question is purely VECTOR, still fill entity_names with any named entities mentioned (useful for the merge step later), and leave the other graph fields null.

EXAMPLES:
Q: "What does NVIDIA's Compute & Networking segment include?"
-> route=VECTOR, entity_names=["NVIDIA"]

Q: "Which companies does AMD compete with in the Data Center segment?"
-> route=GRAPH, query_type=one_hop, entity_names=["AMD"], relationship_type=COMPETES_WITH

Q: "What company did NVIDIA acquire in 2020?"
-> route=GRAPH, query_type=one_hop, entity_names=["NVIDIA"], relationship_type=ACQUIRED
(Not VECTOR: "what did X acquire" is a relationship lookup even though it sounds like a single fact -- ACQUIRED is a first-class relationship type, so treat any acquirer/acquired/subsidiary question as GRAPH.)

Q: "How is Google connected to Intel?"
-> route=GRAPH, query_type=path_between, entity_names=["Google", "Intel"], max_hops=3

Q: "Who competes with NVIDIA's suppliers?"
-> route=GRAPH, query_type=two_hop_chain, entity_names=["NVIDIA"], relationship_type=SUPPLIES, second_relationship_type=COMPETES_WITH

Q: "How many regulatory bodies oversee Amazon?"
-> route=GRAPH, query_type=aggregate_count, entity_names=["Amazon"], relationship_type=REGULATED_BY

Q: "What is NVIDIA's revenue growth strategy and who are its main rivals?"
-> route=BOTH, entity_names=["NVIDIA"], query_type=one_hop, relationship_type=COMPETES_WITH"""

_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["GRAPH", "VECTOR", "BOTH"]},
        "confidence": {"type": "number"},
        "entity_names": {"type": "array", "items": {"type": "string"}},
        "query_type": {"anyOf": [{"type": "string", "enum": QUERY_TYPES}, {"type": "null"}]},
        "relationship_type": {"anyOf": [{"type": "string", "enum": _RELATIONSHIP_TYPES}, {"type": "null"}]},
        "second_relationship_type": {
            "anyOf": [{"type": "string", "enum": _RELATIONSHIP_TYPES}, {"type": "null"}]
        },
        "max_hops": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
    },
    "required": [
        "route",
        "confidence",
        "entity_names",
        "query_type",
        "relationship_type",
        "second_relationship_type",
        "max_hops",
    ],
    "additionalProperties": False,
}


@dataclass
class RouteDecision:
    route: Route
    confidence: float
    entity_names: list[str]
    query_type: str | None = None
    relationship_type: str | None = None
    second_relationship_type: str | None = None
    max_hops: int | None = None
    forced_both_low_confidence: bool = False


def _log_decision(question: str, decision: RouteDecision, latency_ms: float) -> None:
    ROUTING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ROUTING_LOG_PATH, "a") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": time.time(),
                    "question": question,
                    "latency_ms": latency_ms,
                    **asdict(decision),
                }
            )
            + "\n"
        )


def classify_route(
    client: anthropic.Anthropic,
    question: str,
    model: str = "claude-haiku-4-5",
) -> RouteDecision:
    """Classify a question's route, log the decision, and return it."""
    start = time.monotonic()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": _JSON_SCHEMA}},
        messages=[{"role": "user", "content": question}],
    )
    latency_ms = (time.monotonic() - start) * 1000
    raw = json.loads(next(b.text for b in response.content if b.type == "text"))

    decision = RouteDecision(
        route=raw["route"],
        confidence=raw["confidence"],
        entity_names=raw["entity_names"],
        query_type=raw["query_type"],
        relationship_type=raw["relationship_type"],
        second_relationship_type=raw["second_relationship_type"],
        max_hops=raw["max_hops"],
    )

    if decision.confidence < CONFIDENCE_THRESHOLD and decision.route != "BOTH":
        decision.route = "BOTH"
        decision.forced_both_low_confidence = True

    _log_decision(question, decision, latency_ms)
    return decision
