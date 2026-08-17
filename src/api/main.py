"""
FastAPI entrypoint: question in, citation-grounded answer out.

Purpose:
    Tie together router -> retrieval (graph and/or vector) -> merge ->
    citation-validated answer as a single HTTP endpoint. This is the
    "Done when" artifact for Phase 4/5: a FastAPI endpoint that answers
    questions with validated citations.

Design decisions:
    - Connections (Anthropic client, Neo4j driver, Postgres connection)
      are created once at app startup via FastAPI's lifespan context, not
      per-request -- opening a fresh Neo4j driver or Postgres connection
      on every request would add real latency and eventually exhaust
      connection limits under load.
    - The endpoint always runs the router first, then only the retrieval
      path(s) the router chose (GRAPH, VECTOR, or BOTH) -- it does not
      unconditionally run both paths on every request, which would double
      cost and latency for the common case.
    - Response includes route, graph/vector evidence counts, and any
      rejected_claims -- not just the final answer text -- because the
      whole point of this project (per the guide) is that the benchmark
      and any debugging session needs to see *why* an answer was
      produced, not just what it says.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import anthropic
from fastapi import FastAPI
from pydantic import BaseModel

from src.config import get_settings
from src.ingest.embed import get_connection as get_pg_connection
from src.ingest.graph_writer import get_driver
from src.retrieval.graph_query import GraphQueryPlan, execute_graph_query
from src.retrieval.merge import build_answer, graph_rows_to_statements
from src.retrieval.router import classify_route
from src.retrieval.vector_query import vector_search

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _state["anthropic_client"] = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    _state["neo4j_driver"] = get_driver(settings)
    _state["pg_conn"] = get_pg_connection(settings)
    yield
    _state["neo4j_driver"].close()
    _state["pg_conn"].close()


app = FastAPI(title="Graph RAG over SEC Filings", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    vector_k: int = 5


class ClaimOut(BaseModel):
    text: str
    chunk_id: str


class AskResponse(BaseModel):
    question: str
    answer: str
    route: str
    forced_both_low_confidence: bool
    claims: list[ClaimOut]
    rejected_claim_count: int
    graph_evidence_count: int
    vector_evidence_count: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    client: anthropic.Anthropic = _state["anthropic_client"]
    driver = _state["neo4j_driver"]
    pg_conn = _state["pg_conn"]

    decision = classify_route(client, request.question)

    graph_statements = []
    if decision.route in ("GRAPH", "BOTH") and decision.query_type:
        plan = GraphQueryPlan(
            query_type=decision.query_type,  # type: ignore[arg-type]
            entity_names=decision.entity_names,
            relationship_type=decision.relationship_type,
            second_relationship_type=decision.second_relationship_type,
            max_hops=decision.max_hops or 3,
        )
        rows = execute_graph_query(driver, plan)
        graph_statements = graph_rows_to_statements(plan, rows)

    vector_results = []
    if decision.route in ("VECTOR", "BOTH"):
        vector_results = vector_search(pg_conn, request.question, k=request.vector_k)

    merged = build_answer(client, request.question, graph_statements, vector_results)

    return AskResponse(
        question=request.question,
        answer=merged.answer_text,
        route=decision.route,
        forced_both_low_confidence=decision.forced_both_low_confidence,
        claims=[ClaimOut(text=c.text, chunk_id=c.chunk_id) for c in merged.claims],
        rejected_claim_count=len(merged.rejected_claims),
        graph_evidence_count=merged.graph_evidence_count,
        vector_evidence_count=merged.vector_evidence_count,
    )
