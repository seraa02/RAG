"""
Evidence merge and citation-grounded answering.

Purpose:
    Combine graph traversal results and vector search results into one
    context, then produce an answer where every claim carries a citation
    that has been validated to actually resolve to a retrieved chunk.

Design decisions:
    - Graph rows are converted into readable statements BEFORE they reach
      the prompt. A raw Cypher result row is not readable prose, and
      feeding raw triples/paths to the model produces awkward, sometimes
      wrong-sounding text (the guide's explicit warning). Each statement
      keeps the source_chunk_id(s) that supported it.
    - Context is assembled with explicit labels separating graph-derived
      facts from retrieved passages -- the model should not confuse
      "NVIDIA COMPETES_WITH AMD (from the graph)" with "a sentence that
      happens to mention both companies (from a vector passage)".
    - The model does not free-write an answer with citations bolted on.
      It returns a list of {text, chunk_id} claims via structured
      outputs -- every claim is forced to name exactly one chunk_id it's
      grounded in. This makes citation validation mechanical: a claim
      whose chunk_id was never actually retrieved is rejected and the
      whole answer is regenerated once, rather than trusting the model's
      citation to be honest.
    - "No answer" is a valid, expected output when retrieval turned up
      nothing relevant -- the model is explicitly told to say so rather
      than fabricate a citation to force an answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic

from src.retrieval.graph_query import GraphQueryPlan
from src.retrieval.vector_query import VectorResult

# Human-readable phrasing per relationship type, for turning graph rows
# into prose before they reach the prompt.
_RELATIONSHIP_PHRASING = {
    "COMPETES_WITH": "competes with",
    "ACQUIRED": "acquired",
    "SUBSIDIARY_OF": "is a subsidiary of",
    "SUPPLIES": "supplies",
    "CUSTOMER_OF": "is a customer of",
    "PARTNERS_WITH": "partners with",
    "DEPENDS_ON": "depends on",
    "REGULATED_BY": "is regulated by",
    "OFFERS_PRODUCT": "offers the product",
    "PART_OF_SEGMENT": "is part of the segment",
    "HAS_SEGMENT": "has the business segment",
    "OFFICER_OF": "is an officer of",
}


@dataclass
class GraphStatement:
    text: str
    source_chunk_ids: list[str]


@dataclass
class Claim:
    text: str
    chunk_id: str


@dataclass
class MergedAnswer:
    answer_text: str
    claims: list[Claim]
    rejected_claims: list[dict]  # claims dropped for citing an unretrieved chunk_id
    graph_evidence_count: int
    vector_evidence_count: int


def graph_rows_to_statements(plan: GraphQueryPlan, rows: list[dict]) -> list[GraphStatement]:
    """Convert raw Cypher result rows into readable prose statements."""
    statements: list[GraphStatement] = []

    if plan.query_type == "one_hop":
        for row in rows:
            verb = _RELATIONSHIP_PHRASING.get(row["relationship_type"], row["relationship_type"].lower())
            subject, obj = (
                (plan.entity_names[0], row["neighbor"]) if row["outgoing"] else (row["neighbor"], plan.entity_names[0])
            )
            statements.append(
                GraphStatement(text=f"{subject} {verb} {obj}.", source_chunk_ids=row["source_chunk_ids"])
            )

    elif plan.query_type == "two_hop_chain":
        rel1 = _RELATIONSHIP_PHRASING.get(plan.relationship_type, plan.relationship_type.lower())
        rel2 = _RELATIONSHIP_PHRASING.get(plan.second_relationship_type, plan.second_relationship_type.lower())
        for row in rows:
            statements.append(
                GraphStatement(
                    text=(
                        f"{plan.entity_names[0]} {rel1} {row['via']}, "
                        f"and {row['via']} {rel2} {row['result']}."
                    ),
                    source_chunk_ids=row["source_chunk_ids"],
                )
            )

    elif plan.query_type == "path_between":
        for row in rows:
            hops = []
            for i, rel_type in enumerate(row["relationship_path"]):
                verb = _RELATIONSHIP_PHRASING.get(rel_type, rel_type.lower())
                hops.append(f"{row['node_path'][i]} {verb} {row['node_path'][i + 1]}")
            all_chunk_ids = [cid for hop_ids in row["source_chunk_ids_per_hop"] for cid in hop_ids]
            statements.append(GraphStatement(text="; then ".join(hops) + ".", source_chunk_ids=all_chunk_ids))

    elif plan.query_type == "aggregate_count":
        for row in rows:
            verb = _RELATIONSHIP_PHRASING.get(plan.relationship_type, plan.relationship_type.lower())
            statements.append(
                GraphStatement(
                    text=(
                        f"{plan.entity_names[0]} {verb} {row['count']} entities "
                        f"({', '.join(row['neighbors'])})."
                    ),
                    source_chunk_ids=[],  # aggregation summarizes multiple edges; no single supporting chunk
                )
            )

    return statements


_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "chunk_id": {
                        "type": "string",
                        "description": "The exact chunk_id (from GRAPH FACTS or VECTOR PASSAGES below) this claim is grounded in.",
                    },
                },
                "required": ["text", "chunk_id"],
                "additionalProperties": False,
            },
        },
        "no_answer_reason": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "If the retrieved evidence does not answer the question, explain why here and leave claims empty.",
        },
    },
    "required": ["claims", "no_answer_reason"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You answer a question about SEC 10-K filings using ONLY the evidence provided below -- never outside knowledge.

Every claim in your answer must be backed by exactly one chunk_id from the evidence provided. If the evidence doesn't answer the question, return an empty claims list and explain why in no_answer_reason -- do not guess or fabricate a citation to force an answer."""


def build_answer(
    client: anthropic.Anthropic,
    question: str,
    graph_statements: list[GraphStatement],
    vector_results: list[VectorResult],
    model: str | None = None,
) -> MergedAnswer:
    """
    Build a citation-grounded answer from merged graph + vector evidence.
    Every claim's chunk_id is validated against the retrieved set; a claim
    citing an unretrieved chunk_id is dropped (not silently kept) and
    counted in rejected_claims.
    """
    if model is None:
        from src.config import get_settings

        model = get_settings().anthropic_model
    retrieved_chunk_ids: set[str] = set()
    context_parts = []

    if graph_statements:
        context_parts.append("=== GRAPH FACTS ===")
        for i, stmt in enumerate(graph_statements):
            chunk_ref = stmt.source_chunk_ids[0] if stmt.source_chunk_ids else f"graph_fact_{i}"
            retrieved_chunk_ids.add(chunk_ref)
            context_parts.append(f"[{chunk_ref}] {stmt.text}")

    if vector_results:
        context_parts.append("\n=== VECTOR PASSAGES ===")
        for r in vector_results:
            retrieved_chunk_ids.add(r.chunk_id)
            context_parts.append(f"[{r.chunk_id}] ({r.section_path}, {r.company}) {r.text}")

    context = "\n".join(context_parts) if context_parts else "(no evidence retrieved)"
    user_content = f"QUESTION: {question}\n\nEVIDENCE:\n{context}"

    accepted: list[Claim] = []
    rejected: list[dict] = []
    raw: dict = {}

    # Reject-and-regenerate (per the build guide): if any claim cites a
    # chunk_id that was never retrieved, regenerate once with an explicit
    # correction before accepting the (possibly still partial) result --
    # a plain retry of the identical prompt would likely repeat the same
    # mistake, so the second attempt names exactly which chunk_ids were
    # invalid.
    for attempt in range(2):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            output_config={"format": {"type": "json_schema", "schema": _ANSWER_SCHEMA}},
            messages=[{"role": "user", "content": user_content}],
        )
        raw = json.loads(next(b.text for b in response.content if b.type == "text"))

        accepted = []
        rejected = []
        for c in raw["claims"]:
            if c["chunk_id"] in retrieved_chunk_ids:
                accepted.append(Claim(text=c["text"], chunk_id=c["chunk_id"]))
            else:
                rejected.append(c)

        if not rejected or attempt == 1:
            break

        invalid_ids = ", ".join(sorted({c["chunk_id"] for c in rejected}))
        user_content = (
            f"QUESTION: {question}\n\nEVIDENCE:\n{context}\n\n"
            f"Your previous answer cited chunk_id(s) not present in the evidence above: "
            f"{invalid_ids}. Every claim's chunk_id must be copied EXACTLY from a "
            f"[chunk_id] label in the evidence. Regenerate the answer."
        )

    if not accepted:
        answer_text = raw.get("no_answer_reason") or "No grounded answer could be produced from the retrieved evidence."
    else:
        answer_text = " ".join(c.text for c in accepted)

    return MergedAnswer(
        answer_text=answer_text,
        claims=accepted,
        rejected_claims=rejected,
        graph_evidence_count=len(graph_statements),
        vector_evidence_count=len(vector_results),
    )
