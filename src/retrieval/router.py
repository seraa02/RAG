"""
Query router.

Purpose:
    Classify an incoming natural-language question as GRAPH, VECTOR, or
    BOTH, and log every routing decision.

TODO (Phase 3):
    - Route relationship / multi-hop / comparison / aggregation questions
      to GRAPH.
    - Route definition / policy / direct-fact / semantic-retrieval
      questions to VECTOR.
    - Fall back to BOTH when routing confidence is low.
    - Log every decision (question, chosen route, confidence).
    - SECURITY: this module must never let an LLM generate unrestricted
      Cypher. Router output is a structured intent that maps onto a
      parameterized Cypher template from graph_query.py -- never raw
      Cypher text from the LLM.
"""
