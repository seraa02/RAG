"""
Parameterized Neo4j query templates.

Purpose:
    Provide a fixed library of parameterized Cypher templates that the
    router's structured intent gets mapped onto. This is the ONLY module
    allowed to run Cypher against Neo4j.

TODO (Phase 3):
    - Define a template per supported structured intent (e.g.
      "competitors_of(company)", "path_between(entity_a, entity_b)").
    - All templates must use parameters (never string-interpolate
      user/LLM input into Cypher).
    - Each result must retain source_chunk_id so it can be cited.
"""
