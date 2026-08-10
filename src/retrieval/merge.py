"""
Hybrid result merging and citation validation.

Purpose:
    Combine graph-derived facts and vector-retrieved passages into a
    single evidence set, deduplicate overlapping evidence, generate an
    answer, and validate that every claim's citation resolves to an
    actually retrieved chunk_id.

TODO (Phase 4):
    - Distinguish GRAPH-DERIVED FACT from RETRIEVED-PASSAGE FACT in the
      merged evidence.
    - Deduplicate evidence that references the same chunk_id.
    - Reject/regenerate any answer whose citation does not resolve to
      retrieved evidence.
"""
