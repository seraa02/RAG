"""
Entity resolution.

Purpose:
    Map raw entity names extracted by Claude (e.g. "Advanced Micro Devices,
    Inc.", "AMD", "AMD Inc.") onto a single canonical entity, so the graph
    does not end up with duplicate nodes representing the same real-world
    company/product/entity.

TODO (Phase 1):
    - Normalize raw entity name strings.
    - Compare against existing canonical entities using embedding
      similarity.
    - Apply a similarity threshold -- to be tuned against real examples,
      not chosen arbitrarily.
    - Maintain an aliases list per canonical entity.
"""
