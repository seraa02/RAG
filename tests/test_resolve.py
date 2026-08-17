"""
Tests for src/ingest/resolve.py.

What these tests prove:
    - Suffix-variant names ("NVIDIA Corporation" / "NVIDIA CORP" /
      "Nvidia Corp.") collapse to one canonical entity via normalization
      alone -- no embedding call needed, no network dependency.
    - Entities of different entity_types never merge even if their names
      are textually identical (a Company named "Blackwell" must not
      merge with a Product named "Blackwell").
    - Every input entity ends up with a canonical_name assigned.
"""

from src.ingest.resolve import normalize_name, resolve_entities
from src.schema import ExtractedEntity


def _entity(entity_type: str, raw_name: str) -> ExtractedEntity:
    return ExtractedEntity(
        entity_type=entity_type,
        raw_name=raw_name,
        source_chunk_id="TEST_2026_10K_00001",
        confidence=0.9,
    )


def test_normalize_name_strips_corporate_suffixes():
    assert normalize_name("NVIDIA Corporation") == normalize_name("NVIDIA CORP")
    assert normalize_name("NVIDIA CORP") == normalize_name("Nvidia Corp.")
    assert normalize_name("Advanced Micro Devices, Inc.") == "advanced micro devices"


def test_suffix_variants_resolve_to_one_canonical_entity():
    entities = [
        _entity("Company", "NVIDIA Corporation"),
        _entity("Company", "NVIDIA CORP"),
        _entity("Company", "Nvidia Corp."),
    ]
    resolved, registry = resolve_entities(entities)

    canonical_names = {e.canonical_name for e in resolved}
    assert len(canonical_names) == 1
    assert len(registry) == 1
    assert set(registry[0].aliases) | {registry[0].canonical_name} == {
        "NVIDIA Corporation",
        "NVIDIA CORP",
        "Nvidia Corp.",
    }


def test_different_entity_types_never_merge():
    entities = [
        _entity("Company", "Blackwell"),
        _entity("Product", "Blackwell"),
    ]
    resolved, registry = resolve_entities(entities)

    assert len(registry) == 2
    types = {r.entity_type for r in registry}
    assert types == {"Company", "Product"}


def test_every_entity_gets_a_canonical_name():
    entities = [
        _entity("Company", "Intel Corporation"),
        _entity("RegulatoryBody", "Federal Trade Commission"),
        _entity("Company", "Intel Corp"),
    ]
    resolved, _ = resolve_entities(entities)
    assert all(e.canonical_name is not None for e in resolved)


def test_abbreviation_does_not_merge_without_a_seed():
    # Regression guard for the real bug found during the Phase 1 build:
    # "AMD" and "Advanced Micro Devices, Inc." measure ~0.77 cosine
    # similarity -- below the 0.87 threshold -- so without a seed group
    # they resolve to two separate nodes. This documents that as expected
    # default behavior (the fix is the seed group, not a lower threshold
    # -- a lower threshold was tested and catastrophically over-merges
    # unrelated companies, see resolve_entities' docstring).
    entities = [_entity("Company", "AMD"), _entity("Company", "Advanced Micro Devices, Inc.")]
    resolved, registry = resolve_entities(entities)
    assert len(registry) == 2


def test_seed_group_force_merges_known_abbreviation():
    entities = [
        _entity("Company", "AMD"),
        _entity("Company", "Advanced Micro Devices, Inc."),
        _entity("Company", "Intel Corporation"),  # unrelated -- must not get pulled in
    ]
    resolved, registry = resolve_entities(
        entities, seed_groups=[("Company", ["AMD", "Advanced Micro Devices, Inc."])]
    )

    amd_records = [r for r in registry if "AMD" in ([r.canonical_name] + r.aliases)]
    assert len(amd_records) == 1
    assert set(amd_records[0].aliases) | {amd_records[0].canonical_name} == {
        "AMD",
        "Advanced Micro Devices, Inc.",
    }
    intel_records = [r for r in registry if r.canonical_name == "Intel Corporation"]
    assert len(intel_records) == 1
    assert intel_records[0].aliases == []


def test_seed_group_is_a_no_op_when_names_never_appear():
    entities = [_entity("Company", "NVIDIA Corporation")]
    resolved, registry = resolve_entities(
        entities, seed_groups=[("Company", ["AMD", "Advanced Micro Devices, Inc."])]
    )
    assert len(registry) == 1
    assert registry[0].canonical_name == "NVIDIA Corporation"
