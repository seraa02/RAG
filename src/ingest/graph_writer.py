"""
Neo4j graph writer.

Purpose:
    Write resolved entities and their relationships into Neo4j. Writes
    with MERGE (never CREATE) so re-running ingestion is idempotent --
    running the pipeline twice must not duplicate nodes or edges.

Design decisions:
    - One Neo4j label per entity_type (Company, Product, BusinessSegment,
      RegulatoryBody, Person) rather than one generic ":Entity" label with
      a type property. This lets Cypher templates in
      src/retrieval/graph_query.py filter by label directly (fast,
      index-friendly) instead of a WHERE clause on every query.
    - Nodes are MERGEd on (label, canonical_name) -- the unique key entity
      resolution (resolve.py) already computed. aliases is stored as a
      list property on the node so a lookup can also match on any known
      alias later (see graph_query.py's entity-resolution-at-query-time
      step).
    - Relationships are MERGEd on (source, type, target) -- NOT per
      source_chunk_id, because the same real-world relationship (e.g.
      "AMD COMPETES_WITH Intel") is typically stated in multiple chunks
      across a filing (and sometimes across filings). We don't want N
      duplicate edges for one fact. Instead, source_chunk_id becomes a
      list property on the edge (`source_chunk_ids`), accumulating every
      chunk that supports the claim -- MERGE-then-append via
      apoc-free plain Cypher (coalesce + list concatenation), so a
      re-ingested chunk that already contributed to this edge doesn't
      duplicate itself in the list either.
    - The relationship_type string becomes the literal Neo4j relationship
      type (e.g. :COMPETES_WITH). This is safe ONLY because
      relationship_type is drawn from the fixed ontology enum validated
      at extraction time (src/ingest/extract.py) -- it is never raw
      user/model input interpolated without validation. Never do this
      with a string that hasn't been validated against a fixed allowlist.
"""

from __future__ import annotations

from pathlib import Path

from neo4j import Driver, GraphDatabase

from src.config import Settings, get_settings
from src.ingest.resolve import ResolvedEntity
from src.schema import ExtractedRelationship

_MERGE_NODE_QUERY = """
MERGE (n:{label} {{canonical_name: $canonical_name}})
ON CREATE SET n.aliases = $aliases
ON MATCH SET n.aliases = REDUCE(acc = coalesce(n.aliases, []), a IN $aliases |
    CASE WHEN a IN acc THEN acc ELSE acc + a END)
"""

_MERGE_RELATIONSHIP_QUERY = """
MATCH (source {{canonical_name: $source_name}})
MATCH (target {{canonical_name: $target_name}})
MERGE (source)-[r:{rel_type}]->(target)
ON CREATE SET
    r.source_chunk_ids = [$source_chunk_id],
    r.confidence = $confidence
ON MATCH SET
    r.source_chunk_ids = CASE WHEN $source_chunk_id IN r.source_chunk_ids
        THEN r.source_chunk_ids
        ELSE r.source_chunk_ids + $source_chunk_id
    END,
    r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END
"""


def get_driver(settings: Settings | None = None) -> Driver:
    settings = settings or get_settings()
    return GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))


def ensure_constraints(driver: Driver, entity_types: list[str]) -> None:
    """
    One uniqueness constraint per entity label on canonical_name -- this
    is what makes MERGE-by-canonical_name safe/fast and is what actually
    enforces "one node per real-world entity" at the database level (not
    just as an application-level convention).
    """
    with driver.session() as session:
        for label in entity_types:
            session.run(
                f"CREATE CONSTRAINT {label.lower()}_canonical_name IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.canonical_name IS UNIQUE"
            )


def write_entities(driver: Driver, registry: list[ResolvedEntity]) -> None:
    with driver.session() as session:
        for entity in registry:
            session.run(
                _MERGE_NODE_QUERY.format(label=entity.entity_type),
                canonical_name=entity.canonical_name,
                aliases=entity.aliases,
            )


def write_relationships(
    driver: Driver,
    relationships: list[ExtractedRelationship],
    raw_to_canonical: dict[tuple[str, str], str],
    entity_type_by_raw: dict[str, str],
) -> int:
    """
    Write relationships, resolving source_entity/target_entity raw names
    to canonical_name first. Skips (and counts) any relationship whose
    endpoint didn't survive resolution -- this shouldn't happen if
    resolve.py ran over the same entity list first, but is checked rather
    than assumed, since a silently dropped MATCH would otherwise write
    nothing and look like success.
    """
    written = 0
    skipped = 0
    with driver.session() as session:
        for rel in relationships:
            source_type = entity_type_by_raw.get(rel.source_entity)
            target_type = entity_type_by_raw.get(rel.target_entity)
            if source_type is None or target_type is None:
                skipped += 1
                continue
            source_canonical = raw_to_canonical.get((source_type, rel.source_entity))
            target_canonical = raw_to_canonical.get((target_type, rel.target_entity))
            if source_canonical is None or target_canonical is None:
                skipped += 1
                continue

            session.run(
                _MERGE_RELATIONSHIP_QUERY.format(rel_type=rel.relationship_type),
                source_name=source_canonical,
                target_name=target_canonical,
                source_chunk_id=rel.source_chunk_id,
                confidence=rel.confidence,
            )
            written += 1

    if skipped:
        print(f"[graph_writer] skipped {skipped} relationships with unresolved endpoints")
    return written


def _filer_seed_groups() -> list[tuple[str, list[str]]]:
    """
    Build (entity_type, [names]) seed groups from data/raw/manifest.json's
    ticker <-> legal-name pairs -- the one place we hold outside ground
    truth about entity identity, rather than relying on embedding
    similarity for the specific case it measurably fails on (see
    resolve.py's resolve_entities docstring for the AMD/Advanced Micro
    Devices measurement).
    """
    import json as _json

    manifest_path = Path("data/raw/manifest.json")
    if not manifest_path.exists():
        return []
    records = _json.loads(manifest_path.read_text())
    return [("Company", [record["ticker"], record["company"]]) for record in records]


def run_graph_write() -> None:
    """
    Load every data/processed/extracted/{document_id}.json, resolve
    entities corpus-wide, and write nodes + edges into Neo4j. Run after
    extract.py has produced extraction output for at least one document.
    """
    import glob
    import json

    from src.ingest.extract import Ontology
    from src.ingest.resolve import resolve_entities
    from src.schema import ExtractedEntity, ExtractedRelationship

    ontology = Ontology.load()
    all_entities: list[ExtractedEntity] = []
    all_relationships: list[ExtractedRelationship] = []

    for path in sorted(glob.glob("data/processed/extracted/*.json")):
        data = json.loads(open(path).read())
        all_entities.extend(ExtractedEntity(**e) for e in data["entities"])
        all_relationships.extend(ExtractedRelationship(**r) for r in data["relationships"])

    print(f"Loaded {len(all_entities)} raw entities, {len(all_relationships)} raw relationships")

    resolved_entities, registry = resolve_entities(all_entities, seed_groups=_filer_seed_groups())
    print(f"Resolved to {len(registry)} canonical entities")

    raw_to_canonical = {(e.entity_type, e.raw_name): e.canonical_name for e in resolved_entities}
    entity_type_by_raw = {e.raw_name: e.entity_type for e in resolved_entities}

    driver = get_driver()
    try:
        ensure_constraints(driver, ontology.entity_types)
        write_entities(driver, registry)
        written = write_relationships(driver, all_relationships, raw_to_canonical, entity_type_by_raw)
        print(f"=== Graph write complete: {len(registry)} nodes, {written} relationships ===")
    finally:
        driver.close()


if __name__ == "__main__":
    run_graph_write()
