"""
Entity resolution.

Purpose:
    Collapse different surface forms of the same real-world entity into
    one canonical node -- "Acme Corp", "Acme Corporation", and "ACME"
    must resolve to the same graph node, or the graph fragments into
    disconnected islands that can't be traversed (the guide's explicit
    warning: "skipping entity resolution... the same company exists as
    four disconnected nodes").

Design decisions:
    - Resolution only ever merges entities of the SAME entity_type (a
      Company must never merge with a Product just because the names are
      textually similar).
    - Two-stage matching, cheapest first:
        1. Exact match after normalization (strip corporate suffixes --
           Inc./Corp./Corporation/Ltd./LLC/N.V./Co./Company -- lowercase,
           collapse whitespace/punctuation). Catches "NVIDIA Corporation"
           vs "NVIDIA CORP" vs "Nvidia Corp." for free.
        2. Embedding similarity above a tuned threshold for names that
           don't normalize to the same string but are the same entity
           (e.g. "AMD" vs "Advanced Micro Devices, Inc." -- an
           abbreviation, not a suffix variant, so normalization alone
           can't catch it). Reuses the same local embedding model as
           src/ingest/embed.py.
    - Clustering is union-find over a similarity graph, not a fixed
      "compare every name to a canonical list" -- there is no canonical
      list yet; it's discovered from the data.
    - Canonical name = the longest raw_name in the cluster (the fullest,
      most formal form generally wins for a public company: "Advanced
      Micro Devices, Inc." over "AMD"). All other raw_names in the
      cluster become the alias list stored on the node.
    - Threshold is a starting point (0.87 cosine on normalized BGE
      embeddings), not tuned against a labeled set yet -- see README's
      Phase 5 (recall@k) methodology for how thresholds should eventually
      be validated, not just asserted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.schema import ExtractedEntity

SIMILARITY_THRESHOLD = 0.87

_SUFFIX_RE = re.compile(
    r"[,.]?\s+(inc|incorporated|corp|corporation|co|company|ltd|llc|"
    r"l\.?l\.?c\.?|plc|n\.?v\.?|group|holdings?)\.?\s*$",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize_name(raw_name: str) -> str:
    """Strip corporate suffixes and punctuation, lowercase, collapse whitespace."""
    name = raw_name.strip()
    # Suffix stripping can apply more than once, e.g. "X Holdings, Inc." -> "X Holdings" -> "X".
    while True:
        stripped = _SUFFIX_RE.sub("", name)
        if stripped == name:
            break
        name = stripped.strip()
    name = _PUNCT_RE.sub("", name)
    name = _WS_RE.sub(" ", name).strip().lower()
    return name


@dataclass
class ResolvedEntity:
    canonical_name: str
    entity_type: str
    aliases: list[str] = field(default_factory=list)


class _UnionFind:
    def __init__(self, items: list[str]):
        self._parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for item in self._parent:
            out.setdefault(self.find(item), []).append(item)
        return out


def _cosine_similarity_matrix(vectors: list[list[float]]) -> list[list[float]]:
    """Vectors are assumed pre-normalized (embed_texts uses normalize_embeddings=True)."""
    n = len(vectors)
    sims = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            dot = sum(a * b for a, b in zip(vectors[i], vectors[j]))
            sims[i][j] = sims[j][i] = dot
    return sims


def resolve_entities(
    entities: list[ExtractedEntity],
    threshold: float = SIMILARITY_THRESHOLD,
    seed_groups: list[tuple[str, list[str]]] | None = None,
) -> tuple[list[ExtractedEntity], list[ResolvedEntity]]:
    """
    Resolve raw_name -> canonical_name across all extracted entities,
    grouped by entity_type. Returns (entities with canonical_name filled
    in, the deduplicated registry of ResolvedEntity nodes to write to
    Neo4j).

    Mutates and returns the same ExtractedEntity list (canonical_name is
    the only field changed) rather than a copy, since callers pass the
    full corpus-wide entity list and copying it is wasted memory at this
    scale for no benefit.

    seed_groups -- KNOWN, GROUND-TRUTH abbreviation/name equivalences to
    force-merge, as (entity_type, [name_1, name_2, ...]) pairs, e.g.
    ("Company", ["AMD", "Advanced Micro Devices, Inc."]). This exists
    because embedding similarity alone measurably fails on true
    abbreviations: "AMD" vs "Advanced Micro Devices, Inc." cosine-
    similarity at ~0.77, well under the 0.87 threshold. Lowering the
    global threshold to catch this is NOT a fix -- verified empirically
    against this exact corpus, threshold=0.75 merges AMD, Sony, Intel,
    Microsoft, NVIDIA, Samsung, Apple, Qualcomm, IBM, and Oracle into ONE
    node, because those names co-occur in near-identical "OEM/customer
    partner" list sentences and their embeddings cluster on shared
    context, not shared identity. A scoped seed list for entities where
    we hold outside ground truth (here: the 6 known SEC filers, from
    data/raw/manifest.json) is the correct fix; general abbreviation
    resolution beyond that remains an open, documented limitation -- see
    README's Known Limitations.
    """
    from src.ingest.embed import embed_texts  # local import: avoid loading the model unless needed

    by_type: dict[str, list[str]] = {}
    for e in entities:
        by_type.setdefault(e.entity_type, []).append(e.raw_name)

    seed_groups = seed_groups or []
    seeds_by_type: dict[str, list[list[str]]] = {}
    for entity_type, names in seed_groups:
        seeds_by_type.setdefault(entity_type, []).append(names)

    raw_to_canonical: dict[tuple[str, str], str] = {}  # (entity_type, raw_name) -> canonical_name
    registry: list[ResolvedEntity] = []

    for entity_type, raw_names in by_type.items():
        unique_names = sorted(set(raw_names))

        # Stage 1: exact match after normalization.
        norm_groups: dict[str, list[str]] = {}
        for name in unique_names:
            norm_groups.setdefault(normalize_name(name), []).append(name)

        # Stage 2: embedding similarity between the DISTINCT normalized forms
        # (comparing one representative per normalized group keeps the
        # pairwise matrix small -- we don't need to compare "NVIDIA Corp"
        # against "NVIDIA Corporation" again, they already merged in stage 1).
        representatives = list(norm_groups.keys())
        uf = _UnionFind(representatives)

        # Stage 2b: force-union any seed group's members that actually
        # appear in this corpus. A seed name never extracted is simply a
        # no-op, not an error.
        for seed_names in seeds_by_type.get(entity_type, []):
            seed_reps = [normalize_name(n) for n in seed_names if normalize_name(n) in norm_groups]
            for rep in seed_reps[1:]:
                uf.union(seed_reps[0], rep)

        if len(representatives) > 1:
            vectors = embed_texts(representatives)
            sims = _cosine_similarity_matrix(vectors)
            for i in range(len(representatives)):
                for j in range(i + 1, len(representatives)):
                    if sims[i][j] >= threshold:
                        uf.union(representatives[i], representatives[j])

        for _, rep_group in uf.groups().items():
            all_raw_names = sorted(
                {name for rep in rep_group for name in norm_groups[rep]},
                key=len,
                reverse=True,
            )
            canonical = all_raw_names[0]
            aliases = all_raw_names[1:]
            registry.append(ResolvedEntity(canonical_name=canonical, entity_type=entity_type, aliases=aliases))
            for raw_name in all_raw_names:
                raw_to_canonical[(entity_type, raw_name)] = canonical

    for e in entities:
        e.canonical_name = raw_to_canonical[(e.entity_type, e.raw_name)]

    return entities, registry
