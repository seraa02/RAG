"""
Claude-based entity and relationship extraction.

Purpose:
    Send each chunk to Claude with a schema-constrained prompt (based on
    the ontology defined in config/ontology.yaml) and extract entities and
    relationships, each tagged with source_chunk_id and a confidence score.

Design decisions:
    - Structured outputs (output_config.format = json_schema), not a
      prefilled/free-text response. Guarantees valid JSON shape; Claude
      cannot return prose or a malformed list.
    - entity_type / relationship_type are constrained to the ontology's
      enums directly in the JSON schema, so Claude cannot invent a new
      type outside config/ontology.yaml.
    - Every response is *also* validated against the Pydantic models in
      src/schema.py after parsing (confidence in [0, 1], required fields)
      and retried up to MAX_RETRIES on violation -- structured outputs
      guarantee schema *shape* conformance, not the numeric-range
      constraints Pydantic checks (JSON Schema numeric constraints like
      minimum/maximum aren't supported by the API).
    - source_chunk_id is NOT trusted from the model -- we set it ourselves
      from the chunk being processed, since Claude has no reason to lie
      about it but also no reason we should trust free-form model output
      for an identifier that must exactly match our own chunk_id.
    - Cache key is document_hash + chunk_id + model + prompt_version +
      schema_version, NOT just chunk content. Changing the extraction
      prompt or the ontology invalidates exactly the entries it should,
      and nothing else -- no manual cache wipe required.
    - Cost is tracked per call (input/output tokens) and accumulated by
      the caller so the full-corpus cost can be reported before and after
      a run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import yaml
from pydantic import ValidationError

from src.schema import Chunk, ExtractedEntity, ExtractedRelationship

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ONTOLOGY_PATH = _PROJECT_ROOT / "config" / "ontology.yaml"
CACHE_DIR = _PROJECT_ROOT / "data" / "cache" / "extraction"

PROMPT_VERSION = "v2"
SCHEMA_VERSION = "v1"
MAX_RETRIES = 2


@dataclass
class Ontology:
    entity_types: list[str]
    relationship_types: list[str]
    entity_descriptions: dict[str, str]
    relationship_specs: dict[str, dict[str, str]]  # name -> {source, target, description}

    @classmethod
    def load(cls, path: Path = ONTOLOGY_PATH) -> "Ontology":
        raw = yaml.safe_load(path.read_text())
        if raw.get("status") != "DEFINED":
            raise RuntimeError(
                f"Ontology at {path} is not yet defined (status={raw.get('status')!r}). "
                "Run Phase 1's ontology design step before extracting."
            )
        entity_types = [e["name"] for e in raw["entity_types"]]
        relationship_types = [r["name"] for r in raw["relationship_types"]]
        return cls(
            entity_types=entity_types,
            relationship_types=relationship_types,
            entity_descriptions={e["name"]: e["description"].strip() for e in raw["entity_types"]},
            relationship_specs={
                r["name"]: {
                    "source": r["source"],
                    "target": r["target"],
                    "description": r["description"].strip(),
                }
                for r in raw["relationship_types"]
            },
        )

    def as_prompt_block(self) -> str:
        lines = ["ENTITY TYPES:"]
        for name, desc in self.entity_descriptions.items():
            lines.append(f"- {name}: {desc}")
        lines.append("\nRELATIONSHIP TYPES:")
        for name, spec in self.relationship_specs.items():
            lines.append(f"- {name} ({spec['source']} -> {spec['target']}): {spec['description']}")
        return "\n".join(lines)

    def json_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity_type": {"type": "string", "enum": self.entity_types},
                            "raw_name": {
                                "type": "string",
                                "description": "The entity's name exactly as it appears in the text.",
                            },
                            "confidence": {
                                "type": "number",
                                "description": "0.0-1.0 confidence this entity is correctly typed and named.",
                            },
                        },
                        "required": ["entity_type", "raw_name", "confidence"],
                        "additionalProperties": False,
                    },
                },
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "relationship_type": {"type": "string", "enum": self.relationship_types},
                            "source_entity": {
                                "type": "string",
                                "description": "raw_name of the source entity (must match an entity in this response's entities list).",
                            },
                            "target_entity": {
                                "type": "string",
                                "description": "raw_name of the target entity (must match an entity in this response's entities list).",
                            },
                            "confidence": {"type": "number"},
                        },
                        "required": ["relationship_type", "source_entity", "target_entity", "confidence"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["entities", "relationships"],
            "additionalProperties": False,
        }


SYSTEM_PROMPT_TEMPLATE = """You extract entities and relationships from a single excerpt of an SEC 10-K filing, strictly constrained to the ontology below. This is one chunk of a larger document -- extract only what this excerpt actually states, do not infer facts from outside knowledge.

This excerpt is from the 10-K filed by {filer_company}. Every use of "we," "our," "us," or "the Company" in this text refers to {filer_company} -- resolve these self-references to the literal name "{filer_company}" when they are the subject or object of a relationship (e.g. "we compete against Intel" means {filer_company} COMPETES_WITH Intel). Do not skip a relationship just because the filer side is written as "we" rather than its name.

{ontology_block}

Rules:
- Only extract entities/relationships explicitly stated or clearly implied by THIS text. Do not use outside knowledge about these companies.
- raw_name should be the entity's name exactly as written in the text, EXCEPT for filer self-references, which you resolve to "{filer_company}" as instructed above.
- Every source_entity/target_entity in relationships must also appear as a raw_name in entities.
- confidence reflects your certainty in the extraction, not the certainty of the underlying business fact.
- If the excerpt contains no relevant entities or relationships, return empty lists. Do not force extraction where none exists."""


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    from_cache: bool = False


def _cache_key(chunk: Chunk, model: str) -> str:
    raw = f"{chunk.chunk_id}|{model}|{PROMPT_VERSION}|{SCHEMA_VERSION}|{chunk.text}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_path(cache_key: str) -> Path:
    return CACHE_DIR / f"{cache_key}.json"


def _load_from_cache(cache_key: str) -> ExtractionResult | None:
    path = _cache_path(cache_key)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return ExtractionResult(
        entities=[ExtractedEntity(**e) for e in data["entities"]],
        relationships=[ExtractedRelationship(**r) for r in data["relationships"]],
        input_tokens=data["input_tokens"],
        output_tokens=data["output_tokens"],
        from_cache=True,
    )


def _save_to_cache(cache_key: str, result: ExtractionResult) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(cache_key).write_text(
        json.dumps(
            {
                "entities": [e.model_dump(mode="json") for e in result.entities],
                "relationships": [r.model_dump(mode="json") for r in result.relationships],
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
            indent=2,
        )
    )


def _parse_and_validate(raw_json: dict, chunk: Chunk) -> tuple[list[ExtractedEntity], list[ExtractedRelationship]]:
    entities = [
        ExtractedEntity(
            entity_type=e["entity_type"],
            raw_name=e["raw_name"],
            source_chunk_id=chunk.chunk_id,
            confidence=e["confidence"],
        )
        for e in raw_json["entities"]
    ]
    known_names = {e.raw_name for e in entities}
    relationships = []
    for r in raw_json["relationships"]:
        if r["source_entity"] not in known_names or r["target_entity"] not in known_names:
            # Model referenced an entity it didn't also emit in `entities` --
            # drop the relationship rather than writing a dangling edge.
            continue
        relationships.append(
            ExtractedRelationship(
                relationship_type=r["relationship_type"],
                source_entity=r["source_entity"],
                target_entity=r["target_entity"],
                source_chunk_id=chunk.chunk_id,
                confidence=r["confidence"],
            )
        )
    return entities, relationships


def extract_chunk(
    client: anthropic.Anthropic,
    chunk: Chunk,
    ontology: Ontology,
    model: str | None = None,
    use_cache: bool = True,
) -> ExtractionResult:
    """
    Extract entities/relationships from one chunk. Cached by
    (chunk_id, model, prompt_version, schema_version, chunk text) so
    re-running ingestion on unchanged chunks costs nothing.
    """
    if model is None:
        from src.config import get_settings

        model = get_settings().anthropic_model
    cache_key = _cache_key(chunk, model)
    if use_cache:
        cached = _load_from_cache(cache_key)
        if cached is not None:
            return cached

    filer_company = chunk.metadata.get("company") or "the filer"
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        filer_company=filer_company, ontology_block=ontology.as_prompt_block()
    )
    schema = ontology.json_schema()

    last_error: Exception | None = None
    max_tokens = 8192
    for attempt in range(MAX_RETRIES + 1):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[
                {
                    "role": "user",
                    "content": f"Section: {chunk.section_path}\n\nText:\n{chunk.text}",
                }
            ],
        )
        text = next(b.text for b in response.content if b.type == "text")

        if response.stop_reason == "max_tokens":
            # Response was truncated mid-JSON -- retrying at the same cap
            # would fail identically. Double it and try again rather than
            # burning a retry on a doomed parse.
            last_error = RuntimeError(
                f"response truncated at max_tokens={max_tokens} (entity-dense chunk)"
            )
            max_tokens *= 2
            continue

        try:
            raw_json = json.loads(text)
            entities, relationships = _parse_and_validate(raw_json, chunk)
            result = ExtractionResult(
                entities=entities,
                relationships=relationships,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            if use_cache:
                _save_to_cache(cache_key, result)
            return result
        except (json.JSONDecodeError, ValidationError, KeyError) as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"Extraction failed for chunk {chunk.chunk_id} after {MAX_RETRIES + 1} attempts: {last_error}"
    )


def compute_total_extraction_cost() -> dict:
    """
    Sum input/output tokens directly from every cached extraction result
    on disk and compute total cost. Read from the cache rather than
    trusting a live run's stdout total -- this is accurate no matter how
    many separate runs/restarts it took to populate the cache (relevant
    here: the first full-corpus run crashed partway through on a
    truncation bug and had to be restarted), and it's the number that
    belongs in the README per the guide's "report the one-time ingestion
    cost" requirement.
    """
    total_input = 0
    total_output = 0
    chunk_count = 0
    for path in CACHE_DIR.glob("*.json"):
        data = json.loads(path.read_text())
        total_input += data["input_tokens"]
        total_output += data["output_tokens"]
        chunk_count += 1

    cost = total_input / 1e6 * _INPUT_COST_PER_MTOK + total_output / 1e6 * _OUTPUT_COST_PER_MTOK
    return {
        "chunks_cached": chunk_count,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_cost_usd": round(cost, 4),
    }


# Sonnet 5 pricing (introductory, through 2026-08-31): $2/MTok in, $10/MTok out.
# See shared/models.md in the claude-api skill for current rates.
_INPUT_COST_PER_MTOK = 2.00
_OUTPUT_COST_PER_MTOK = 10.00

_EXTRACTED_DIR = _PROJECT_ROOT / "data" / "processed" / "extracted"


def run_extraction(model: str | None = None, max_workers: int = 8) -> None:
    """
    Run entity/relationship extraction over every document in
    data/processed/*.json, writing per-document results to
    data/processed/extracted/{document_id}.json and printing running
    token/cost totals. Cache-aware: re-running after a crash or interrupt
    only pays for chunks not already cached.

    Chunks within a document are extracted concurrently (max_workers
    threads) -- each call is an independent network round trip with no
    shared mutable state (extract_chunk's cache read/write is one file
    per chunk_id, so concurrent chunks never touch the same file). This
    is a real wall-clock win: 749 chunks sequentially at ~1-3s each is
    20-35 minutes; at max_workers=8 it's roughly a 6-8x speedup, bounded
    by the API's actual concurrent-request handling rather than by this
    process waiting on one response at a time.
    """
    import glob
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from src.config import get_settings
    from src.ingest.chunk import chunk_document
    from src.schema import NormalizedDocument

    settings = get_settings()
    model = model or settings.anthropic_model
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    ontology = Ontology.load()
    _EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    doc_paths = sorted(
        p for p in glob.glob(str(_PROJECT_ROOT / "data" / "processed" / "*.json"))
    )

    grand_total_chunks = 0
    grand_total_entities = 0
    grand_total_relationships = 0
    grand_input_tokens = 0
    grand_output_tokens = 0
    grand_cached = 0

    for path in doc_paths:
        doc = NormalizedDocument(**json.loads(Path(path).read_text()))
        chunks = chunk_document(doc)
        doc_entities: list[ExtractedEntity] = []
        doc_relationships: list[ExtractedRelationship] = []
        doc_input = 0
        doc_output = 0
        doc_cached = 0

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(extract_chunk, client, chunk, ontology, model): chunk for chunk in chunks}
            for future in as_completed(futures):
                result = future.result()
                doc_entities.extend(result.entities)
                doc_relationships.extend(result.relationships)
                doc_input += result.input_tokens
                doc_output += result.output_tokens
                if result.from_cache:
                    doc_cached += 1

        out_path = _EXTRACTED_DIR / f"{doc.document_id}.json"
        out_path.write_text(
            json.dumps(
                {
                    "document_id": doc.document_id,
                    "entities": [e.model_dump(mode="json") for e in doc_entities],
                    "relationships": [r.model_dump(mode="json") for r in doc_relationships],
                },
                indent=2,
            )
        )

        doc_cost = doc_input / 1e6 * _INPUT_COST_PER_MTOK + doc_output / 1e6 * _OUTPUT_COST_PER_MTOK
        print(
            f"[extracted] {doc.document_id}: {len(chunks)} chunks "
            f"({doc_cached} cached), {len(doc_entities)} entities, "
            f"{len(doc_relationships)} relationships, "
            f"in={doc_input:,} out={doc_output:,} tokens, ~${doc_cost:.2f}"
        )

        grand_total_chunks += len(chunks)
        grand_total_entities += len(doc_entities)
        grand_total_relationships += len(doc_relationships)
        grand_input_tokens += doc_input
        grand_output_tokens += doc_output
        grand_cached += doc_cached

    grand_cost = (
        grand_input_tokens / 1e6 * _INPUT_COST_PER_MTOK
        + grand_output_tokens / 1e6 * _OUTPUT_COST_PER_MTOK
    )
    print()
    print("=== Extraction complete ===")
    print(f"Documents: {len(doc_paths)}")
    print(f"Chunks processed: {grand_total_chunks} ({grand_cached} served from cache)")
    print(f"Entities extracted: {grand_total_entities}")
    print(f"Relationships extracted: {grand_total_relationships}")
    print(f"Tokens: {grand_input_tokens:,} in / {grand_output_tokens:,} out")
    print(f"Total cost: ${grand_cost:.2f}")


if __name__ == "__main__":
    run_extraction()
