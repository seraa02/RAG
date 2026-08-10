# Graph RAG over SEC Filings

A hybrid **Knowledge Graph + Retrieval-Augmented Generation** system, combining Neo4j (graph traversal) and PostgreSQL/pgvector (semantic search) with a query router and a citation-grounded answering layer.

SEC 10-K filings are the **first data source**, not the architecture. The core pipeline (normalize → chunk → embed/extract → vector + graph → router → retrieval → citation validation → answer) is designed to accept other document sources later (PDFs, internal docs, web pages) without being rewritten — see [Design Principles](#design-principles).

## Benchmark Results

> **Status: not yet run.** This section will report accuracy, accuracy-by-hop-count, latency, cost/query, and one-time ingestion cost for the hybrid system versus a vector-only baseline, once Phase 5 (evaluation) is complete.

| Metric | Vector-only baseline | Graph + Vector (hybrid) |
|---|---|---|
| Accuracy | TBD | TBD |
| Accuracy (1-hop) | TBD | TBD |
| Accuracy (multi-hop) | TBD | TBD |
| Latency / query | TBD | TBD |
| Cost / query | TBD | TBD |
| One-time ingestion cost | TBD | TBD |

---

## Why Knowledge Graph + RAG?

Plain vector RAG retrieves passages that are *semantically similar* to a question. That works well for definitions, policies, and direct facts, but it struggles with questions that require **connecting** information across documents — "which of NVIDIA's competitors also mentioned supply chain risk in their filing?" is a relationship/multi-hop question, not a similarity-search question.

A knowledge graph makes entities (companies, products, people, regulators) and the relationships between them ("competes with," "acquired," "supplies," "regulated by") explicit and queryable. This project combines both retrieval strategies and routes each question to whichever (or both) actually answers it, rather than betting the whole system on one retrieval paradigm.

## Why SEC Filings?

We need a corpus with real, meaningful **entity-to-entity relationships** — not a pile of unrelated documents. [SEC EDGAR](https://www.sec.gov/edgar) (the U.S. Securities and Exchange Commission's public filing database) is exactly that: public companies are required to disclose competitors, customers, suppliers, acquisitions, subsidiaries, regulators, and business relationships in structured, factual language.

## Why 10-Ks?

A **10-K** is a company's annual report to the SEC. It's the single richest filing type for this purpose because it must describe, in detail: business operations, products, competitors, customers, suppliers, acquisitions, subsidiaries, regulators, risk factors, business segments, and financial performance. That density of named entities and relationships is exactly what a knowledge graph needs to be worth building.

## Why These Six Companies?

NVIDIA (NVDA), AMD, Intel (INTC), Microsoft (MSFT), Amazon (AMZN), and Alphabet (GOOGL) — six large tech companies with well-documented, real competitive and business relationships (e.g., NVIDIA/AMD/Intel compete directly in semiconductors; Microsoft/Amazon/Alphabet compete in cloud). This gives the graph genuine cross-document edges to demonstrate, rather than isolated per-company facts.

We start with **one 10-K per company (6 documents total)** deliberately small, so we can:
1. Inspect the actual source documents.
2. Understand their real HTML structure.
3. Verify parsing and cleaning work correctly.
4. Design the ontology from real content, not guesswork.
5. Inspect real extracted entities/relationships before trusting them.
6. Estimate Claude extraction cost from real numbers.
7. Prove the pipeline works before scaling to more companies/years.

## Data Acquisition

Filings are downloaded from SEC EDGAR by the SEC source adapter (`src/ingest/sources/sec.py`), identifying our requests with a proper `SEC_USER_AGENT` per SEC's fair-access requirements. Every download is recorded in a manifest (company, ticker, CIK, accession number, filing type, filing date, local path, download timestamp, SHA-256 hash), so re-running the downloader is idempotent — it won't re-fetch what's already there. *(Not yet implemented — Phase 0.)*

## Raw vs. Processed Data

- **`data/raw/`** — the original downloaded filings, byte-for-byte, never modified. This is our ground truth; every later claim must be traceable back to it.
- **`data/processed/`** — parsed/normalized/cleaned text, chunks, and metadata derived from raw.
- **`data/cache/`** — cached Claude extraction results and embeddings, keyed by a *versioned* cache identity (see Design Principles), so we never pay to re-extract or re-embed unchanged content.

## Cleaning Philosophy

Cleaning is **deterministic**, not LLM-based, and lives inside the SEC source adapter (it's genuinely SEC/HTML-specific — see Design Principles below). We strip structural HTML noise (nav menus, repeated headers/footers, page numbers, excessive whitespace) using BeautifulSoup and regex — but we never rewrite, summarize, or otherwise alter the actual wording of the filing. This preserves provenance: the cleaned sentence a user sees cited is the sentence the company actually filed.

## Why Chunk IDs Matter

Every chunk gets a stable `chunk_id` used identically by **both** pipelines:

```
DOCUMENT → CHUNKS → ┬─ pgvector (embeddings)
                     └─ Claude extraction → entities/relationships (Neo4j, tagged source_chunk_id)
```

Because both branches share the same `chunk_id`, any graph fact or retrieved passage can be traced back to the exact source text it came from — this is what makes citations and provenance validation possible later.

## Design Principles

**Central principle: change the data source without rewriting the core pipeline.**

```
Source ──▶ Source Adapter ──▶ NormalizedDocument ──▶ Chunk ──▶ Embedding + Extraction ──▶ Vector + Graph ──▶ Router ──▶ Retrieval ──▶ Evidence Merge ──▶ Citation Validation ──▶ LLM ──▶ Answer
```

- **Source adapters** (`src/ingest/sources/`) are the only place source-specific logic is allowed to live. `sec.py` is Project 1's only adapter — it owns SEC EDGAR downloading, SEC HTML parsing, and deterministic HTML cleaning. A future PDF or internal-wiki source would be a new adapter file, not a rewrite of anything downstream.
- **NormalizedDocument** (`src/ingest/normalize.py`, schema in `src/schema.py`) is the boundary. Every adapter emits this same shape (`document_id`, `source`, `document_type`, `title`, `date`, `metadata`, `sections`, `text`, `source_uri`, `content_hash`). Everything after this point — chunking, embedding, extraction, entity resolution, graph writing, retrieval — only ever sees a `NormalizedDocument` or a `Chunk`, never SEC-specific structure.
- **Content hashing:** every document gets a SHA-256 `content_hash` of its actual content (not filename), used for idempotent ingestion, duplicate detection, and cache invalidation.
- **Versioned caching:** expensive computation (Claude extraction, embeddings) is cached by a *versioned* key, not just a document hash — e.g. extraction cache key = `document_hash + chunk_id + model_version + prompt_version + schema_version`. Changing the extraction prompt or schema automatically invalidates only the affected cache entries; nothing needs to be manually wiped.
- **Configuration-driven ontology:** entity/relationship types live in `config/ontology.yaml`, not hardcoded in Python. Changing domains later (e.g. a different filing type, or an entirely different document domain) means editing config, not rewriting the ingestion/retrieval layer.
- **Golden dataset separate from system data:** `eval/golden_dataset.json` (human-verified gold entities, relationships, source chunk IDs, expected route, hop count) and `eval/annotations.json` (verification audit trail) are evaluation-only — the system under test never reads them while answering. `eval/questions.json` holds just the question text; `run_benchmark.py` is the only thing that reads the gold data.
- **No unnecessary infrastructure now:** Docker + Python + PostgreSQL/pgvector + Neo4j is sufficient for Project 1's scale (6 → dozens of documents). No Kafka/Kubernetes/Airflow/Redis/S3/microservices — those get introduced later only if an actual scale requirement appears, at which point the clean adapter/normalization/pipeline boundaries here are what make that swap possible without a rewrite (local filesystem → object storage, synchronous ingestion → queue + workers, local Docker DBs → managed DBs).

## Architecture

```
                     Question
                        │
                     Router (GRAPH / VECTOR / BOTH)
                 ┌──────┴──────┐
                 ▼             ▼
         Graph Query      Vector Query
         (Neo4j, via      (pgvector
        parameterized     similarity
          templates)         search)
                 └──────┬──────┘
                         ▼
                  Merge + Dedupe
                         ▼
              Answer + Citation Validation
```

The LLM never writes raw Cypher — the router produces a structured intent that maps onto a fixed library of parameterized query templates. This is a deliberate security boundary (see Engineering Rules).

## Repository Structure

```
RAG/
├── docker-compose.yml      # Neo4j + PostgreSQL/pgvector, local infra
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── config/
│   └── ontology.yaml       # entity/relationship types (defined in Phase 1)
│
├── data/
│   ├── raw/                # original filings, untouched
│   ├── processed/          # parsed/cleaned text, chunks
│   └── cache/               # cached Claude extraction results
│
├── src/
│   ├── config.py            # env/config loading
│   ├── schema.py             # Pydantic data contracts (incl. NormalizedDocument)
│   ├── ingest/
│   │   ├── sources/
│   │   │   └── sec.py         # SEC EDGAR adapter: download, parse, clean -> NormalizedDocument
│   │   ├── normalize.py        # NormalizedDocument boundary (shared, source-agnostic)
│   │   ├── chunk.py             # NormalizedDocument -> chunks (shared)
│   │   ├── embed.py              # chunks -> pgvector embeddings (shared)
│   │   ├── extract.py             # chunks -> Claude entity/relationship extraction (shared)
│   │   ├── resolve.py              # entity resolution (shared)
│   │   └── graph_writer.py          # Neo4j MERGE writes (shared)
│   ├── retrieval/            # router, graph query, vector query, merge
│   └── api/                   # HTTP entrypoint
│
├── eval/
│   ├── questions.json        # benchmark question text
│   ├── golden_dataset.json    # human-verified gold entities/relationships/chunk IDs/hop count
│   ├── annotations.json        # verification audit trail for golden_dataset.json
│   └── run_benchmark.py         # baseline vs. hybrid comparison
│
└── tests/
```

## Technology Stack

| Tool | Role |
|---|---|
| Python 3.11+ | implementation language |
| Claude / Anthropic API | entity & relationship extraction |
| Neo4j | knowledge graph |
| PostgreSQL + pgvector | vector retrieval |
| Docker / Docker Compose | local infrastructure |
| Pydantic | schema validation |
| BeautifulSoup | deterministic HTML parsing |
| Pytest | testing |

## Project Phases

- **Phase 0** — Environment setup, SEC access verification, download the 6-document corpus, inspect raw filings, plan cleaning strategy, estimate extraction cost.
- **Phase 1** — Knowledge graph: ontology design (from real corpus), Claude extraction, entity resolution, Neo4j writes.
- **Phase 2** — Vector database: embeddings, pgvector, HNSW indexing, recall@k evaluation.
- **Phase 3** — Query router: GRAPH / VECTOR / BOTH classification via parameterized templates.
- **Phase 4** — Hybrid answering: merge, dedupe, citation validation.
- **Phase 5** — Benchmark: build `golden_dataset.json` from the real corpus (~20-30 questions in development, 50-100 final), stratified by hop count/aggregation/out-of-scope; evaluate hybrid vs. vector-only baseline across extraction, entity resolution, vector retrieval, router accuracy, graph retrieval, and end-to-end answering.

## Setup Instructions

See the **Setup & Verification** checkpoint below (current scaffold stage) for exact commands.

## Docker Instructions

```bash
docker compose up -d
docker compose ps
docker compose down        # stop (data persists in named volumes)
```

## Testing Instructions

```bash
pytest
```

## Current Project Status

```
PHASE 0 — NOT STARTED
SCAFFOLD — COMPLETE
```

No SEC filings have been downloaded, no API calls have been made, and no ontology, cleaning, chunking, embedding, extraction, graph-writing, retrieval, or benchmarking logic has been implemented. This README will be updated as each phase completes.
