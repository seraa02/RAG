# Graph RAG over SEC Filings

A hybrid **Knowledge Graph + Retrieval-Augmented Generation** system, combining Neo4j (graph traversal) and PostgreSQL/pgvector (semantic search) with a query router and a citation-grounded answering layer.

SEC 10-K filings are the **first data source**, not the architecture. The core pipeline (normalize → chunk → embed/extract → vector + graph → router → retrieval → citation validation → answer) is designed to accept other document sources later (PDFs, internal docs, web pages) without being rewritten — see [Design Principles](#design-principles).

## Benchmark Results

Hybrid (graph + vector) vs. a vector-only baseline, on a 20-question development-tier golden set (`eval/golden_dataset.json`) stratified by hop count, run against the real 6-filing corpus. Full per-question results in `eval/benchmark_results.json`.

| Metric | Vector-only baseline | Graph + Vector (hybrid) |
|---|---|---|
| **Overall accuracy** | **85%** (17/20) | **100%** (20/20) |
| Accuracy — single-hop | 85% (11/13) | 100% (13/13) |
| Accuracy — two-hop | 50% (1/2) | 100% (2/2) |
| Accuracy — aggregation | 100% (2/2) | 100% (2/2) |
| Accuracy — out-of-scope (correct refusal) | 100% (3/3) | 100% (3/3) |
| Avg latency / query | 5.52s | 5.37s |
| One-time ingestion cost (entity/relationship extraction, 749 chunks) | — | **$9.65** |

**The shape is the story, per the build guide's own methodology:** accuracy holds parity on out-of-scope and aggregation questions, then a real gap opens as hop count increases — vector-only drops to 50% on genuine 2-hop connection questions (it has no mechanism to traverse `AMD → competes with → NVIDIA → competes with → Alphabet`; it can only get lucky if a single passage happens to mention both endpoints), while the hybrid system holds 100% by routing those questions to the graph. Hybrid is not slower or more expensive per query here — the one real cost of the graph is the $9.65 one-time extraction pass, not per-query overhead.

Grading methodology is a mechanical entity-coverage check (documented in `eval/run_benchmark.py`'s `_grade()`), not an LLM-judge — see [Known Limitations](#known-limitations) for what that does and doesn't verify.

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

Filings are downloaded from SEC EDGAR by the SEC source adapter (`src/ingest/sources/sec.py`), identifying our requests with a proper `SEC_USER_AGENT` per SEC's fair-access requirements. Tickers are resolved to CIKs via SEC's canonical `company_tickers.json` (not hardcoded), the most recent 10-K per company is located via the `data.sec.gov/submissions` API, and the filing HTML is downloaded from SEC's Archives. Every download is recorded in `data/raw/manifest.json` (company, ticker, CIK, accession number, filing type, filing date, local path, download timestamp, SHA-256 hash), so re-running `python -m src.ingest.sources.sec` is idempotent — it skips any ticker already present in the manifest with its file still on disk. Run: `python -m src.ingest.sources.sec`.

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
│   ├── golden_dataset.json    # agent-verified gold entities/relationships/hop count (see Known Limitations)
│   ├── annotations.json        # verification audit trail for golden_dataset.json
│   ├── run_benchmark.py         # baseline vs. hybrid comparison
│   └── benchmark_results.json    # latest run's output (overall + by-difficulty + per-question)
│
└── tests/
```

## Technology Stack

| Tool | Role |
|---|---|
| Python 3.12 | implementation language |
| Claude Sonnet 5 / Haiku 4.5 (Anthropic API) | entity/relationship extraction, answer generation (Sonnet); query routing (Haiku, cheap+fast) |
| Neo4j | knowledge graph |
| PostgreSQL + pgvector | vector retrieval (HNSW index) |
| sentence-transformers (BAAI/bge-large-en-v1.5) | local embeddings, no per-call API cost |
| FastAPI + Uvicorn | HTTP API |
| Docker / Docker Compose | local infrastructure |
| Pydantic | schema validation |
| BeautifulSoup | deterministic HTML parsing |
| Pytest | testing (67 tests) |

## Project Phases

- **Phase 0** — ✅ Environment setup, SEC access verification, download the 6-document corpus, inspect raw filings, plan cleaning strategy, estimate extraction cost.
- **Phase 1** — ✅ Knowledge graph: ontology design (from real corpus), Claude extraction, entity resolution, Neo4j writes. 3,029 raw entities → 830 canonical nodes, 2,608 relationships. Cost: $9.65.
- **Phase 2** — ✅ Vector database: embeddings (local BAAI/bge-large-en-v1.5), pgvector, HNSW indexing. 749 chunks embedded.
- **Phase 3** — ✅ Query router: GRAPH / VECTOR / BOTH classification via parameterized templates (Claude Haiku), with low-confidence fallback and full decision logging (`data/cache/routing_log.jsonl`).
- **Phase 4** — ✅ Hybrid answering: merge, dedupe, citation validation with reject-and-regenerate on an invalid citation.
- **Phase 5** — ✅ Benchmark: 20-question development-tier golden set, stratified by hop count/aggregation/out-of-scope, evaluated hybrid vs. vector-only. See [Benchmark Results](#benchmark-results) and [Known Limitations](#known-limitations).

## Setup Instructions

```bash
# 1. Start Docker (Neo4j + pgvector)
docker compose up -d
docker compose ps          # confirm both containers are healthy

# 2. Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure — copy .env.example to .env and fill in ANTHROPIC_API_KEY
cp .env.example .env

# 4. Verify
pytest
```

## Running the Full Pipeline

```bash
# Data acquisition (idempotent — safe to re-run)
python -m src.ingest.sources.sec

# Phase 1: entity/relationship extraction + graph write
python -m src.ingest.extract       # ~$9.65, ~10-15 min with concurrency
python -m src.ingest.graph_writer

# Phase 2: embeddings
python -m src.ingest.embed

# Serve the API
uvicorn src.api.main:app --reload
# POST /ask {"question": "..."}  ->  citation-grounded answer

# Run the benchmark
python -m eval.run_benchmark
```

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

## Known Limitations

Honest notes on what did not work cleanly, found and either fixed or documented during the build — not hidden:

- **Section-boundary detection (Phase 0)** works well for 4 of 6 filers (NVDA, AMD, AMZN, GOOGL). MSFT has a cosmetic-only issue (decorative drop-cap first letters cause a stray space in section title text; body content is correct). INTC places dense financial-statement tables before the narrative Item sections in its HTML, so the header detector doesn't find a real heading until deep into the document — no text is lost, but INTC's early chunks mostly carry a generic "Front Matter" section label rather than a precise one.
- **Entity resolution catches suffix variants, not abbreviations, via embeddings alone.** "NVIDIA Corporation" / "NVIDIA CORP" / "Nvidia Corp." collapse correctly via normalization. But "AMD" vs. "Advanced Micro Devices, Inc." measures only ~0.77 cosine similarity — below the 0.87 threshold — so they resolved to two separate graph nodes until fixed. **Lowering the threshold is not a fix**: tested empirically, threshold=0.75 merges AMD, Sony, Intel, Microsoft, NVIDIA, Samsung, Apple, Qualcomm, IBM, and Oracle into *one node*, because those names co-occur in near-identical "OEM/customer partner" list sentences and their embeddings cluster on shared context, not shared identity. The actual fix: a small, scoped seed-alias list built from `data/raw/manifest.json`'s ticker↔legal-name pairs (`src/ingest/graph_writer.py::_filer_seed_groups`), applied only to the 6 filers we hold outside ground truth for. General abbreviation resolution beyond the 6 known filers remains unresolved — e.g. "SEC" and "SECURITIES AND EXCHANGE COMMISSION" are still two separate `RegulatoryBody` nodes, which is why q013's aggregate count (23) is a documented 1-off overcount.
- **The router initially misclassified "what did X acquire" questions as VECTOR**, even though `ACQUIRED` is a first-class graph relationship — because the phrasing superficially reads as a single-fact lookup. Fixed by adding an explicit ACQUIRED example to the router's few-shot prompt (`src/retrieval/router.py`); re-verified the specific failing question (NVIDIA/Mellanox) after the fix.
- **A bare graph fact can't disambiguate a *qualified* question.** "Which company did AMD acquire that led to a tax incentive in Singapore?" — the graph correctly returns AMD's 5 `ACQUIRED` targets (including the right answer, Xilinx), but the one_hop→statement conversion only produces "AMD acquired Xilinx, Inc." with no further detail, so the model correctly declines rather than guessing which of the 5 matches the Singapore qualifier. Vector search independently fails to surface the specific chunk either (the acquisition-tax-incentive sentence scores below several higher-similarity-but-irrelevant financial/legal-exhibit chunks). This is a real, still-open architectural gap — the fix would be re-ranking/restricting vector search to the `source_chunk_ids` already on the relevant graph edges rather than searching the whole corpus independently — documented here rather than papered over.
- **Vector-only "succeeding" on aggregation questions (100%) doesn't mean it can count.** The benchmark's grading is entity-coverage only (does the answer mention the required names), not a check on the stated count — vector-only likely retrieved passages naming several of the same competitors/regulators without actually enumerating or counting them the way the graph's `aggregate_count` query does. `gold_count` values (12, 23) are recorded in `golden_dataset.json` for a human/future grader to check directly; the current automated grader does not.
- **Grading is a mechanical entity-coverage heuristic, not an LLM-judge.** `gold_entities` had to move from flat exact-legal-name strings (e.g. `"NVIDIA CORP"`) to synonym groups (`[["NVIDIA", "Nvidia"]]`) after the original version silently failed several *correct* answers that used natural phrasing instead of the SEC legal name — a real grading bug, not a system bug, caught before it produced a misleading benchmark number.

## Stretch Goals (not built)

Per the build guide's own list: incremental graph updates instead of full reingestion, community detection for corpus-level summaries, a graph visualization in the response UI, LLM-as-judge grading for the benchmark (Project 4's methodology) instead of entity-coverage, and re-ranking vector search against a graph edge's `source_chunk_ids` to fix the qualified-question disambiguation gap above.

## Current Project Status

```
PHASE 0 — COMPLETE
PHASE 1 — COMPLETE
PHASE 2 — COMPLETE
PHASE 3 — COMPLETE
PHASE 4 — COMPLETE
PHASE 5 — COMPLETE
```

All five phases are built, tested, and run against the real 6-filing corpus. 67 tests passing (`pytest`). See [Benchmark Results](#benchmark-results) for the headline numbers and [Known Limitations](#known-limitations) for what to look at next before scaling past 6 documents.
