"""
Central configuration for the Graph RAG system.

Purpose:
    Load environment variables (from .env) and expose them as a single,
    validated settings object so the rest of the codebase never reads
    os.environ directly.

TODO (Phase 0+):
    - Define a Pydantic settings model covering:
        ANTHROPIC_API_KEY, SEC_USER_AGENT,
        NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD,
        POSTGRES_HOST / PORT / DB / USER / PASSWORD,
        EMBEDDING_PROVIDER / EMBEDDING_MODEL
    - Load values via python-dotenv.
    - Fail fast (clear error) if required variables are missing.
"""
