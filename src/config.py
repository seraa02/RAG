"""
Central configuration for the Graph RAG system.

Purpose:
    Load environment variables (from .env) and expose them as a single,
    validated settings object so the rest of the codebase never reads
    os.environ directly. Missing required variables raise a
    pydantic.ValidationError immediately on first use ("fail fast"),
    rather than failing deep inside an ingestion run.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Anthropic (Claude) ---
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-5"

    # --- SEC EDGAR access ---
    sec_user_agent: str

    # --- Neo4j (Knowledge Graph) ---
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # --- PostgreSQL + pgvector (Vector DB) ---
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str

    # --- Embedding configuration (finalized in Phase 2) ---
    embedding_provider: str = "undecided"
    embedding_model: str = "undecided"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Load once and cache. Import this function, never instantiate Settings() directly,
    so the whole process shares one validated config object."""
    return Settings()
