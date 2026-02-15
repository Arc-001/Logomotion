"""
Centralized configuration for the Manim Graph RAG Agent.

Loads all settings from environment variables (with .env support)
so that no other module needs to call os.getenv() directly.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application-wide settings loaded from environment variables."""

    # OpenRouter / LLM
    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemini-3-flash-preview"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"

    # Manim execution
    max_retries: int = 3
    render_timeout: int = 120

    # Video generation defaults
    video_length: float = 1.0
    explanation_depth: str = "detailed"
    video_orientation: str = "landscape"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance populated from the environment."""
    return Settings(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "google/gemini-3-flash-preview"),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        render_timeout=int(os.getenv("RENDER_TIMEOUT", "120")),
        video_length=float(os.getenv("VIDEO_LENGTH", "1.0")),
        explanation_depth=os.getenv("EXPLANATION_DEPTH", "detailed"),
        video_orientation=os.getenv("VIDEO_ORIENTATION", "landscape"),
    )
