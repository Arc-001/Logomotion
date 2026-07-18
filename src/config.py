"""
Centralized configuration for the Manim Graph RAG Agent.

Loads all settings from environment variables (with .env support)
so that no other module needs to call os.getenv() directly.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

_QUALITY_ALIASES = {
    "low": "l", "medium": "m", "high": "h", "production": "p", "4k": "k",
    "l": "l", "m": "m", "h": "h", "p": "p", "k": "k",
}


def normalize_render_quality(value: str) -> str:
    """Map quality names (low/medium/high/...) to manim's single-letter flags."""
    normalized = _QUALITY_ALIASES.get(value.strip().lower())
    if normalized is None:
        raise ValueError(
            f"Invalid render quality {value!r}; expected one of {sorted(set(_QUALITY_ALIASES))}"
        )
    return normalized


@dataclass(frozen=True)
class Settings:
    """Application-wide settings loaded from environment variables."""

    # API server
    cors_allow_origins: tuple = ("http://localhost:5173", "http://localhost:8000")

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
    render_quality: str = "m"  # manim quality flag: l, m, h, p, or k
    render_fps: Optional[int] = None  # None = manim's default for the quality

    # LLM call retries (transport-level, with exponential backoff)
    llm_retries: int = 3

    # Video generation defaults
    video_length: float = 1.0
    explanation_depth: str = "detailed"
    video_orientation: str = "landscape"
    duration_mode: str = "guide"  # "guide" = soft hint, "strict" = ffmpeg speed adjust
    storyboard_enabled: bool = True  # plan timed sections before writing code

    # Visual QA (multimodal review of rendered frames; opt-in — adds a
    # vision call and possibly a full re-render per attempt)
    visual_qa_enabled: bool = False
    visual_qa_max_attempts: int = 1
    visual_qa_frames: int = 6


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance populated from the environment."""
    return Settings(
        cors_allow_origins=tuple(
            origin.strip()
            for origin in os.getenv(
                "CORS_ALLOW_ORIGINS", "http://localhost:5173,http://localhost:8000"
            ).split(",")
            if origin.strip()
        ),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "google/gemini-3-flash-preview"),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        render_timeout=int(os.getenv("RENDER_TIMEOUT", "120")),
        render_quality=normalize_render_quality(os.getenv("RENDER_QUALITY", "m")),
        render_fps=int(os.getenv("RENDER_FPS")) if os.getenv("RENDER_FPS") else None,
        llm_retries=int(os.getenv("LLM_RETRIES", "3")),
        video_length=float(os.getenv("VIDEO_LENGTH", "1.0")),
        explanation_depth=os.getenv("EXPLANATION_DEPTH", "detailed"),
        video_orientation=os.getenv("VIDEO_ORIENTATION", "landscape"),
        duration_mode=os.getenv("DURATION_MODE", "guide"),
        storyboard_enabled=os.getenv("STORYBOARD_ENABLED", "true").strip().lower()
        in ("1", "true", "yes", "on"),
        visual_qa_enabled=os.getenv("VISUAL_QA_ENABLED", "false").strip().lower()
        in ("1", "true", "yes", "on"),
        visual_qa_max_attempts=int(os.getenv("VISUAL_QA_MAX_ATTEMPTS", "1")),
        visual_qa_frames=int(os.getenv("VISUAL_QA_FRAMES", "6")),
    )
