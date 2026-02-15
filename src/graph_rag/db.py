"""
Shared database client management for Graph RAG.

Provides lazy-initialised Neo4j and ChromaDB connections used
by both the Indexer and the Retriever.
"""

from neo4j import GraphDatabase

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None

from ..config import get_settings


class GraphRAGClients:
    """Mixin providing lazy-loaded Neo4j driver, ChromaDB client, and collection."""

    def __init__(self):
        self._neo4j_driver = None
        self._chroma_client = None
        self._collection = None

    @property
    def neo4j_driver(self):
        """Lazy-load Neo4j driver."""
        if self._neo4j_driver is None:
            settings = get_settings()
            self._neo4j_driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
        return self._neo4j_driver

    @property
    def chroma_client(self):
        """Lazy-load ChromaDB client."""
        if self._chroma_client is None and chromadb:
            settings = get_settings()
            self._chroma_client = chromadb.PersistentClient(
                path=settings.chroma_persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._chroma_client

    @property
    def collection(self):
        """Get or create ChromaDB collection with OpenRouter embeddings."""
        if self._collection is None and self.chroma_client:
            try:
                from .embeddings import OpenRouterEmbeddingFunction

                embedding_fn = OpenRouterEmbeddingFunction(
                    model="google/gemini-embedding-001",
                )
                self._collection = self.chroma_client.get_or_create_collection(
                    name="manim_examples",
                    metadata={"description": "Manim code examples with embeddings"},
                    embedding_function=embedding_fn,
                )
            except Exception as e:
                print(f"OpenRouter embedding failed: {e}")
                raise
        return self._collection

    def close(self):
        """Close database connections."""
        if self._neo4j_driver:
            self._neo4j_driver.close()
