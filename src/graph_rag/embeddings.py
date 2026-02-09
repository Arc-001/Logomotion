"""
OpenRouter Embedding Function for ChromaDB.

Uses OpenRouter's embedding API with Google's Gemini embedding model.
"""

import os
import requests
from typing import List

try:
    from chromadb import EmbeddingFunction, Documents, Embeddings
    CHROMADB_AVAILABLE = True
except ImportError:
    EmbeddingFunction = object
    Documents = List[str]
    Embeddings = List[List[float]]
    CHROMADB_AVAILABLE = False


class OpenRouterEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function using OpenRouter's API.
    
    Uses google/gemini-embedding-001 by default.
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "google/gemini-embedding-001",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.base_url = base_url.rstrip("/")
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")
    
    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for a list of documents."""
        if not input:
            return []
        
        # OpenRouter embedding endpoint
        url = f"{self.base_url}/embeddings"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://manim-agent.local",
            "X-Title": "Manim Graph RAG Agent",
        }
        
        payload = {
            "model": self.model,
            "input": input,
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract embeddings in order
            embeddings = [None] * len(input)
            for item in data.get("data", []):
                idx = item.get("index", 0)
                embedding = item.get("embedding", [])
                if idx < len(embeddings):
                    embeddings[idx] = embedding
            
            # Verify all embeddings were received
            if None in embeddings:
                raise ValueError("Some embeddings were not returned")
            
            return embeddings
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenRouter embedding request failed: {e}")
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Failed to parse embedding response: {e}")


def get_embedding_function() -> OpenRouterEmbeddingFunction:
    """Get the configured embedding function."""
    return OpenRouterEmbeddingFunction()
