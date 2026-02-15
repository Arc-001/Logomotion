"""
Hybrid Retriever for Graph RAG.

Combines vector similarity search with graph traversal
to find the most relevant Manim code examples.
"""

from typing import Optional
from dataclasses import dataclass

from .db import GraphRAGClients


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    example_id: str
    prompt: str
    code: str
    score: float
    scene_class: Optional[str] = None
    used_classes: list[str] = None
    used_animations: list[str] = None
    related_concepts: list[str] = None

    def __post_init__(self):
        self.used_classes = self.used_classes or []
        self.used_animations = self.used_animations or []
        self.related_concepts = self.related_concepts or []


class ManimRetriever(GraphRAGClients):
    """Hybrid retriever for Manim code examples."""

    def __init__(self):
        super().__init__()

    def search_by_vector(self, query: str, limit: int = 5) -> list[str]:
        """Search for similar examples using vector embeddings."""
        if not self.collection or self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(limit, self.collection.count())
        )

        return results.get("ids", [[]])[0]

    def search_by_classes(self, class_names: list[str], limit: int = 5) -> list[str]:
        """Find examples that use specific Manim classes."""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (e:Example)-[:USES]->(c:ManimClass)
                WHERE c.name IN $class_names
                WITH e, COUNT(c) as match_count
                ORDER BY match_count DESC
                LIMIT $limit
                RETURN e.id as id
            """, class_names=class_names, limit=limit)
            return [record["id"] for record in result]

    def search_by_animations(self, animation_names: list[str], limit: int = 5) -> list[str]:
        """Find examples that use specific animations."""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (e:Example)-[:USES]->(a:Animation)
                WHERE a.name IN $animation_names
                WITH e, COUNT(a) as match_count
                ORDER BY match_count DESC
                LIMIT $limit
                RETURN e.id as id
            """, animation_names=animation_names, limit=limit)
            return [record["id"] for record in result]

    def search_by_concept(self, concept: str, limit: int = 5) -> list[str]:
        """Find examples that demonstrate a concept."""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (e:Example)-[:DEMONSTRATES]->(c:Concept)
                WHERE toLower(c.name) CONTAINS toLower($concept)
                LIMIT $limit
                RETURN e.id as id
            """, concept=concept, limit=limit)
            return [record["id"] for record in result]

    def get_related_examples(self, example_id: str, limit: int = 3) -> list[str]:
        """Find examples related through shared classes/animations."""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (e1:Example {id: $example_id})-[:USES]->(shared)<-[:USES]-(e2:Example)
                WHERE e1 <> e2
                WITH e2, COUNT(shared) as shared_count
                ORDER BY shared_count DESC
                LIMIT $limit
                RETURN e2.id as id
            """, example_id=example_id, limit=limit)
            return [record["id"] for record in result]

    def get_example_details(self, example_id: str) -> Optional[RetrievalResult]:
        """Get full details for an example."""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (e:Example {id: $example_id})
                OPTIONAL MATCH (e)-[:USES]->(c:ManimClass)
                OPTIONAL MATCH (e)-[:USES]->(a:Animation)
                OPTIONAL MATCH (e)-[:DEMONSTRATES]->(concept:Concept)
                RETURN e.id as id, e.prompt as prompt, e.code as code, 
                       e.scene_class as scene_class,
                       COLLECT(DISTINCT c.name) as used_classes,
                       COLLECT(DISTINCT a.name) as used_animations,
                       COLLECT(DISTINCT concept.name) as concepts
            """, example_id=example_id)

            record = result.single()
            if not record:
                return None

            return RetrievalResult(
                example_id=record["id"],
                prompt=record["prompt"],
                code=record["code"],
                score=1.0,
                scene_class=record["scene_class"],
                used_classes=record["used_classes"],
                used_animations=record["used_animations"],
                related_concepts=record["concepts"],
            )

    def hybrid_search(
        self,
        query: str,
        class_hints: Optional[list[str]] = None,
        animation_hints: Optional[list[str]] = None,
        limit: int = 5,
        vector_weight: float = 0.6,
    ) -> list[RetrievalResult]:
        """
        Hybrid search combining vector similarity and graph traversal.

        Args:
            query: Natural language query
            class_hints: Optional list of Manim classes to prioritize
            animation_hints: Optional list of animations to prioritize
            limit: Maximum results to return
            vector_weight: Weight for vector results (0-1)

        Returns:
            List of retrieval results sorted by relevance
        """
        candidates: dict[str, float] = {}

        vector_ids = self.search_by_vector(query, limit=limit * 2)
        for i, id_ in enumerate(vector_ids):
            score = (1 - i / len(vector_ids)) * vector_weight
            candidates[id_] = candidates.get(id_, 0) + score

        graph_weight = 1 - vector_weight

        if class_hints:
            class_ids = self.search_by_classes(class_hints, limit=limit)
            for i, id_ in enumerate(class_ids):
                score = (1 - i / max(len(class_ids), 1)) * graph_weight * 0.5
                candidates[id_] = candidates.get(id_, 0) + score

        if animation_hints:
            anim_ids = self.search_by_animations(animation_hints, limit=limit)
            for i, id_ in enumerate(anim_ids):
                score = (1 - i / max(len(anim_ids), 1)) * graph_weight * 0.5
                candidates[id_] = candidates.get(id_, 0) + score

        concept_ids = self.search_by_concept(query, limit=limit)
        for i, id_ in enumerate(concept_ids):
            score = (1 - i / max(len(concept_ids), 1)) * graph_weight * 0.3
            candidates[id_] = candidates.get(id_, 0) + score

        sorted_ids = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:limit]

        results = []
        for id_, score in sorted_ids:
            details = self.get_example_details(id_)
            if details:
                details.score = score
                results.append(details)

        return results

    def get_class_info(self, class_name: str) -> Optional[dict]:
        """Get information about a Manim class."""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (c:ManimClass {name: $name})
                OPTIONAL MATCH (e:Example)-[:USES]->(c)
                RETURN c.name as name, c.module as module, 
                       c.description as description,
                       c.is_scene as is_scene, c.is_mobject as is_mobject,
                       COUNT(e) as example_count
            """, name=class_name)

            record = result.single()
            if not record:
                return None

            return {
                "name": record["name"],
                "module": record["module"],
                "description": record["description"],
                "is_scene": record["is_scene"],
                "is_mobject": record["is_mobject"],
                "example_count": record["example_count"],
            }

    def get_animation_info(self, animation_name: str) -> Optional[dict]:
        """Get information about an animation."""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (a:Animation {name: $name})
                OPTIONAL MATCH (e:Example)-[:USES]->(a)
                RETURN a.name as name, a.description as description,
                       COUNT(e) as example_count
            """, name=animation_name)

            record = result.single()
            if not record:
                return None

            return {
                "name": record["name"],
                "description": record["description"],
                "example_count": record["example_count"],
            }
