"""
Graph RAG Indexer for Manim Dataset.

Parses JSONL files and builds:
1. Knowledge graph in Neo4j
2. Vector embeddings in ChromaDB
"""

import json
import re
import hashlib
from pathlib import Path
from typing import Generator, Optional

from pydantic import BaseModel
from neo4j import GraphDatabase

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None


from .schema import (
    NodeType,
    RelationType,
    ExampleNode,
    KNOWN_MANIM_CLASSES,
    KNOWN_ANIMATIONS,
    SCHEMA_CONSTRAINTS,
    SCHEMA_INDEXES,
)


class IndexerConfig(BaseModel):
    """Configuration for the indexer."""
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    chroma_persist_dir: str = "./chroma_db"
    batch_size: int = 100


class ManimIndexer:
    """Indexes Manim code examples into Graph RAG system."""
    
    def __init__(self, config: Optional[IndexerConfig] = None):
        self.config = config or IndexerConfig()
        self._neo4j_driver = None
        self._chroma_client = None
        self._collection = None
    
    @property
    def neo4j_driver(self):
        """Lazy-load Neo4j driver."""
        if self._neo4j_driver is None:
            self._neo4j_driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
        return self._neo4j_driver
    
    @property
    def chroma_client(self):
        """Lazy-load ChromaDB client."""
        if self._chroma_client is None and chromadb:
            self._chroma_client = chromadb.PersistentClient(
                path=self.config.chroma_persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )
        return self._chroma_client
    
    @property
    def collection(self):
        """Get or create ChromaDB collection with OpenRouter embeddings."""
        if self._collection is None and self.chroma_client:
            try:
                from .embeddings import OpenRouterEmbeddingFunction
                embedding_fn = OpenRouterEmbeddingFunction(
                    model="google/gemini-embedding-001"
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
    
    def _generate_id(self, text: str) -> str:
        """Generate unique ID from text."""
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def _parse_jsonl(self, file_path: Path) -> Generator[dict, None, None]:
        """Parse JSONL file and yield examples."""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    messages = data.get("messages", [])
                    if len(messages) >= 2:
                        user_msg = next((m for m in messages if m.get("role") == "user"), None)
                        assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)
                        if user_msg and assistant_msg:
                            yield {
                                "prompt": user_msg.get("content", ""),
                                "code": assistant_msg.get("content", ""),
                                "line_num": line_num,
                                "file": str(file_path)
                            }
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num} in {file_path}: {e}")
    
    def _extract_scene_class(self, code: str) -> Optional[str]:
        """Extract the main Scene class name from code."""
        pattern = r'class\s+(\w+)\s*\([^)]*Scene[^)]*\)'
        match = re.search(pattern, code)
        return match.group(1) if match else None
    
    def _extract_imports(self, code: str) -> list[str]:
        """Extract import statements from code."""
        patterns = [
            r'^from\s+[\w.]+\s+import\s+.+$',
            r'^import\s+[\w.]+.*$'
        ]
        imports = []
        for line in code.split('\n'):
            for pattern in patterns:
                if re.match(pattern, line.strip()):
                    imports.append(line.strip())
                    break
        return imports
    
    def _extract_used_classes(self, code: str) -> list[str]:
        """Extract Manim class names used in code."""
        known_names = {c.name for c in KNOWN_MANIM_CLASSES}
        used = set()
        for name in known_names:
            if re.search(rf'\b{name}\b', code):
                used.add(name)
        return list(used)
    
    def _extract_used_animations(self, code: str) -> list[str]:
        """Extract animation names used in code."""
        known_names = {a.name for a in KNOWN_ANIMATIONS}
        used = set()
        for name in known_names:
            if re.search(rf'\b{name}\b', code):
                used.add(name)
        return list(used)
    
    def _extract_concepts(self, prompt: str) -> list[str]:
        """Extract concepts from the prompt."""
        concept_keywords = [
            "3d", "3-d", "three dimensional",
            "vector field", "magnetic field", "electric field",
            "graph", "function", "derivative", "integral", 
            "matrix", "transformation", "rotation",
            "circle", "square", "triangle", "polygon",
            "animation", "morph", "transform",
            "complex", "plane", "coordinate",
            "physics", "math", "geometry",
            "probability", "statistics",
        ]
        prompt_lower = prompt.lower()
        return [c for c in concept_keywords if c in prompt_lower]
    
    def init_schema(self):
        """Initialize Neo4j schema with constraints and indexes."""
        with self.neo4j_driver.session() as session:
            for constraint in SCHEMA_CONSTRAINTS.strip().split(';'):
                constraint = constraint.strip()
                if constraint:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        print(f"Constraint may already exist: {e}")
            
            for index in SCHEMA_INDEXES.strip().split(';'):
                index = index.strip()
                if index:
                    try:
                        session.run(index)
                    except Exception as e:
                        print(f"Index may already exist: {e}")
    
    def seed_known_entities(self):
        """Seed the graph with known Manim classes and animations."""
        with self.neo4j_driver.session() as session:
            for cls in KNOWN_MANIM_CLASSES:
                session.run("""
                    MERGE (c:ManimClass {name: $name})
                    SET c.module = $module,
                        c.description = $description,
                        c.is_scene = $is_scene,
                        c.is_mobject = $is_mobject
                """, name=cls.name, module=cls.module, description=cls.description,
                    is_scene=cls.is_scene, is_mobject=cls.is_mobject)
            
            for anim in KNOWN_ANIMATIONS:
                session.run("""
                    MERGE (a:Animation {name: $name})
                    SET a.description = $description
                """, name=anim.name, description=anim.description)
            
            print(f"Seeded {len(KNOWN_MANIM_CLASSES)} classes and {len(KNOWN_ANIMATIONS)} animations")
    
    def index_example(self, prompt: str, code: str, example_id: Optional[str] = None) -> str:
        """Index a single example into both Neo4j and ChromaDB."""
        example_id = example_id or self._generate_id(prompt + code)
        scene_class = self._extract_scene_class(code)
        used_classes = self._extract_used_classes(code)
        used_animations = self._extract_used_animations(code)
        concepts = self._extract_concepts(prompt)
        
        with self.neo4j_driver.session() as session:
            session.run("""
                MERGE (e:Example {id: $id})
                SET e.prompt = $prompt,
                    e.code = $code,
                    e.scene_class = $scene_class
            """, id=example_id, prompt=prompt, code=code, scene_class=scene_class)
            
            for cls_name in used_classes:
                session.run("""
                    MATCH (e:Example {id: $example_id})
                    MATCH (c:ManimClass {name: $class_name})
                    MERGE (e)-[:USES]->(c)
                """, example_id=example_id, class_name=cls_name)
            
            for anim_name in used_animations:
                session.run("""
                    MATCH (e:Example {id: $example_id})
                    MATCH (a:Animation {name: $anim_name})
                    MERGE (e)-[:USES]->(a)
                """, example_id=example_id, anim_name=anim_name)
            
            for concept in concepts:
                session.run("""
                    MERGE (c:Concept {name: $name})
                    WITH c
                    MATCH (e:Example {id: $example_id})
                    MERGE (e)-[:DEMONSTRATES]->(c)
                """, name=concept, example_id=example_id)
        
        # Add to ChromaDB with embeddings (uses ChromaDB's built-in embedding function)
        if self.collection:
            # Create embedding text (combine prompt and code summary)
            embedding_text = f"Prompt: {prompt}\nScene: {scene_class or 'Unknown'}\nUses: {', '.join(used_classes + used_animations)}"
            
            self.collection.upsert(
                ids=[example_id],
                documents=[embedding_text],
                metadatas=[{
                    "prompt": prompt[:1000],  # Truncate for metadata
                    "scene_class": scene_class or "",
                    "used_classes": ",".join(used_classes),
                    "used_animations": ",".join(used_animations),
                }]
            )
        
        return example_id
    
    def index_directory(self, data_dir: str, pattern: str = "*.jsonl"):
        """Index all JSONL files in a directory."""
        data_path = Path(data_dir)
        if not data_path.exists():
            raise ValueError(f"Directory not found: {data_dir}")
        
        print("Initializing schema...")
        self.init_schema()
        
        print("Seeding known entities...")
        self.seed_known_entities()
        
        files = list(data_path.glob(pattern))
        print(f"Found {len(files)} JSONL files")
        
        total_indexed = 0
        for file_path in files:
            print(f"Processing {file_path.name}...")
            file_count = 0
            for example in self._parse_jsonl(file_path):
                self.index_example(
                    prompt=example["prompt"],
                    code=example["code"],
                    example_id=self._generate_id(f"{file_path.name}:{example['line_num']}")
                )
                file_count += 1
                total_indexed += 1
                
                if total_indexed % 100 == 0:
                    print(f"  Indexed {total_indexed} examples...")
            
            print(f"  Completed {file_path.name}: {file_count} examples")
        
        print(f"\nTotal indexed: {total_indexed} examples")
        return total_indexed


def index_dataset(data_dir: str, config: Optional[IndexerConfig] = None):
    """Main function to index the Manim dataset."""
    indexer = ManimIndexer(config)
    try:
        return indexer.index_directory(data_dir)
    finally:
        indexer.close()
