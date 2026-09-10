"""
Graph RAG Indexer for Manim Dataset.

Parses JSONL files and builds:
1. Knowledge graph in Neo4j
2. Vector embeddings in ChromaDB
"""

import json
import re
import hashlib
from collections import Counter
from pathlib import Path
from typing import Generator, Optional

from .curation import example_rejection_reason
from .db import GraphRAGClients
from .schema import (
    ExampleNode,
    KNOWN_MANIM_CLASSES,
    KNOWN_ANIMATIONS,
    SCHEMA_CONSTRAINTS,
    SCHEMA_INDEXES,
)


class ManimIndexer(GraphRAGClients):
    """Indexes Manim code examples into Graph RAG system.

    Connection settings come from ``Settings`` (via ``GraphRAGClients``).
    """

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

    # Examples per write batch. One ChromaDB upsert means one embedding HTTP
    # call, so batching is the difference between ~1200 round trips and ~20.
    BATCH_SIZE = 64

    def _prepare_example(self, prompt: str, code: str, example_id: Optional[str] = None) -> dict:
        """Extract everything an example contributes, without touching a database."""
        return {
            "id": example_id or self._generate_id(prompt + code),
            "prompt": prompt,
            "code": code,
            "scene_class": self._extract_scene_class(code),
            "used_classes": self._extract_used_classes(code),
            "used_animations": self._extract_used_animations(code),
            "concepts": self._extract_concepts(prompt),
        }

    def _write_batch(self, rows: list[dict]) -> None:
        """Write a batch of prepared examples to Neo4j and ChromaDB.

        Four Cypher statements for the whole batch rather than roughly ten per
        example, and a single embedding call for all the documents.
        """
        if not rows:
            return

        class_pairs = [
            {"id": row["id"], "name": name}
            for row in rows for name in row["used_classes"]
        ]
        animation_pairs = [
            {"id": row["id"], "name": name}
            for row in rows for name in row["used_animations"]
        ]
        concept_pairs = [
            {"id": row["id"], "name": name}
            for row in rows for name in row["concepts"]
        ]

        with self.neo4j_driver.session() as session:
            session.run("""
                UNWIND $rows AS row
                MERGE (e:Example {id: row.id})
                SET e.prompt = row.prompt,
                    e.code = row.code,
                    e.scene_class = row.scene_class
            """, rows=[
                {k: row[k] for k in ("id", "prompt", "code", "scene_class")}
                for row in rows
            ])

            if class_pairs:
                session.run("""
                    UNWIND $pairs AS pair
                    MATCH (e:Example {id: pair.id})
                    MATCH (c:ManimClass {name: pair.name})
                    MERGE (e)-[:USES]->(c)
                """, pairs=class_pairs)

            if animation_pairs:
                session.run("""
                    UNWIND $pairs AS pair
                    MATCH (e:Example {id: pair.id})
                    MATCH (a:Animation {name: pair.name})
                    MERGE (e)-[:USES]->(a)
                """, pairs=animation_pairs)

            if concept_pairs:
                session.run("""
                    UNWIND $pairs AS pair
                    MERGE (c:Concept {name: pair.name})
                    WITH c, pair
                    MATCH (e:Example {id: pair.id})
                    MERGE (e)-[:DEMONSTRATES]->(c)
                """, pairs=concept_pairs)

        if self.collection:
            self.collection.upsert(
                ids=[row["id"] for row in rows],
                documents=[
                    f"Prompt: {row['prompt']}\n"
                    f"Scene: {row['scene_class'] or 'Unknown'}\n"
                    f"Uses: {', '.join(row['used_classes'] + row['used_animations'])}"
                    for row in rows
                ],
                metadatas=[
                    {
                        "prompt": row["prompt"][:1000],
                        "scene_class": row["scene_class"] or "",
                        "used_classes": ",".join(row["used_classes"]),
                        "used_animations": ",".join(row["used_animations"]),
                    }
                    for row in rows
                ],
            )

    def index_example(self, prompt: str, code: str, example_id: Optional[str] = None) -> str:
        """Index a single example into both Neo4j and ChromaDB."""
        row = self._prepare_example(prompt, code, example_id)
        self._write_batch([row])
        return row["id"]

    def reset(self) -> None:
        """Drop every indexed example so the store can be rebuilt from scratch.

        Curation only changes what a *new* indexing run writes; entries an
        earlier run already stored stay until they are deleted. Known classes
        and animations are seeded state, not example data, so they survive.
        """
        print("Clearing indexed examples...")

        with self.neo4j_driver.session() as session:
            session.run("MATCH (e:Example) DETACH DELETE e")
            session.run("MATCH (c:Concept) DETACH DELETE c")

        if self.chroma_client:
            try:
                self.chroma_client.delete_collection("manim_examples")
            except Exception as e:
                print(f"  ChromaDB collection not dropped ({e}); continuing")
            self._collection = None  # recreated on next access

    def index_directory(self, data_dir: str, pattern: str = "*.jsonl", rebuild: bool = False):
        """Index all JSONL files in a directory."""
        data_path = Path(data_dir)
        if not data_path.exists():
            raise ValueError(f"Directory not found: {data_dir}")

        if rebuild:
            self.reset()

        print("Initializing schema...")
        self.init_schema()

        print("Seeding known entities...")
        self.seed_known_entities()

        files = list(data_path.glob(pattern))
        print(f"Found {len(files)} JSONL files")

        total_indexed = 0
        total_skipped = 0
        skip_reasons: Counter = Counter()
        batch: list[dict] = []

        for file_path in files:
            print(f"Processing {file_path.name}...")
            file_count = 0
            file_skipped = 0
            for example in self._parse_jsonl(file_path):
                # Retrieved examples are shown to the code generator as things
                # to imitate, so anything that cannot run on its own is worse
                # than no example at all.
                reason = example_rejection_reason(example["code"])
                if reason:
                    file_skipped += 1
                    total_skipped += 1
                    skip_reasons[reason.split(":")[0]] += 1
                    continue

                batch.append(self._prepare_example(
                    prompt=example["prompt"],
                    code=example["code"],
                    example_id=self._generate_id(f"{file_path.name}:{example['line_num']}"),
                ))
                file_count += 1
                total_indexed += 1

                if len(batch) >= self.BATCH_SIZE:
                    self._write_batch(batch)
                    batch = []
                    print(f"  Indexed {total_indexed} examples...")

            self._write_batch(batch)
            batch = []
            print(f"  Completed {file_path.name}: {file_count} indexed, {file_skipped} skipped")

        print(f"\nTotal indexed: {total_indexed} examples ({total_skipped} skipped)")
        for reason, count in skip_reasons.most_common():
            print(f"  skipped {count:5d}: {reason}")
        return total_indexed


def index_dataset(data_dir: str):
    """Main function to index the Manim dataset."""
    indexer = ManimIndexer()
    try:
        return indexer.index_directory(data_dir)
    finally:
        indexer.close()
