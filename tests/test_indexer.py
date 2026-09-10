"""
Tests for src.graph_rag.indexer batching.

No live Neo4j or ChromaDB: the driver and collection are replaced with
recorders so the tests assert how many round trips a batch costs, which is
the whole point of batching.
"""

import pytest

from src.graph_rag.indexer import ManimIndexer


CLEAN = """from manim import *

class DemoScene(Scene):
    def construct(self):
        self.play(Create(Circle()))
"""


class _RecordingSession:
    def __init__(self, queries):
        self.queries = queries

    def run(self, query, **params):
        self.queries.append((query.strip().split("\n")[0].strip(), params))
        return []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _RecordingDriver:
    def __init__(self):
        self.queries = []

    def session(self):
        return _RecordingSession(self.queries)


class _RecordingCollection:
    def __init__(self):
        self.upserts = []

    def upsert(self, ids, documents, metadatas):
        self.upserts.append({"ids": ids, "documents": documents, "metadatas": metadatas})


@pytest.fixture
def indexer():
    idx = ManimIndexer.__new__(ManimIndexer)
    idx._neo4j_driver = _RecordingDriver()
    idx._chroma_client = None
    idx._collection = _RecordingCollection()
    return idx


class TestPrepareExample:
    def test_extracts_classes_animations_and_concepts(self, indexer):
        row = indexer._prepare_example("draw a circle", CLEAN, "abc")

        assert row["id"] == "abc"
        assert row["scene_class"] == "DemoScene"
        assert "Circle" in row["used_classes"]
        assert "Create" in row["used_animations"]
        assert "circle" in row["concepts"]

    def test_touches_no_database(self, indexer):
        indexer._prepare_example("draw a circle", CLEAN, "abc")

        assert indexer._neo4j_driver.queries == []
        assert indexer._collection.upserts == []


class TestWriteBatch:
    def _rows(self, indexer, n):
        return [
            indexer._prepare_example(f"draw a circle {i}", CLEAN, f"id-{i}")
            for i in range(n)
        ]

    def test_a_whole_batch_costs_one_embedding_call(self, indexer):
        indexer._write_batch(self._rows(indexer, 20))

        assert len(indexer._collection.upserts) == 1
        assert len(indexer._collection.upserts[0]["ids"]) == 20

    def test_a_whole_batch_costs_four_cypher_statements(self, indexer):
        indexer._write_batch(self._rows(indexer, 20))

        assert len(indexer._neo4j_driver.queries) == 4
        assert all(q.startswith("UNWIND") for q, _ in indexer._neo4j_driver.queries)

    def test_relationship_rows_are_flattened_across_the_batch(self, indexer):
        indexer._write_batch(self._rows(indexer, 3))

        # Every example contributes its own pairs, each tagged with its own id.
        pairs = [
            pair
            for _, params in indexer._neo4j_driver.queries
            for pair in params.get("pairs", [])
        ]
        assert {pair["id"] for pair in pairs} == {"id-0", "id-1", "id-2"}
        assert {pair["name"] for pair in pairs} >= {"Circle", "Create", "circle"}

    def test_empty_batch_is_a_no_op(self, indexer):
        indexer._write_batch([])

        assert indexer._neo4j_driver.queries == []
        assert indexer._collection.upserts == []

    def test_statements_with_nothing_to_write_are_skipped(self, indexer):
        # A scene with no known classes, animations or concepts: only the
        # example MERGE should run.
        row = indexer._prepare_example("zzz", "from manim import *\n\nclass A(Foo):\n    pass\n", "solo")
        row["used_classes"] = []
        row["used_animations"] = []
        row["concepts"] = []
        indexer._write_batch([row])

        assert len(indexer._neo4j_driver.queries) == 1

    def test_index_example_still_writes_a_single_row(self, indexer):
        returned = indexer.index_example("draw a circle", CLEAN, "solo")

        assert returned == "solo"
        assert indexer._collection.upserts[0]["ids"] == ["solo"]
