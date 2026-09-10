"""
Tests for src.graph_rag.retriever fusion, concept matching, and curation gate.

No live Neo4j or ChromaDB: each source method is stubbed so the tests assert
how the sources are combined and which results are allowed through.
"""

import pytest

from src.graph_rag.retriever import ManimRetriever, RetrievalResult, _tokenize


CLEAN = """from manim import *

class DemoScene(Scene):
    def construct(self):
        self.play(Create(Circle()))
"""

MANIMGL = CLEAN.replace("from manim import *", "from manimlib import *")


def _result(example_id, code=CLEAN):
    return RetrievalResult(
        example_id=example_id, prompt=f"prompt {example_id}", code=code, score=0.0
    )


@pytest.fixture
def retriever():
    return ManimRetriever.__new__(ManimRetriever)


def _stub_sources(retriever, vector=(), classes=(), animations=(), concepts=(), details=None):
    retriever.search_by_vector = lambda query, limit=5: list(vector)
    retriever.search_by_classes = lambda names, limit=5: list(classes)
    retriever.search_by_animations = lambda names, limit=5: list(animations)
    retriever.search_by_concept = lambda query, limit=5: list(concepts)
    lookup = details if details is not None else {}
    retriever.get_example_details = lambda id_: lookup.get(id_, _result(id_))


class TestTokenize:
    def test_drops_stopwords_and_short_tokens(self):
        assert _tokenize("Explain the concept of a derivative") == ["concept", "derivative"]

    def test_empty_query_yields_no_tokens(self):
        assert _tokenize("the a of to") == []


class TestConceptSearch:
    def test_matches_on_query_tokens_not_the_whole_sentence(self, retriever):
        """A Concept node is named 'derivative'; the old query asked whether
        'derivative' CONTAINS the entire sentence, which never matched."""
        captured = {}

        class _Session:
            def run(self, query, **params):
                captured.update(params)
                captured["query"] = query
                return []

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        retriever._neo4j_driver = type("D", (), {"session": lambda self: _Session()})()

        retriever.search_by_concept("Explain the derivative of a function", limit=5)

        assert captured["tokens"] == ["derivative", "function"]
        assert "ANY(token IN $tokens" in captured["query"]

    def test_query_with_only_stopwords_skips_the_database(self, retriever):
        def _boom():
            raise AssertionError("must not open a session")

        retriever._neo4j_driver = property(lambda self: _boom())

        assert retriever.search_by_concept("the a of", limit=5) == []


class TestHybridSearchFusion:
    def test_an_id_ranked_by_several_sources_outranks_a_single_source_hit(self, retriever):
        _stub_sources(
            retriever,
            vector=["solo", "shared"],
            classes=["shared"],
            concepts=["shared"],
        )

        results = retriever.hybrid_search("circle", class_hints=["Circle"], limit=5)

        assert [r.example_id for r in results][0] == "shared"

    def test_source_contribution_does_not_depend_on_list_length(self, retriever):
        """Rank-position scoring gave the last item of a short list a score of
        zero and the last of a long list something else; RRF depends only on
        rank."""
        _stub_sources(retriever, vector=["a", "b"])
        short = {r.example_id: r.score for r in retriever.hybrid_search("q", limit=5)}

        _stub_sources(retriever, vector=["a", "b", "c", "d", "e"])
        long = {r.example_id: r.score for r in retriever.hybrid_search("q", limit=5)}

        assert short["a"] == pytest.approx(long["a"])
        assert short["b"] == pytest.approx(long["b"])

    def test_limit_is_respected(self, retriever):
        _stub_sources(retriever, vector=["a", "b", "c", "d"])

        assert len(retriever.hybrid_search("q", limit=2)) == 2


class TestHybridSearchCurationGate:
    def test_manimgl_example_never_reaches_the_caller(self, retriever):
        _stub_sources(
            retriever,
            vector=["gl", "ce"],
            details={"gl": _result("gl", MANIMGL), "ce": _result("ce", CLEAN)},
        )

        results = retriever.hybrid_search("q", limit=5)

        assert [r.example_id for r in results] == ["ce"]

    def test_gate_does_not_shrink_the_result_set_when_clean_ones_remain(self, retriever):
        _stub_sources(
            retriever,
            vector=["gl", "ce1", "ce2"],
            details={
                "gl": _result("gl", MANIMGL),
                "ce1": _result("ce1", CLEAN),
                "ce2": _result("ce2", CLEAN),
            },
        )

        results = retriever.hybrid_search("q", limit=2)

        assert [r.example_id for r in results] == ["ce1", "ce2"]

    def test_missing_details_are_skipped(self, retriever):
        _stub_sources(retriever, vector=["gone", "ce"])
        retriever.get_example_details = lambda id_: None if id_ == "gone" else _result(id_)

        assert [r.example_id for r in retriever.hybrid_search("q", limit=5)] == ["ce"]


class TestProcessWideRetriever:
    def test_get_retriever_returns_the_same_instance(self, monkeypatch):
        import src.graph_rag.retriever as module

        created = []

        class _Fake:
            def __init__(self):
                created.append(self)

            def close(self):
                pass

        monkeypatch.setattr(module, "ManimRetriever", _Fake)
        monkeypatch.setattr(module, "_retriever", None)

        first = module.get_retriever()
        second = module.get_retriever()

        assert first is second
        assert len(created) == 1
        module.close_retriever()

    def test_close_retriever_closes_and_forgets_it(self, monkeypatch):
        import src.graph_rag.retriever as module

        closed = []

        class _Fake:
            def close(self):
                closed.append(True)

        monkeypatch.setattr(module, "ManimRetriever", _Fake)
        monkeypatch.setattr(module, "_retriever", None)

        module.get_retriever()
        module.close_retriever()

        assert closed == [True]
        assert module._retriever is None
