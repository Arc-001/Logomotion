"""
Tests for src.agent.nodes.

These tests run with no live Neo4j/Chroma databases, no network access,
and no OPENROUTER_API_KEY: every LLM call goes through a monkeypatched
`llm_chat`, and the Graph RAG retriever is replaced with an in-memory stub
patched at its source module (`src.graph_rag.retriever.ManimRetriever`),
matching how `video_code_gen_node` imports it lazily inside the function.
"""

import shutil
import tempfile
import uuid
from pathlib import Path

import pytest

from src.agent.nodes import (
    _extract_code_block,
    _build_atempo_chain,
    _cleanup_temp_artifacts,
    _finish_merge,
    video_code_gen_node,
    recorrector_node,
    should_retry_or_continue,
)


# ============================================================================
# _extract_code_block
# ============================================================================

class TestExtractCodeBlock:
    def test_code_start_end_markers(self):
        text = "some preamble\n# CODE_START\nprint('hi')\n# CODE_END\nsome trailer"
        assert _extract_code_block(text) == "print('hi')"

    def test_python_fence(self):
        text = "Here you go:\n```python\nprint('hi')\n```\nThanks"
        assert _extract_code_block(text) == "print('hi')"

    def test_bare_fence(self):
        text = "```\nprint('hi')\n```"
        assert _extract_code_block(text) == "print('hi')"

    def test_raw_text_fallback(self):
        text = "  print('hi')  "
        assert _extract_code_block(text) == "print('hi')"

    def test_markers_take_priority_over_fences(self):
        text = "```python\nwrong\n```\n# CODE_START\nright\n# CODE_END"
        assert _extract_code_block(text) == "right"


# ============================================================================
# video_code_gen_node
# ============================================================================

CANNED_RESPONSE = '''Here is your animation:

```python
# CODE_START
from manim import *

class DemoScene(Scene):
    def construct(self):
        title = Text("Demo")
        self.play(Write(title))
        self.wait(1)
# CODE_END
```

```python
# TRANSCRIPT_START
transcript = {0: "hi", 5: "there"}
# TRANSCRIPT_END
```
'''


class _EmptyRetriever:
    """Stub ManimRetriever: no results, never touches Neo4j/Chroma."""

    def __init__(self):
        self.closed = False

    def hybrid_search(self, query, limit=3):
        return []

    def close(self):
        self.closed = True


class _RaisingRetriever:
    """Stub ManimRetriever whose search fails, simulating a DB outage."""

    def __init__(self):
        pass

    def hybrid_search(self, query, limit=3):
        raise RuntimeError("db unreachable")

    def close(self):
        pass


def _base_state(**overrides):
    state = {
        "system_message": "You are a helpful Manim animator.",
        "scene_title": "Pythagorean Theorem",
        "scene_prompt_description": "Explain the Pythagorean theorem visually.",
        "scene_length": 1.0,
    }
    state.update(overrides)
    return state


class TestVideoCodeGenNode:
    def test_generates_code_and_transcript_from_llm_response(self, monkeypatch):
        monkeypatch.setattr(
            "src.agent.nodes.llm_chat",
            lambda messages, temperature=0.2: CANNED_RESPONSE,
        )
        monkeypatch.setattr("src.graph_rag.retriever.ManimRetriever", _EmptyRetriever)

        result = video_code_gen_node(_base_state())

        assert "class DemoScene(Scene):" in result["code"]
        assert result["scene_class_name"] == "DemoScene"
        assert result["transcript"] == {0: "hi", 5: "there"}
        assert result["retrieved_examples"] == []
        assert "No similar examples found." in result["retrieved_context"]

    def test_malformed_transcript_falls_back_to_empty_dict(self, monkeypatch):
        response = CANNED_RESPONSE.replace(
            'transcript = {0: "hi", 5: "there"}',
            'transcript = {0: undefined_name}',
        )
        monkeypatch.setattr(
            "src.agent.nodes.llm_chat", lambda messages, temperature=0.2: response
        )
        monkeypatch.setattr("src.graph_rag.retriever.ManimRetriever", _EmptyRetriever)

        result = video_code_gen_node(_base_state())

        assert result["transcript"] == {}
        assert result["scene_class_name"] == "DemoScene"

    def test_llm_returns_none_uses_fallback_scene(self, monkeypatch):
        monkeypatch.setattr(
            "src.agent.nodes.llm_chat", lambda messages, temperature=0.2: None
        )
        monkeypatch.setattr("src.graph_rag.retriever.ManimRetriever", _EmptyRetriever)

        state = _base_state(scene_title="Fallback Title")
        result = video_code_gen_node(state)

        assert result["scene_class_name"] == "GeneratedScene"
        assert "class GeneratedScene(Scene):" in result["code"]
        assert result["transcript"] == {0: "Welcome to Fallback Title"}

    def test_rag_failure_is_recovered_and_code_still_generated(self, monkeypatch):
        monkeypatch.setattr(
            "src.agent.nodes.llm_chat",
            lambda messages, temperature=0.2: CANNED_RESPONSE,
        )
        monkeypatch.setattr("src.graph_rag.retriever.ManimRetriever", _RaisingRetriever)

        result = video_code_gen_node(_base_state())

        assert result["retrieved_examples"] == []
        assert "RAG retrieval failed" in result["retrieved_context"]
        # Code generation still succeeds despite the RAG outage.
        assert result["scene_class_name"] == "DemoScene"

    def test_accepts_optional_depth_orientation_duration_mode(self, monkeypatch):
        monkeypatch.setattr(
            "src.agent.nodes.llm_chat",
            lambda messages, temperature=0.2: CANNED_RESPONSE,
        )
        monkeypatch.setattr("src.graph_rag.retriever.ManimRetriever", _EmptyRetriever)

        state = _base_state(
            explanation_depth="comprehensive",
            orientation="portrait",
            duration_mode="strict",
        )
        result = video_code_gen_node(state)

        assert result["scene_class_name"] == "DemoScene"
        assert result["transcript"] == {0: "hi", 5: "there"}


# ============================================================================
# _build_atempo_chain
# ============================================================================

class TestBuildAtempoChain:
    def test_factor_within_single_stage_range(self):
        assert _build_atempo_chain(1.5) == "atempo=1.5000"

    def test_factor_above_2x_chains_stages(self):
        result = _build_atempo_chain(4.0)
        stages = result.split(",")
        assert len(stages) == 2
        for stage in stages:
            assert stage.startswith("atempo=")
            assert float(stage.split("=")[1]) == pytest.approx(2.0)

    def test_factor_below_half_chains_stages(self):
        result = _build_atempo_chain(0.25)
        stages = result.split(",")
        assert len(stages) == 2
        for stage in stages:
            assert float(stage.split("=")[1]) == pytest.approx(0.5)

    def test_non_positive_factor_defaults_to_unity(self):
        assert _build_atempo_chain(0) == "atempo=1.0"
        assert _build_atempo_chain(-3.0) == "atempo=1.0"


# ============================================================================
# should_retry_or_continue
# ============================================================================

class TestShouldRetryOrContinue:
    def test_error_below_max_retries_goes_to_recorrector(self):
        result = should_retry_or_continue(
            {"error": "boom", "error_count": 1, "max_retries": 3}
        )
        assert result == "recorrector"

    def test_error_at_max_retries_goes_to_render_checker(self):
        result = should_retry_or_continue(
            {"error": "boom", "error_count": 3, "max_retries": 3}
        )
        assert result == "render_checker"

    def test_no_error_goes_to_render_checker(self):
        result = should_retry_or_continue(
            {"error": None, "error_count": 0, "max_retries": 3}
        )
        assert result == "render_checker"


# ============================================================================
# _cleanup_temp_artifacts
# ============================================================================

class TestCleanupTempArtifacts:
    def test_removes_known_prefixed_dir_and_kokoro_wav(self):
        tmp_root = Path(tempfile.gettempdir())
        exec_dir = Path(tempfile.mkdtemp(prefix="manim_exec_test_"))
        (exec_dir / "scene.py").write_text("# dummy")

        wav_path = tmp_root / f"kokoro_{uuid.uuid4().hex}.wav"
        wav_path.write_bytes(b"RIFF....")

        unrelated_dir = Path(tempfile.mkdtemp(prefix="unrelated_prefix_"))

        state = {
            "temp_dirs": [str(exec_dir), str(unrelated_dir), None],
            "audio_segments": [str(wav_path), None, ""],
        }

        try:
            _cleanup_temp_artifacts(state)

            assert not exec_dir.exists()
            assert not wav_path.exists()
            assert unrelated_dir.exists()
        finally:
            shutil.rmtree(unrelated_dir, ignore_errors=True)
            shutil.rmtree(exec_dir, ignore_errors=True)
            if wav_path.exists():
                wav_path.unlink()

    def test_refuses_etc_paths(self):
        state = {"temp_dirs": ["/etc"], "audio_segments": []}

        _cleanup_temp_artifacts(state)

        assert Path("/etc").exists()

    def test_tolerates_none_and_empty_entries(self):
        state = {"temp_dirs": [None, ""], "audio_segments": [None, ""]}

        # Should not raise.
        _cleanup_temp_artifacts(state)

    def test_extra_dirs_argument_is_also_cleaned(self):
        extra_dir = Path(tempfile.mkdtemp(prefix="manim_merge_test_"))
        state = {"temp_dirs": [], "audio_segments": []}

        _cleanup_temp_artifacts(state, extra_dirs=[str(extra_dir)])

        assert not extra_dir.exists()


# ============================================================================
# _finish_merge
# ============================================================================

class TestFinishMerge:
    def test_persists_video_to_output_dir_and_removes_temp_dir(self):
        merge_dir = tempfile.mkdtemp(prefix="manim_merge_test_")
        video_path = Path(merge_dir) / "final.mp4"
        video_path.write_bytes(b"fake mp4 bytes")

        state = {"temp_dirs": [], "audio_segments": []}
        result = None
        try:
            result = _finish_merge(state, str(video_path), merge_dir)

            final_path = result["final_output_path"]
            assert final_path is not None
            assert Path(final_path).exists()
            assert Path(final_path).parent == Path("output").resolve()
            # Temp dir should be removed once the file has been persisted.
            assert not Path(merge_dir).exists()
        finally:
            if result and result.get("final_output_path"):
                Path(result["final_output_path"]).unlink(missing_ok=True)
            shutil.rmtree(merge_dir, ignore_errors=True)

    def test_none_video_returns_none_output_path(self):
        state = {"temp_dirs": [], "audio_segments": []}

        result = _finish_merge(state, None)

        assert result == {"final_output_path": None}


# ============================================================================
# recorrector_node
# ============================================================================

class TestRecorrectorNode:
    def test_returns_stripped_fixed_code_on_success(self, monkeypatch):
        monkeypatch.setattr(
            "src.agent.nodes.llm_chat",
            lambda messages, temperature=0.1: "```python\nfixed = True\n```",
        )
        state = {"code": "broken = True", "error": "NameError: broken", "error_count": 0}

        result = recorrector_node(state)

        assert result["code"] == "fixed = True"
        assert result["error"] is None

    def test_preserves_code_with_error_prefix_when_llm_unavailable(self, monkeypatch):
        monkeypatch.setattr(
            "src.agent.nodes.llm_chat", lambda messages, temperature=0.1: None
        )
        state = {"code": "broken = True", "error": "NameError: broken", "error_count": 1}

        result = recorrector_node(state)

        assert result["code"].startswith("# Error was: NameError: broken")
        assert "broken = True" in result["code"]
        assert result["error"] is None
