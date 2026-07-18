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

    def hybrid_search(self, query, limit=3, **kwargs):
        return []

    def close(self):
        self.closed = True


class _RaisingRetriever:
    """Stub ManimRetriever whose search fails, simulating a DB outage."""

    def __init__(self):
        pass

    def hybrid_search(self, query, limit=3, **kwargs):
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


# ============================================================================
# llm_chat retry behavior
# ============================================================================

class _FlakyCompletions:
    """chat.completions stub that fails a set number of times, then succeeds."""

    def __init__(self, failures: int, content: str = "ok"):
        self.failures = failures
        self.calls = 0
        self.content = content

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("transient failure")

        class _Msg:
            content = self.content

        class _Choice:
            message = _Msg()

        class _Completion:
            choices = [_Choice()]

        return _Completion()


def _fake_client(completions):
    class _Chat:
        pass

    class _Client:
        chat = _Chat()

    _Client.chat.completions = completions
    return _Client()


class TestLlmChatRetry:
    def test_retries_transient_failures_then_succeeds(self, monkeypatch):
        from src.agent import nodes

        completions = _FlakyCompletions(failures=2)
        monkeypatch.setattr(nodes, "get_llm_client", lambda: _fake_client(completions))
        monkeypatch.setattr(nodes.time, "sleep", lambda s: None)

        result = nodes.llm_chat([{"role": "user", "content": "hi"}])

        assert result == "ok"
        assert completions.calls == 3

    def test_returns_none_after_exhausting_retries(self, monkeypatch):
        from src.agent import nodes

        completions = _FlakyCompletions(failures=100)
        monkeypatch.setattr(nodes, "get_llm_client", lambda: _fake_client(completions))
        monkeypatch.setattr(nodes.time, "sleep", lambda s: None)

        result = nodes.llm_chat([{"role": "user", "content": "hi"}])

        assert result is None
        assert completions.calls == 3

    def test_empty_response_is_retried(self, monkeypatch):
        from src.agent import nodes

        completions = _FlakyCompletions(failures=0, content="")
        monkeypatch.setattr(nodes, "get_llm_client", lambda: _fake_client(completions))
        monkeypatch.setattr(nodes.time, "sleep", lambda s: None)

        result = nodes.llm_chat([{"role": "user", "content": "hi"}])

        assert result is None
        assert completions.calls == 3


# ============================================================================
# _extract_rag_hints / _truncate_code_example
# ============================================================================

class TestExtractRagHints:
    def test_direct_class_and_animation_names(self):
        from src.agent.nodes import _extract_rag_hints

        classes, animations = _extract_rag_hints("Show a Circle and FadeIn a Square")
        assert "Circle" in classes and "Square" in classes
        assert "FadeIn" in animations

    def test_keyword_mapping(self):
        from src.agent.nodes import _extract_rag_hints

        classes, animations = _extract_rag_hints("Plot the graph of a quadratic equation")
        assert "Axes" in classes
        assert "MathTex" in classes

    def test_no_hints_for_unrelated_text(self):
        from src.agent.nodes import _extract_rag_hints

        classes, animations = _extract_rag_hints("history of the roman empire")
        assert classes == []
        assert animations == []

    def test_hints_deduplicated(self):
        from src.agent.nodes import _extract_rag_hints

        classes, _ = _extract_rag_hints("graph graph axes plot")
        assert classes.count("Axes") == 1


class TestTruncateCodeExample:
    def test_short_code_untouched(self):
        from src.agent.nodes import _truncate_code_example

        code = "line1\nline2"
        assert _truncate_code_example(code) == code

    def test_long_code_cut_at_line_boundary_with_marker(self):
        from src.agent.nodes import _truncate_code_example

        code = "\n".join(f"line_{i} = {i}" for i in range(500))
        result = _truncate_code_example(code, limit=100)
        assert result.endswith("# ... truncated")
        body = result.rsplit("\n", 1)[0]
        assert len(body) <= 100
        assert all(line in code for line in body.split("\n"))


# ============================================================================
# storyboard_node
# ============================================================================

class TestStoryboardNode:
    def _state(self):
        return {
            "scene_title": "Binary Search",
            "scene_prompt_description": "Explain binary search",
            "scene_length": 0.5,  # 30 seconds
            "explanation_depth": "detailed",
        }

    def test_parses_valid_storyboard(self, monkeypatch):
        from src.agent.nodes import storyboard_node

        response = (
            '{"sections": ['
            '{"title": "Intro", "duration_seconds": 10, "visuals": "title card", "narration": "welcome"},'
            '{"title": "Steps", "duration_seconds": 15, "visuals": "array", "narration": "we halve"},'
            '{"title": "Wrap", "duration_seconds": 5, "visuals": "summary", "narration": "done"}'
            "]}"
        )
        monkeypatch.setattr(
            "src.agent.nodes.llm_chat", lambda messages, temperature=0.4: response
        )

        result = storyboard_node(self._state())

        assert result["storyboard"] is not None
        assert len(result["storyboard"]) == 3
        assert result["storyboard"][0]["title"] == "Intro"
        assert sum(s["duration_seconds"] for s in result["storyboard"]) == pytest.approx(30, abs=1)

    def test_rescales_durations_to_target(self, monkeypatch):
        from src.agent.nodes import storyboard_node

        response = (
            '{"sections": ['
            '{"title": "A", "duration_seconds": 30, "visuals": "v", "narration": "n"},'
            '{"title": "B", "duration_seconds": 30, "visuals": "v", "narration": "n"}'
            "]}"
        )
        monkeypatch.setattr(
            "src.agent.nodes.llm_chat", lambda messages, temperature=0.4: response
        )

        result = storyboard_node(self._state())  # target 30s, plan sums to 60s

        total = sum(s["duration_seconds"] for s in result["storyboard"])
        assert total == pytest.approx(30, abs=1)

    def test_json_in_markdown_fences(self, monkeypatch):
        from src.agent.nodes import storyboard_node

        response = (
            "Here is the plan:\n```json\n"
            '{"sections": [{"title": "A", "duration_seconds": 30, "visuals": "v", "narration": "n"}]}'
            "\n```"
        )
        monkeypatch.setattr(
            "src.agent.nodes.llm_chat", lambda messages, temperature=0.4: response
        )

        result = storyboard_node(self._state())
        assert result["storyboard"] is not None

    def test_malformed_json_falls_back_with_warning(self, monkeypatch):
        from src.agent.nodes import storyboard_node

        monkeypatch.setattr(
            "src.agent.nodes.llm_chat", lambda messages, temperature=0.4: "not json at all"
        )

        result = storyboard_node(self._state())

        assert result["storyboard"] is None
        assert any("unusable" in w for w in result["pipeline_warnings"])

    def test_llm_unavailable_falls_back_with_warning(self, monkeypatch):
        from src.agent.nodes import storyboard_node

        monkeypatch.setattr(
            "src.agent.nodes.llm_chat", lambda messages, temperature=0.4: None
        )

        result = storyboard_node(self._state())

        assert result["storyboard"] is None
        assert any("LLM unavailable" in w for w in result["pipeline_warnings"])


class TestStoryboardPromptInjection:
    def test_storyboard_block_reaches_code_gen_prompt(self, monkeypatch):
        from src.agent import nodes

        captured = {}

        def fake_llm(messages, temperature=0.2):
            captured["prompt"] = messages[1]["content"]
            return None  # fall back to stub scene; we only care about the prompt

        monkeypatch.setattr("src.graph_rag.retriever.ManimRetriever", _EmptyRetriever)
        monkeypatch.setattr(nodes, "llm_chat", fake_llm)

        state = _base_state()
        state["storyboard"] = [
            {"title": "Intro", "duration_seconds": 10, "visuals": "title card", "narration": "welcome"},
            {"title": "Wrap", "duration_seconds": 20, "visuals": "summary", "narration": "bye"},
        ]
        nodes.video_code_gen_node(state)

        prompt = captured["prompt"]
        assert "STORYBOARD" in prompt
        assert "[0s–10s] Intro" in prompt
        assert "[10s–30s] Wrap" in prompt
