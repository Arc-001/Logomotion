"""
Tests for src.agent.nodes.

These tests run with no live Neo4j/Chroma databases, no network access,
and no OPENROUTER_API_KEY: every LLM call goes through a monkeypatched
`llm_chat`, and the Graph RAG retriever is replaced with an in-memory stub
patched at its source module (`src.graph_rag.retriever.get_retriever`),
matching how `video_code_gen_node` imports it lazily inside the function.
"""

import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import pytest

from src.agent.nodes import (
    _extract_code_block,
    _build_atempo_chain,
    _cleanup_temp_artifacts,
    _finish_merge,
    audio_video_merger_node,
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
        monkeypatch.setattr("src.graph_rag.retriever.get_retriever", lambda: _EmptyRetriever())

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
        monkeypatch.setattr("src.graph_rag.retriever.get_retriever", lambda: _EmptyRetriever())

        result = video_code_gen_node(_base_state())

        assert result["transcript"] == {}
        assert result["scene_class_name"] == "DemoScene"

    def test_llm_returns_none_uses_fallback_scene(self, monkeypatch):
        monkeypatch.setattr(
            "src.agent.nodes.llm_chat", lambda messages, temperature=0.2: None
        )
        monkeypatch.setattr("src.graph_rag.retriever.get_retriever", lambda: _EmptyRetriever())

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
        monkeypatch.setattr("src.graph_rag.retriever.get_retriever", lambda: _RaisingRetriever())

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
        monkeypatch.setattr("src.graph_rag.retriever.get_retriever", lambda: _EmptyRetriever())

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

        monkeypatch.setattr("src.graph_rag.retriever.get_retriever", lambda: _EmptyRetriever())
        monkeypatch.setattr(nodes, "llm_chat", fake_llm)

        state = _base_state()
        state["storyboard"] = [
            {"title": "Intro", "duration_seconds": 10, "visuals": "title card", "narration": "welcome"},
            {"title": "Wrap", "duration_seconds": 20, "visuals": "summary", "narration": "bye"},
        ]
        nodes.video_code_gen_node(state)

        prompt = captured["prompt"]
        assert "STORYBOARD" in prompt
        assert "[0.0s–10.0s] Intro" in prompt
        assert "[10.0s–30.0s] Wrap" in prompt


# ============================================================================
# Visual QA
# ============================================================================

VERDICT_BAD = (
    '{"acceptable": false, "issues": ['
    '{"frame_index": 1, "timestamp": 5.0, "problem": "title overlaps diagram", '
    '"fix": "FadeOut the title before showing the diagram"}]}'
)


class TestVisualQaRouting:
    def test_no_error_with_qa_enabled_routes_to_visual_qa(self):
        assert should_retry_or_continue({"error": None, "visual_qa_enabled": True}) == "visual_qa"

    def test_no_error_without_qa_routes_to_render_checker(self):
        assert should_retry_or_continue({"error": None}) == "render_checker"

    def test_error_still_routes_to_recorrector_first(self):
        state = {"error": "boom", "error_count": 0, "max_retries": 3, "visual_qa_enabled": True}
        assert should_retry_or_continue(state) == "recorrector"

    def test_acceptable_video_proceeds(self):
        from src.agent.nodes import should_fix_visuals

        assert should_fix_visuals({"visual_acceptable": True}) == "render_checker"

    def test_unacceptable_video_routes_to_fix_until_cap(self):
        from src.agent.nodes import should_fix_visuals

        state = {"visual_acceptable": False, "visual_fix_count": 0}
        assert should_fix_visuals(state) == "visual_recorrector"

        state["visual_fix_count"] = 99
        assert should_fix_visuals(state) == "render_checker"


class TestParseVisualVerdict:
    def test_parses_verdict(self):
        from src.agent.nodes import _parse_visual_verdict

        verdict = _parse_visual_verdict(VERDICT_BAD)
        assert verdict["acceptable"] is False
        assert len(verdict["issues"]) == 1
        assert "overlaps" in verdict["issues"][0]["problem"]

    def test_parses_fenced_verdict(self):
        from src.agent.nodes import _parse_visual_verdict

        verdict = _parse_visual_verdict(f"```json\n{VERDICT_BAD}\n```")
        assert verdict["acceptable"] is False

    def test_missing_acceptable_key_returns_none(self):
        from src.agent.nodes import _parse_visual_verdict

        assert _parse_visual_verdict('{"issues": []}') is None

    def test_malformed_json_raises(self):
        import json as json_module

        from src.agent.nodes import _parse_visual_verdict

        with pytest.raises(json_module.JSONDecodeError):
            _parse_visual_verdict("not json")


class TestVisualQaNode:
    def test_missing_video_accepts_without_review(self):
        from src.agent.nodes import visual_qa_node

        result = visual_qa_node({"rendered_video_path": None})
        assert result["visual_acceptable"] is True

    def test_verdict_flows_through(self, monkeypatch, tmp_path):
        from src.agent import nodes

        video = tmp_path / "video.mp4"
        video.write_bytes(b"x" * 2000)
        frame = tmp_path / "frame_00.jpg"
        frame.write_bytes(b"\xff\xd8\xff\xe0fakejpg")

        monkeypatch.setattr(
            "src.manim_runner.frames.extract_frames",
            lambda path, count=6, out_dir=None: [{"path": str(frame), "timestamp": 1.0}],
        )
        monkeypatch.setattr(nodes, "llm_chat", lambda messages, temperature=0.1: VERDICT_BAD)

        result = nodes.visual_qa_node({"rendered_video_path": str(video), "visual_fix_count": 0})

        assert result["visual_acceptable"] is False
        assert len(result["visual_issues"]) == 1

    def test_llm_failure_never_blocks_pipeline(self, monkeypatch, tmp_path):
        from src.agent import nodes

        video = tmp_path / "video.mp4"
        video.write_bytes(b"x" * 2000)
        frame = tmp_path / "frame_00.jpg"
        frame.write_bytes(b"\xff\xd8\xff\xe0fakejpg")

        monkeypatch.setattr(
            "src.manim_runner.frames.extract_frames",
            lambda path, count=6, out_dir=None: [{"path": str(frame), "timestamp": 1.0}],
        )
        monkeypatch.setattr(nodes, "llm_chat", lambda messages, temperature=0.1: None)

        result = nodes.visual_qa_node({"rendered_video_path": str(video)})

        assert result["visual_acceptable"] is True
        assert any("unreviewed" in w for w in result["pipeline_warnings"])


class TestVisualRecorrectorNode:
    def test_fixes_code_and_increments_count(self, monkeypatch):
        from src.agent.nodes import visual_recorrector_node

        monkeypatch.setattr(
            "src.agent.nodes.llm_chat",
            lambda messages, temperature=0.1: "```python\nfixed_layout = True\n```",
        )
        state = {
            "code": "original = True",
            "visual_issues": [{"timestamp": 5.0, "problem": "overlap", "fix": "FadeOut first"}],
        }

        result = visual_recorrector_node(state)

        assert result["code"] == "fixed_layout = True"
        assert result["visual_fix_count"] == 1
        assert result["error"] is None

    def test_llm_failure_keeps_code_with_warning(self, monkeypatch):
        from src.agent.nodes import visual_recorrector_node

        monkeypatch.setattr("src.agent.nodes.llm_chat", lambda messages, temperature=0.1: None)
        state = {"code": "original = True", "visual_issues": []}

        result = visual_recorrector_node(state)

        assert "code" not in result  # state code untouched
        assert result["visual_fix_count"] == 1
        assert any("fix failed" in w for w in result["pipeline_warnings"])


# ============================================================================
# narration_tts_node — narration is recorded and measured before code exists
# ============================================================================

class _StubTTSResult:
    def __init__(self, duration, path="/tmp/kokoro_stub.wav", error=None):
        self.success = error is None
        self.audio_path = None if error else path
        self.duration = duration
        self.error = error


class _StubTTS:
    """Kokoro stand-in: speech length is proportional to the text length."""

    def __init__(self, seconds_per_char=0.1, fail_on=()):
        self.seconds_per_char = seconds_per_char
        self.fail_on = fail_on
        self.calls = []

    def synthesize(self, text):
        self.calls.append(text)
        if text in self.fail_on:
            return _StubTTSResult(None, error="synthesis exploded")
        return _StubTTSResult(
            duration=len(text) * self.seconds_per_char,
            path=f"/tmp/kokoro_{len(self.calls):03d}.wav",
        )


class TestNarrationTtsNode:
    def _state(self, storyboard, **overrides):
        state = {
            "scene_length": 1.0,
            "target_duration": 60.0,
            "storyboard": storyboard,
        }
        state.update(overrides)
        return state

    def _storyboard(self):
        return [
            {"title": "Intro", "duration_seconds": 10, "visuals": "v", "narration": "a" * 20},
            {"title": "Body", "duration_seconds": 30, "visuals": "v", "narration": "b" * 50},
            {"title": "Wrap", "duration_seconds": 20, "visuals": "v", "narration": "c" * 10},
        ]

    def test_section_budget_is_at_least_the_measured_speech(self, monkeypatch):
        from src.agent import nodes

        # 50 chars * 0.1 = 5.0s of speech, well under the 30s plan
        monkeypatch.setattr(nodes, "_get_tts", lambda: _StubTTS())
        result = nodes.narration_tts_node(self._state(self._storyboard()))

        for section, segment in zip(result["storyboard"], result["narration_segments"]):
            assert section["duration_seconds"] >= segment["audio_duration"]

    def test_budget_grows_when_speech_overruns_the_plan(self, monkeypatch):
        from src.agent import nodes

        # 0.5s/char makes the 50-char middle section 25s of speech vs a 10s plan
        monkeypatch.setattr(nodes, "_get_tts", lambda: _StubTTS(seconds_per_char=0.5))
        storyboard = [
            {"title": "Only", "duration_seconds": 10, "visuals": "v", "narration": "b" * 50},
        ]
        result = nodes.narration_tts_node(self._state(storyboard))

        assert result["storyboard"][0]["duration_seconds"] == pytest.approx(26.0)
        assert result["target_duration"] == pytest.approx(26.0)

    def test_timestamps_are_cumulative_section_starts(self, monkeypatch):
        from src.agent import nodes

        monkeypatch.setattr(nodes, "_get_tts", lambda: _StubTTS())
        result = nodes.narration_tts_node(self._state(self._storyboard()))

        starts = [seg["timestamp"] for seg in result["narration_segments"]]
        assert starts == [0.0, 10.0, 40.0]

        sections = result["transcript_sections"]
        assert [s["timestamp"] for s in sections] == starts
        assert all(s["audio_path"] for s in sections)

    def test_target_duration_is_the_sum_of_budgets(self, monkeypatch):
        from src.agent import nodes

        monkeypatch.setattr(nodes, "_get_tts", lambda: _StubTTS())
        result = nodes.narration_tts_node(self._state(self._storyboard()))

        assert result["target_duration"] == pytest.approx(60.0)

    def test_warns_when_narration_stretches_the_video(self, monkeypatch):
        from src.agent import nodes

        monkeypatch.setattr(nodes, "_get_tts", lambda: _StubTTS(seconds_per_char=1.0))
        result = nodes.narration_tts_node(self._state(self._storyboard()))

        assert result["target_duration"] > 60.0
        assert any("timed to the narration" in w for w in result["pipeline_warnings"])

    def test_failed_segment_is_warned_and_skipped(self, monkeypatch):
        from src.agent import nodes

        storyboard = self._storyboard()
        monkeypatch.setattr(
            nodes, "_get_tts", lambda: _StubTTS(fail_on=(storyboard[1]["narration"],))
        )
        result = nodes.narration_tts_node(self._state(storyboard))

        assert result["narration_segments"][1]["audio_path"] is None
        assert len(result["audio_segments"]) == 2
        assert any("section 2" in w for w in result["pipeline_warnings"])

    def test_no_storyboard_is_a_no_op(self, monkeypatch):
        from src.agent import nodes

        monkeypatch.setattr(nodes, "_get_tts", lambda: _StubTTS())
        assert nodes.narration_tts_node(self._state(None)) == {}

    def test_tts_unavailable_falls_back_with_a_warning(self, monkeypatch):
        from src.agent import nodes

        monkeypatch.setattr(nodes, "_get_tts", lambda: None)
        result = nodes.narration_tts_node(self._state(self._storyboard()))

        assert "narration_segments" not in result
        assert any("TTS unavailable" in w for w in result["pipeline_warnings"])


class TestTranscriptProcessorSkipsPreRecordedNarration:
    def test_no_op_when_narration_already_recorded(self, monkeypatch):
        from src.agent import nodes

        def _boom():
            raise AssertionError("_get_tts must not be called")

        monkeypatch.setattr(nodes, "_get_tts", _boom)
        state = {
            "narration_segments": [{"index": 0, "text": "hi", "audio_path": "/tmp/a.wav"}],
            "transcript": {0: "hi"},
        }
        assert nodes.transcript_processor_node(state) == {}

    def test_still_runs_on_the_single_shot_path(self, monkeypatch):
        from src.agent import nodes

        monkeypatch.setattr(nodes, "_get_tts", lambda: _StubTTS())
        state = {"narration_segments": [], "transcript": {0: "hello", 5: "world"}}
        result = nodes.transcript_processor_node(state)

        assert [s["timestamp"] for s in result["transcript_sections"]] == [0.0, 5.0]
        assert len(result["audio_segments"]) == 2


class TestPreRecordedNarrationPrompt:
    """With narration already recorded, code gen is timed to it and asks for no transcript."""

    def _capture(self, monkeypatch, state):
        from src.agent import nodes

        captured = {}

        def fake_llm(messages, temperature=0.2):
            captured["prompt"] = messages[1]["content"]
            return "```python\nfrom manim import *\n\nclass S(Scene):\n    def construct(self):\n        pass\n```"

        monkeypatch.setattr("src.graph_rag.retriever.get_retriever", lambda: _EmptyRetriever())
        monkeypatch.setattr(nodes, "llm_chat", fake_llm)
        result = nodes.video_code_gen_node(state)
        return captured["prompt"], result

    def _state(self):
        state = _base_state()
        state["storyboard"] = [
            {"title": "Intro", "duration_seconds": 12.5, "visuals": "title", "narration": "welcome"},
            {"title": "Wrap", "duration_seconds": 7.5, "visuals": "summary", "narration": "bye"},
        ]
        state["narration_segments"] = [
            {"index": 0, "text": "welcome", "audio_path": "/tmp/a.wav", "audio_duration": 11.5,
             "timestamp": 0.0},
            {"index": 1, "text": "bye", "audio_path": "/tmp/b.wav", "audio_duration": 6.5,
             "timestamp": 12.5},
        ]
        state["target_duration"] = 20.0
        return state

    def test_prompt_states_the_measured_audio_and_forbids_a_transcript(self, monkeypatch):
        prompt, _ = self._capture(monkeypatch, self._state())

        assert "ALREADY been synthesised" in prompt
        assert "Spoken audio: 11.5s (already recorded)" in prompt
        assert "Spoken audio: 6.5s (already recorded)" in prompt
        assert "TRANSCRIPT_START" not in prompt
        assert "TRANSCRIPT / NARRATION" not in prompt

    def test_prompt_targets_the_narration_derived_duration(self, monkeypatch):
        prompt, _ = self._capture(monkeypatch, self._state())

        assert "MUST come to 20 seconds" in prompt
        assert "**Target Duration:** 20 seconds" in prompt

    def test_node_does_not_publish_a_transcript(self, monkeypatch):
        _, result = self._capture(monkeypatch, self._state())

        assert "transcript" not in result

    def test_single_shot_path_still_asks_for_a_transcript(self, monkeypatch):
        state = _base_state()
        prompt, result = self._capture(monkeypatch, state)

        assert "TRANSCRIPT_START" in prompt
        assert "TRANSCRIPT / NARRATION" in prompt
        assert "transcript" in result


# ============================================================================
# audio_video_merger_node — timeline integrity
# ============================================================================

FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


def _make_silent_video(path, seconds):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s=160x120:d={seconds}",
         "-r", "10", "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True,
    )


def _make_tone(path, seconds):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(path)],
        capture_output=True, check=True,
    )


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg/ffprobe not installed")
class TestAudioVideoMergerTimeline:
    @pytest.fixture
    def workdir(self):
        d = Path(tempfile.mkdtemp(prefix="manim_merge_test_"))
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def _run(self, workdir, video_seconds, timed_clips):
        """timed_clips: list of (timestamp, clip_seconds)."""
        video = workdir / "video.mp4"
        _make_silent_video(video, video_seconds)

        sections = []
        for i, (ts, clip_seconds) in enumerate(timed_clips):
            wav = workdir / f"seg_{i}.wav"
            _make_tone(wav, clip_seconds)
            sections.append({"timestamp": ts, "text": f"line {i}", "audio_path": str(wav)})

        state = {
            "synced_video_path": str(video),
            "audio_segments": [s["audio_path"] for s in sections],
            "transcript_sections": sections,
            "temp_dirs": [],
        }
        result = audio_video_merger_node(state)
        if result.get("final_output_path"):
            Path(result["final_output_path"]).unlink(missing_ok=True)
        return result

    def test_narration_past_the_video_end_is_reported(self, workdir):
        # 6s video, but the last two lines start at 8s and 12s
        result = self._run(workdir, 6, [(0.0, 1.0), (8.0, 1.0), (12.0, 1.0)])

        warnings = result.get("pipeline_warnings") or []
        assert any("fell past the end" in w for w in warnings), warnings
        assert any("2 narration line(s)" in w for w in warnings), warnings

    def test_segments_within_bounds_produce_no_warning(self, workdir):
        result = self._run(workdir, 10, [(0.0, 1.0), (3.0, 1.0), (6.0, 1.0)])

        assert not (result.get("pipeline_warnings") or [])
        assert result["final_output_path"] is not None

    def test_dense_narration_is_not_sped_up_past_the_cap(self, workdir):
        from src.agent.nodes import _MAX_NARRATION_TEMPO

        # 4s of speech crammed into a 1s window would need 4x to fit
        result = self._run(workdir, 10, [(0.0, 4.0), (1.0, 0.5)])

        warnings = result.get("pipeline_warnings") or []
        assert any(f"{_MAX_NARRATION_TEMPO}x" in w for w in warnings), warnings
        assert any("4.0x needed to fit" in w for w in warnings), warnings

    def test_the_cap_is_only_reported_once(self, workdir):
        result = self._run(workdir, 20, [(0.0, 4.0), (1.0, 4.0), (2.0, 4.0), (3.0, 0.5)])

        warnings = [w for w in (result.get("pipeline_warnings") or []) if "speech kept at" in w]
        assert len(warnings) == 1

    def test_audio_that_fits_is_left_at_natural_speed(self, workdir):
        result = self._run(workdir, 10, [(0.0, 1.0), (5.0, 1.0)])

        assert not any("speech kept at" in w for w in (result.get("pipeline_warnings") or []))


class TestCoordinateSystemGuidance:
    """The blanket ban on coordinate arrays also banned ax.c2p, the only correct
    way to place anything on a set of axes."""

    def _prompt(self, monkeypatch):
        from src.agent import nodes

        captured = {}

        def fake_llm(messages, temperature=0.2):
            captured["system"] = messages[0]["content"]
            captured["user"] = messages[1]["content"]
            return None

        monkeypatch.setattr("src.graph_rag.retriever.get_retriever", lambda: _EmptyRetriever())
        monkeypatch.setattr(nodes, "llm_chat", fake_llm)
        nodes.video_code_gen_node(_base_state())
        return captured

    def test_axes_derived_coordinates_are_required_not_banned(self, monkeypatch):
        captured = self._prompt(monkeypatch)

        assert "ax.c2p" in captured["user"]
        assert "REQUIRED ON A GRAPH" in captured["user"]

    def test_literal_coordinates_are_still_banned(self, monkeypatch):
        captured = self._prompt(monkeypatch)

        assert "move_to([2, -1, 0])" in captured["user"]
        assert "BANNED" in captured["user"]

    def test_default_system_message_carries_the_same_exception(self):
        from src.agent.state import create_initial_state

        state = create_initial_state("T", "d")

        assert "ax.c2p" in state["system_message"]
        assert "EXCEPTION" in state["system_message"]


class TestRecorrectorHistory:
    def _capture(self, monkeypatch, state):
        from src.agent import nodes

        captured = {}

        def fake_llm(messages, temperature=0.1):
            captured["prompt"] = messages[1]["content"]
            return "```python\nfixed = True\n```"

        monkeypatch.setattr(nodes, "llm_chat", fake_llm)
        result = nodes.recorrector_node(state)
        return captured["prompt"], result

    def test_first_attempt_carries_no_history_block(self, monkeypatch):
        prompt, result = self._capture(
            monkeypatch, {"code": "x = 1", "error": "NameError: y", "error_count": 0}
        )

        assert "not the first attempt" not in prompt
        assert result["fix_history"] == [{"attempt": 1, "error": "NameError: y"}]

    def test_later_attempts_see_what_was_already_tried(self, monkeypatch):
        prompt, result = self._capture(
            monkeypatch,
            {
                "code": "x = 1",
                "error": "TypeError: bad tip_length",
                "error_count": 2,
                "fix_history": [
                    {"attempt": 1, "error": "AttributeError: no ShowCreation"},
                    {"attempt": 2, "error": "TypeError: unexpected length="},
                ],
            },
        )

        assert "not the first attempt" in prompt
        assert "Attempt 1 tried to fix:\nAttributeError: no ShowCreation" in prompt
        assert "Attempt 2 tried to fix:\nTypeError: unexpected length=" in prompt
        assert "Do not repeat them" in prompt

    def test_history_entry_appends_rather_than_replaces(self, monkeypatch):
        """fix_history uses an add reducer, so a node returns only its own entry."""
        _, result = self._capture(
            monkeypatch,
            {
                "code": "x = 1",
                "error": "boom",
                "error_count": 1,
                "fix_history": [{"attempt": 1, "error": "earlier"}],
            },
        )

        assert result["fix_history"] == [{"attempt": 2, "error": "boom"}]

    def test_long_errors_are_truncated_in_history(self, monkeypatch):
        _, result = self._capture(
            monkeypatch, {"code": "x = 1", "error": "E" * 900, "error_count": 0}
        )

        assert len(result["fix_history"][0]["error"]) == 500
