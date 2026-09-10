"""
Tests for src.manim_runner.executor.ManimExecutor.

Covers the pure-Python helpers (`_parse_error`, `_find_video`, `cleanup`)
without ever invoking a real Manim render.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from src.manim_runner.executor import (
    ManimExecutor,
    ExecutionResult,
    compute_render_timeout,
)


# ============================================================================
# _parse_error
# ============================================================================

class TestParseError:
    def test_traceback_text_returns_traceback_lines(self):
        executor = ManimExecutor()
        text = (
            "Traceback (most recent call last):\n"
            '  File "scene.py", line 3, in <module>\n'
            "    class Foo(Scene):\n"
            "NameError: name 'Scene' is not defined\n"
        )

        result = executor._parse_error(text)

        # Once "Traceback" is seen, every subsequent line (including the
        # trailing empty split artifact) is captured, so the full text
        # round-trips unchanged.
        assert result == text
        assert "Traceback (most recent call last):" in result
        assert "NameError: name 'Scene' is not defined" in result

    def test_more_than_20_traceback_lines_returns_last_20_only(self):
        executor = ManimExecutor()
        lines = ["Traceback (most recent call last):"]
        lines += [f'  File "scene.py", line {i}, in <module>' for i in range(24)]
        text = "\n".join(lines)

        result = executor._parse_error(text)
        result_lines = result.split("\n")

        assert len(result_lines) == 20
        assert result_lines == lines[-20:]

    def test_error_line_without_traceback_is_captured(self):
        executor = ManimExecutor()
        text = "some noise\nValueError: something bad happened\nmore noise"

        result = executor._parse_error(text)

        assert result == "ValueError: something bad happened"

    def test_plain_text_falls_back_to_first_1000_chars(self):
        executor = ManimExecutor()
        text = "x" * 1500

        result = executor._parse_error(text)

        assert result == "x" * 1000
        assert len(result) == 1000


# ============================================================================
# _find_video
# ============================================================================

class TestFindVideo:
    def test_prefers_scene_name_match(self, tmp_path):
        executor = ManimExecutor()
        nested = tmp_path / "videos" / "1080p30"
        nested.mkdir(parents=True)
        (nested / "OtherScene.mp4").write_bytes(b"x")
        target = nested / "MyScene.mp4"
        target.write_bytes(b"x")

        result = executor._find_video(tmp_path, "MyScene")

        assert result == str(target)

    def test_empty_dir_returns_none(self, tmp_path):
        executor = ManimExecutor()

        result = executor._find_video(tmp_path, "MyScene")

        assert result is None

    def test_missing_dir_returns_none(self, tmp_path):
        executor = ManimExecutor()
        missing = tmp_path / "does_not_exist"

        result = executor._find_video(missing, "MyScene")

        assert result is None

    def test_partial_movie_files_are_never_chosen(self, tmp_path):
        """Manim writes every animation as its own clip before stitching them;
        returning one of those yields a fragment instead of the video."""
        executor = ManimExecutor()
        nested = tmp_path / "videos" / "scene" / "1080p30"
        partials = nested / "partial_movie_files" / "MyScene"
        partials.mkdir(parents=True)
        (partials / "MyScene_0000.mp4").write_bytes(b"x")
        target = nested / "MyScene.mp4"
        target.write_bytes(b"x")

        assert executor._find_video(tmp_path, "MyScene") == str(target)

    def test_only_partial_movie_files_means_no_video(self, tmp_path):
        executor = ManimExecutor()
        partials = tmp_path / "videos" / "partial_movie_files" / "MyScene"
        partials.mkdir(parents=True)
        (partials / "MyScene_0000.mp4").write_bytes(b"x")

        assert executor._find_video(tmp_path, "MyScene") is None

    def test_falls_back_to_any_stitched_video(self, tmp_path):
        executor = ManimExecutor()
        nested = tmp_path / "videos" / "1080p30"
        nested.mkdir(parents=True)
        other = nested / "SomethingElse.mp4"
        other.write_bytes(b"x")

        assert executor._find_video(tmp_path, "MyScene") == str(other)


# ============================================================================
# cleanup
# ============================================================================

class TestCleanup:
    def test_removes_manim_exec_dir(self):
        temp_dir = tempfile.mkdtemp(prefix="manim_exec_")
        code_path = Path(temp_dir) / "scene.py"
        code_path.write_text("# dummy")

        result = ExecutionResult(
            success=True,
            video_path=None,
            error=None,
            stdout="",
            stderr="",
            code_path=str(code_path),
            output_dir=str(Path(temp_dir) / "media"),
        )

        executor = ManimExecutor()
        executor.cleanup(result)

        assert not Path(temp_dir).exists()

    def test_survives_non_matching_dir(self):
        temp_dir = tempfile.mkdtemp(prefix="other_prefix_")
        code_path = Path(temp_dir) / "scene.py"
        code_path.write_text("# dummy")

        result = ExecutionResult(
            success=True,
            video_path=None,
            error=None,
            stdout="",
            stderr="",
            code_path=str(code_path),
            output_dir=str(Path(temp_dir) / "media"),
        )

        executor = ManimExecutor()
        try:
            executor.cleanup(result)
            assert Path(temp_dir).exists()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Quality / fps configuration
# ============================================================================

class TestQualityConfiguration:
    def test_normalize_render_quality(self):
        from src.config import normalize_render_quality

        assert normalize_render_quality("low") == "l"
        assert normalize_render_quality("Medium") == "m"
        assert normalize_render_quality("HIGH") == "h"
        assert normalize_render_quality("m") == "m"
        assert normalize_render_quality("k") == "k"
        with pytest.raises(ValueError):
            normalize_render_quality("ultra")

    def test_portrait_resolution_tracks_quality(self):
        from src.manim_runner.executor import _PORTRAIT_RESOLUTIONS

        assert _PORTRAIT_RESOLUTIONS["l"] == "480,854"
        assert _PORTRAIT_RESOLUTIONS["m"] == "720,1280"
        assert _PORTRAIT_RESOLUTIONS["h"] == "1080,1920"

    def test_executor_render_command_includes_quality_and_fps(self, monkeypatch, tmp_path):
        """Build the render command via execute() with subprocess mocked out."""
        recorded = {}

        def fake_run(cmd, **kwargs):
            recorded["cmd"] = cmd

            class _R:
                returncode = 1
                stdout = ""
                stderr = "Error: stop here"

            return _R()

        executor = ManimExecutor(quality="h", fps=24)
        monkeypatch.setattr("src.manim_runner.executor.subprocess.run", fake_run)
        code = "from manim import *\n\nclass SceneX(Scene):\n    def construct(self):\n        pass\n"
        executor.execute(code, "SceneX", orientation="portrait")

        cmd = recorded["cmd"]
        assert "-qh" in cmd
        assert "--fps" in cmd and cmd[cmd.index("--fps") + 1] == "24"
        assert "--resolution" in cmd and cmd[cmd.index("--resolution") + 1] == "1080,1920"


# ============================================================================
# validate_manim_code (pre-render static validation)
# ============================================================================

class TestValidateManimCode:
    def test_valid_code_passes(self):
        from src.manim_runner.validator import validate_manim_code

        code = "from manim import *\n\nclass MyScene(Scene):\n    def construct(self):\n        pass\n"
        assert validate_manim_code(code, "MyScene") is None

    def test_syntax_error_reports_line(self):
        from src.manim_runner.validator import validate_manim_code

        code = "from manim import *\nclass MyScene(Scene):\n    def construct(self)\n        pass\n"
        error = validate_manim_code(code, "MyScene")
        assert error is not None
        assert "SyntaxError on line 3" in error

    def test_missing_scene_class(self):
        from src.manim_runner.validator import validate_manim_code

        code = "from manim import *\n\nclass OtherScene(Scene):\n    pass\n"
        error = validate_manim_code(code, "MyScene")
        assert "MyScene" in error and "not defined" in error

    def test_missing_manim_import(self):
        from src.manim_runner.validator import validate_manim_code

        code = "class MyScene:\n    pass\n"
        error = validate_manim_code(code, "MyScene")
        assert "manim import" in error

    def test_legacy_api_flagged_with_replacement(self):
        from src.manim_runner.validator import validate_manim_code

        code = (
            "from manim import *\n\nclass MyScene(Scene):\n"
            "    def construct(self):\n        self.play(ShowCreation(Circle()))\n"
        )
        error = validate_manim_code(code, "MyScene")
        assert "ShowCreation" in error and "Create" in error

    def test_executor_short_circuits_without_subprocess(self, monkeypatch):
        from src.manim_runner import executor as executor_module

        def _explode(*args, **kwargs):
            raise AssertionError("subprocess.run must not be called for invalid code")

        monkeypatch.setattr(executor_module.subprocess, "run", _explode)
        result = ManimExecutor().execute("def broken(:", "MyScene")

        assert result.success is False
        assert "SyntaxError" in result.error
        assert result.code_path == ""


class TestComputeRenderTimeout:
    """A fixed 120s ceiling could not cover the 30-minute videos the API accepts."""

    def test_short_clips_keep_the_configured_floor(self):
        assert compute_render_timeout(30, "m", 120) == 120

    def test_long_targets_scale_past_the_floor(self):
        # 5 minutes at 720p: 300s of video, 2s of rendering per second
        assert compute_render_timeout(300, "m", 120) == 600

    def test_higher_quality_costs_more_time(self):
        assert compute_render_timeout(300, "h", 120) > compute_render_timeout(300, "m", 120)
        assert compute_render_timeout(300, "m", 120) > compute_render_timeout(300, "l", 120)

    def test_unknown_quality_falls_back_to_the_medium_factor(self):
        assert compute_render_timeout(300, "z", 120) == compute_render_timeout(300, "m", 120)

    @pytest.mark.parametrize("target", [0, None, -5])
    def test_missing_target_keeps_the_floor(self, target):
        assert compute_render_timeout(target, "m", 120) == 120


class TestValidatorReadsTheAst:
    """Pattern matching on raw source failed code that only mentioned an API."""

    def _code(self, body):
        return f"from manim import *\n\nclass MyScene(Scene):\n    def construct(self):\n{body}\n"

    def test_removed_api_in_a_comment_is_not_an_error(self):
        from src.manim_runner.validator import validate_manim_code

        code = self._code("        # do not use ShowCreation(...) here\n        pass")

        assert validate_manim_code(code, "MyScene") is None

    def test_removed_api_in_a_docstring_is_not_an_error(self):
        from src.manim_runner.validator import validate_manim_code

        code = self._code('        """Prefer Create over Code(...) and TextMobject(...)."""\n        pass')

        assert validate_manim_code(code, "MyScene") is None

    def test_removed_api_actually_called_is_still_an_error(self):
        from src.manim_runner.validator import validate_manim_code

        code = self._code("        self.play(ShowCreation(Circle()))")

        assert "ShowCreation" in validate_manim_code(code, "MyScene")

    def test_scene_name_inside_a_string_does_not_satisfy_the_class_check(self):
        from src.manim_runner.validator import validate_manim_code

        code = 'from manim import *\n\nname = "class MyScene(Scene)"\n'

        assert "not defined" in validate_manim_code(code, "MyScene")

    def test_manim_mentioned_in_a_string_does_not_satisfy_the_import_check(self):
        from src.manim_runner.validator import validate_manim_code

        code = 'note = "from manim import *"\n\nclass MyScene:\n    pass\n'

        assert "Missing manim import" in validate_manim_code(code, "MyScene")


class TestFailedRenderIsCleanedUpImmediately:
    def test_executor_cleanup_is_called_and_no_temp_dir_is_registered(self, monkeypatch):
        from src.agent import nodes

        cleaned = []

        class _FailingExecutor:
            def __init__(self, **kwargs):
                pass

            def execute(self, code, scene_class_name, orientation="landscape"):
                return ExecutionResult(
                    success=False, video_path=None, error="boom",
                    stdout="", stderr="", code_path="/tmp/manim_exec_x/scene.py",
                    output_dir="/tmp/manim_exec_x/media",
                )

            def cleanup(self, result):
                cleaned.append(result.code_path)

        monkeypatch.setattr(nodes, "ManimExecutor", _FailingExecutor)
        state = {"code": "x", "scene_class_name": "MyScene", "render_quality": "m",
                 "scene_length": 1.0, "target_duration": 60.0}

        result = nodes.code_executor_node(state)

        assert cleaned == ["/tmp/manim_exec_x/scene.py"]
        assert "temp_dirs" not in result
        assert result["error"] == "boom"
