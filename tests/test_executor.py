"""
Tests for src.manim_runner.executor.ManimExecutor.

Covers the pure-Python helpers (`_parse_error`, `_find_video`, `cleanup`)
without ever invoking a real Manim render.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from src.manim_runner.executor import ManimExecutor, ExecutionResult


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
        executor.execute("code", "SceneX", orientation="portrait")

        cmd = recorded["cmd"]
        assert "-qh" in cmd
        assert "--fps" in cmd and cmd[cmd.index("--fps") + 1] == "24"
        assert "--resolution" in cmd and cmd[cmd.index("--resolution") + 1] == "1080,1920"
