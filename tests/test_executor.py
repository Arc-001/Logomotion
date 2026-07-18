"""
Tests for src.manim_runner.executor.ManimExecutor.

Covers the pure-Python helpers (`_parse_error`, `_find_video`, `cleanup`)
without ever invoking a real Manim render.
"""

import shutil
import tempfile
from pathlib import Path

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
