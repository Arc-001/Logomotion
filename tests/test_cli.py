"""
Tests for src.main argument plumbing.

Nothing here generates a video: generate_video_sync is stubbed so the tests
assert what the CLI passes through.
"""

import pytest

from src import main as cli


@pytest.fixture
def captured(monkeypatch):
    calls = {}

    def fake_generate(**kwargs):
        calls.update(kwargs)
        return {"final_output_path": "/tmp/out.mp4", "code": "x"}

    import src.agent.graph as graph_module
    monkeypatch.setattr(graph_module, "generate_video_sync", fake_generate)
    return calls


def _run(monkeypatch, argv):
    import sys
    monkeypatch.setattr(sys, "argv", ["manim-agent", "generate", "--prompt", "circles"] + argv)
    cli.main()


class TestLengthFlag:
    def test_explicit_length_is_honoured_even_when_it_equals_the_old_sentinel(
        self, monkeypatch, captured
    ):
        """1.0 used to double as "flag not passed" and was silently replaced
        by VIDEO_LENGTH."""
        from src.config import Settings

        monkeypatch.setattr(cli, "get_settings", lambda: Settings(video_length=5.0))
        _run(monkeypatch, ["--length", "1.0"])

        assert captured["scene_length"] == 1.0

    def test_omitted_length_falls_back_to_settings(self, monkeypatch, captured):
        from src.config import Settings

        monkeypatch.setattr(cli, "get_settings", lambda: Settings(video_length=5.0))
        _run(monkeypatch, [])

        assert captured["scene_length"] == 5.0


class TestWebSearchFlag:
    def test_flag_reaches_the_pipeline(self, monkeypatch, captured):
        _run(monkeypatch, ["--web-search"])

        assert captured["web_search_enabled"] is True

    def test_default_is_off(self, monkeypatch, captured):
        _run(monkeypatch, [])

        assert captured["web_search_enabled"] is False


class TestOtherFlagsStillPlumbThrough:
    def test_quality_name_is_normalised(self, monkeypatch, captured):
        _run(monkeypatch, ["--quality", "high"])

        assert captured["render_quality"] == "h"

    def test_visual_qa_opt_in(self, monkeypatch, captured):
        _run(monkeypatch, ["--visual-qa"])

        assert captured["visual_qa"] is True
