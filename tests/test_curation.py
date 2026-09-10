"""
Tests for src.graph_rag.curation.

The corpus these rules filter is real: of the 1214 examples in
`extracted data/*.jsonl`, over half import `manimlib` (ManimGL) rather than
Manim Community, and others depend on helper modules that lived beside the
original file. Every retrieved example is shown to the code generator as
something to imitate, so those must not reach the prompt.
"""

import pytest

from src.graph_rag.curation import example_rejection_reason, is_usable_example


CLEAN = """from manim import *

class DemoScene(Scene):
    def construct(self):
        self.play(Create(Circle()))
        self.wait(1)
"""


class TestAcceptedExamples:
    def test_plain_community_scene_is_kept(self):
        assert example_rejection_reason(CLEAN) is None
        assert is_usable_example(CLEAN)

    def test_numpy_and_stdlib_imports_are_fine(self):
        code = "from manim import *\nimport numpy as np\nimport math\n" + CLEAN.split("\n", 1)[1]
        assert example_rejection_reason(code) is None

    def test_three_d_scene_counts_as_a_scene(self):
        code = CLEAN.replace("class DemoScene(Scene)", "class DemoScene(ThreeDScene)")
        assert example_rejection_reason(code) is None


class TestRejectedExamples:
    def test_manimgl_example_is_rejected(self):
        code = CLEAN.replace("from manim import *", "from manimlib import *")
        assert "ManimGL" in example_rejection_reason(code)

    def test_local_helper_module_is_rejected(self):
        code = "from manim import *\nfrom utilities import *\n" + CLEAN.split("\n", 1)[1]
        assert "unavailable module" in example_rejection_reason(code)

    def test_missing_manim_import_is_rejected(self):
        code = CLEAN.replace("from manim import *\n", "")
        assert example_rejection_reason(code) == "does not import manim"

    def test_removed_api_is_rejected(self):
        code = CLEAN.replace("Create(Circle())", "ShowCreation(Circle())")
        assert "ShowCreation" in example_rejection_reason(code)

    def test_local_asset_is_rejected(self):
        code = CLEAN.replace(
            "self.play(Create(Circle()))",
            "logo = ImageMobject('./media/images/logo.png')",
        )
        assert "local asset" in example_rejection_reason(code)

    def test_code_without_a_scene_is_rejected(self):
        assert example_rejection_reason("from manim import *\n\nx = 1\n") == "no Scene subclass"

    def test_unparseable_code_is_rejected(self):
        assert "does not parse" in example_rejection_reason("from manim import *\ndef (:\n")

    @pytest.mark.parametrize("code", ["", "   \n"])
    def test_empty_code_is_rejected(self, code):
        assert example_rejection_reason(code) == "empty"


class TestRealCorpusSample:
    def test_the_shipped_dataset_entry_that_needs_external_helpers_is_rejected(self):
        """First record of manim_finetuning_dataset_1.jsonl: imports TTS.TTS and
        utils, and loads ./media/images/logo.png."""
        code = (
            "from manim import *\n"
            "from TTS.TTS import get_mp3_file\n"
            "from utils import cut, get_duration, deal_text\n"
            "import time\n\n"
            "class Video(Scene):\n"
            "    def construct(self):\n"
            "        LOGO = ImageMobject('./media/images/logo.png')\n"
        )
        reason = example_rejection_reason(code)
        assert "unavailable module" in reason
        assert "TTS" in reason and "utils" in reason
