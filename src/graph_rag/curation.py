"""
Corpus curation for the Manim example knowledge base.

The retrieved examples are shown to the code generator as things to imitate,
so an example that cannot run on its own is worse than no example at all.
This module decides which dataset entries earn that role.

Used at index time (to keep the store clean) and again at retrieval time (as
a safety net for whatever a previous indexing run already wrote).
"""

import ast
import re
import sys
from typing import Optional

from ..manim_runner.validator import (
    REMOVED_APIS,
    called_names,
    has_scene_class,
    import_roots,
)

# Third-party packages an example may import on top of the standard library.
# Anything else is a helper module that lived beside the original file and
# does not exist here, so the example cannot be copied even in spirit.
ALLOWED_THIRD_PARTY = {"manim", "numpy", "scipy", "mpmath", "PIL"}

ALLOWED_IMPORT_ROOTS = ALLOWED_THIRD_PARTY | set(sys.stdlib_module_names)

# ManimGL (3Blue1Brown's fork) and Manim Community have diverged; the whole
# pipeline targets CE, so a ManimGL example teaches the generator the wrong
# API surface even when it is otherwise well written.
_WRONG_LIBRARY_ROOTS = {"manimlib", "manim_projects", "ManimProjects"}

# Local media the example would load from disk: fonts, images, audio, video.
_ASSET_SUFFIXES = ("png", "jpg", "jpeg", "svg", "gif", "mp3", "wav", "mp4", "mov", "ttf", "otf")
_LOCAL_ASSET_RE = re.compile(
    r"""['"][^'"]*\.(?:%s)['"]""" % "|".join(_ASSET_SUFFIXES),
    re.IGNORECASE,
)


def example_rejection_reason(code: str) -> Optional[str]:
    """Why this example must not be shown to the generator, or None if it is fine."""
    if not code or not code.strip():
        return "empty"

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"does not parse ({e.msg})"

    if not has_scene_class(tree):
        return "no Scene subclass"

    roots = import_roots(tree)

    wrong_library = sorted(roots & _WRONG_LIBRARY_ROOTS)
    if wrong_library:
        return f"targets ManimGL rather than Manim Community: {', '.join(wrong_library)}"

    if "manim" not in roots:
        return "does not import manim"

    foreign = sorted(roots - ALLOWED_IMPORT_ROOTS)
    if foreign:
        return f"imports unavailable module(s): {', '.join(foreign)}"

    removed = sorted(called_names(tree) & set(REMOVED_APIS))
    if removed:
        return f"uses removed Manim API: {', '.join(removed)}"

    asset = _LOCAL_ASSET_RE.search(code)
    if asset:
        return f"loads a local asset that does not exist here: {asset.group(0)}"

    return None


def is_usable_example(code: str) -> bool:
    """True when the example is self-contained, current, and runnable as shown."""
    return example_rejection_reason(code) is None
