"""
Manim Code and Video Validators.

Validates generated code before rendering (cheap static checks) and
rendered videos for integrity and quality. Provides shared helpers for
ffprobe operations.
"""

import ast
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# Known-fatal legacy/forbidden Manim APIs with actionable replacements.
# Shared with corpus curation, which must reject the same APIs it would
# otherwise show the generator as examples to imitate.
REMOVED_APIS = {
    "ShowCreation": "ShowCreation was removed in Manim CE; use Create(...)",
    "TextMobject": "TextMobject was removed; use Text(...)",
    "TexMobject": "TexMobject was removed; use MathTex(r\"...\")",
    "FadeInFromDown": "FadeInFromDown was removed; use FadeIn(..., shift=UP)",
    "Code": "Do not use the Code() class; use Text() with a monospace font",
}

def import_roots(tree: ast.AST) -> set:
    """Top-level package name of every import in the module."""
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import — depends on a package we don't have
                roots.add(node.module.split(".")[0] if node.module else ".")
            elif node.module:
                roots.add(node.module.split(".")[0])
    return roots


def called_names(tree: ast.AST) -> set:
    """Names used in call position, e.g. ``ShowCreation(...)`` -> ShowCreation."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def defined_classes(tree: ast.AST) -> set:
    """Every class name defined in the module."""
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def has_scene_class(tree: ast.AST) -> bool:
    """Whether the module defines a class inheriting from some kind of Scene."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
                if name.endswith("Scene"):
                    return True
    return False


def validate_manim_code(code: str, scene_class_name: str) -> Optional[str]:
    """
    Run cheap static checks on generated code before spending a render on it.

    Every check reads the parsed module rather than the raw text: matching
    patterns against the source meant a docstring or comment that merely
    mentioned ``Code(`` or ``ShowCreation(`` failed the render before manim
    was ever invoked.

    Returns an actionable error message suitable for feeding back to the
    code corrector, or None when the code looks runnable.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        offending = (e.text or "").strip()
        message = f"SyntaxError on line {e.lineno}: {e.msg}"
        if offending:
            message += f"\n    {offending}"
        return message

    if scene_class_name not in defined_classes(tree):
        return f"Scene class '{scene_class_name}' is not defined in the code"

    if "manim" not in import_roots(tree):
        return "Missing manim import (add 'from manim import *')"

    problems = [
        REMOVED_APIS[name] for name in sorted(called_names(tree) & set(REMOVED_APIS))
    ]
    if problems:
        return "Forbidden or removed API usage:\n- " + "\n- ".join(problems)

    return None


@dataclass
class ValidationResult:
    """Result from video validation."""
    valid: bool
    errors: list[str]
    warnings: list[str]
    duration: Optional[float]
    width: Optional[int]
    height: Optional[int]
    codec: Optional[str]


def get_video_duration(video_path: str, timeout: int = 30) -> Optional[float]:
    """
    Get the duration of a video file in seconds using ffprobe.

    Returns None if the duration cannot be determined.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


class VideoValidator:
    """Validates Manim output videos."""

    def __init__(
        self,
        min_duration: float = 0.5,
        min_file_size: int = 1000,  # bytes
    ):
        self.min_duration = min_duration
        self.min_file_size = min_file_size

    def validate(self, video_path: str) -> ValidationResult:
        """
        Validate a video file.

        Args:
            video_path: Path to the video file

        Returns:
            ValidationResult with status and metadata
        """
        errors = []
        warnings = []

        path = Path(video_path)

        if not path.exists():
            return ValidationResult(
                valid=False,
                errors=["Video file does not exist"],
                warnings=[],
                duration=None,
                width=None,
                height=None,
                codec=None,
            )

        file_size = path.stat().st_size
        if file_size < self.min_file_size:
            errors.append(f"File too small: {file_size} bytes")

        metadata = self._get_metadata(video_path)

        if metadata is None:
            errors.append("Could not read video metadata")
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                duration=None,
                width=None,
                height=None,
                codec=None,
            )

        duration = metadata.get("duration")
        width = metadata.get("width")
        height = metadata.get("height")
        codec = metadata.get("codec")

        if duration is not None and duration < self.min_duration:
            warnings.append(f"Video very short: {duration:.2f}s")

        if width and height:
            if width < 100 or height < 100:
                warnings.append(f"Low resolution: {width}x{height}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            duration=duration,
            width=width,
            height=height,
            codec=codec,
        )

    def _get_metadata(self, video_path: str) -> Optional[dict]:
        """Get video metadata using ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return None

            data = json.loads(result.stdout)

            metadata = {}

            if "format" in data:
                duration_str = data["format"].get("duration")
                if duration_str:
                    metadata["duration"] = float(duration_str)

            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    metadata["width"] = stream.get("width")
                    metadata["height"] = stream.get("height")
                    metadata["codec"] = stream.get("codec_name")
                    break

            return metadata

        except FileNotFoundError:
            return {}
        except Exception:
            return None


def validate_video(video_path: str) -> ValidationResult:
    """Convenience function to validate a video."""
    validator = VideoValidator()
    return validator.validate(video_path)
