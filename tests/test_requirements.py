"""
Guards requirements.txt and requirements.lock against drifting apart.

The Docker image installs the lock file, so a package listed in
requirements.txt but missing from the lock produces an image that fails at
import time rather than at build time — which is how duckduckgo-search and
lxml, both imported by src/search/web.py, went missing from the lock.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _names(path):
    names = set()
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        name = re.split(r"[=<>!~\[]", line)[0].strip()
        if name:
            names.add(name.lower().replace("_", "-"))
    return names


def test_every_declared_dependency_is_pinned_in_the_lock():
    declared = _names(REPO_ROOT / "requirements.txt")
    locked = _names(REPO_ROOT / "requirements.lock")

    missing = sorted(declared - locked)

    assert not missing, f"declared in requirements.txt but absent from the lock: {missing}"


def test_the_lock_pins_exact_versions():
    unpinned = [
        line.strip()
        for line in (REPO_ROOT / "requirements.lock").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#") and "==" not in line
    ]

    assert not unpinned, f"lock entries without an exact version: {unpinned}"
