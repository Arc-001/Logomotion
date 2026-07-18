"""
Shared pytest configuration.

Ensures the repository root is importable as `src...` regardless of the
directory pytest is invoked from, matching how the test suite has always
imported project modules (`from src.agent.nodes import ...`).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
