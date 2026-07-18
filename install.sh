#!/bin/bash
# Thin wrapper kept for backward compatibility.
# All functionality lives in start.sh — this is equivalent to:
#   ./start.sh install [--docker]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/start.sh" install "$@"
