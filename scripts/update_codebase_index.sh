#!/usr/bin/env bash
# Regenerate the AST-based codebase index.
#
# Usage:
#   ./tool/update_codebase_index.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Generating AST-based codebase index ==="
python tool/generate_codebase_index.py --root pixeltable --output .claude/CODEBASE_INDEX.md
