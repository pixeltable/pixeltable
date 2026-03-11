#!/usr/bin/env bash
# Update the codebase index (AST-based) and optionally use Claude to
# generate/refresh per-directory CLAUDE.md files.
#
# Usage:
#   ./tool/update_codebase_index.sh              # AST index only
#   ./tool/update_codebase_index.sh --with-claude # AST index + Claude-generated summaries

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Step 1: Generating AST-based codebase index ==="
python tool/generate_codebase_index.py --root pixeltable --output CODEBASE_INDEX.md

if [[ "${1:-}" == "--with-claude" ]]; then
    echo "=== Step 2: Generating Claude-powered module summaries ==="

    if ! command -v claude &>/dev/null; then
        echo "Error: 'claude' CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
        exit 1
    fi

    # Generate per-directory CLAUDE.md summaries for key modules
    MODULES=(
        "pixeltable/catalog"
        "pixeltable/exec"
        "pixeltable/exprs"
        "pixeltable/func"
        "pixeltable/functions"
        "pixeltable/index"
        "pixeltable/io"
        "pixeltable/iterators"
        "pixeltable/metadata"
        "pixeltable/utils"
    )

    for module in "${MODULES[@]}"; do
        if [[ ! -d "$module" ]]; then
            continue
        fi

        echo "  Summarizing $module ..."
        claude -p \
            --model sonnet \
            --max-budget-usd 0.05 \
            "Read all Python files in $module/ and the existing CODEBASE_INDEX.md. Write a concise CLAUDE.md file for this directory that explains: (1) what this module does, (2) key classes and their responsibilities, (3) important patterns or non-obvious behaviors. Keep it under 60 lines. Output ONLY the markdown content, no preamble." \
            > "$module/CLAUDE.md"
    done

    echo "=== Done. Per-directory CLAUDE.md files updated. ==="
else
    echo "=== Done. Run with --with-claude to also generate module summaries. ==="
fi
