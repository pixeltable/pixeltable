#!/usr/bin/env bash
# Verify all sample apps: nuke Pixeltable, uv sync, idempotent schema init, smoke test.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PASS=0
FAIL=0
declare -a RESULTS=()

nuke_pixeltable() {
    echo "=== Nuking Pixeltable database ==="
    local pxt_home="${PIXELTABLE_HOME:-$HOME/.pixeltable}"
    local pg_bin
    pg_bin=$(python3 -c 'import pixeltable_pgserver; print(pixeltable_pgserver._commands.POSTGRES_BIN_PATH)')
    local url="postgresql://postgres:@/postgres?host=${pxt_home}/pgdata"

    if [ ! -d "${pxt_home}/pgdata" ]; then
        echo "No existing pgdata — fresh start"
        return 0
    fi

    "$pg_bin/psql" "$url" -U postgres -c \
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid <> pg_backend_pid() AND datname = 'pixeltable';" \
        2>/dev/null || true
    "$pg_bin/psql" "$url" -U postgres -c "DROP DATABASE IF EXISTS pixeltable;" 2>/dev/null || true
    echo "Dropped pixeltable database"
}

run_app() {
    local name="$1"
    local dir="$2"
    local schema_cmd="$3"
    local smoke_cmd="${4:-}"

    echo ""
    echo "################################################################"
    echo "# APP: $name"
    echo "################################################################"

    nuke_pixeltable
    cd "$ROOT/$dir"

    echo "--- uv sync ---"
    if ! uv sync -q; then
        RESULTS+=("FAIL  $name — uv sync")
        FAIL=$((FAIL + 1))
        return
    fi

    echo "--- schema init (run 1) ---"
    if ! bash -c "$schema_cmd"; then
        RESULTS+=("FAIL  $name — schema init run 1")
        FAIL=$((FAIL + 1))
        return
    fi

    echo "--- schema init (run 2 — idempotency) ---"
    if ! bash -c "$schema_cmd"; then
        RESULTS+=("FAIL  $name — schema init run 2 (not idempotent)")
        FAIL=$((FAIL + 1))
        return
    fi

    if [ -n "$smoke_cmd" ]; then
        echo "--- smoke test ---"
        if ! bash -c "$smoke_cmd"; then
            RESULTS+=("FAIL  $name — smoke test")
            FAIL=$((FAIL + 1))
            return
        fi
    fi

    RESULTS+=("PASS  $name")
    PASS=$((PASS + 1))
    echo ">>> $name: OK"
}

run_app "intelligence-hub" "docs/sample-apps/intelligence-hub" \
    "uv run python -m spacy download en_core_web_sm && uv run python setup_pixeltable.py" \
    'uv run python -c "import ingest; print(\"ingest ok\")"'

run_app "cli-media-toolkit" "docs/sample-apps/cli-media-toolkit" \
    "uv run python schema.py" \
    "uv run python cli.py list"

run_app "context-aware-discord-bot" "docs/sample-apps/context-aware-discord-bot" \
    "uv run python setup_pixeltable.py" \
    'uv run python -c "import pixeltable as pxt; pxt.get_table(\"discord_bot.messages\"); print(\"tables ok\")"'

run_app "reddit-agentic-bot" "docs/sample-apps/reddit-agentic-bot" \
    "uv run python setup_pixeltable.py" \
    'uv run python -c "import config, pixeltable as pxt; pxt.get_table(config.BASE_DIR + \".questions\"); print(\"ok\")"'

run_app "jfk-files-mcp-server" "docs/sample-apps/jfk-files-mcp-server" \
    "uv run python schema.py" \
    'uv run python -c "import pixeltable as pxt; pxt.get_table(\"JFK.documents\"); print(\"schema ok\")"'

run_app "prompt-engineering-studio" "docs/sample-apps/prompt-engineering-studio-gradio-application" \
    'uv run python -c "import pixeltable; print(\"pxt ok\")"' \
    'uv run python -c "import gradio, textblob, nltk, mistralai; print(\"deps ok\")"'

run_app "trading-extension" "docs/sample-apps/ai-based-trading-insight-chrome-extension" \
    "cd server && uv run python schema.py" \
    'cd server && ANTHROPIC_API_KEY=sk-test-placeholder uv run python -c "import main; print(\"main ok\")"'

run_app "text-and-image-search" "docs/sample-apps/text-and-image-similarity-search-nextjs-fastapi" \
    "uv run python schema.py" \
    'uv run python -c "import main, pixeltable as pxt; pxt.get_table(\"media_search.videos\"); print(\"ok\")"'

run_app "multimodal-chat" "docs/sample-apps/multimodal-chat/backend" \
    "uv run python -m spacy download en_core_web_sm && uv run python setup_pixeltable.py" \
    'uv run python -c "import sys; sys.path.insert(0,\"api\"); sys.path.insert(0,\".\"); import routes; print(\"routes ok\")"'

echo ""
echo "================================================================"
echo "VERIFICATION SUMMARY"
echo "================================================================"
for r in "${RESULTS[@]}"; do echo "$r"; done
echo ""
echo "PASSED: $PASS  FAILED: $FAIL"
exit $FAIL
