#!/usr/bin/env bash
# Deep verification for sample apps: baseline + no-key fixture/API/UI tests.
# Standard tier — no API keys required except documented KEY_GATED skips.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURE_IMAGE="$ROOT/tests/data/_verify_fixture.png"
FIXTURE_DOC="$ROOT/tests/data/documents/simple.md"
PORT=8765

create_fixture_image() {
    if [ ! -f "$FIXTURE_IMAGE" ]; then
        echo "Creating verification fixture image..."
        echo 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==' \
            | base64 -d > "$FIXTURE_IMAGE"
    fi
}

wait_for_px_table() {
    local table_path="$1"
    local retries="${2:-60}"
    for _ in $(seq 1 "$retries"); do
        if uv run python -c "import pixeltable as pxt; pxt.get_table('$table_path'); print('ready')" 2>/dev/null; then
            return 0
        fi
        sleep 2
    done
    echo "Timed out waiting for table $table_path"
    return 1
}

npm_build() {
    if [ -f package-lock.json ]; then
        npm install
    else
        npm install
    fi
    npm run build
}

create_fixture_image

declare -a MATRIX=()
BASELINE_PASS=0
BASELINE_FAIL=0
DEEP_PASS=0
DEEP_FAIL=0
DEEP_SKIP=0

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

wait_for_url() {
    local url="$1"
    local retries="${2:-30}"
    for _ in $(seq 1 "$retries"); do
        if curl -sf "$url" -o /dev/null 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    return 1
}

start_server() {
    local cmd="$1"
    eval "$cmd" &
    SERVER_PID=$!
    echo "Started server PID $SERVER_PID"
}

stop_server() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}

record() {
    local name="$1" baseline="$2" deep="$3" notes="${4:-}"
    MATRIX+=("$(printf '%-32s %-10s %-8s %s' "$name" "$baseline" "$deep" "$notes")")
}

run_baseline() {
    local schema_cmd="$1"
    local smoke_cmd="${2:-}"

    if ! uv sync -q; then
        return 1
    fi
    if ! bash -c "$schema_cmd"; then
        return 1
    fi
    if ! bash -c "$schema_cmd"; then
        return 1
    fi
    if [ -n "$smoke_cmd" ] && ! bash -c "$smoke_cmd"; then
        return 1
    fi
    return 0
}

# ── Per-app deep runners ──────────────────────────────────────────────────────

deep_cli_media_toolkit() {
    uv run python cli.py add "$FIXTURE_IMAGE"
    uv run python -c "
import pixeltable as pxt
t = pxt.get_table('ai_media_toolkit')
assert t.count() > 0, 'add did not insert media'
print(f'media count: {t.count()}')
"
    uv run python cli.py status
    uv run python cli.py functions
}

deep_intelligence_hub() {
    uv run python -c "
import pixeltable as pxt
import config
import pandas as pd
from datetime import datetime, timezone

t = pxt.get_table(f'{config.APP_NAMESPACE}.sources')
df = pd.read_csv('sample_sources.csv')
for row in df.to_dict('records'):
    t.insert([{
        'url': row['url'],
        'title': row['title'],
        'doc': row['url'],
        'origin': 'csv',
        'metadata': {},
        'timestamp': datetime.now(tz=timezone.utc),
    }], on_error='ignore')
assert t.count() > 0, 'CSV ingest failed'
print(f'sources count: {t.count()}')
"
}

deep_discord_bot() {
    uv run python -c "
import pixeltable as pxt
from datetime import datetime

m = pxt.get_table('discord_bot.messages')
m.insert([{
    'channel_id': 'test',
    'username': 'u',
    'content': 'Pixeltable stores multimodal data.',
    'timestamp': datetime.now(),
}])
s = pxt.get_table('discord_bot.sentences')
sim = s.text.similarity(string='multimodal')
results = s.order_by(sim, asc=False).limit(3).collect()
assert len(results) >= 1, 'similarity search returned no results'
print(f'search hits: {len(results)}')
"
}

deep_reddit_bot() {
    uv run python -c "
import config
import pixeltable as pxt

doc_chunks = pxt.get_table(f'{config.BASE_DIR}.doc_chunks')
assert doc_chunks.count() > 0, 'no document chunks indexed'
sim = doc_chunks.text.similarity(string='What is Pixeltable?', idx='doc_chunks_text_idx')
r = doc_chunks.order_by(sim, asc=False).limit(3).collect()
assert len(r) > 0, 'RAG search returned no results'
print(f'chunks: {doc_chunks.count()}, RAG hits: {len(r)}')
"
}

deep_prompt_studio() {
    uv run python -c "
import nltk
from textblob import TextBlob
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
sentiment = TextBlob('great response').sentiment
print(f'sentiment: {sentiment}')
"
}

deep_trading_extension() {
    python3 -c "import json; json.load(open('../extension/manifest.json')); print('manifest ok')"
    ANTHROPIC_API_KEY=sk-test-placeholder uv run uvicorn main:app --host 127.0.0.1 --port "$PORT" &
    SERVER_PID=$!
    sleep 3
    # Root path may 404 — server reachable is enough
    curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/" | grep -qE '^(200|404)$'
    stop_server
}

deep_text_image_search() {
    wait_for_px_table "media_search.images"
    uv run uvicorn main:app --host 127.0.0.1 --port "$PORT" &
    SERVER_PID=$!
    wait_for_url "http://127.0.0.1:$PORT/openapi.json"
    wait_for_px_table "media_search.images"
    curl -sf "http://127.0.0.1:$PORT/openapi.json" | python3 -c "import sys,json; json.load(sys.stdin)"
    curl -sf -F 'tags=["test"]' -F "file=@$FIXTURE_IMAGE" "http://127.0.0.1:$PORT/api/upload-image"
    sleep 5
    curl -sf -F "search_type=image" -F "num_results=3" -F "file=@$FIXTURE_IMAGE" \
        "http://127.0.0.1:$PORT/api/search-images"
    stop_server
    cd frontend
    npm_build
}

deep_multimodal_chat() {
    wait_for_px_table "chatbot.documents"
    (cd api && uv run uvicorn main:app --host 127.0.0.1 --port "$PORT") &
    SERVER_PID=$!
    wait_for_url "http://127.0.0.1:$PORT/health"
    curl -sf "http://127.0.0.1:$PORT/health" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok'"
    curl -sf -X POST -F "file=@$FIXTURE_DOC" "http://127.0.0.1:$PORT/api/upload"
    curl -sf "http://127.0.0.1:$PORT/api/files" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'files' in d"
    stop_server
    cd "$ROOT/docs/sample-apps/multimodal-chat/frontend"
    npm_build
}

run_app() {
    local name="$1"
    local dir="$2"
    local schema_cmd="$3"
    local smoke_cmd="$4"
    local deep_fn="${5:-}"
    local deep_mode="${6:-run}"  # run | skip | key_gated

    echo ""
    echo "################################################################"
    echo "# APP: $name"
    echo "################################################################"

    nuke_pixeltable
    cd "$ROOT/$dir"

    local baseline_result="FAIL"
    echo "--- baseline ---"
    if run_baseline "$schema_cmd" "$smoke_cmd"; then
        baseline_result="PASS"
        BASELINE_PASS=$((BASELINE_PASS + 1))
    else
        BASELINE_FAIL=$((BASELINE_FAIL + 1))
        record "$name" "$baseline_result" "SKIP" "baseline failed"
        return
    fi

    local deep_result="SKIP"
    local notes=""

    if [ "$deep_mode" = "skip" ]; then
        notes="needs MISTRAL_API_KEY"
        DEEP_SKIP=$((DEEP_SKIP + 1))
        record "$name" "$baseline_result" "$deep_result" "$notes"
        echo ">>> $name: baseline OK, deep SKIP ($notes)"
        return
    fi

    if [ "$deep_mode" = "key_gated" ]; then
        notes="NLTK only; Gradio/Mistral KEY_GATED"
        echo "--- deep (partial) ---"
        if $deep_fn; then
            deep_result="PASS"
            DEEP_PASS=$((DEEP_PASS + 1))
        else
            deep_result="FAIL"
            DEEP_FAIL=$((DEEP_FAIL + 1))
        fi
        record "$name" "$baseline_result" "$deep_result" "$notes"
        echo ">>> $name: baseline $baseline_result, deep $deep_result"
        return
    fi

    echo "--- deep ---"
    if $deep_fn; then
        deep_result="PASS"
        DEEP_PASS=$((DEEP_PASS + 1))
        echo ">>> $name: baseline $baseline_result, deep $deep_result"
    else
        deep_result="FAIL"
        DEEP_FAIL=$((DEEP_FAIL + 1))
        echo ">>> $name: baseline $baseline_result, deep $deep_result FAILED"
    fi
    record "$name" "$baseline_result" "$deep_result" "$notes"
}

# ── App matrix ────────────────────────────────────────────────────────────────

run_app "intelligence-hub" "docs/sample-apps/intelligence-hub" \
    "uv run python -m spacy download en_core_web_sm && uv run python setup_pixeltable.py" \
    'uv run python -c "import ingest; print(\"ingest ok\")"' \
    deep_intelligence_hub

run_app "cli-media-toolkit" "docs/sample-apps/cli-media-toolkit" \
    "uv run python schema.py" \
    "uv run python cli.py list" \
    deep_cli_media_toolkit

run_app "context-aware-discord-bot" "docs/sample-apps/context-aware-discord-bot" \
    "uv run python -m spacy download en_core_web_sm && uv run python setup_pixeltable.py" \
    'uv run python -c "import pixeltable as pxt; pxt.get_table(\"discord_bot.messages\"); print(\"tables ok\")"' \
    deep_discord_bot

run_app "reddit-agentic-bot" "docs/sample-apps/reddit-agentic-bot" \
    "uv run python setup_pixeltable.py" \
    'uv run python -c "import config, pixeltable as pxt; pxt.get_table(config.BASE_DIR + \".questions\"); print(\"ok\")"' \
    deep_reddit_bot

run_app "jfk-files-mcp-server" "docs/sample-apps/jfk-files-mcp-server" \
    "uv run python schema.py" \
    'uv run python -c "import pixeltable as pxt; pxt.get_table(\"JFK.documents\"); print(\"schema ok\")"' \
    "" skip

run_app "prompt-engineering-studio" "docs/sample-apps/prompt-engineering-studio-gradio-application" \
    'uv run python -c "import pixeltable; print(\"pxt ok\")"' \
    'uv run python -c "import gradio, textblob, nltk, mistralai; print(\"deps ok\")"' \
    deep_prompt_studio key_gated

run_app "trading-extension" "docs/sample-apps/ai-based-trading-insight-chrome-extension/server" \
    "uv run python schema.py" \
    'ANTHROPIC_API_KEY=sk-test-placeholder uv run python -c "import main; print(\"main ok\")"' \
    deep_trading_extension

run_app "text-and-image-search" "docs/sample-apps/text-and-image-similarity-search-nextjs-fastapi" \
    "uv run python schema.py" \
    'uv run python -c "import main, pixeltable as pxt; pxt.get_table(\"media_search.videos\"); print(\"ok\")"' \
    deep_text_image_search

run_app "multimodal-chat" "docs/sample-apps/multimodal-chat/backend" \
    "uv run python -m spacy download en_core_web_sm && uv run python setup_pixeltable.py" \
    'uv run python -c "import sys; sys.path.insert(0,\"api\"); sys.path.insert(0,\".\"); import routes; print(\"routes ok\")"' \
    deep_multimodal_chat

# ── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo "================================================================"
echo "DEEP VERIFICATION SUMMARY"
echo "================================================================"
printf '%-32s %-10s %-8s %s\n' "APP" "BASELINE" "DEEP" "NOTES"
for row in "${MATRIX[@]}"; do
    echo "$row"
done
echo ""
echo "BASELINE: PASS=$BASELINE_PASS  FAIL=$BASELINE_FAIL"
echo "DEEP:     PASS=$DEEP_PASS  FAIL=$DEEP_FAIL  SKIP=$DEEP_SKIP"
echo ""
echo "KEY_GATED (manual, needs API keys):"
echo "  - context-aware-discord-bot: bot.py (DISCORD_TOKEN + OPENAI_API_KEY)"
echo "  - reddit-agentic-bot: reddit_bot.py (Reddit creds + ANTHROPIC_API_KEY)"
echo "  - jfk-files-mcp-server: server.py load (MISTRAL_API_KEY)"
echo "  - prompt-engineering-studio: Gradio + Mistral LLM cells"
echo "  - trading-extension: POST /analyze (real ANTHROPIC_API_KEY)"
echo "  - multimodal-chat: /api/chat, video/audio upload (OPENAI_API_KEY)"
echo ""
echo "UI spot-checks (optional):"
echo "  cd docs/sample-apps/multimodal-chat/frontend && npm run dev"
echo "  cd docs/sample-apps/text-and-image-similarity-search-nextjs-fastapi/frontend && npm run dev"
echo "  Chrome -> Load unpacked -> docs/sample-apps/ai-based-trading-insight-chrome-extension/extension/"

TOTAL_FAIL=$((BASELINE_FAIL + DEEP_FAIL))
exit $TOTAL_FAIL
