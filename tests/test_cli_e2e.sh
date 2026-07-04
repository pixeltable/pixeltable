#!/usr/bin/env bash
# End-to-end CLI test for pxt cloud commands.
# Exercises: org list, db list, SDK table create, service lifecycle.
#
# Required env vars (or set them below):
#   PXT_TEST_API_KEY       - WorkOS API key (sk_...)
#
# Optional overrides:
#   PXT_TEST_CLOUD_HOST    - proxy domain (default: dev.pxt.run)
#   PXT_TEST_ORG_SLUG      - org slug to use (default: pixeltable)
#   PXT_TEST_DB_SLUG       - existing db slug to use (default: main)
#   PXT_TEST_SVC_NAME      - service name to create (default: clisvc-e2e)
#   SKIP_CLEANUP           - set to 1 to leave resources after failure
#
# NOTE: pxt db create/delete/stop/start require async Lambda support (provisioning
#       takes >60s synchronously). Those operations are excluded from this test.
#       Use SKIP_DB_LIFECYCLE=0 to enable them (expect longer runs or 504s).

set -uo pipefail   # -e intentionally omitted; we handle failures via assert_*

# ── Configuration ────────────────────────────────────────────────────────────
CLOUD_HOST="${PXT_TEST_CLOUD_HOST:-dev.pxt.run}"
INTERNAL_API_URL="${PXT_TEST_INTERNAL_API_URL:-https://dev-internal-api.pixeltable.com}"
ORG_SLUG="${PXT_TEST_ORG_SLUG:-pixeltable}"
DB_SLUG="${PXT_TEST_DB_SLUG:-main}"
SVC_NAME="${PXT_TEST_SVC_NAME:-clisvc-e2e}"
NEW_DB_SLUG="${PXT_TEST_NEW_DB_SLUG:-clitest-e2e-$$}"
SKIP_CLEANUP="${SKIP_CLEANUP:-0}"
SKIP_DB_LIFECYCLE="${SKIP_DB_LIFECYCLE:-0}"

API_KEY="${PXT_TEST_API_KEY:-}"
if [[ -z "$API_KEY" ]]; then
  echo "ERROR: PXT_TEST_API_KEY is not set." >&2
  exit 1
fi

DB_URI="pxt://${ORG_SLUG}:${DB_SLUG}"
TABLE_URI="${DB_URI}/tables/test_table"
SVC_URI="${DB_URI}/services/${SVC_NAME}"

PXT="${PXT:-pxt}"
PYTHON="${PYTHON:-python}"

export PIXELTABLE_API_KEY="$API_KEY"
export PIXELTABLE_CLOUD_HOST="$CLOUD_HOST"
export PIXELTABLE_API_URL="$INTERNAL_API_URL"

PASS=0
FAIL=0

pass() {
  echo "  PASS: $1"
  PASS=$((PASS + 1))
}

fail() {
  echo "  FAIL: $1"
  FAIL=$((FAIL + 1))
}

assert_contains() {
  local label="$1" needle="$2" haystack="$3"
  # reject if the output is a JSON error object
  if echo "$haystack" | grep -qF '"error"'; then
    fail "$label  (got error response)"
    echo "    output: $haystack"
    return
  fi
  if echo "$haystack" | grep -qF "$needle"; then
    pass "$label"
  else
    fail "$label  (expected '$needle' in output)"
    echo "    output: $haystack"
  fi
}

assert_ok() {
  local label="$1"; shift
  local out
  out=$("$@" 2>&1)
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    pass "$label"
  else
    fail "$label  (exit $rc)"
    echo "    output: $out"
  fi
}

cleanup() {
  if [[ "$SKIP_CLEANUP" == "1" ]]; then
    echo "SKIP_CLEANUP=1 — leaving resources intact"
    return
  fi
  echo ""
  echo "── Cleanup ──────────────────────────────────────────────────────────────────"
  $PXT service delete "$SVC_URI" --json 2>/dev/null || true
  if [[ "$SKIP_DB_LIFECYCLE" != "1" ]]; then
    NEW_DB_URI="pxt://${ORG_SLUG}:${NEW_DB_SLUG}"
    $PXT db delete "$NEW_DB_URI" --json 2>/dev/null || true
  fi
  echo "Cleanup done."
}
trap cleanup EXIT

echo "══════════════════════════════════════════════════════════════════"
echo "  pxt CLI e2e test"
echo "  api=$INTERNAL_API_URL"
echo "  host=$CLOUD_HOST  org=$ORG_SLUG  db=$DB_SLUG  svc=$SVC_NAME"
echo "══════════════════════════════════════════════════════════════════"

# ── 1. Help smoke tests ───────────────────────────────────────────────────────
echo ""
echo "── 1. Help smoke tests ──────────────────────────────────────────────────────"

out=$($PXT db --help 2>&1 || true)
assert_contains "pxt db --help lists 'create'" "create" "$out"
assert_contains "pxt db --help lists 'list'" "list" "$out"

out=$($PXT service --help 2>&1 || true)
assert_contains "pxt service --help lists 'create'" "create" "$out"
assert_contains "pxt service --help lists 'list'" "list" "$out"

out=$($PXT org --help 2>&1 || true)
assert_contains "pxt org --help lists 'list'" "list" "$out"

# ── 2. pxt org list ───────────────────────────────────────────────────────────
echo ""
echo "── 2. pxt org list ──────────────────────────────────────────────────────────"
out=$($PXT org list --json 2>&1 || true)
assert_contains "org list returns JSON array" "[" "$out"

# ── 3. pxt db list ───────────────────────────────────────────────────────────
echo ""
echo "── 3. pxt db list ───────────────────────────────────────────────────────────"
out=$($PXT db list "pxt://${ORG_SLUG}" --json 2>&1 || true)
assert_contains "db list shows existing db" "$DB_SLUG" "$out"

# ── 3b. pxt db create / delete lifecycle ────────────────────────────────────
if [[ "$SKIP_DB_LIFECYCLE" != "1" ]]; then
  echo ""
  echo "── 3b. pxt db create / delete lifecycle ─────────────────────────────────────"
  NEW_DB_URI="pxt://${ORG_SLUG}:${NEW_DB_SLUG}"
  out=$($PXT db create "$NEW_DB_URI" --json 2>&1 || true)
  assert_contains "db create returns ACTIVE after provisioning" "ACTIVE" "$out"
  out=$($PXT db list "pxt://${ORG_SLUG}" --json 2>&1 || true)
  assert_contains "db list shows newly created db" "$NEW_DB_SLUG" "$out"
  out=$($PXT db delete "$NEW_DB_URI" --json 2>&1 || true)
  assert_contains "db delete succeeds" "$NEW_DB_SLUG" "$out"
  SKIP_CLEANUP=1   # db already deleted above
fi

# ── 4. SDK: create a table in the cloud DB ───────────────────────────────────
# SDK accepts pxt://org:db/table_name and routes to the cloud proxy.
echo ""
echo "── 4. SDK: create table in cloud DB ────────────────────────────────────────"
CLOUD_TABLE_PATH="${DB_URI}/e2e_test_table"
out=$(PXT_CLOUD_TABLE_PATH="$CLOUD_TABLE_PATH" $PYTHON - <<'EOF' 2>&1 || true
import os, pixeltable as pxt
cloud_path = os.environ['PXT_CLOUD_TABLE_PATH']
pxt.init()
t = pxt.create_table(cloud_path, {'prompt': pxt.String})
print('table created:', t._path)
EOF
)
assert_contains "sdk creates table in cloud db" "e2e_test_table" "$out"

# ── 5. pxt service create ────────────────────────────────────────────────────
echo ""
echo "── 5. pxt service create ────────────────────────────────────────────────────"
out=$($PXT service create "${DB_URI}/tables/e2e_test_table" --name "$SVC_NAME" --workers 1 --json 2>&1 || true)
assert_contains "service create returns service name" "$SVC_NAME" "$out"

# ── 6. pxt service list ──────────────────────────────────────────────────────
echo ""
echo "── 6. pxt service list ──────────────────────────────────────────────────────"
out=$($PXT service list "$DB_URI" --json 2>&1 || true)
assert_contains "service list shows new service" "$SVC_NAME" "$out"

# ── 7. pxt service stop ──────────────────────────────────────────────────────
echo ""
echo "── 7. pxt service stop ──────────────────────────────────────────────────────"
out=$($PXT service stop "$SVC_URI" --json 2>&1 || true)
assert_contains "service stop returns service name" "$SVC_NAME" "$out"

# ── 8. pxt service start ─────────────────────────────────────────────────────
echo ""
echo "── 8. pxt service start ─────────────────────────────────────────────────────"
out=$($PXT service start "$SVC_URI" --json 2>&1 || true)
assert_contains "service start returns service name" "$SVC_NAME" "$out"

# ── 9. pxt service delete ───────────────────────────────────────────────────
echo ""
echo "── 9. pxt service delete ───────────────────────────────────────────────────"
assert_ok "service delete" $PXT service delete "$SVC_URI" --json

SKIP_CLEANUP=1   # already deleted service above

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  Results: ${PASS} passed, ${FAIL} failed"
echo "══════════════════════════════════════════════════════════════════"
[[ $FAIL -eq 0 ]] || exit 1
