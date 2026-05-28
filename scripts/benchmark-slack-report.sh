#!/usr/bin/env bash
# Posts a daily Slack digest of the 100K-row batch-insert benchmarks, comparing
# today's mean execution time against the value from ~24h ago. The numbers come
# from the same Grafana Cloud Prometheus store that tool/benchmark_reporter.py
# pushes to in the stress-tests workflow.
# temp2
set -euo pipefail

: "${GRAFANA_PROM_URL:?set GRAFANA_PROM_URL to the Grafana Cloud Prometheus query base (ending in /api/prom)}"
: "${GRAFANA_INSTANCE_ID:?GRAFANA_INSTANCE_ID is required}"
: "${GRAFANA_SERVICE_ACCOUNT_TOKEN:?GRAFANA_SERVICE_ACCOUNT_TOKEN is required}"
: "${SLACK_WEBHOOK_URL:?SLACK_WEBHOOK_URL is required}"

BRANCH="${BENCHMARK_BRANCH:-main}"
METRIC="${BENCHMARK_METRIC:-benchmark_mean_seconds}"
PROM_USER="${GRAFANA_PROM_USER:-$GRAFANA_INSTANCE_ID}"

# The three benchmarks we report, in display order: pytest-benchmark test_name -> label.
ORDER=(
  'test_insert_batch_scaling_pxt[100000]'
  'test_insert_batch_scaling_pyarrow[100000]'
  'test_insert_batch_scaling_sql[100000]'
)
declare -A LABELS=(
  ['test_insert_batch_scaling_pxt[100000]']='Pixeltable'
  ['test_insert_batch_scaling_pyarrow[100000]']='PyArrow'
  ['test_insert_batch_scaling_sql[100000]']='SQLAlchemy'
)

# --- helpers -----------------------------------------------------------------

# Run a PromQL instant query; print the scalar value, or empty string if no data.
prom_query() {
  curl -sf -G "${GRAFANA_PROM_URL%/}/api/v1/query" \
    -u "${PROM_USER}:${GRAFANA_SERVICE_ACCOUNT_TOKEN}" \
    --data-urlencode "query=$1" \
    | jq -r '.data.result[0].value[1] // empty'
}

# Fetch the latest value for a test, now and ~24h ago. Prints "<cur> <prev>"
# (either may be empty). last_over_time tolerates the once-a-day push cadence;
# a plain instant query would go stale 5 min after each push.
fetch_values() {
  local sel="${METRIC}{test_name=\"$1\",branch=\"${BRANCH}\"}"
  local cur prev
  cur=$(prom_query "last_over_time(${sel}[3d])") || cur=''
  prev=$(prom_query "last_over_time(${sel}[3d] offset 24h)") || prev=''
  echo "$cur $prev"
}

# Format a duration in seconds as a human-friendly string (ms below 1s).
format_duration() {
  awk -v s="$1" 'BEGIN { if (s < 1) printf "%.2f ms", s * 1000; else printf "%.2f s", s }'
}

# Build the Slack mrkdwn line for one benchmark. Args: label, cur_seconds, prev_seconds.
# Lower runtime is better, so a decrease vs yesterday is reported as "faster".
format_line() {
  local label="$1" cur="$2" prev="$3"
  if [[ -z "$cur" ]]; then
    printf '*%s*\n`n/a` (no data reported)' "$label"
    return
  fi
  local cur_fmt; cur_fmt=$(format_duration "$cur")
  if [[ -z "$prev" ]]; then
    printf '*%s*\n`%s` (no prior value to compare)' "$label" "$cur_fmt"
    return
  fi
  local pct abs change
  pct=$(awk -v c="$cur" -v p="$prev" 'BEGIN { printf "%.1f", (c - p) / p * 100 }')
  abs="${pct#-}"
  if   awk -v x="$pct" 'BEGIN { exit !(x < -1) }'; then change="${abs}% faster than yesterday"
  elif awk -v x="$pct" 'BEGIN { exit !(x >  1) }'; then change="${abs}% slower than yesterday"
  else change='unchanged vs yesterday'; fi
  printf '*%s*\n`%s`   %s' "$label" "$cur_fmt" "$change"
}

# --- gather data -------------------------------------------------------------

declare -A CUR PREV
for tn in "${ORDER[@]}"; do
  read -r cur prev < <(fetch_values "$tn")
  CUR[$tn]="$cur"
  PREV[$tn]="$prev"
done

# Headline: Pixeltable's insert time relative to each baseline.
pxt_cur="${CUR['test_insert_batch_scaling_pxt[100000]']}"
pa_cur="${CUR['test_insert_batch_scaling_pyarrow[100000]']}"
sql_cur="${CUR['test_insert_batch_scaling_sql[100000]']}"
headline=''
if [[ -n "$pxt_cur" ]]; then
  parts=()
  if [[ -n "$sql_cur" ]]; then
    r=$(awk -v a="$pxt_cur" -v b="$sql_cur" 'BEGIN { if (b > 0) printf "%.1f", a / b }')
    [[ -n "$r" ]] && parts+=("*${r}×* the time of SQLAlchemy")
  fi
  if [[ -n "$pa_cur" ]]; then
    r=$(awk -v a="$pxt_cur" -v b="$pa_cur" 'BEGIN { if (b > 0) printf "%.1f", a / b }')
    [[ -n "$r" ]] && parts+=("*${r}×* the time of PyArrow")
  fi
  if [[ ${#parts[@]} -eq 2 ]]; then
    headline="Pixeltable takes ${parts[0]} and ${parts[1]} for 100K-row inserts."
  elif [[ ${#parts[@]} -eq 1 ]]; then
    headline="Pixeltable takes ${parts[0]} for 100K-row inserts."
  fi
fi

# --- build Slack message blocks ----------------------------------------------

sections='[]'
for tn in "${ORDER[@]}"; do
  line=$(format_line "${LABELS[$tn]}" "${CUR[$tn]}" "${PREV[$tn]}")
  sections=$(jq -c --arg t "$line" '. += [{type: "section", text: {type: "mrkdwn", text: $t}}]' <<<"$sections")
done

date_str=$(date -u +'%Y-%m-%d %H:%M UTC')

payload=$(jq -n \
  --argjson sections "$sections" \
  --arg branch "$BRANCH" \
  --arg date "$date_str" \
  --arg headline "$headline" \
  '{
    blocks: (
      [
        {type: "header", text: {type: "plain_text", text: "Batch Insert Benchmark · 100K rows"}},
        {type: "context", elements: [{type: "mrkdwn", text: "*Branch:* `\($branch)`   ·   \($date)"}]},
        {type: "divider"}
      ]
      + (if $headline == "" then [] else [{type: "section", text: {type: "mrkdwn", text: $headline}}, {type: "divider"}] end)
      + $sections
      + [{type: "context", elements: [{type: "mrkdwn",
           text: "Baselines: PyArrow and raw SQLAlchemy · group `batch_insert_scaling` · mean of pytest-benchmark rounds · source: Grafana"}]}]
    )
  }')

curl -sf -X POST -H 'Content-type: application/json' --data "$payload" "$SLACK_WEBHOOK_URL" >/dev/null
echo 'Posted benchmark digest to Slack.'
