#!/usr/bin/env bash
set -euo pipefail

cd /app

if [[ $# -gt 0 ]]; then
  exec python nested_workflow_stress.py "$@"
fi

TARGET=${TARGET:-demo-agent.synthetic_nested}
BASE_URL=${BASE_URL:-http://host.docker.internal:8080}
MODE=${MODE:-async}
REQUESTS=${REQUESTS:-200}
CONCURRENCY=${CONCURRENCY:-16}
DEPTH=${DEPTH:-0}
WIDTH=${WIDTH:-0}
PAYLOAD_BYTES=${PAYLOAD_BYTES:-1024}
PRINT_FAILURES=${PRINT_FAILURES:-false}
SAVE_METRICS=${SAVE_METRICS:-}
POLL_INTERVAL=${POLL_INTERVAL:-0.25}
MAX_POLL_INTERVAL=${MAX_POLL_INTERVAL:-5.0}
BACKOFF_MULTIPLIER=${BACKOFF_MULTIPLIER:-1.7}
EXECUTION_TIMEOUT=${EXECUTION_TIMEOUT:-600}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-60}
HEADERS=${HEADERS:-}
METRICS_URL=${METRICS_URL:-}
METRICS=${METRICS:-}
METRICS_TIMEOUT=${METRICS_TIMEOUT:-5}
SCENARIO_FILE=${SCENARIO_FILE:-}

ARGS=(
  --mode "$MODE"
  --base-url "$BASE_URL"
  --target "$TARGET"
  --requests "$REQUESTS"
  --concurrency "$CONCURRENCY"
  --depth "$DEPTH"
  --width "$WIDTH"
  --payload-bytes "$PAYLOAD_BYTES"
  --poll-interval "$POLL_INTERVAL"
  --max-poll-interval "$MAX_POLL_INTERVAL"
  --backoff-multiplier "$BACKOFF_MULTIPLIER"
  --execution-timeout "$EXECUTION_TIMEOUT"
  --request-timeout "$REQUEST_TIMEOUT"
)

if [[ -n "$HEADERS" ]]; then
  IFS=',' read -ra HEADER_LIST <<< "$HEADERS"
  for header in "${HEADER_LIST[@]}"; do
    ARGS+=(--header "$header")
  done
fi

if [[ "$PRINT_FAILURES" =~ ^(true|1|yes)$ ]]; then
  ARGS+=(--print-failures)
fi

if [[ -n "$SAVE_METRICS" ]]; then
  ARGS+=(--save-metrics "$SAVE_METRICS")
fi

if [[ -n "$METRICS_URL" ]]; then
  ARGS+=(--metrics-url "$METRICS_URL" --metrics-timeout "$METRICS_TIMEOUT")
  if [[ -n "$METRICS" ]]; then
    IFS=',' read -ra METRIC_LIST <<< "$METRICS"
    for metric in "${METRIC_LIST[@]}"; do
      ARGS+=(--metrics "$metric")
    done
  fi
fi

if [[ -n "$SCENARIO_FILE" ]]; then
  ARGS+=(--scenario-file "$SCENARIO_FILE")
fi

echo "Running stress harness -> target=$TARGET base_url=$BASE_URL requests=$REQUESTS concurrency=$CONCURRENCY"
exec python nested_workflow_stress.py "${ARGS[@]}"
