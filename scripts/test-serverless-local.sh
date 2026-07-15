#!/usr/bin/env bash
# Local, no-Docker/no-cloud REST validation of the Go SDK serverless node flow:
# control plane triggers a plain HTTP process exactly as it would a Lambda/
# Cloudflare Worker sitting behind an HTTP adapter. Exercises registration,
# CP-triggered execution, the /status health-check endpoint, parent/child
# chaining, the origin-auth flag surfaced in discovery (and its refresh on
# re-registration), and that a control-plane restart doesn't spuriously mark
# a serverless node unhealthy.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${AGENTFIELD_SERVERLESS_TEST_DATA:-/tmp/agentfield-serverless-local-test}"
CP_PORT="${AGENTFIELD_SERVERLESS_TEST_CP_PORT:-18080}"
CP_URL="http://127.0.0.1:${CP_PORT}"
TOKEN="serverless-local-test-token"

PARENT_PORT=18091
CHILD_PORT=18092
PARENT_ID="svless-local-parent"
CHILD_ID="svless-local-child"

CP_LOG="${DATA_DIR}/control-plane.log"
PARENT_LOG="${DATA_DIR}/parent-node.log"
CHILD_LOG="${DATA_DIR}/child-node.log"

CP_PID=""
PARENT_PID=""
CHILD_PID=""

pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; tail -n 40 "${2:-/dev/null}" 2>/dev/null || true; exit 1; }

cleanup() {
  for pid in "${CHILD_PID}" "${PARENT_PID}" "${CP_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

rm -rf "${DATA_DIR}"
mkdir -p "${DATA_DIR}"

echo "==> Building control plane (af) and the Go serverless example node..."
(cd "${REPO_ROOT}/control-plane" && go build -o "${DATA_DIR}/af" ./cmd/af)
(cd "${REPO_ROOT}/examples/go_agent_nodes" && go build -o "${DATA_DIR}/serverless-node" ./cmd/serverless)

wait_for_http() {
  local url="$1" name="$2" log="$3"
  for _ in $(seq 1 60); do
    if curl -sfS --max-time 2 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  fail "${name} did not become reachable at ${url}" "${log}"
}

start_control_plane() {
  AGENTFIELD_HOME="${DATA_DIR}/cp-home" \
  AGENTFIELD_STORAGE_MODE=local \
  AGENTFIELD_STORAGE_LOCAL_DATABASE_PATH="${DATA_DIR}/cp-home/agentfield.db" \
  AGENTFIELD_STORAGE_LOCAL_KV_STORE_PATH="${DATA_DIR}/cp-home/agentfield.bolt" \
  AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN="${TOKEN}" \
  nohup "${DATA_DIR}/af" server --port "${CP_PORT}" --open=false >>"${CP_LOG}" 2>&1 &
  CP_PID=$!
  wait_for_http "${CP_URL}/api/v1/health" "control plane" "${CP_LOG}"
  echo "==> Control plane up (PID ${CP_PID})"
}

# $1=node_id $2=port $3=logfile $4=pid_var_name $5=child_target(optional) $6=extra_env(optional, e.g. AUTH_TOKEN=x)
start_node() {
  local node_id="$1" port="$2" log="$3"
  local child_target="${4:-}"
  local auth_token="${5:-}"
  local env_args=(
    "AGENT_NODE_ID=${node_id}"
    "AGENTFIELD_URL=${CP_URL}"
    "PORT=${port}"
    "CHILD_TARGET=${child_target}"
  )
  if [[ -n "${auth_token}" ]]; then
    env_args+=("AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN=${auth_token}")
  fi
  env "${env_args[@]}" nohup "${DATA_DIR}/serverless-node" >>"${log}" 2>&1 &
  echo $!
  wait_for_http "http://127.0.0.1:${port}/health" "node ${node_id}" "${log}" >&2
}

register_serverless() {
  local port="$1"
  "${DATA_DIR}/af" nodes register-serverless --server "${CP_URL}" --url "http://127.0.0.1:${port}" --json
}

# Deliberately uses /api/v1/execute/:target, not the legacy /api/v1/reasoners/:id
# endpoint: the legacy handler (ExecuteReasonerHandler) never forwards the
# control plane's internal token to the node, so once a node enables
# RequireOriginAuth, calls through the legacy endpoint 401 even from the
# control plane itself. That's a real, separate bug - out of scope here, but
# worth knowing before reaching for the legacy endpoint against an auth'd node.
call_reasoner() {
  local target="$1" body="$2"
  curl -sfS --max-time 10 -X POST "${CP_URL}/api/v1/execute/${target}" \
    -H 'Content-Type: application/json' -d "${body}"
}

start_control_plane

echo "==> Starting nodes (no auth) and registering..."
CHILD_PID=$(start_node "${CHILD_ID}" "${CHILD_PORT}" "${CHILD_LOG}")
register_serverless "${CHILD_PORT}" >/dev/null

PARENT_PID=$(start_node "${PARENT_ID}" "${PARENT_PORT}" "${PARENT_LOG}" "${CHILD_ID}.hello")
register_serverless "${PARENT_PORT}" >/dev/null

echo "==> [1] CP-triggered execution"
resp=$(call_reasoner "${PARENT_ID}.hello" '{"input":{"name":"Local"}}') || fail "hello call failed" "${CP_LOG}"
echo "${resp}" | grep -q '"greeting":"Hello, Local!"' && pass "control plane triggered node execution end to end" || fail "unexpected response: ${resp}"
echo "${resp}" | grep -q '"execution_id"' && pass "execution_id propagated" || fail "execution_id missing: ${resp}"

echo "==> [2] origin_auth_required defaults to false and is visible via /api/v1/nodes"
nodes_json=$(curl -sfS --max-time 5 "${CP_URL}/api/v1/nodes?show_all=true")
echo "${nodes_json}" | python3 -c "
import json,sys
nodes = json.load(sys.stdin)
nodes = nodes.get('nodes', nodes) if isinstance(nodes, dict) else nodes
node = next(n for n in nodes if n.get('id') == '${PARENT_ID}')
assert node['metadata']['custom']['origin_auth_required'] is False, node
" && pass "unauthenticated node correctly captured as origin_auth_required=false" || fail "origin_auth_required check failed"

echo "==> [3] /status endpoint"
status_resp=$(curl -sfS --max-time 5 "http://127.0.0.1:${PARENT_PORT}/status")
echo "${status_resp}" | grep -q '"status":"running"' && pass "/status returns running" || fail "unexpected /status response: ${status_resp}"

echo "==> [4] parent -> child chain"
resp=$(call_reasoner "${PARENT_ID}.relay" "{\"input\":{\"target\":\"${CHILD_ID}.hello\",\"message\":\"hi-child\"}}") || fail "relay call failed" "${CP_LOG}"
echo "${resp}" | grep -q '"greeting":"Hello, hi-child!"' && pass "parent/child chain executed" || fail "unexpected relay response: ${resp}"

echo "==> [5] flip auth on, re-register, confirm the flag swaps and unauthenticated direct calls are rejected"
kill "${PARENT_PID}"; wait "${PARENT_PID}" 2>/dev/null || true
PARENT_PID=$(start_node "${PARENT_ID}" "${PARENT_PORT}" "${PARENT_LOG}" "${CHILD_ID}.hello" "${TOKEN}")
register_serverless "${PARENT_PORT}" >/dev/null

nodes_json=$(curl -sfS --max-time 5 "${CP_URL}/api/v1/nodes?show_all=true")
echo "${nodes_json}" | python3 -c "
import json,sys
nodes = json.load(sys.stdin)
nodes = nodes.get('nodes', nodes) if isinstance(nodes, dict) else nodes
node = next(n for n in nodes if n.get('id') == '${PARENT_ID}')
assert node['metadata']['custom']['origin_auth_required'] is True, node
" && pass "re-registration swapped origin_auth_required to true" || fail "origin_auth_required did not flip"

http_code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 -X POST "http://127.0.0.1:${PARENT_PORT}/execute/hello" \
  -H 'Content-Type: application/json' -d '{"input":{"name":"Intruder"}}')
[[ "${http_code}" == "401" ]] && pass "direct unauthenticated call now rejected (401)" || fail "expected 401, got ${http_code}"

resp=$(call_reasoner "${PARENT_ID}.hello" '{"input":{"name":"Local"}}') || fail "hello call via CP failed after auth flip" "${CP_LOG}"
echo "${resp}" | grep -q '"greeting":"Hello, Local!"' && pass "CP-mediated call still succeeds (shared internal token)" || fail "unexpected response: ${resp}"

echo "==> [6] control-plane restart doesn't break the serverless node"
kill "${CP_PID}"; wait "${CP_PID}" 2>/dev/null || true
start_control_plane

resp=$(call_reasoner "${PARENT_ID}.hello" '{"input":{"name":"Local"}}') || fail "hello call failed after CP restart" "${CP_LOG}"
echo "${resp}" | grep -q '"greeting":"Hello, Local!"' && pass "execution still works after control-plane restart" || fail "unexpected response: ${resp}"

echo
echo "All serverless local REST checks passed."
