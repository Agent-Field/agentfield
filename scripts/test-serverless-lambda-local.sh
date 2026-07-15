#!/usr/bin/env bash
# Local, no-Docker/no-cloud simulation of the Go SDK running as an AWS Lambda
# function, driven end to end by the control plane. Unlike
# test-serverless-local.sh (which hits the SDK's http.Handler directly),
# this script exercises the *actual* Lambda code path:
#
#   control plane --plain HTTP--> fake Function URL front end
#     --translates to a v2 proxy event--> fake Lambda Runtime API
#     --GET next / POST response, exactly like the real service-->
#     the compiled cmd/lambda binary (lambda.Start + httpadapter.NewV2)
#     --ServeHTTP--> the Go SDK's srv.Handler()
#
# scripts/lambda-rie-fake is a hand-rolled stand-in for the AWS Lambda
# Runtime Interface Emulator: no Docker image, no AWS account, just two
# local HTTP listeners. See that package's doc comment for exactly which
# part of the real Lambda contract it implements.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${AGENTFIELD_SERVERLESS_TEST_DATA:-/tmp/agentfield-serverless-lambda-local-test}"
CP_PORT="${AGENTFIELD_SERVERLESS_TEST_CP_PORT:-18280}"
CP_URL="http://127.0.0.1:${CP_PORT}"
TOKEN="serverless-lambda-local-test-token"

PARENT_RUNTIME_PORT=18291
PARENT_PUBLIC_PORT=18292
CHILD_RUNTIME_PORT=18293
CHILD_PUBLIC_PORT=18294
PARENT_ID="svless-lambda-parent"
CHILD_ID="svless-lambda-child"

CP_LOG="${DATA_DIR}/control-plane.log"
PARENT_RIE_LOG="${DATA_DIR}/parent-rie.log"
PARENT_FN_LOG="${DATA_DIR}/parent-fn.log"
CHILD_RIE_LOG="${DATA_DIR}/child-rie.log"
CHILD_FN_LOG="${DATA_DIR}/child-fn.log"

CP_PID=""
PARENT_RIE_PID=""
PARENT_FN_PID=""
CHILD_RIE_PID=""
CHILD_FN_PID=""

pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; tail -n 40 "${2:-/dev/null}" 2>/dev/null || true; exit 1; }

cleanup() {
  for pid in "${PARENT_FN_PID}" "${CHILD_FN_PID}" "${PARENT_RIE_PID}" "${CHILD_RIE_PID}" "${CP_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

rm -rf "${DATA_DIR}"
mkdir -p "${DATA_DIR}"

echo "==> Building control plane (af), the fake Lambda runtime, and the Lambda-adapter node..."
(cd "${REPO_ROOT}/control-plane" && go build -o "${DATA_DIR}/af" ./cmd/af)
(cd "${REPO_ROOT}/scripts/lambda-rie-fake" && go build -o "${DATA_DIR}/fake-rie" .)
(cd "${REPO_ROOT}/examples/go_agent_nodes" && go build -o "${DATA_DIR}/lambda-node" ./cmd/lambda)

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

# Starts (or, on a redeploy, restarts) just the Lambda binary against an
# already-running fake-rie's runtime port. Echoes the new PID.
start_function() {
  local node_id="$1" runtime_port="$2" public_port="$3" fn_log="$4"
  local child_target="${5:-}" auth_token="${6:-}"

  local env_args=(
    "AWS_LAMBDA_RUNTIME_API=127.0.0.1:${runtime_port}"
    "AGENT_NODE_ID=${node_id}"
    "AGENTFIELD_URL=${CP_URL}"
    "CHILD_TARGET=${child_target}"
  )
  if [[ -n "${auth_token}" ]]; then
    env_args+=("AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN=${auth_token}")
  fi
  env "${env_args[@]}" nohup "${DATA_DIR}/lambda-node" >>"${fn_log}" 2>&1 &
  local fn_pid=$!

  wait_for_http "http://127.0.0.1:${public_port}/discover" "lambda function ${node_id}" "${fn_log}" >&2
  echo "${fn_pid}"
}

# Starts one simulated Lambda: a fake-rie process (runtime API + public
# "Function URL" front end) plus the compiled Lambda binary pointed at it via
# AWS_LAMBDA_RUNTIME_API. Echoes "<rie_pid> <fn_pid>".
start_lambda() {
  local node_id="$1" runtime_port="$2" public_port="$3" rie_log="$4" fn_log="$5"
  local child_target="${6:-}" auth_token="${7:-}"

  PORT="${runtime_port}" PUBLIC_PORT="${public_port}" \
    nohup "${DATA_DIR}/fake-rie" >>"${rie_log}" 2>&1 &
  local rie_pid=$!
  # /discover isn't servable until the function process is attached, so just
  # wait for the public listener's port to accept connections first.
  for _ in $(seq 1 60); do
    if curl -s --max-time 1 -o /dev/null "http://127.0.0.1:${public_port}/" 2>/dev/null; then break; fi
    sleep 0.2
  done

  local fn_pid
  fn_pid=$(start_function "${node_id}" "${runtime_port}" "${public_port}" "${fn_log}" "${child_target}" "${auth_token}")
  echo "${rie_pid} ${fn_pid}"
}

register_serverless() {
  local public_port="$1"
  "${DATA_DIR}/af" nodes register-serverless --server "${CP_URL}" --url "http://127.0.0.1:${public_port}" --json
}

call_reasoner() {
  local target="$1" body="$2"
  curl -sfS --max-time 10 -X POST "${CP_URL}/api/v1/execute/${target}" \
    -H 'Content-Type: application/json' -d "${body}"
}

start_control_plane

echo "==> Starting simulated Lambda functions (no auth) and registering..."
read -r CHILD_RIE_PID CHILD_FN_PID <<<"$(start_lambda "${CHILD_ID}" "${CHILD_RUNTIME_PORT}" "${CHILD_PUBLIC_PORT}" "${CHILD_RIE_LOG}" "${CHILD_FN_LOG}")"
register_serverless "${CHILD_PUBLIC_PORT}" >/dev/null

read -r PARENT_RIE_PID PARENT_FN_PID <<<"$(start_lambda "${PARENT_ID}" "${PARENT_RUNTIME_PORT}" "${PARENT_PUBLIC_PORT}" "${PARENT_RIE_LOG}" "${PARENT_FN_LOG}" "${CHILD_ID}.hello")"
register_serverless "${PARENT_PUBLIC_PORT}" >/dev/null

echo "==> [1] CP-triggered execution, through the simulated Lambda invoke path"
resp=$(call_reasoner "${PARENT_ID}.hello" '{"input":{"name":"Local"}}') || fail "hello call failed" "${CP_LOG}"
echo "${resp}" | grep -q '"greeting":"Hello, Local!"' && pass "control plane triggered Lambda-shaped node execution end to end" || fail "unexpected response: ${resp}"
echo "${resp}" | grep -q '"execution_id"' && pass "execution_id propagated" || fail "execution_id missing: ${resp}"

echo "==> [2] origin_auth_required defaults to false and is visible via /api/v1/nodes"
nodes_json=$(curl -sfS --max-time 5 "${CP_URL}/api/v1/nodes?show_all=true")
echo "${nodes_json}" | python3 -c "
import json,sys
nodes = json.load(sys.stdin)
nodes = nodes.get('nodes', nodes) if isinstance(nodes, dict) else nodes
node = next(n for n in nodes if n.get('id') == '${PARENT_ID}')
assert node['metadata']['custom']['origin_auth_required'] is False, node
" && pass "unauthenticated Lambda node correctly captured as origin_auth_required=false" || fail "origin_auth_required check failed"

echo "==> [3] /status endpoint, round-tripped through the Lambda invoke path"
status_resp=$(curl -sfS --max-time 5 "http://127.0.0.1:${PARENT_PUBLIC_PORT}/status")
echo "${status_resp}" | grep -q '"status":"running"' && pass "/status returns running" || fail "unexpected /status response: ${status_resp}"

echo "==> [4] parent -> child chain across two simulated Lambda functions"
resp=$(call_reasoner "${PARENT_ID}.relay" "{\"input\":{\"target\":\"${CHILD_ID}.hello\",\"message\":\"hi-child\"}}") || fail "relay call failed" "${CP_LOG}"
echo "${resp}" | grep -q '"greeting":"Hello, hi-child!"' && pass "parent/child chain executed across two Lambda-shaped nodes" || fail "unexpected relay response: ${resp}"

echo "==> [5] flip auth on (redeploy the function), re-register, confirm the flag swaps and unauthenticated direct calls are rejected"
kill "${PARENT_FN_PID}"; wait "${PARENT_FN_PID}" 2>/dev/null || true
PARENT_FN_PID=$(start_function "${PARENT_ID}" "${PARENT_RUNTIME_PORT}" "${PARENT_PUBLIC_PORT}" "${PARENT_FN_LOG}" "${CHILD_ID}.hello" "${TOKEN}")
register_serverless "${PARENT_PUBLIC_PORT}" >/dev/null

nodes_json=$(curl -sfS --max-time 5 "${CP_URL}/api/v1/nodes?show_all=true")
echo "${nodes_json}" | python3 -c "
import json,sys
nodes = json.load(sys.stdin)
nodes = nodes.get('nodes', nodes) if isinstance(nodes, dict) else nodes
node = next(n for n in nodes if n.get('id') == '${PARENT_ID}')
assert node['metadata']['custom']['origin_auth_required'] is True, node
" && pass "re-registration swapped origin_auth_required to true" || fail "origin_auth_required did not flip"

http_code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 -X POST "http://127.0.0.1:${PARENT_PUBLIC_PORT}/execute/hello" \
  -H 'Content-Type: application/json' -d '{"input":{"name":"Intruder"}}')
[[ "${http_code}" == "401" ]] && pass "direct unauthenticated call now rejected (401), through the Lambda invoke path" || fail "expected 401, got ${http_code}"

resp=$(call_reasoner "${PARENT_ID}.hello" '{"input":{"name":"Local"}}') || fail "hello call via CP failed after auth flip" "${CP_LOG}"
echo "${resp}" | grep -q '"greeting":"Hello, Local!"' && pass "CP-mediated call still succeeds (shared internal token, forwarded through the Lambda event headers)" || fail "unexpected response: ${resp}"

echo "==> [6] control-plane restart doesn't break the Lambda-shaped node"
kill "${CP_PID}"; wait "${CP_PID}" 2>/dev/null || true
start_control_plane

resp=$(call_reasoner "${PARENT_ID}.hello" '{"input":{"name":"Local"}}') || fail "hello call failed after CP restart" "${CP_LOG}"
echo "${resp}" | grep -q '"greeting":"Hello, Local!"' && pass "execution still works after control-plane restart" || fail "unexpected response: ${resp}"

echo
echo "All Lambda-simulated serverless checks passed."
