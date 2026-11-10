#!/usr/bin/env bash
# Check B: checkout SDKs (Python sdk/python, Go sdk/go) against stable v0.1.145 and old cloud v0.1.57.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/smoke/existing/lib.sh
source "${SCRIPT_DIR}/lib.sh"

CP_TAG="${1:-v0.1.145}"
SDK_SPEC="${2:-python-main}"
CP_PORT="${SMOKE_CP_PORT:-18082}"

case "${CP_TAG}" in
  v0.1.145|stable)
    CP_IMAGE="${STABLE_IMAGE}"
    CP_LABEL="v0.1.145"
    ;;
  v0.1.57|old-cloud)
    CP_IMAGE="${OLD_CLOUD_IMAGE}"
    CP_LABEL="v0.1.57"
    ;;
  *)
    CP_IMAGE="agentfield/control-plane:${CP_TAG}"
    CP_LABEL="${CP_TAG}"
    ;;
esac

init_smoke_env "check-b-${CP_LABEL}-${SDK_SPEC}" "Check B (${SDK_SPEC} -> control-plane ${CP_LABEL})"

DATA_DIR="${SMOKE_WORK_DIR}/cp-data"
CP_NAME="af-smoke-cp-b-${CP_LABEL//[^a-zA-Z0-9-]/-}-${SDK_SPEC//[^a-zA-Z0-9-]/-}"
CP_URL="http://127.0.0.1:${CP_PORT}"

docker pull "${CP_IMAGE}" >/dev/null
start_cp "${CP_NAME}" "${CP_IMAGE}" "${CP_PORT}" "${DATA_DIR}" "local"
add_evidence "Control plane \`${CP_IMAGE}\` healthy at \`${CP_URL}\`"

NODE_PORT=18121
NODE_ID="node-${SDK_SPEC//[^a-zA-Z0-9-]/-}-${CP_LABEL//[^a-zA-Z0-9-]/-}"

case "${SDK_SPEC}" in
  python-main)
    VENV_DIR="${SMOKE_WORK_DIR}/venv-py-main"
    PY_VER="$(setup_python_venv "${VENV_DIR}" "source")"
    start_python_node "${VENV_DIR}" "${NODE_ID}" "${CP_URL}" "${NODE_PORT}" 5
    add_evidence "Python SDK from source (\`sdk/python\`, version \`${PY_VER}\`) started as \`${NODE_ID}\`"
    ;;
  go-main)
    GO_BIN="${SMOKE_WORK_DIR}/go-node-main"
    build_go_node "go-main" "${GO_BIN}"
    start_go_node "${GO_BIN}" "${NODE_ID}" "${CP_URL}" "${NODE_PORT}" "5s"
    add_evidence "Go SDK from source (\`sdk/go\`) started as \`${NODE_ID}\`"
    ;;
  *)
    log_err "Unknown Check B SDK spec: ${SDK_SPEC}"
    exit 1
    ;;
esac

wait_for_node_registered "${CP_URL}" "${NODE_ID}" "${NODE_PORT}"
add_evidence "Node \`${NODE_ID}\` registered and appeared active in \`GET /api/v1/nodes\` on \`${CP_IMAGE}\`"

SYNC_RESP="$(run_sync_execute "${CP_URL}" "${NODE_ID}.echo" '{"message":"check-b-sync","seq":31,"extra":{"cp":"'"${CP_LABEL}"'"}}')"
assert_echo_result "${SYNC_RESP}" "check-b-sync" 31 "${NODE_ID}"
add_evidence "Sync execute \`POST /api/v1/execute/${NODE_ID}.echo\` succeeded on \`${CP_IMAGE}\`"

ASYNC_RESP="$(run_async_execute "${CP_URL}" "${NODE_ID}.echo" '{"message":"check-b-async","seq":42,"extra":{"cp":"'"${CP_LABEL}"'"}}')"
assert_echo_result "${ASYNC_RESP}" "check-b-async" 42 "${NODE_ID}"
add_evidence "Async execute \`POST /api/v1/execute/async/${NODE_ID}.echo\` + status poll succeeded on \`${CP_IMAGE}\`"

if [[ "${SDK_SPEC}" == "go-main" ]]; then
  python3 - "${SYNC_RESP}" "${ASYNC_RESP}" <<'PY'
import json
import sys

for idx, raw in enumerate(sys.argv[1:], start=1):
    res = (json.loads(raw).get("result") or {})
    if res.get("run_watched") is not False:
        raise SystemExit(f"expected run_watched=False on old control plane (call {idx}): {res}")
PY
  add_evidence "Go main node reported \`RunWatched == false\` on both sync and async calls against \`${CP_IMAGE}\`"
fi

HB_PROOF="$(verify_heartbeat_survival "${CP_URL}" "${NODE_ID}" 12)"
add_evidence "Heartbeat survival verified for \`${NODE_ID}\` on \`${CP_IMAGE}\`: \`${HB_PROOF}\`"

write_check_summary "PASS" "Checkout SDK (${SDK_SPEC}) registered, executed sync + async, and survived heartbeat sweep against ${CP_IMAGE}."
log_info "Check B (${SDK_SPEC} -> ${CP_LABEL}) PASSED"
