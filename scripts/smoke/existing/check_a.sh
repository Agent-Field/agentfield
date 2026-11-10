#!/usr/bin/env bash
# Check A: old SDKs (Python 0.1.130, Python 0.1.145, Go v0.1.130) against the candidate control plane.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/smoke/existing/lib.sh
source "${SCRIPT_DIR}/lib.sh"

NODE_SPEC="${1:-all}"
CP_PORT="${SMOKE_CP_PORT:-18081}"

init_smoke_env "check-a-${NODE_SPEC}" "Check A (${NODE_SPEC} -> candidate control plane)"

DATA_DIR="${SMOKE_WORK_DIR}/cp-data"
CP_NAME="af-smoke-cp-a-${NODE_SPEC//[^a-zA-Z0-9-]/-}"
CP_URL="http://127.0.0.1:${CP_PORT}"

start_cp "${CP_NAME}" "${CANDIDATE_IMAGE}" "${CP_PORT}" "${DATA_DIR}" "local"
add_evidence "Candidate control plane image \`${CANDIDATE_IMAGE}\` healthy at \`${CP_URL}\`"

run_one_sdk() {
  local spec="$1"
  local port="$2"
  local node_id="node-${spec//[^a-zA-Z0-9-]/-}"

  case "${spec}" in
    py-0.1.130)
      local venv_dir="${SMOKE_WORK_DIR}/venv-py-130"
      local ver
      ver="$(setup_python_venv "${venv_dir}" "0.1.130")"
      start_python_node "${venv_dir}" "${node_id}" "${CP_URL}" "${port}" 5
      add_evidence "Python SDK \`agentfield==${ver}\` started as \`${node_id}\`"
      ;;
    py-0.1.145)
      local venv_dir="${SMOKE_WORK_DIR}/venv-py-145"
      local ver
      ver="$(setup_python_venv "${venv_dir}" "0.1.145")"
      start_python_node "${venv_dir}" "${node_id}" "${CP_URL}" "${port}" 5
      add_evidence "Python SDK \`agentfield==${ver}\` started as \`${node_id}\`"
      ;;
    go-v0.1.130)
      local bin_path="${SMOKE_WORK_DIR}/go-node-130"
      build_go_node "go-v0.1.130" "${bin_path}"
      start_go_node "${bin_path}" "${node_id}" "${CP_URL}" "${port}" "5s"
      add_evidence "Go SDK \`sdk/go v0.1.130\` binary started as \`${node_id}\`"
      ;;
    *)
      log_err "Unknown Check A node spec: ${spec}"
      return 1
      ;;
  esac

  wait_for_node_registered "${CP_URL}" "${node_id}" "${port}"
  add_evidence "Node \`${node_id}\` registered and appeared active in \`GET /api/v1/nodes\`"

  local sync_resp async_resp
  sync_resp="$(run_sync_execute "${CP_URL}" "${node_id}.echo" '{"message":"check-a-sync","seq":11,"extra":{"lane":"sync"}}')"
  assert_echo_result "${sync_resp}" "check-a-sync" 11 "${node_id}"
  add_evidence "Sync execute \`POST /api/v1/execute/${node_id}.echo\` succeeded"

  async_resp="$(run_async_execute "${CP_URL}" "${node_id}.echo" '{"message":"check-a-async","seq":22,"extra":{"lane":"async"}}')"
  assert_echo_result "${async_resp}" "check-a-async" 22 "${node_id}"
  add_evidence "Async execute \`POST /api/v1/execute/async/${node_id}.echo\` + status poll succeeded"

  local hb_proof
  hb_proof="$(verify_heartbeat_survival "${CP_URL}" "${node_id}" 12)"
  add_evidence "Heartbeat survival verified for \`${node_id}\`: \`${hb_proof}\`"
}

if [[ "${NODE_SPEC}" == "all" ]]; then
  run_one_sdk "py-0.1.130" 18111
  run_one_sdk "py-0.1.145" 18112
  run_one_sdk "go-v0.1.130" 18113
else
  run_one_sdk "${NODE_SPEC}" 18111
fi

write_check_summary "PASS" "Old SDK node(s) (${NODE_SPEC}) registered, executed sync + async, and survived heartbeat sweep against candidate control plane."
log_info "Check A (${NODE_SPEC}) PASSED"
