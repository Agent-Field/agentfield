#!/usr/bin/env bash
# Check D: migrations and rollback (v0.1.145 -> candidate -> v0.1.145) for local (SQLite) and Postgres (postgres:16).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/smoke/existing/lib.sh
source "${SCRIPT_DIR}/lib.sh"

STORAGE_MODE="${1:-local}"
CP_PORT="${SMOKE_CP_PORT:-18084}"
PG_PORT="${SMOKE_PG_PORT:-15434}"

init_smoke_env "check-d-${STORAGE_MODE}" "Check D (migrations & rollback [${STORAGE_MODE}])"

DATA_DIR="${SMOKE_WORK_DIR}/cp-data"
CP_URL="http://127.0.0.1:${CP_PORT}"
PG_DSN=""

if [[ "${STORAGE_MODE}" == "postgres" ]]; then
  PG_NAME="af-smoke-pg-d"
  start_postgres "${PG_NAME}" "${PG_PORT}"
  PG_DSN="postgres://agentfield:agentfield@127.0.0.1:${PG_PORT}/agentfield?sslmode=disable"
elif [[ "${STORAGE_MODE}" != "local" ]]; then
  log_err "Unsupported storage mode: ${STORAGE_MODE}"
  exit 1
fi

docker pull "${STABLE_IMAGE}" >/dev/null

# ── Step 1: Start stable v0.1.145 on a fresh store ──────────────────────────
CP_STEP1="af-smoke-cp-d-${STORAGE_MODE}-step1-v145"
start_cp "${CP_STEP1}" "${STABLE_IMAGE}" "${CP_PORT}" "${DATA_DIR}" "${STORAGE_MODE}" "${PG_DSN}"
add_evidence "Step 1: started \`${STABLE_IMAGE}\` on fresh \`${STORAGE_MODE}\` store"

PY_VENV="${SMOKE_WORK_DIR}/venv-py-145"
PY_VER="$(setup_python_venv "${PY_VENV}" "0.1.145")"
NODE_ID="py-145-d"
NODE_PORT=18141
start_python_node "${PY_VENV}" "${NODE_ID}" "${CP_URL}" "${NODE_PORT}" 2
wait_for_node_registered "${CP_URL}" "${NODE_ID}" "${NODE_PORT}"

EXEC1_RESP="$(run_sync_execute "${CP_URL}" "${NODE_ID}.echo" '{"message":"step1-exec-1","seq":101}')"
assert_echo_result "${EXEC1_RESP}" "step1-exec-1" 101 "${NODE_ID}"
EXEC1_ID="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["execution_id"])' "${EXEC1_RESP}")"

EXEC2_RESP="$(run_async_execute "${CP_URL}" "${NODE_ID}.echo" '{"message":"step1-exec-2","seq":102}')"
assert_echo_result "${EXEC2_RESP}" "step1-exec-2" 102 "${NODE_ID}"
EXEC2_ID="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["execution_id"])' "${EXEC2_RESP}")"

MEM_SET_BODY='{"key":"smoke.migration.state","data":{"origin":"v0.1.145","storage_mode":"'"${STORAGE_MODE}"'","executions":2},"scope":"global"}'
MEM_SET_RESP="$(curl -fsS --max-time 10 -X POST "${CP_URL}/api/v1/memory/set" -H "Content-Type: application/json" -d "${MEM_SET_BODY}")"
MEM_STEP1="$(curl -fsS --max-time 10 -X POST "${CP_URL}/api/v1/memory/get" -H "Content-Type: application/json" -d '{"key":"smoke.migration.state","scope":"global"}')"
NODES_STEP1="$(curl -fsS --max-time 10 "${CP_URL}/api/v1/nodes?show_all=true")"
E1_STEP1="$(curl -fsS --max-time 10 "${CP_URL}/api/v1/executions/${EXEC1_ID}")"
E2_STEP1="$(curl -fsS --max-time 10 "${CP_URL}/api/v1/executions/${EXEC2_ID}")"

python3 - "${MEM_SET_RESP}" "${MEM_STEP1}" "${NODES_STEP1}" "${NODE_ID}" "${STORAGE_MODE}" <<'PY'
import json
import sys

_, mem_get, nodes_raw, node_id, mode = sys.argv[1:6]
mem = json.loads(mem_get)
want_data = {"origin": "v0.1.145", "storage_mode": mode, "executions": 2}
if mem.get("data") != want_data:
    raise SystemExit(f"Step 1 memory readback mismatch: {mem}")
nodes = json.loads(nodes_raw).get("nodes") or []
if not any(n.get("id") == node_id for n in nodes):
    raise SystemExit(f"Step 1 node {node_id} missing from nodes list: {nodes}")
PY

add_evidence "Step 1 (\`v0.1.145\`): registered \`${NODE_ID}\` (\`agentfield==${PY_VER}\`), ran executions \`${EXEC1_ID}\` & \`${EXEC2_ID}\`, wrote & verified global memory key \`smoke.migration.state\`"

stop_container "${CP_STEP1}"

# ── Step 2: Start candidate on the same store ───────────────────────────────
CP_STEP2="af-smoke-cp-d-${STORAGE_MODE}-step2-candidate"
start_cp "${CP_STEP2}" "${CANDIDATE_IMAGE}" "${CP_PORT}" "${DATA_DIR}" "${STORAGE_MODE}" "${PG_DSN}"

NODES_STEP2="$(curl -fsS --max-time 10 "${CP_URL}/api/v1/nodes?show_all=true")"
E1_STEP2="$(curl -fsS --max-time 10 "${CP_URL}/api/v1/executions/${EXEC1_ID}")"
E2_STEP2="$(curl -fsS --max-time 10 "${CP_URL}/api/v1/executions/${EXEC2_ID}")"
MEM_STEP2="$(curl -fsS --max-time 10 -X POST "${CP_URL}/api/v1/memory/get" -H "Content-Type: application/json" -d '{"key":"smoke.migration.state","scope":"global"}')"

python3 - "${NODE_ID}" "${NODES_STEP2}" "${E1_STEP1}" "${E1_STEP2}" "${E2_STEP1}" "${E2_STEP2}" "${MEM_STEP1}" "${MEM_STEP2}" <<'PY'
import json
import sys

node_id = sys.argv[1]
nodes2 = json.loads(sys.argv[2]).get("nodes") or []
e1_1, e1_2 = json.loads(sys.argv[3]), json.loads(sys.argv[4])
e2_1, e2_2 = json.loads(sys.argv[5]), json.loads(sys.argv[6])
m1, m2 = json.loads(sys.argv[7]), json.loads(sys.argv[8])

if not any(n.get("id") == node_id for n in nodes2):
    raise SystemExit(f"Step 2 candidate missing stored node {node_id}: {nodes2}")
for label, a, b in (("exec1", e1_1, e1_2), ("exec2", e2_1, e2_2)):
    if b.get("status") != "succeeded" or b.get("result") != a.get("result"):
        raise SystemExit(f"Step 2 candidate {label} mismatch: before={a}, after={b}")
if m2.get("data") != m1.get("data"):
    raise SystemExit(f"Step 2 candidate memory mismatch: before={m1}, after={m2}")
PY

wait_for_node_registered "${CP_URL}" "${NODE_ID}" "${NODE_PORT}"
EXEC3_RESP="$(run_sync_execute "${CP_URL}" "${NODE_ID}.echo" '{"message":"step2-candidate-exec","seq":203}')"
assert_echo_result "${EXEC3_RESP}" "step2-candidate-exec" 203 "${NODE_ID}"
EXEC3_ID="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["execution_id"])' "${EXEC3_RESP}")"

capture_container_log "${CP_STEP2}"
CANDIDATE_LOG="${SMOKE_ARTIFACT_DIR}/logs/${CP_STEP2}.log"
python3 - "${CANDIDATE_LOG}" <<'PY'
import re
import sys

log_path = sys.argv[1]
with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
    lines = fh.readlines()

bad_patterns = re.compile(
    r'("level"\s*:\s*"error"|\bERR\b|migration.*fail|failed to (auto-migrate|run .*migrations|create .*schema)|create execution_relay:|create run_owners:|SQLSTATE|SQL logic error)',
    re.IGNORECASE,
)
hits = [ln.rstrip() for ln in lines if bad_patterns.search(ln)]
if hits:
    raise SystemExit("Step 2 candidate log contained migration/runtime errors:\n" + "\n".join(hits[:20]))
PY

add_evidence "Step 2 (\`candidate\`): migrations applied with zero log errors; read back \`${NODE_ID}\`, \`${EXEC1_ID}\`, \`${EXEC2_ID}\` & \`smoke.migration.state\`; ran new execution \`${EXEC3_ID}\`"

stop_container "${CP_STEP2}"

# ── Step 3: Rollback to stable v0.1.145 on the now-migrated store ───────────
CP_STEP3="af-smoke-cp-d-${STORAGE_MODE}-step3-rollback-v145"
start_cp "${CP_STEP3}" "${STABLE_IMAGE}" "${CP_PORT}" "${DATA_DIR}" "${STORAGE_MODE}" "${PG_DSN}"

HEALTH_STEP3="$(curl -fsS --max-time 5 "${CP_URL}/health")"
NODES_STEP3="$(curl -fsS --max-time 10 "${CP_URL}/api/v1/nodes?show_all=true")"
E1_STEP3="$(curl -fsS --max-time 10 "${CP_URL}/api/v1/executions/${EXEC1_ID}")"
E2_STEP3="$(curl -fsS --max-time 10 "${CP_URL}/api/v1/executions/${EXEC2_ID}")"
E3_STEP3="$(curl -fsS --max-time 10 "${CP_URL}/api/v1/executions/${EXEC3_ID}")"
MEM_STEP3="$(curl -fsS --max-time 10 -X POST "${CP_URL}/api/v1/memory/get" -H "Content-Type: application/json" -d '{"key":"smoke.migration.state","scope":"global"}')"

python3 - "${NODE_ID}" "${HEALTH_STEP3}" "${NODES_STEP3}" "${E1_STEP1}" "${E1_STEP3}" "${E2_STEP1}" "${E2_STEP3}" "${E3_STEP3}" "${MEM_STEP1}" "${MEM_STEP3}" <<'PY'
import json
import sys

node_id = sys.argv[1]
health = json.loads(sys.argv[2])
nodes3 = json.loads(sys.argv[3]).get("nodes") or []
e1_1, e1_3 = json.loads(sys.argv[4]), json.loads(sys.argv[5])
e2_1, e2_3 = json.loads(sys.argv[6]), json.loads(sys.argv[7])
e3_3 = json.loads(sys.argv[8])
m1, m3 = json.loads(sys.argv[9]), json.loads(sys.argv[10])

if health.get("status") not in ("healthy", "ok"):
    raise SystemExit(f"Step 3 rollback /health not healthy: {health}")
if not any(n.get("id") == node_id for n in nodes3):
    raise SystemExit(f"Step 3 rollback missing stored node {node_id}: {nodes3}")
for label, a, b in (("exec1", e1_1, e1_3), ("exec2", e2_1, e2_3)):
    if b.get("status") != "succeeded" or b.get("result") != a.get("result"):
        raise SystemExit(f"Step 3 rollback {label} mismatch: before={a}, after={b}")
if e3_3.get("status") != "succeeded":
    raise SystemExit(f"Step 3 rollback failed reading Step 2 execution: {e3_3}")
if m3.get("data") != m1.get("data"):
    raise SystemExit(f"Step 3 rollback memory mismatch: before={m1}, after={m3}")
PY

wait_for_node_registered "${CP_URL}" "${NODE_ID}" "${NODE_PORT}"
EXEC4_RESP="$(run_sync_execute "${CP_URL}" "${NODE_ID}.echo" '{"message":"step3-rollback-exec","seq":304}')"
assert_echo_result "${EXEC4_RESP}" "step3-rollback-exec" 304 "${NODE_ID}"
EXEC4_ID="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["execution_id"])' "${EXEC4_RESP}")"

capture_container_log "${CP_STEP3}"
ROLLBACK_LOG="${SMOKE_ARTIFACT_DIR}/logs/${CP_STEP3}.log"
WARNINGS_FILE="${SMOKE_ARTIFACT_DIR}/rollback-schema-warnings.txt"

WARNING_SUMMARY="$(python3 - "${ROLLBACK_LOG}" "${WARNINGS_FILE}" <<'PY'
import re
import sys

log_path, out_path = sys.argv[1:3]
with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
    lines = fh.readlines()

pat = re.compile(
    r'(unknown\s+(table|column)|no such (table|column)|relation .* does not exist|column .* does not exist|("level"\s*:\s*"warn"|\bWRN\b|\bwarn\b).*(table|column|schema|migration|execution_relay|run_owners|approval_question))',
    re.IGNORECASE,
)
matches = [ln.rstrip() for ln in lines if pat.search(ln)]
with open(out_path, "w", encoding="utf-8") as out:
    if matches:
        out.write("\n".join(matches) + "\n")
    else:
        out.write("none\n")

if matches:
    print(f"{len(matches)} warning(s): " + "; ".join(matches[:5]))
else:
    print("none")
PY
)"

add_evidence "Step 3 (\`v0.1.145\` rollback): \`/health\` healthy; read back \`${NODE_ID}\`, \`${EXEC1_ID}\`, \`${EXEC2_ID}\`, \`${EXEC3_ID}\` & \`smoke.migration.state\`; ran new execution \`${EXEC4_ID}\`"
add_evidence "Step 3 rollback log warnings about unknown tables or columns: \`${WARNING_SUMMARY}\`"

write_check_summary "PASS" "Upgrade & rollback verified on ${STORAGE_MODE} (v0.1.145 -> candidate -> v0.1.145); unknown-table/column warnings on rollback: ${WARNING_SUMMARY}."
log_info "Check D (${STORAGE_MODE}) PASSED"
