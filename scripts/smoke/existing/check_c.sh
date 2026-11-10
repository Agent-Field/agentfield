#!/usr/bin/env bash
# Check C: relay gate (unwatched runs leave 0 rows in execution_relay & run_owners) + positive control.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/smoke/existing/lib.sh
source "${SCRIPT_DIR}/lib.sh"

STORAGE_MODE="${1:-local}"
CP_PORT="${SMOKE_CP_PORT:-18083}"
PG_PORT="${SMOKE_PG_PORT:-15433}"

init_smoke_env "check-c-${STORAGE_MODE}" "Check C (relay gate & positive control [${STORAGE_MODE}])"

DATA_DIR="${SMOKE_WORK_DIR}/cp-data"
CP_NAME="af-smoke-cp-c-${STORAGE_MODE}"
CP_URL="http://127.0.0.1:${CP_PORT}"
PG_DSN=""

if [[ "${STORAGE_MODE}" == "postgres" ]]; then
  PG_NAME="af-smoke-pg-c"
  start_postgres "${PG_NAME}" "${PG_PORT}"
  PG_DSN="postgres://agentfield:agentfield@127.0.0.1:${PG_PORT}/agentfield?sslmode=disable"
elif [[ "${STORAGE_MODE}" != "local" ]]; then
  log_err "Unsupported storage mode: ${STORAGE_MODE}"
  exit 1
fi

start_cp "${CP_NAME}" "${CANDIDATE_IMAGE}" "${CP_PORT}" "${DATA_DIR}" "${STORAGE_MODE}" "${PG_DSN}"
add_evidence "Candidate control plane \`${CANDIDATE_IMAGE}\` started on \`${STORAGE_MODE}\` store"

PY_VENV="${SMOKE_WORK_DIR}/venv-py-145"
PY_VER="$(setup_python_venv "${PY_VENV}" "0.1.145")"
PY_NODE="py-145-c"
start_python_node "${PY_VENV}" "${PY_NODE}" "${CP_URL}" 18131 5

GO_BIN="${SMOKE_WORK_DIR}/go-main-c"
build_go_node "go-main" "${GO_BIN}"
GO_NODE="go-main-c"
start_go_node "${GO_BIN}" "${GO_NODE}" "${CP_URL}" 18132 "5s"

wait_for_node_registered "${CP_URL}" "${PY_NODE}" 18131
wait_for_node_registered "${CP_URL}" "${GO_NODE}" 18132
add_evidence "Registered \`${PY_NODE}\` (\`agentfield==${PY_VER}\`) and \`${GO_NODE}\` (\`sdk/go\` main)"

query_table_count() {
  local table_name="$1"
  if [[ "${STORAGE_MODE}" == "local" ]]; then
    ensure_world_rw "${DATA_DIR}"
    local db_file="${DATA_DIR}/agentfield.db"
    local exists
    exists="$(host_sqlite "${db_file}" "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='${table_name}';")"
    if [[ "${exists}" == "0" ]]; then
      printf '0'
      return 0
    fi
    host_sqlite "${db_file}" "SELECT COUNT(*) FROM ${table_name};" | tr -d '[:space:]'
  else
    local exists
    exists="$(psql "${PG_DSN}" -t -A -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public' AND table_name='${table_name}';")"
    if [[ "${exists}" == "0" ]]; then
      printf '0'
      return 0
    fi
    psql "${PG_DSN}" -t -A -c "SELECT COUNT(*) FROM ${table_name};" | tr -d '[:space:]'
  fi
}

# ── Part 1: Ordinary unwatched runs (no X-AgentField-Run-Owner) ─────────────
PY_ORD_RESP="$(run_sync_execute "${CP_URL}" "${PY_NODE}.echo" '{"message":"unwatched-py","seq":1}')"
assert_echo_result "${PY_ORD_RESP}" "unwatched-py" 1 "${PY_NODE}"

GO_ORD_RESP="$(run_sync_execute "${CP_URL}" "${GO_NODE}.echo" '{"message":"unwatched-go","seq":2}')"
assert_echo_result "${GO_ORD_RESP}" "unwatched-go" 2 "${GO_NODE}"

python3 - "${PY_ORD_RESP}" "${GO_ORD_RESP}" <<'PY'
import json
import sys

py_res = (json.loads(sys.argv[1]).get("result") or {})
go_res = (json.loads(sys.argv[2]).get("result") or {})

py_headers = py_res.get("headers")
if not isinstance(py_headers, dict) or len(py_headers) == 0:
    raise SystemExit(f"expected Python node to expose inbound HTTP headers, got: {py_headers!r}")

lower_headers = {k.lower(): v for k, v in py_headers.items()}
if "x-agentfield-run-watched" in lower_headers:
    raise SystemExit(f"unwatched Python execution received X-AgentField-Run-Watched: {lower_headers}")
if "x-agentfield-run-owner" in lower_headers:
    raise SystemExit(f"Python execution received X-AgentField-Run-Owner: {lower_headers}")

if go_res.get("run_watched") is not False:
    raise SystemExit(f"expected Go main node RunWatched == False on unwatched run, got: {go_res}")
go_ctx = go_res.get("execution_context") or {}
if go_ctx.get("run_watched") is not False:
    raise SystemExit(f"expected Go execution_context.run_watched == False, got: {go_ctx}")
PY

add_evidence "Unwatched \`${GO_NODE}\` saw \`RunWatched == false\`; unwatched \`${PY_NODE}\` saw no \`X-AgentField-Run-Watched\` (or \`X-AgentField-Run-Owner\`) in HTTP headers"

RELAY_ROWS_UNWATCHED="$(query_table_count "execution_relay")"
OWNERS_ROWS_UNWATCHED="$(query_table_count "run_owners")"

if [[ "${RELAY_ROWS_UNWATCHED}" != "0" ]]; then
  log_err "Expected 0 rows in execution_relay after unwatched runs, got ${RELAY_ROWS_UNWATCHED}"
  exit 1
fi
if [[ "${OWNERS_ROWS_UNWATCHED}" != "0" ]]; then
  log_err "Expected 0 rows in run_owners after unwatched runs, got ${OWNERS_ROWS_UNWATCHED}"
  exit 1
fi
add_evidence "Direct \`${STORAGE_MODE}\` query after unwatched executions: \`execution_relay = ${RELAY_ROWS_UNWATCHED}\`, \`run_owners = ${OWNERS_ROWS_UNWATCHED}\`"

# ── Part 1b: Concurrent owned executions ────────────────────────────────────
# Before the relay tables were created with the schema, two requests reaching
# them at once raced CREATE TABLE IF NOT EXISTS on Postgres, lost the run's
# owner, and the cached error kept every later owned run unwatched until a
# restart. So the first owned executions this control plane ever sees arrive
# together: each must be watched and each run must get exactly one owner.
BURST=8
BURST_DIR="${SMOKE_WORK_DIR}/owned-burst"
mkdir -p "${BURST_DIR}"
# Wait on the burst's own PIDs only: a bare `wait` would also wait on the test
# nodes this shell started in the background, which never exit.
BURST_PIDS=()
for i in $(seq 1 "${BURST}"); do
  (
    run_sync_execute \
      "${CP_URL}" \
      "${GO_NODE}.echo" \
      '{"message":"owned-burst","seq":'"$((100 + i))"'}' \
      "X-AgentField-Run-Owner: smoke-burst-${i}" > "${BURST_DIR}/${i}.json" 2> "${BURST_DIR}/${i}.err"
  ) &
  BURST_PIDS+=("$!")
done
wait "${BURST_PIDS[@]}" || true

python3 - "${BURST_DIR}" "${BURST}" <<'PY'
import json
import pathlib
import sys

d, n = pathlib.Path(sys.argv[1]), int(sys.argv[2])
runs = set()
for i in range(1, n + 1):
    raw = (d / f"{i}.json").read_text().strip()
    if not raw:
        raise SystemExit(f"owned burst call {i} returned nothing: {(d / f'{i}.err').read_text()[:400]}")
    resp = json.loads(raw)
    res = resp.get("result") or {}
    if res.get("run_watched") is not True:
        raise SystemExit(f"owned burst call {i} was not watched: {res}")
    runs.add(resp.get("run_id"))
if len(runs) != n or None in runs or "" in runs:
    raise SystemExit(f"expected {n} distinct run ids from the burst, got {runs}")
(d / "runs.txt").write_text("\n".join(sorted(runs)))
PY

OWNERS_AFTER_BURST="$(query_table_count "run_owners")"
if [[ "${OWNERS_AFTER_BURST}" != "${BURST}" ]]; then
  log_err "Expected exactly ${BURST} run_owners rows after ${BURST} concurrent owned executions, got ${OWNERS_AFTER_BURST}"
  exit 1
fi
add_evidence "${BURST} concurrent owned executions on a fresh \`${STORAGE_MODE}\` control plane: every one watched (\`RunWatched == true\`), ${OWNERS_AFTER_BURST} owner rows, one per run"

# ── Part 2: Positive control (X-AgentField-Run-Owner + POST /executions/:id/relay) ──
OWNED_RESP="$(run_sync_execute \
  "${CP_URL}" \
  "${GO_NODE}.echo" \
  '{"message":"watched-positive-control","seq":3,"call_target":"'"${PY_NODE}"'.echo"}' \
  "X-AgentField-Run-Owner: smoke-factory")"

OWNED_IDS="$(python3 - "${OWNED_RESP}" "${GO_NODE}" "${PY_NODE}" <<'PY'
import json
import sys

resp = json.loads(sys.argv[1])
go_node, py_node = sys.argv[2], sys.argv[3]
exec_id = resp.get("execution_id") or ""
run_id = resp.get("run_id") or ""
res = resp.get("result") or {}

if res.get("node_id") != go_node:
    raise SystemExit(f"unexpected node_id on owned execute: {res}")
if res.get("run_watched") is not True:
    raise SystemExit(f"expected Go main node RunWatched == True on owned run, got: {res}")

child = res.get("child_result") or {}
if child.get("node_id") != py_node:
    raise SystemExit(f"expected nested child result from {py_node}, got: {child}")
child_headers = {k.lower(): v for k, v in (child.get("headers") or {}).items()}
if child_headers.get("x-agentfield-run-watched") != "1":
    raise SystemExit(f"expected X-AgentField-Run-Watched: 1 on owned child HTTP request, got: {child_headers}")
if "x-agentfield-run-owner" in child_headers:
    raise SystemExit(f"X-AgentField-Run-Owner must never be forwarded to nodes: {child_headers}")

print(f"{exec_id} {run_id}")
PY
)"
read -r OWNED_EXEC_ID OWNED_RUN_ID <<<"${OWNED_IDS}"

add_evidence "Positive control owned run (\`run_id=${OWNED_RUN_ID}\`, \`execution_id=${OWNED_EXEC_ID}\`) set \`RunWatched == true\` on \`${GO_NODE}\` and \`X-AgentField-Run-Watched: 1\` on nested \`${PY_NODE}\`"

RELAY_PUT_RESP="$(curl -fsS --max-time 15 \
  -X POST "${CP_URL}/api/v1/executions/${OWNED_EXEC_ID}/relay" \
  -H "Content-Type: application/json" \
  -d '{"entries":[{"seq":1,"kind":"started","data":{"harness":"codeaf","steerable":true}}]}')"

RELAY_GET_RESP="$(curl -fsS --max-time 15 \
  -H "X-AgentField-Run-Owner: smoke-factory" \
  "${CP_URL}/api/v1/executions/${OWNED_EXEC_ID}/relay?after=0")"

python3 - "${RELAY_PUT_RESP}" "${RELAY_GET_RESP}" <<'PY'
import json
import sys

put_d = json.loads(sys.argv[1])
get_d = json.loads(sys.argv[2])
if put_d.get("stored") != 1 or put_d.get("through") != 1:
    raise SystemExit(f"unexpected relay put response: {put_d}")
entries = get_d.get("entries") or []
if len(entries) != 1 or entries[0].get("seq") != 1 or entries[0].get("kind") != "started":
    raise SystemExit(f"unexpected relay read response: {get_d}")
PY

RELAY_ROWS_OWNED="$(query_table_count "execution_relay")"
OWNERS_ROWS_OWNED="$(query_table_count "run_owners")"

if [[ "${OWNERS_ROWS_OWNED}" -lt 1 ]]; then
  log_err "Positive control expected >=1 row in run_owners, got ${OWNERS_ROWS_OWNED}"
  exit 1
fi
if [[ "${RELAY_ROWS_OWNED}" -lt 1 ]]; then
  log_err "Positive control expected >=1 row in execution_relay, got ${RELAY_ROWS_OWNED}"
  exit 1
fi

if [[ "${STORAGE_MODE}" == "local" ]]; then
  ensure_world_rw "${DATA_DIR}"
  OWNER_DB_ROW="$(host_sqlite "${DATA_DIR}/agentfield.db" "SELECT run_id || ':' || owner_node_id FROM run_owners WHERE run_id='${OWNED_RUN_ID}';")"
else
  OWNER_DB_ROW="$(psql "${PG_DSN}" -t -A -c "SELECT run_id || ':' || owner_node_id FROM run_owners WHERE run_id='${OWNED_RUN_ID}';" | tr -d '[:space:]')"
fi
if [[ "${OWNER_DB_ROW}" != "${OWNED_RUN_ID}:smoke-factory" ]]; then
  log_err "Unexpected run_owners row: ${OWNER_DB_ROW}"
  exit 1
fi

add_evidence "Positive control direct \`${STORAGE_MODE}\` query: \`run_owners = ${OWNERS_ROWS_OWNED}\` (\`${OWNER_DB_ROW}\`), \`execution_relay = ${RELAY_ROWS_OWNED}\`"

write_check_summary "PASS" "Relay gate verified on ${STORAGE_MODE}: 0 rows in execution_relay & run_owners for unwatched runs; positive control set X-AgentField-Run-Watched: 1 and populated both tables."
log_info "Check C (${STORAGE_MODE}) PASSED"
