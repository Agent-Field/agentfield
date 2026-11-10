#!/usr/bin/env bash
# Shared helpers for existing-deployment smoke checks (#1174).

set -euo pipefail

SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SMOKE_DIR}/../../.." && pwd)"

CANDIDATE_IMAGE="${CANDIDATE_IMAGE:-agentfield-control-plane:candidate}"
STABLE_IMAGE="${STABLE_IMAGE:-agentfield/control-plane:v0.1.145}"
OLD_CLOUD_IMAGE="${OLD_CLOUD_IMAGE:-agentfield/control-plane:v0.1.57}"
POSTGRES_IMAGE="${POSTGRES_IMAGE:-postgres:16}"

SMOKE_CHECK_ID="${SMOKE_CHECK_ID:-check}"
SMOKE_CHECK_TITLE="${SMOKE_CHECK_TITLE:-Smoke Check}"
SMOKE_WORK_DIR="${SMOKE_WORK_DIR:-}"
SMOKE_ARTIFACT_DIR="${SMOKE_ARTIFACT_DIR:-}"
SMOKE_SUMMARY_WRITTEN=0

declare -a SMOKE_CONTAINERS=()
declare -a SMOKE_STOP_FILES=()
declare -a SMOKE_NODE_PIDS=()
declare -a SMOKE_LOG_FILES=()
declare -a SMOKE_EVIDENCE_LINES=()

log_info() {
  printf '[%s] INFO: %s\n' "$(date -u +%H:%M:%S)" "$*"
}

log_err() {
  printf '[%s] ERROR: %s\n' "$(date -u +%H:%M:%S)" "$*" >&2
}

add_evidence() {
  SMOKE_EVIDENCE_LINES+=("$*")
  log_info "EVIDENCE: $*"
}

pid_alive() {
  local pid="$1"
  [[ -n "${pid}" && -d "/proc/${pid}" ]]
}

init_smoke_env() {
  local check_id="$1"
  local check_title="$2"
  SMOKE_CHECK_ID="${check_id}"
  SMOKE_CHECK_TITLE="${check_title}"

  if [[ -z "${SMOKE_WORK_DIR}" ]]; then
    SMOKE_WORK_DIR="${REPO_ROOT}/.smoke-work/${check_id}"
  fi
  if [[ -z "${SMOKE_ARTIFACT_DIR}" ]]; then
    SMOKE_ARTIFACT_DIR="${REPO_ROOT}/smoke-artifacts/${check_id}"
  fi
  mkdir -p "${SMOKE_WORK_DIR}" "${SMOKE_ARTIFACT_DIR}/logs"
  trap 'smoke_on_exit $?' EXIT
  log_info "Starting ${SMOKE_CHECK_TITLE} (${SMOKE_CHECK_ID})"
  log_info "Work dir: ${SMOKE_WORK_DIR} | Artifact dir: ${SMOKE_ARTIFACT_DIR}"
}

capture_container_log() {
  local cname="$1"
  local out="${SMOKE_ARTIFACT_DIR}/logs/${cname}.log"
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${cname}"; then
    docker logs "${cname}" >"${out}" 2>&1 || true
    local already=0
    local existing
    for existing in "${SMOKE_LOG_FILES[@]:-}"; do
      if [[ "${existing}" == "${out}" ]]; then
        already=1
        break
      fi
    done
    if [[ "${already}" -eq 0 ]]; then
      SMOKE_LOG_FILES+=("${out}")
    fi
  fi
}

dump_all_logs() {
  log_err "=== DUMPING CONTROL-PLANE AND NODE LOGS FOR ${SMOKE_CHECK_ID} ==="
  local cname
  for cname in "${SMOKE_CONTAINERS[@]:-}"; do
    [[ -n "${cname}" ]] || continue
    capture_container_log "${cname}"
  done
  local lf
  for lf in "${SMOKE_LOG_FILES[@]:-}"; do
    [[ -n "${lf}" && -f "${lf}" ]] || continue
    printf '\n===== LOG: %s =====\n' "${lf}" >&2
    tail -n 250 "${lf}" >&2 || true
    printf '===== END LOG: %s =====\n' "${lf}" >&2
  done
}

write_check_summary() {
  local status="$1"
  local headline="$2"
  local summary_md="${SMOKE_ARTIFACT_DIR}/summary.md"
  local evidence_json="${SMOKE_ARTIFACT_DIR}/evidence.json"

  {
    printf '### %s — **%s**\n\n' "${SMOKE_CHECK_TITLE}" "${status}"
    printf '%s\n\n' "${headline}"
    if [[ "${#SMOKE_EVIDENCE_LINES[@]}" -gt 0 ]]; then
      printf '#### Evidence\n'
      local line
      for line in "${SMOKE_EVIDENCE_LINES[@]}"; do
        printf -- '- %s\n' "${line}"
      done
      printf '\n'
    fi
  } >"${summary_md}"

  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    cat "${summary_md}" >>"${GITHUB_STEP_SUMMARY}"
  fi

  python3 - "${SMOKE_CHECK_ID}" "${SMOKE_CHECK_TITLE}" "${status}" "${headline}" "${evidence_json}" ${SMOKE_EVIDENCE_LINES[@]+"${SMOKE_EVIDENCE_LINES[@]}"} <<'PY'
import datetime
import json
import sys

check_id, title, status, headline, out_path = sys.argv[1:6]
evidence = sys.argv[6:]
doc = {
    "check_id": check_id,
    "title": title,
    "status": status,
    "headline": headline,
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "evidence": evidence,
}
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(doc, fh, indent=2)
    fh.write("\n")
PY

  SMOKE_SUMMARY_WRITTEN=1
}

stop_all_nodes() {
  local sf
  for sf in "${SMOKE_STOP_FILES[@]:-}"; do
    [[ -n "${sf}" ]] || continue
    touch "${sf}" || true
  done

  local pid
  for pid in "${SMOKE_NODE_PIDS[@]:-}"; do
    [[ -n "${pid}" ]] || continue
    local waited=0
    while pid_alive "${pid}" && [[ "${waited}" -lt 50 ]]; do
      sleep 0.2
      waited=$((waited + 1))
    done
    wait "${pid}" 2>/dev/null || true
  done
}

stop_container() {
  local cname="$1"
  if docker ps -a --format '{{.Names}}' | grep -Fxq "${cname}"; then
    capture_container_log "${cname}"
    docker stop -t 5 "${cname}" >/dev/null 2>&1 || true
    docker rm -f "${cname}" >/dev/null 2>&1 || true
  fi
}

smoke_on_exit() {
  local rc="$1"
  set +e
  local cname
  for cname in "${SMOKE_CONTAINERS[@]:-}"; do
    [[ -n "${cname}" ]] || continue
    capture_container_log "${cname}"
  done

  if [[ "${rc}" -ne 0 ]]; then
    if [[ "${SMOKE_SUMMARY_WRITTEN}" -eq 0 ]]; then
      write_check_summary "FAIL" "Check exited with status ${rc}; see logs in workflow output and uploaded artifact."
    fi
    dump_all_logs
  fi

  stop_all_nodes
  for cname in "${SMOKE_CONTAINERS[@]:-}"; do
    [[ -n "${cname}" ]] || continue
    docker stop -t 3 "${cname}" >/dev/null 2>&1 || true
    docker rm -f "${cname}" >/dev/null 2>&1 || true
  done
}

ensure_world_rw() {
  local dir="$1"
  if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    sudo -n chmod -R 0777 "${dir}"
  else
    chmod -R 0777 "${dir}" 2>/dev/null || true
  fi
}

# The control-plane images run as the distroless nonroot user (UID 65532) and
# tighten their own key directory to 0700 on start. That chmod fails with
# "operation not permitted" on a directory the runner created, even a 0777 one,
# so the data directory is handed to 65532 empty and the control plane creates
# its own subdirectories. Host-side reads then go through host_sqlite.
prepare_data_dir() {
  local dir="$1"
  mkdir -p "${dir}"
  if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    sudo -n chown -R 65532:65532 "${dir}"
  else
    chmod 0777 "${dir}"
  fi
}

# The store the control plane writes belongs to UID 65532, and opening a WAL
# database needs write access beside it, so the runner reads it with sudo.
host_sqlite() {
  if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    sudo -n sqlite3 "$@"
  else
    sqlite3 "$@"
  fi
}

start_postgres() {
  local cname="$1"
  local pg_port="$2"
  local pg_data="${SMOKE_WORK_DIR}/${cname}-pgdata"
  mkdir -p "${pg_data}"

  docker rm -f "${cname}" >/dev/null 2>&1 || true
  SMOKE_CONTAINERS+=("${cname}")

  log_info "Starting Postgres (${POSTGRES_IMAGE}) container ${cname} on 127.0.0.1:${pg_port}"
  docker run -d \
    --name "${cname}" \
    -p "127.0.0.1:${pg_port}:5432" \
    -e POSTGRES_USER=agentfield \
    -e POSTGRES_PASSWORD=agentfield \
    -e POSTGRES_DB=agentfield \
    -v "${pg_data}:/var/lib/postgresql/data" \
    "${POSTGRES_IMAGE}" >/dev/null

  local attempt=0
  while [[ "${attempt}" -lt 60 ]]; do
    if docker exec "${cname}" pg_isready -U agentfield -d agentfield >/dev/null 2>&1; then
      log_info "Postgres ${cname} is ready on 127.0.0.1:${pg_port}"
      return 0
    fi
    sleep 1
    attempt=$((attempt + 1))
  done

  log_err "Postgres ${cname} failed readiness check"
  return 1
}

start_cp() {
  local cname="$1"
  local image="$2"
  local port="$3"
  local data_dir="$4"
  local storage_mode="$5"
  local pg_dsn="${6:-}"

  prepare_data_dir "${data_dir}"
  docker rm -f "${cname}" >/dev/null 2>&1 || true
  SMOKE_CONTAINERS+=("${cname}")

  local -a env_args=(
    -e "AGENTFIELD_HOME=/data/agentfield"
    -e "AGENTFIELD_CONFIG_FILE=/etc/agentfield/smoke-cp.yaml"
    -e "AGENTFIELD_PORT=${port}"
    -e "AGENTFIELD_STORAGE_MODE=${storage_mode}"
    -e "AGENTFIELD_TELEMETRY_ENABLED=false"
  )
  local -a cmd_args=(
    "--open=false"
    "--port" "${port}"
    "--config" "/etc/agentfield/smoke-cp.yaml"
    "--storage-mode" "${storage_mode}"
  )

  if [[ "${storage_mode}" == "postgres" ]]; then
    env_args+=(
      -e "AGENTFIELD_POSTGRES_URL=${pg_dsn}"
      -e "AGENTFIELD_STORAGE_POSTGRES_URL=${pg_dsn}"
      -e "AGENTFIELD_DATABASE_URL=${pg_dsn}"
    )
    cmd_args+=("--postgres-url" "${pg_dsn}")
  fi

  log_info "Starting control-plane ${cname} (${image}) on 127.0.0.1:${port} [storage=${storage_mode}]"
  docker run -d \
    --name "${cname}" \
    --network=host \
    "${env_args[@]}" \
    -v "${data_dir}:/data" \
    -v "${SMOKE_DIR}/smoke-cp.yaml:/etc/agentfield/smoke-cp.yaml:ro" \
    "${image}" \
    "${cmd_args[@]}" >/dev/null

  wait_for_cp_health "${cname}" "http://127.0.0.1:${port}"
  chmod -R a+rX "${data_dir}" 2>/dev/null || true
}

wait_for_cp_health() {
  local cname="$1"
  local cp_url="$2"
  local attempt=0
  local body=""

  while [[ "${attempt}" -lt 90 ]]; do
    if ! docker ps --format '{{.Names}}' | grep -Fxq "${cname}"; then
      log_err "Control-plane container ${cname} exited unexpectedly"
      capture_container_log "${cname}"
      return 1
    fi
    if body="$(curl -fsS --max-time 3 "${cp_url}/health" 2>/dev/null)" || \
       body="$(curl -fsS --max-time 3 "${cp_url}/api/v1/health" 2>/dev/null)"; then
      if python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sys.exit(0 if d.get("status") in ("healthy","ok") else 1)' "${body}" 2>/dev/null; then
        log_info "Control-plane ${cname} healthy at ${cp_url}"
        return 0
      fi
    fi
    sleep 1
    attempt=$((attempt + 1))
  done

  log_err "Timed out waiting for control-plane ${cname} health at ${cp_url} (last=${body})"
  return 1
}

setup_python_venv() {
  local venv_dir="$1"
  local spec="$2"

  log_info "Creating Python venv at ${venv_dir} (${spec})" >&2
  python3 -m venv "${venv_dir}"
  "${venv_dir}/bin/python" -m pip install --quiet --upgrade pip
  if [[ "${spec}" == "source" ]]; then
    "${venv_dir}/bin/pip" install --quiet "${REPO_ROOT}/sdk/python"
  else
    "${venv_dir}/bin/pip" install --quiet "agentfield==${spec}"
  fi
  local installed_ver
  installed_ver="$("${venv_dir}/bin/python" -c 'import agentfield; print(getattr(agentfield, "__version__", "unknown"))')"
  log_info "Installed Python agentfield=${installed_ver} in ${venv_dir}" >&2
  printf '%s' "${installed_ver}"
}

build_go_node() {
  local node_subdir="$1"
  local out_bin="$2"
  local src_dir="${SMOKE_DIR}/nodes/${node_subdir}"

  log_info "Building Go test node ${node_subdir} -> ${out_bin}"
  (
    cd "${src_dir}"
    GOFLAGS=-buildvcs=false GOMAXPROCS=4 nice go build -o "${out_bin}" .
  )
}

start_python_node() {
  local venv_dir="$1"
  local node_id="$2"
  local cp_url="$3"
  local port="$4"
  local hb_seconds="${5:-5}"

  local stop_file="${SMOKE_WORK_DIR}/stop-${node_id}"
  local log_file="${SMOKE_ARTIFACT_DIR}/logs/node-${node_id}.log"
  rm -f "${stop_file}"

  AGENT_NODE_ID="${node_id}" \
  AGENTFIELD_URL="${cp_url}" \
  AGENT_HOST="127.0.0.1" \
  PORT="${port}" \
  AGENT_CALLBACK_URL="http://127.0.0.1:${port}" \
  SMOKE_HEARTBEAT_SECONDS="${hb_seconds}" \
  SMOKE_STOP_FILE="${stop_file}" \
  PYTHONUNBUFFERED=1 \
  "${venv_dir}/bin/python" "${SMOKE_DIR}/nodes/python/node.py" >"${log_file}" 2>&1 &
  local pid=$!

  SMOKE_STOP_FILES+=("${stop_file}")
  SMOKE_NODE_PIDS+=("${pid}")
  SMOKE_LOG_FILES+=("${log_file}")
  log_info "Started Python node ${node_id} (pid=${pid}, port=${port}, hb=${hb_seconds}s)"
}

start_go_node() {
  local bin_path="$1"
  local node_id="$2"
  local cp_url="$3"
  local port="$4"
  local hb_interval="${5:-5s}"

  local stop_file="${SMOKE_WORK_DIR}/stop-${node_id}"
  local log_file="${SMOKE_ARTIFACT_DIR}/logs/node-${node_id}.log"
  rm -f "${stop_file}"

  AGENT_NODE_ID="${node_id}" \
  AGENTFIELD_URL="${cp_url}" \
  AGENT_LISTEN_ADDR="127.0.0.1:${port}" \
  AGENT_PUBLIC_URL="http://127.0.0.1:${port}" \
  SMOKE_HEARTBEAT_INTERVAL="${hb_interval}" \
  SMOKE_STOP_FILE="${stop_file}" \
  "${bin_path}" >"${log_file}" 2>&1 &
  local pid=$!

  SMOKE_STOP_FILES+=("${stop_file}")
  SMOKE_NODE_PIDS+=("${pid}")
  SMOKE_LOG_FILES+=("${log_file}")
  log_info "Started Go node ${node_id} (pid=${pid}, port=${port}, hb=${hb_interval})"
}

wait_for_node_registered() {
  local cp_url="$1"
  local node_id="$2"
  local node_port="$3"
  local attempt=0

  while [[ "${attempt}" -lt 60 ]]; do
    if curl -fsS --max-time 2 "http://127.0.0.1:${node_port}/health" >/dev/null 2>&1; then
      local nodes_json=""
      if nodes_json="$(curl -fsS --max-time 3 "${cp_url}/api/v1/nodes" 2>/dev/null)"; then
        if python3 - "${node_id}" "${nodes_json}" <<'PY'
import json
import sys

want = sys.argv[1]
data = json.loads(sys.argv[2])
nodes = data.get("nodes") or []
for n in nodes:
    if n.get("id") == want and n.get("health_status") == "active":
        sys.exit(0)
sys.exit(1)
PY
        then
          log_info "Node ${node_id} registered and active in ${cp_url}/api/v1/nodes"
          return 0
        fi
      fi
    fi
    sleep 1
    attempt=$((attempt + 1))
  done

  log_err "Node ${node_id} did not appear active in ${cp_url}/api/v1/nodes"
  return 1
}

run_sync_execute() {
  local cp_url="$1"
  local target="$2"
  local input_json="$3"
  local extra_header="${4:-}"

  local req_body
  req_body="$(python3 -c 'import json,sys; print(json.dumps({"input": json.loads(sys.argv[1])}))' "${input_json}")"

  local -a curl_args=(
    -fsS
    --max-time 45
    -X POST
    "${cp_url}/api/v1/execute/${target}"
    -H "Content-Type: application/json"
    -d "${req_body}"
  )
  if [[ -n "${extra_header}" ]]; then
    curl_args+=(-H "${extra_header}")
  fi

  local resp
  resp="$(curl "${curl_args[@]}")"
  python3 - "${resp}" "${target}" <<'PY'
import json
import sys

resp = json.loads(sys.argv[1])
target = sys.argv[2]
if resp.get("status") != "succeeded":
    raise SystemExit(f"sync execute {target} failed: {resp}")
if not resp.get("execution_id"):
    raise SystemExit(f"sync execute {target} missing execution_id: {resp}")
PY
  printf '%s' "${resp}"
}

run_async_execute() {
  local cp_url="$1"
  local target="$2"
  local input_json="$3"
  local extra_header="${4:-}"

  local req_body
  req_body="$(python3 -c 'import json,sys; print(json.dumps({"input": json.loads(sys.argv[1])}))' "${input_json}")"

  local -a curl_args=(
    -fsS
    --max-time 30
    -X POST
    "${cp_url}/api/v1/execute/async/${target}"
    -H "Content-Type: application/json"
    -d "${req_body}"
  )
  if [[ -n "${extra_header}" ]]; then
    curl_args+=(-H "${extra_header}")
  fi

  local submit_resp exec_id
  submit_resp="$(curl "${curl_args[@]}")"
  exec_id="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); eid=d.get("execution_id",""); assert eid, f"missing execution_id in {d}"; print(eid)' "${submit_resp}")"

  local attempt=0
  local poll_resp=""
  while [[ "${attempt}" -lt 90 ]]; do
    poll_resp="$(curl -fsS --max-time 5 "${cp_url}/api/v1/executions/${exec_id}")"
    local st
    st="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("status",""))' "${poll_resp}")"
    case "${st}" in
      succeeded)
        printf '%s' "${poll_resp}"
        return 0
        ;;
      failed|cancelled|timeout|timed_out)
        log_err "Async execution ${exec_id} for ${target} ended with status=${st}: ${poll_resp}"
        return 1
        ;;
    esac
    sleep 0.5
    attempt=$((attempt + 1))
  done

  log_err "Timed out polling async execution ${exec_id} for ${target}: ${poll_resp}"
  return 1
}

assert_echo_result() {
  local exec_resp="$1"
  local expected_msg="$2"
  local expected_seq="$3"
  local expected_node="$4"

  python3 - "${exec_resp}" "${expected_msg}" "${expected_seq}" "${expected_node}" <<'PY'
import json
import sys

resp = json.loads(sys.argv[1])
want_msg = sys.argv[2]
want_seq = int(sys.argv[3])
want_node = sys.argv[4]

res = resp.get("result") or {}
echo = res.get("echo") or {}
if echo.get("message") != want_msg:
    raise SystemExit(f"echo.message mismatch: got {echo.get('message')!r}, want {want_msg!r}; full={resp}")
if int(echo.get("seq", -1)) != want_seq:
    raise SystemExit(f"echo.seq mismatch: got {echo.get('seq')!r}, want {want_seq!r}; full={resp}")
if res.get("node_id") != want_node:
    raise SystemExit(f"node_id mismatch: got {res.get('node_id')!r}, want {want_node!r}; full={resp}")
exec_ctx = res.get("execution_context") or {}
if not exec_ctx.get("execution_id"):
    raise SystemExit(f"missing execution_context.execution_id in {resp}")
PY
}

verify_heartbeat_survival() {
  local cp_url="$1"
  local node_id="$2"
  local wait_secs="${3:-12}"

  local before_json after_json list_json
  before_json="$(curl -fsS --max-time 5 "${cp_url}/api/v1/nodes/${node_id}")"
  local hb_before
  hb_before="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("last_heartbeat",""))' "${before_json}")"

  log_info "Waiting ${wait_secs}s (>1 heartbeat + health sweep) to verify ${node_id} stays active (initial last_heartbeat=${hb_before})"
  sleep "${wait_secs}"

  list_json="$(curl -fsS --max-time 5 "${cp_url}/api/v1/nodes")"
  after_json="$(curl -fsS --max-time 5 "${cp_url}/api/v1/nodes/${node_id}")"

  python3 - "${node_id}" "${before_json}" "${after_json}" "${list_json}" <<'PY'
import json
import sys

node_id = sys.argv[1]
before = json.loads(sys.argv[2])
after = json.loads(sys.argv[3])
node_list = json.loads(sys.argv[4]).get("nodes") or []

active_ids = [n.get("id") for n in node_list if n.get("health_status") == "active"]
if node_id not in active_ids:
    raise SystemExit(f"node {node_id} missing from active node list after heartbeat wait: {node_list}")

health = after.get("health_status")
lifecycle = after.get("lifecycle_status")
if health != "active":
    raise SystemExit(f"node {node_id} health_status={health!r} (expected 'active'): {after}")
if lifecycle in ("offline", "degraded", "error"):
    raise SystemExit(f"node {node_id} lifecycle_status={lifecycle!r}: {after}")

hb_before = before.get("last_heartbeat") or ""
hb_after = after.get("last_heartbeat") or ""
if not hb_after or hb_after == hb_before:
    raise SystemExit(
        f"node {node_id} last_heartbeat did not advance across heartbeat interval: before={hb_before!r}, after={hb_after!r}"
    )
print(f"{hb_before} -> {hb_after} (health={health}, lifecycle={lifecycle})")
PY
}
