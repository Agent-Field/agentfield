#!/usr/bin/env bash
# End-to-end plain SWE-AF build smoke comparison:
#   new (PR #181) vs baseline (da78a77d + command-code@1.88.5)
# against agentfield/control-plane:v0.1.57 with local storage and OpenCode.
#
# Never enable `set -x` in this file: OPENROUTER_API_KEY and AGENTFIELD_API_KEY
# are in the process environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SWEAF_NEW_DIR="${SWEAF_NEW_DIR:-${REPO_ROOT}/sweaf-new}"
SWEAF_BASE_DIR="${SWEAF_BASE_DIR:-${REPO_ROOT}/sweaf-base}"
SMOKE_OUT_DIR="${SMOKE_OUT_DIR:-${REPO_ROOT}/smoke-realbuild-out}"
CP_IMAGE="${CP_IMAGE:-agentfield/control-plane:v0.1.57}"
CP_CONFIG_HOST="${SCRIPT_DIR}/fixtures/agentfield.yaml"
FIXTURE_SRC_DIR="${SCRIPT_DIR}/fixtures/textutil"
BUILD_REQUEST_JSON="${SCRIPT_DIR}/fixtures/build_request.json"

DOCKER_NET="sweaf-smoke-net"
BUILD_POLL_TIMEOUT_SECONDS="${BUILD_POLL_TIMEOUT_SECONDS:-1800}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-10}"

log() {
  printf '[smoke-sweaf %s] %s\n' "$(date -u +'%H:%M:%S')" "$*"
}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "::error::OPENROUTER_API_KEY secret is not set or empty." >&2
  exit 1
fi

AGENTFIELD_API_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
export AGENTFIELD_API_KEY
if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
  echo "::add-mask::${AGENTFIELD_API_KEY}"
fi

mkdir -p "${SMOKE_OUT_DIR}/new" "${SMOKE_OUT_DIR}/baseline"

CURL_AUTH_CFG="$(mktemp "${TMPDIR:-/tmp}/sweaf-curl-auth-XXXXXX")"
chmod 600 "${CURL_AUTH_CFG}"
printf 'header = "X-API-Key: %s"\n' "${AGENTFIELD_API_KEY}" > "${CURL_AUTH_CFG}"

ACTIVE_WORK_DIRS=()

sanitize_stream() {
  PYTHONPATH="${SCRIPT_DIR}" python3 -c '
import sys
from compare_runs import redact_secrets
sys.stdout.write(redact_secrets(sys.stdin.read()))
'
}

dump_failure_logs() {
  log "Smoke check failed — printing sanitized container logs (tail -n 200)."
  local label
  for label in new baseline; do
    for logfile in "${SMOKE_OUT_DIR}/${label}/control-plane.log" "${SMOKE_OUT_DIR}/${label}/node.log" "${SMOKE_OUT_DIR}/${label}/docker-build.log"; do
      if [[ -f "${logfile}" ]]; then
        echo "==================== ${label}: $(basename "${logfile}") ===================="
        tail -n 200 "${logfile}" | sanitize_stream || true
      fi
    done
  done
}

cleanup_run_containers() {
  local label="$1"
  docker rm -f "sweaf-node-${label}" >/dev/null 2>&1 || true
  docker rm -f "sweaf-cp-${label}" >/dev/null 2>&1 || true
  docker volume rm -f "sweaf-cp-data-${label}" >/dev/null 2>&1 || true
}

cleanup_all() {
  local exit_code=$?
  rm -f "${CURL_AUTH_CFG}" || true
  for label in new baseline; do
    # Capture logs if a container is still up when an unexpected exit triggers
    if docker inspect "sweaf-node-${label}" >/dev/null 2>&1 && [[ ! -s "${SMOKE_OUT_DIR}/${label}/node.log" ]]; then
      docker logs "sweaf-node-${label}" 2>&1 | sanitize_stream > "${SMOKE_OUT_DIR}/${label}/node.log" || true
    fi
    if docker inspect "sweaf-cp-${label}" >/dev/null 2>&1 && [[ ! -s "${SMOKE_OUT_DIR}/${label}/control-plane.log" ]]; then
      docker logs "sweaf-cp-${label}" 2>&1 | sanitize_stream > "${SMOKE_OUT_DIR}/${label}/control-plane.log" || true
    fi
    cleanup_run_containers "${label}"
  done
  docker network rm "${DOCKER_NET}" >/dev/null 2>&1 || true
  if (( ${#ACTIVE_WORK_DIRS[@]} > 0 )); then
    local dir
    for dir in "${ACTIVE_WORK_DIRS[@]}"; do
      if [[ -n "${dir}" && -d "${dir}" ]]; then
        rm -rf "${dir}" || true
      fi
    done
  fi
  if [[ "${exit_code}" -ne 0 ]]; then
    dump_failure_logs
  fi
}
trap cleanup_all EXIT

# Ensure pytest is available on the runner for post-build fixture verification
if ! python3 -m pytest --version >/dev/null 2>&1; then
  log "Installing pytest on host runner..."
  python3 -m pip install --quiet pytest
fi

# 1. Verify checkouts and patch baseline Dockerfile (command-code@1.82.2 -> 1.88.5)
if [[ ! -f "${SWEAF_NEW_DIR}/Dockerfile" ]]; then
  echo "::error::Missing new SWE-AF checkout at ${SWEAF_NEW_DIR}" >&2
  exit 1
fi
if [[ ! -f "${SWEAF_BASE_DIR}/Dockerfile" ]]; then
  echo "::error::Missing baseline SWE-AF checkout at ${SWEAF_BASE_DIR}" >&2
  exit 1
fi

grep -q 'command-code@1.88.5' "${SWEAF_NEW_DIR}/Dockerfile" || {
  echo "::error::Expected command-code@1.88.5 in ${SWEAF_NEW_DIR}/Dockerfile" >&2
  exit 1
}
grep -q 'command-code@1.82.2' "${SWEAF_BASE_DIR}/Dockerfile" || {
  echo "::error::Expected command-code@1.82.2 in unpatched ${SWEAF_BASE_DIR}/Dockerfile" >&2
  exit 1
}

log "Patching baseline Dockerfile: command-code@1.82.2 -> command-code@1.88.5"
sed -i 's/command-code@1.82.2/command-code@1.88.5/' "${SWEAF_BASE_DIR}/Dockerfile"
grep -q 'command-code@1.88.5' "${SWEAF_BASE_DIR}/Dockerfile"
if grep -q 'command-code@1.82.2' "${SWEAF_BASE_DIR}/Dockerfile"; then
  echo "::error::command-code@1.82.2 still present in baseline Dockerfile after sed" >&2
  exit 1
fi
git -c safe.directory='*' -C "${SWEAF_BASE_DIR}" diff -- Dockerfile > "${SMOKE_OUT_DIR}/baseline-dockerfile.patch"

NEW_COMMIT="$(git -c safe.directory='*' -C "${SWEAF_NEW_DIR}" rev-parse HEAD)"
BASE_COMMIT="$(git -c safe.directory='*' -C "${SWEAF_BASE_DIR}" rev-parse HEAD)"
log "New SWE-AF commit:      ${NEW_COMMIT}"
log "Baseline SWE-AF commit: ${BASE_COMMIT} (+ command-code@1.88.5 sed)"

# 2. Build both images and pull control plane v0.1.57
NEW_IMAGE_BUILT=true
log "Building new SWE-AF image (sweaf-smoke:new)..."
set +e
DOCKER_BUILDKIT=1 docker build -t sweaf-smoke:new "${SWEAF_NEW_DIR}" 2>&1 \
  | sanitize_stream > "${SMOKE_OUT_DIR}/new/docker-build.log"
new_build_rc=${PIPESTATUS[0]}
set -e
if [[ "${new_build_rc}" -ne 0 ]]; then
  log "ERROR: docker build failed for sweaf-smoke:new (rc=${new_build_rc})"
  NEW_IMAGE_BUILT=false
fi

BASE_IMAGE_BUILT=true
log "Building baseline SWE-AF image (sweaf-smoke:base)..."
set +e
DOCKER_BUILDKIT=1 docker build -t sweaf-smoke:base "${SWEAF_BASE_DIR}" 2>&1 \
  | sanitize_stream > "${SMOKE_OUT_DIR}/baseline/docker-build.log"
base_build_rc=${PIPESTATUS[0]}
set -e
if [[ "${base_build_rc}" -ne 0 ]]; then
  log "ERROR: docker build failed for sweaf-smoke:base (rc=${base_build_rc})"
  BASE_IMAGE_BUILT=false
fi

log "Pulling control plane image (${CP_IMAGE})..."
docker pull "${CP_IMAGE}" >/dev/null

docker network rm "${DOCKER_NET}" >/dev/null 2>&1 || true
docker network create "${DOCKER_NET}" >/dev/null

prepare_fixture_workspace() {
  local workspaces_dir="$1"
  local origin_dir="${workspaces_dir}/origin.git"
  local fixture_dir="${workspaces_dir}/fixture"

  mkdir -p "${workspaces_dir}"
  chmod 0777 "${workspaces_dir}"

  git init --bare --initial-branch=main "${origin_dir}" >/dev/null
  mkdir -p "${fixture_dir}"
  cp -a "${FIXTURE_SRC_DIR}/." "${fixture_dir}/"

  git -C "${fixture_dir}" init --initial-branch=main >/dev/null
  git -C "${fixture_dir}" add -A
  git -C "${fixture_dir}" \
    -c user.name="Smoke Fixture" \
    -c user.email="smoke@agentfield.local" \
    commit -m "Initial textutil fixture" >/dev/null
  git -C "${fixture_dir}" remote add origin "${origin_dir}"
  git -C "${fixture_dir}" push -u origin main >/dev/null
  # Rewrite origin to the path mounted inside the SWE-AF container (/workspaces/origin.git)
  git -C "${fixture_dir}" remote set-url origin /workspaces/origin.git

  chmod -R a+rwX "${workspaces_dir}"
  git -c safe.directory='*' -C "${fixture_dir}" rev-parse HEAD
}

wait_for_control_plane() {
  local label="$1"
  local cp_name="sweaf-cp-${label}"
  local attempt
  for attempt in $(seq 1 60); do
    if [[ "$(docker inspect -f '{{.State.Running}}' "${cp_name}" 2>/dev/null || echo false)" != "true" ]]; then
      log "[${label}] Control plane container exited prematurely."
      return 1
    fi
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/api/v1/health || echo 000)"
    if [[ "${code}" == "200" ]]; then
      local unauth_code auth_code
      unauth_code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/api/v1/nodes || echo 000)"
      auth_code="$(curl -s -K "${CURL_AUTH_CFG}" -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/api/v1/nodes || echo 000)"
      if [[ "${unauth_code}" == "401" && "${auth_code}" == "200" ]]; then
        log "[${label}] Control plane ready on :8080 (API key auth verified: unauth=401, auth=200)."
        return 0
      fi
    fi
    sleep 1
  done
  log "[${label}] Timed out waiting for control plane readiness."
  return 1
}

wait_for_node_registration() {
  local label="$1"
  local node_name="sweaf-node-${label}"
  local out_file="$2"
  local attempt
  for attempt in $(seq 1 90); do
    if [[ "$(docker inspect -f '{{.State.Running}}' "${node_name}" 2>/dev/null || echo false)" != "true" ]]; then
      log "[${label}] SWE-AF node container exited before registering."
      return 1
    fi
    local tmp_body="${out_file}.tmp"
    local code
    code="$(curl -s -K "${CURL_AUTH_CFG}" -o "${tmp_body}" -w '%{http_code}' http://127.0.0.1:8080/api/v1/nodes/swe-planner || echo 000)"
    if [[ "${code}" == "200" ]]; then
      if python3 -c '
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
reasoners = [r.get("id") for r in (data.get("reasoners") or []) if isinstance(r, dict)]
sys.exit(0 if "build" in reasoners else 1)
' "${tmp_body}" 2>/dev/null; then
        sanitize_stream < "${tmp_body}" > "${out_file}"
        rm -f "${tmp_body}"
        log "[${label}] Node swe-planner registered with build reasoner."
        return 0
      fi
    fi
    rm -f "${tmp_body}"
    sleep 1
  done
  log "[${label}] Timed out waiting for swe-planner registration."
  return 1
}

run_single_build() {
  local label="$1"
  local image_tag="$2"
  local commit_sha="$3"
  local image_built="$4"

  local run_out="${SMOKE_OUT_DIR}/${label}"
  mkdir -p "${run_out}"
  cp "${BUILD_REQUEST_JSON}" "${run_out}/build_request.json"

  local registered=false
  local dispatched=false
  local timed_out_polling=false
  local container_alive_at_end=false
  local execution_id=""
  local run_id=""
  local dispatch_error=""
  local wall_seconds=0
  local harness_model=""
  local agentfield_sdk_version=""
  local command_code_version=""
  local opencode_version=""
  local checked_out_branch=""
  local new_commits_count=0
  local pytest_rc=1
  local slugify_probe="not_run"

  if [[ "${image_built}" != "true" ]]; then
    python3 -c '
import json, sys
meta_path = sys.argv[1]
json.dump({
    "label": sys.argv[2],
    "image_tag": sys.argv[3],
    "commit_sha": sys.argv[4],
    "image_built": False,
    "registered": False,
    "dispatched": False,
    "timed_out_polling": False,
    "container_alive_at_end": False,
    "execution_id": "",
    "run_id": "",
    "dispatch_error": "docker build failed",
    "wall_seconds": 0,
    "harness_model": "",
    "agentfield_sdk_version": "",
    "command_code_version": "",
    "opencode_version": "",
    "checked_out_branch": "",
    "new_commits_count": 0,
    "pytest_rc": 1,
    "slugify_probe": "not_run",
}, open(meta_path, "w", encoding="utf-8"), indent=2)
' "${run_out}/meta.json" "${label}" "${image_tag}" "${commit_sha}"
    return 0
  fi

  local work_root
  work_root="$(mktemp -d "${TMPDIR:-/tmp}/sweaf-work-${label}-XXXXXX")"
  ACTIVE_WORK_DIRS+=("${work_root}")
  local workspaces_dir="${work_root}/workspaces"
  local initial_sha
  initial_sha="$(prepare_fixture_workspace "${workspaces_dir}")"
  log "[${label}] Prepared fixture workspace (initial SHA ${initial_sha:0:12})."

  cleanup_run_containers "${label}"
  local cp_vol="sweaf-cp-data-${label}"
  local cp_name="sweaf-cp-${label}"
  local node_name="sweaf-node-${label}"
  docker volume create "${cp_vol}" >/dev/null

  log "[${label}] Starting control plane (${CP_IMAGE})..."
  docker run -d \
    --name "${cp_name}" \
    --network "${DOCKER_NET}" \
    --network-alias control-plane \
    -p 127.0.0.1:8080:8080 \
    -v "${cp_vol}:/data" \
    -v "${CP_CONFIG_HOST}:/etc/agentfield/smoke-agentfield.yaml:ro" \
    -e AGENTFIELD_HOME=/data/agentfield \
    -e AGENTFIELD_STORAGE_MODE=local \
    -e AGENTFIELD_PORT=8080 \
    -e AGENTFIELD_CONFIG_FILE=/etc/agentfield/smoke-agentfield.yaml \
    -e AGENTFIELD_API_KEY \
    "${CP_IMAGE}" \
    server --port 8080 --config /etc/agentfield/smoke-agentfield.yaml >/dev/null

  if wait_for_control_plane "${label}"; then
    log "[${label}] Starting SWE-AF Python node (${image_tag})..."
    docker run -d \
      --name "${node_name}" \
      --network "${DOCKER_NET}" \
      --network-alias swe-planner \
      -v "${workspaces_dir}:/workspaces" \
      -e AGENTFIELD_SERVER=http://control-plane:8080 \
      -e AGENTFIELD_API_KEY \
      -e NODE_ID=swe-planner \
      -e PORT=8003 \
      -e AGENT_CALLBACK_URL=http://swe-planner:8003 \
      -e OPENROUTER_API_KEY \
      -e SWE_DEFAULT_RUNTIME=open_code \
      "${image_tag}" >/dev/null

    # Probe non-secret container versions and verify forbidden env vars are unset
    local probe_json=""
    if probe_json="$(docker exec "${node_name}" python3 -c '
import importlib.metadata, json, os, subprocess
forbidden = [k for k in ("SWE_FENCE", "SWE_STRICT_BUILD", "GH_TOKEN") if os.environ.get(k)]
if forbidden:
    raise SystemExit(f"Forbidden env vars set: {forbidden}")
def cmd_out(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr).strip() else ""
    except Exception as exc:
        return f"err:{exc}"
print(json.dumps({
    "harness_model": os.environ.get("HARNESS_MODEL", ""),
    "runtime_env": os.environ.get("SWE_DEFAULT_RUNTIME", ""),
    "agentfield_sdk_version": importlib.metadata.version("agentfield"),
    "command_code_version": cmd_out(["command-code", "--version"]),
    "opencode_version": cmd_out(["opencode", "--version"]),
}))
' 2>/dev/null)"; then
      harness_model="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("harness_model",""))' "${probe_json}")"
      agentfield_sdk_version="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("agentfield_sdk_version",""))' "${probe_json}")"
      command_code_version="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("command_code_version",""))' "${probe_json}")"
      opencode_version="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("opencode_version",""))' "${probe_json}")"
      log "[${label}] Runtime metadata: agentfield=${agentfield_sdk_version}, command-code=${command_code_version}, opencode=${opencode_version}, HARNESS_MODEL=${harness_model}"
    fi

    if wait_for_node_registration "${label}" "${run_out}/node_registration.json"; then
      registered=true
      local start_epoch
      start_epoch="$(date +%s)"

      local dispatch_tmp="${run_out}/dispatch_response.tmp"
      local dispatch_code
      dispatch_code="$(curl -s -K "${CURL_AUTH_CFG}" \
        -H "Content-Type: application/json" \
        -d @"${BUILD_REQUEST_JSON}" \
        -o "${dispatch_tmp}" \
        -w '%{http_code}' \
        http://127.0.0.1:8080/api/v1/execute/async/swe-planner.build || echo 000)"
      sanitize_stream < "${dispatch_tmp}" > "${run_out}/dispatch_response.json" || true
      rm -f "${dispatch_tmp}"

      if [[ "${dispatch_code}" == "202" || "${dispatch_code}" == "200" ]]; then
        execution_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("execution_id",""))' "${run_out}/dispatch_response.json" 2>/dev/null || echo "")"
        run_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("run_id",""))' "${run_out}/dispatch_response.json" 2>/dev/null || echo "")"
        if [[ -n "${execution_id}" ]]; then
          dispatched=true
          log "[${label}] Dispatched swe-planner.build: execution_id=${execution_id} run_id=${run_id}"
        else
          dispatch_error="HTTP ${dispatch_code} without execution_id"
        fi
      else
        dispatch_error="HTTP ${dispatch_code}: $(tr '\n' ' ' < "${run_out}/dispatch_response.json" | cut -c1-240)"
        log "[${label}] Dispatch failed: ${dispatch_error}"
      fi

      if [[ "${dispatched}" == "true" ]]; then
        local dead_node_polls=0
        local last_logged_elapsed=-30
        while true; do
          local now_epoch elapsed
          now_epoch="$(date +%s)"
          elapsed=$(( now_epoch - start_epoch ))

          local status_tmp="${run_out}/execution_status.tmp"
          local poll_code
          poll_code="$(curl -s -K "${CURL_AUTH_CFG}" \
            -o "${status_tmp}" \
            -w '%{http_code}' \
            "http://127.0.0.1:8080/api/v1/executions/${execution_id}" || echo 000)"

          local cur_status=""
          if [[ "${poll_code}" == "200" && -s "${status_tmp}" ]]; then
            sanitize_stream < "${status_tmp}" > "${run_out}/execution_status.json"
            cur_status="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1])).get("status","")).lower())' "${run_out}/execution_status.json" 2>/dev/null || echo "")"
          fi
          rm -f "${status_tmp}"

          if (( elapsed - last_logged_elapsed >= 30 )); then
            log "[${label}] t=${elapsed}s status=${cur_status:-unknown}"
            last_logged_elapsed=${elapsed}
          fi

          case "${cur_status}" in
            succeeded|failed|cancelled|timeout)
              wall_seconds=${elapsed}
              log "[${label}] Execution finished at t=${elapsed}s with status=${cur_status}."
              break
              ;;
          esac

          if [[ "$(docker inspect -f '{{.State.Running}}' "${node_name}" 2>/dev/null || echo false)" != "true" ]]; then
            dead_node_polls=$(( dead_node_polls + 1 ))
            if (( dead_node_polls >= 2 )); then
              wall_seconds=${elapsed}
              dispatch_error="SWE-AF node container exited unexpectedly while execution status=${cur_status:-unknown}"
              log "[${label}] ${dispatch_error}"
              break
            fi
          else
            dead_node_polls=0
          fi

          if (( elapsed >= BUILD_POLL_TIMEOUT_SECONDS )); then
            timed_out_polling=true
            wall_seconds=${elapsed}
            log "[${label}] Poll limit (${BUILD_POLL_TIMEOUT_SECONDS}s) reached; cancelling execution ${execution_id}."
            curl -s -K "${CURL_AUTH_CFG}" -X POST \
              -H "Content-Type: application/json" \
              -d '{"reason":"smoke 30m poll timeout"}' \
              "http://127.0.0.1:8080/api/v1/executions/${execution_id}/cancel" >/dev/null 2>&1 || true
            break
          fi

          sleep "${POLL_INTERVAL_SECONDS}"
        done

        # Final fetch of status, notes, and workflow run summary
        local final_tmp="${run_out}/final.tmp"
        if [[ "$(curl -s -K "${CURL_AUTH_CFG}" -o "${final_tmp}" -w '%{http_code}' "http://127.0.0.1:8080/api/v1/executions/${execution_id}" || echo 000)" == "200" ]]; then
          sanitize_stream < "${final_tmp}" > "${run_out}/execution_status.json"
        fi
        if [[ "$(curl -s -K "${CURL_AUTH_CFG}" -o "${final_tmp}" -w '%{http_code}' "http://127.0.0.1:8080/api/v1/executions/${execution_id}/notes" || echo 000)" == "200" ]]; then
          sanitize_stream < "${final_tmp}" > "${run_out}/execution_notes.json"
        fi
        if [[ -n "${run_id}" && "$(curl -s -K "${CURL_AUTH_CFG}" -o "${final_tmp}" -w '%{http_code}' "http://127.0.0.1:8080/api/ui/v2/workflow-runs/${run_id}" || echo 000)" == "200" ]]; then
          sanitize_stream < "${final_tmp}" > "${run_out}/workflow_run.json"
        fi
        rm -f "${final_tmp}"
      fi
    fi

    if [[ "$(docker inspect -f '{{.State.Running}}' "${node_name}" 2>/dev/null || echo false)" == "true" ]]; then
      container_alive_at_end=true
      docker exec "${node_name}" chmod -R a+rwX /workspaces >/dev/null 2>&1 || true
    fi
    docker logs "${node_name}" 2>&1 | sanitize_stream > "${run_out}/node.log" || true
  fi

  docker logs "${cp_name}" 2>&1 | sanitize_stream > "${run_out}/control-plane.log" || true
  cleanup_run_containers "${label}"

  # Inspect fixture workspace after build
  local fixture_dir="${workspaces_dir}/fixture"
  local origin_dir="${workspaces_dir}/origin.git"
  if [[ -d "${fixture_dir}/.git" ]]; then
    checked_out_branch="$(git -c safe.directory='*' -C "${fixture_dir}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")"
    git -c safe.directory='*' -C "${fixture_dir}" log --oneline -n 30 2>&1 | sanitize_stream > "${run_out}/git_log.txt" || true
    git -c safe.directory='*' -C "${fixture_dir}" status --short 2>&1 | sanitize_stream > "${run_out}/git_status.txt" || true
    git -c safe.directory='*' -C "${fixture_dir}" diff --stat "${initial_sha}..HEAD" 2>&1 | sanitize_stream > "${run_out}/git_diff_stat.txt" || true
    new_commits_count="$(git -c safe.directory='*' -C "${fixture_dir}" rev-list --count "${initial_sha}..HEAD" 2>/dev/null || echo 0)"
    git -c safe.directory='*' -C "${origin_dir}" for-each-ref --format='%(refname:short) %(objectname:short)' refs/heads/ 2>&1 | sanitize_stream > "${run_out}/origin_refs.txt" || true

    set +e
    (
      cd "${fixture_dir}" && \
      python3 -m pytest -q -o "cache_dir=${work_root}/pytest_cache"
    ) > "${run_out}/pytest_output.tmp" 2>&1
    pytest_rc=$?
    set -e
    sanitize_stream < "${run_out}/pytest_output.tmp" > "${run_out}/pytest_output.txt"
    rm -f "${run_out}/pytest_output.tmp"

    slugify_probe="$(python3 -c '
import sys
sys.path.insert(0, sys.argv[1])
try:
    import textutil
except Exception as exc:
    print(f"import_error:{type(exc).__name__}")
    raise SystemExit(0)
fn = getattr(textutil, "slugify", None)
if not callable(fn):
    print("missing")
    raise SystemExit(0)
try:
    a = fn("  Hello, World!  ")
    b = fn("---")
    if a == "hello-world" and b == "":
        print("present_and_verified")
    else:
        print(f"present_unexpected_output(a={a!r},b={b!r})")
except Exception as exc:
    print(f"present_raised:{type(exc).__name__}")
' "${fixture_dir}" 2>/dev/null || echo "probe_error")"
  fi

  PYTHONPATH="${SCRIPT_DIR}" python3 -c '
import json, sys
from compare_runs import redact_secrets

meta_path = sys.argv[1]
payload = {
    "label": sys.argv[2],
    "image_tag": sys.argv[3],
    "commit_sha": sys.argv[4],
    "image_built": sys.argv[5] == "true",
    "registered": sys.argv[6] == "true",
    "dispatched": sys.argv[7] == "true",
    "timed_out_polling": sys.argv[8] == "true",
    "container_alive_at_end": sys.argv[9] == "true",
    "execution_id": sys.argv[10],
    "run_id": sys.argv[11],
    "dispatch_error": redact_secrets(sys.argv[12]),
    "wall_seconds": int(sys.argv[13]),
    "harness_model": sys.argv[14],
    "agentfield_sdk_version": sys.argv[15],
    "command_code_version": sys.argv[16],
    "opencode_version": sys.argv[17],
    "checked_out_branch": sys.argv[18],
    "new_commits_count": int(sys.argv[19]),
    "pytest_rc": int(sys.argv[20]),
    "slugify_probe": sys.argv[21],
}
with open(meta_path, "w", encoding="utf-8") as fp:
    json.dump(payload, fp, indent=2)
' \
    "${run_out}/meta.json" \
    "${label}" \
    "${image_tag}" \
    "${commit_sha}" \
    "${image_built}" \
    "${registered}" \
    "${dispatched}" \
    "${timed_out_polling}" \
    "${container_alive_at_end}" \
    "${execution_id}" \
    "${run_id}" \
    "${dispatch_error}" \
    "${wall_seconds}" \
    "${harness_model}" \
    "${agentfield_sdk_version}" \
    "${command_code_version}" \
    "${opencode_version}" \
    "${checked_out_branch}" \
    "${new_commits_count}" \
    "${pytest_rc}" \
    "${slugify_probe}"

  rm -rf "${work_root}" || true
}

# 3. Run plain build once per image: new first, then baseline
log "=== Running plain build on NEW image (PR #181) ==="
run_single_build "new" "sweaf-smoke:new" "${NEW_COMMIT}" "${NEW_IMAGE_BUILT}"

log "=== Running plain build on BASELINE image ==="
run_single_build "baseline" "sweaf-smoke:base" "${BASE_COMMIT}" "${BASE_IMAGE_BUILT}"

# 4. Compare runs, write $GITHUB_STEP_SUMMARY and artifacts, and assert gate
log "=== Comparing NEW vs BASELINE runs ==="
python3 "${SCRIPT_DIR}/compare_runs.py" \
  --new-dir "${SMOKE_OUT_DIR}/new" \
  --base-dir "${SMOKE_OUT_DIR}/baseline" \
  --out-dir "${SMOKE_OUT_DIR}"
