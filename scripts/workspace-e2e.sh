#!/usr/bin/env bash
# Self-contained local end-to-end check for the workspace-artifacts feature
# (docs/design/workspace-artifacts.md), exercised through examples/workspace-demo.
#
# No Docker, no cloud dependency: builds `af` from source, starts a control
# plane and the workspace-demo node as local background processes, then
# drives the `--dir` / `af diff` / `af apply` flow against a throwaway
# fixture folder.
#
# This targets a CLI/SDK surface (`af call --dir`, `af diff`, `af apply`)
# that may not be merged yet when this script is run -- see the capability
# probe below. When it isn't, the script prints a clear "NOT_WIRED" message
# and exits nonzero rather than hanging or failing confusingly deep into
# the run.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${AGENTFIELD_WORKSPACE_E2E_DATA:-/tmp/agentfield-workspace-e2e}"
FIXTURE_DIR="${DATA_DIR}/fixture"

CP_PORT="${AGENTFIELD_WORKSPACE_E2E_CP_PORT:-18280}"
NODE_PORT="${AGENTFIELD_WORKSPACE_E2E_NODE_PORT:-18291}"
CP_URL="http://127.0.0.1:${CP_PORT}"
NODE_ID="workspace-demo"

AF_BIN="${DATA_DIR}/af"
CP_LOG="${DATA_DIR}/control-plane.log"
NODE_LOG="${DATA_DIR}/node.log"
VENV_DIR="${DATA_DIR}/venv"

CP_PID=""
NODE_PID=""
FAILURES=0

# ---------------------------------------------------------------- reporting
pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; FAILURES=$((FAILURES + 1)); }
hard_fail() {
  echo "FAIL: $1"
  [[ -n "${2:-}" && -f "${2:-}" ]] && { echo "----- last 40 lines of $2 -----"; tail -n 40 "$2"; }
  exit 1
}
not_wired() {
  echo "NOT_WIRED: $1"
  echo "This script targets docs/design/workspace-artifacts.md, whose CLI/SDK"
  echo "implementation may still be landing in parallel branches. Re-run this"
  echo "script once that work is merged into this branch."
  exit 1
}

cleanup() {
  for pid in "${NODE_PID}" "${CP_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

wait_for_http() {
  local url="$1" name="$2" log="$3"
  for _ in $(seq 1 60); do
    if curl -sfS --max-time 2 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  hard_fail "${name} did not become reachable at ${url}" "${log}"
}

# --------------------------------------------------------------- prep dirs
rm -rf "${DATA_DIR}"
mkdir -p "${DATA_DIR}/cp-home"

# ------------------------------------------------------------ prerequisites
echo "==> Checking prerequisites..."
command -v go >/dev/null 2>&1 || hard_fail "go toolchain not found on PATH"
command -v python3 >/dev/null 2>&1 || hard_fail "python3 not found on PATH"
python3 -c "import agentfield" >/dev/null 2>&1 \
  || hard_fail "Python 'agentfield' package not importable. Run: pip install -e ${REPO_ROOT}/sdk/python (or activate a venv that has it)."

echo "==> Building control plane CLI (af)..."
(cd "${REPO_ROOT}/control-plane" && go build -o "${AF_BIN}" ./cmd/af) \
  || hard_fail "failed to build af from control-plane/cmd/af"

# ------------------------------------------------------- capability probe
# Fail fast and clearly if the workspace-artifacts CLI surface this script
# needs (docs/design/workspace-artifacts.md "CLI surface") isn't wired yet,
# rather than starting servers only to fail deep into the run.
echo "==> Probing for workspace-artifacts CLI support..."
missing=0
if ! "${AF_BIN}" call --help 2>&1 | grep -qE -- '--dir\b'; then
  echo "  - af call --dir: MISSING"
  missing=1
else
  echo "  - af call --dir: present"
fi
if ! "${AF_BIN}" diff --help >/dev/null 2>&1; then
  echo "  - af diff: MISSING"
  missing=1
else
  echo "  - af diff: present"
fi
if ! "${AF_BIN}" apply --help >/dev/null 2>&1; then
  echo "  - af apply: MISSING"
  missing=1
else
  echo "  - af apply: present"
fi
if [[ "${missing}" -ne 0 ]]; then
  not_wired "af call --dir / af diff / af apply are not available in this af build"
fi

# --------------------------------------------------------------- fixture
echo "==> Creating fixture folder..."
mkdir -p "${FIXTURE_DIR}/nested"
cat > "${FIXTURE_DIR}/a.py" <<'EOF'
def hello():
    return "hi"
EOF
cat > "${FIXTURE_DIR}/nested/util.py" <<'EOF'
def util():
    pass
EOF
cat > "${FIXTURE_DIR}/README.txt" <<'EOF'
Just a plain text file, not source code.
EOF
cat > "${FIXTURE_DIR}/.gitignore" <<'EOF'
junk.log
EOF
cat > "${FIXTURE_DIR}/junk.log" <<'EOF'
This file is gitignored and must never leave the caller's machine.
EOF

# ----------------------------------------------------- start control plane
echo "==> Starting control plane..."
AGENTFIELD_HOME="${DATA_DIR}/cp-home" \
AGENTFIELD_STORAGE_MODE=local \
AGENTFIELD_STORAGE_LOCAL_DATABASE_PATH="${DATA_DIR}/cp-home/agentfield.db" \
AGENTFIELD_STORAGE_LOCAL_KV_STORE_PATH="${DATA_DIR}/cp-home/agentfield.bolt" \
nohup "${AF_BIN}" server --port "${CP_PORT}" --open=false >>"${CP_LOG}" 2>&1 &
CP_PID=$!
wait_for_http "${CP_URL}/api/v1/health" "control plane" "${CP_LOG}"
echo "    control plane up (PID ${CP_PID})"

export AGENTFIELD_SERVER="${CP_URL}"

# ------------------------------------------------------- start demo node
echo "==> Starting workspace-demo node..."
(
  cd "${REPO_ROOT}/examples/workspace-demo" && \
  exec env \
    AGENT_NODE_ID="${NODE_ID}" \
    AGENTFIELD_URL="${CP_URL}" \
    PORT="${NODE_PORT}" \
    AGENTFIELD_DEV_MODE=true \
    python3 main.py
) >>"${NODE_LOG}" 2>&1 &
NODE_PID=$!
wait_for_http "http://127.0.0.1:${NODE_PORT}/health" "workspace-demo node" "${NODE_LOG}"
echo "    node up (PID ${NODE_PID})"

echo "==> Waiting for node to register with the control plane..."
registered=0
for _ in $(seq 1 30); do
  if curl -sfS --max-time 2 "${CP_URL}/api/v1/nodes?show_all=true" 2>/dev/null | grep -q "\"id\":\"${NODE_ID}\""; then
    registered=1
    break
  fi
  sleep 0.5
done
[[ "${registered}" -eq 1 ]] || hard_fail "workspace-demo node never appeared in ${CP_URL}/api/v1/nodes" "${NODE_LOG}"
pass "workspace-demo node registered with the control plane"

# --------------------------------------------------------------- run call
echo "==> Calling ${NODE_ID}.report --dir ${FIXTURE_DIR} ..."
RUN_JSON="$("${AF_BIN}" call "${NODE_ID}.report" --dir "${FIXTURE_DIR}" --async -o json 2>"${DATA_DIR}/call.err")"
CALL_STATUS=$?
if [[ ${CALL_STATUS} -ne 0 ]]; then
  hard_fail "af call ${NODE_ID}.report --dir ${FIXTURE_DIR} --async failed" "${DATA_DIR}/call.err"
fi
RUN_ID="$(echo "${RUN_JSON}" | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("run_id",""))
except Exception:
    print("")' 2>/dev/null)"
if [[ -z "${RUN_ID}" ]]; then
  hard_fail "could not extract run_id from: ${RUN_JSON}"
fi
echo "    run_id: ${RUN_ID}"

echo "==> Waiting for the run to finish..."
WAIT_JSON="$("${AF_BIN}" wait "${RUN_ID}" -o json --timeout 60 2>"${DATA_DIR}/wait.err")"
WAIT_STATUS=$?
if [[ ${WAIT_STATUS} -ne 0 ]]; then
  hard_fail "af wait ${RUN_ID} failed or run did not succeed" "${DATA_DIR}/wait.err"
fi
echo "${WAIT_JSON}" | grep -q '"status": *"succeeded"\|"status":"succeeded"' \
  && pass "run ${RUN_ID} succeeded" \
  || fail "run ${RUN_ID} did not report status=succeeded: ${WAIT_JSON}"

# The reasoner's own result must never mention the gitignored file -- it
# should have been excluded before the workspace was ever transported.
if echo "${WAIT_JSON}" | grep -q "junk.log"; then
  fail "reasoner result mentions junk.log -- gitignored file appears to have been transported"
else
  pass "reasoner result does not mention the gitignored junk.log"
fi

# ------------------------------------------------------------------- diff
echo "==> Checking af diff ${RUN_ID} ..."
DIFF_OUTPUT="$("${AF_BIN}" diff "${RUN_ID}" 2>"${DATA_DIR}/diff.err")"
DIFF_STATUS=$?
if [[ ${DIFF_STATUS} -ne 0 ]]; then
  hard_fail "af diff ${RUN_ID} failed" "${DATA_DIR}/diff.err"
fi
echo "${DIFF_OUTPUT}" | grep -q "REPORT.md" \
  && pass "REPORT.md appears in the staged diff" \
  || fail "REPORT.md missing from af diff output: ${DIFF_OUTPUT}"

if echo "${DIFF_OUTPUT}" | grep -q "junk.log"; then
  fail "junk.log appears in af diff output -- gitignored file was transported"
else
  pass "the gitignored file (junk.log) was not transported"
fi

# ------------------------------------------------------------------ apply
echo "==> Applying ${RUN_ID} back onto the fixture..."
APPLY_OUTPUT="$("${AF_BIN}" apply "${RUN_ID}" 2>"${DATA_DIR}/apply.err")"
APPLY_STATUS=$?
if [[ ${APPLY_STATUS} -ne 0 ]]; then
  hard_fail "af apply ${RUN_ID} failed" "${DATA_DIR}/apply.err"
fi

if [[ -f "${FIXTURE_DIR}/REPORT.md" ]]; then
  pass "af apply materialized REPORT.md into the fixture"
else
  fail "REPORT.md was not written to ${FIXTURE_DIR} after af apply"
fi

if grep -rl "# reviewed: workspace-demo" "${FIXTURE_DIR}"/*.py "${FIXTURE_DIR}"/nested/*.py >/dev/null 2>&1; then
  pass "the reviewed-marker edit was applied to a fixture .py file"
else
  fail "no fixture .py file contains the reviewed marker after af apply"
fi

if [[ -f "${FIXTURE_DIR}/junk.log" ]] && ! grep -q "REPORT" "${FIXTURE_DIR}/junk.log"; then
  pass "junk.log on disk is untouched"
else
  fail "junk.log missing or unexpectedly modified after af apply"
fi

# ------------------------------------------------------- bonus: apply_note
echo "==> Bonus check: ${NODE_ID}.apply_note --dir ${FIXTURE_DIR} ..."
NOTE_RUN_JSON="$("${AF_BIN}" call "${NODE_ID}.apply_note" --dir "${FIXTURE_DIR}" \
  --in '{"filename":"NOTES.md","note":"checked by workspace-e2e.sh"}' --async -o json 2>"${DATA_DIR}/note-call.err")"
if [[ $? -eq 0 ]]; then
  NOTE_RUN_ID="$(echo "${NOTE_RUN_JSON}" | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("run_id",""))
except Exception:
    print("")' 2>/dev/null)"
  if [[ -n "${NOTE_RUN_ID}" ]] && "${AF_BIN}" wait "${NOTE_RUN_ID}" -o json --timeout 60 >/dev/null 2>"${DATA_DIR}/note-wait.err" \
     && "${AF_BIN}" apply "${NOTE_RUN_ID}" >/dev/null 2>"${DATA_DIR}/note-apply.err"; then
    if [[ -f "${FIXTURE_DIR}/NOTES.md" ]] && grep -q "checked by workspace-e2e.sh" "${FIXTURE_DIR}/NOTES.md"; then
      pass "apply_note materialized NOTES.md with the expected note (bonus check)"
    else
      fail "NOTES.md missing or missing expected note after apply_note + af apply (bonus check)"
    fi
  else
    fail "apply_note wait/apply failed (bonus check)"
  fi
else
  fail "af call ${NODE_ID}.apply_note --dir failed (bonus check)"
fi

# --------------------------------------------------------------- summary
echo
if [[ "${FAILURES}" -eq 0 ]]; then
  echo "All workspace e2e checks passed."
  exit 0
else
  echo "${FAILURES} workspace e2e check(s) failed. Logs kept at ${DATA_DIR}."
  exit 1
fi
