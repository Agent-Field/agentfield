#!/usr/bin/env bash
# Run the log-demo stack on the host when Docker Desktop is not available.
# Uses the same internal token as docker-compose.log-demo.yml.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${AGENTFIELD_LOG_DEMO_DATA:-/tmp/agentfield-log-demo}"
TOKEN="${AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN:-log-demo-internal-token}"
VENV="${AGENTFIELD_LOG_DEMO_VENV:-/tmp/agentfield-log-demo-venv}"
CP_LOG="${DATA_DIR}/control-plane.log"
PY_LOG="${DATA_DIR}/demo-python.log"
GO_LOG="${DATA_DIR}/demo-go.log"
TS_LOG="${DATA_DIR}/demo-ts.log"

mkdir -p "${DATA_DIR}/keys"

port_listener_pid() {
  local port="$1"
  if ! command -v lsof >/dev/null 2>&1; then
    return 1
  fi
  lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null | head -n 1
}

ensure_port_available() {
  local port="$1"
  local service="$2"
  local listener_pid
  listener_pid="$(port_listener_pid "${port}" || true)"
  if [[ -n "${listener_pid}" ]]; then
    echo "Port ${port} is already in use by PID ${listener_pid}; refusing to start ${service}."
    echo "Stop the conflicting service or run ./scripts/stop-log-demo-native.sh if it belongs to this demo stack."
    exit 1
  fi
}

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Creating venv at ${VENV} (Python 3.12+) and installing sdk/python..."
  if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv "${VENV}"
  else
    python3 -m venv "${VENV}"
  fi
  "${VENV}/bin/pip" install -q -e "${REPO_ROOT}/sdk/python"
fi

export AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN="${TOKEN}"
export AGENTFIELD_STORAGE_MODE=local
export AGENTFIELD_STORAGE_LOCAL_DATABASE_PATH="${DATA_DIR}/agentfield.db"
export AGENTFIELD_STORAGE_LOCAL_KV_STORE_PATH="${DATA_DIR}/agentfield.bolt"
export AGENTFIELD_FEATURES_DID_KEYSTORE_PATH="${DATA_DIR}/keys"
export AGENTFIELD_CONFIG_FILE="${REPO_ROOT}/tests/functional/docker/agentfield-test.yaml"
export AGENTFIELD_HOME="${DATA_DIR}"

cd "${REPO_ROOT}/control-plane"
echo "Building control plane binary (avoids long go run compile on each start)..."
go build -o "${DATA_DIR}/agentfield-server" ./cmd/agentfield-server

if [[ -f "${DATA_DIR}/cp.pid" ]] && kill -0 "$(cat "${DATA_DIR}/cp.pid")" 2>/dev/null; then
  echo "Control plane already running (PID $(cat "${DATA_DIR}/cp.pid"))."
else
  ensure_port_available 8080 "control plane"
  nohup "${DATA_DIR}/agentfield-server" server \
    --port 8080 \
    --config "${REPO_ROOT}/tests/functional/docker/agentfield-test.yaml" \
    --vc-execution \
    --open=false >>"${CP_LOG}" 2>&1 &
  echo $! >"${DATA_DIR}/cp.pid"
  echo "Control plane PID $(cat "${DATA_DIR}/cp.pid") (log: ${CP_LOG})"
fi

echo "Waiting for http://127.0.0.1:8080/api/v1/health ..."
for i in $(seq 1 60); do
  if curl -sfS --max-time 2 http://127.0.0.1:8080/api/v1/health >/dev/null; then
    echo "Control plane is healthy."
    curl -sfS http://127.0.0.1:8080/api/v1/health
    echo
    break
  fi
  sleep 1
done
if ! curl -sfS --max-time 2 http://127.0.0.1:8080/api/v1/health >/dev/null; then
  echo "Control plane did not become healthy in time. Tail ${CP_LOG}:"
  tail -40 "${CP_LOG}" || true
  exit 1
fi

start_agent() {
  local name="$1" pidfile="$2" logfile="$3" port="$4"
  shift 4
  if [[ -f "${pidfile}" ]] && kill -0 "$(cat "${pidfile}")" 2>/dev/null; then
    echo "${name} already running (PID $(cat "${pidfile}"))."
    return 0
  fi
  ensure_port_available "${port}" "${name}"
  nohup "$@" >>"${logfile}" 2>&1 &
  echo $! >"${pidfile}"
  echo "${name} PID $(cat "${pidfile}") (log: ${logfile})"
}

if [[ ! -f "${REPO_ROOT}/sdk/typescript/dist/index.js" ]]; then
  echo "Building TypeScript SDK dist for the native log demo..."
  (cd "${REPO_ROOT}/sdk/typescript" && npm run build)
fi

start_agent "demo-python" "${DATA_DIR}/demo-python.pid" "${PY_LOG}" 8001 env \
  AGENTFIELD_URL=http://127.0.0.1:8080 \
  AGENT_NODE_ID=demo-python-logs \
  PORT=8001 \
  AGENT_CALLBACK_URL=http://127.0.0.1:8001 \
  AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN="${TOKEN}" \
  "${VENV}/bin/python" "${REPO_ROOT}/examples/python_agent_nodes/docker_hello_world/main.py"

start_agent "demo-go" "${DATA_DIR}/demo-go.pid" "${GO_LOG}" 8002 env \
  AGENTFIELD_URL=http://127.0.0.1:8080 \
  AGENT_NODE_ID=demo-go-logs \
  AGENT_LISTEN_ADDR=:8002 \
  AGENT_PUBLIC_URL=http://127.0.0.1:8002 \
  AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN="${TOKEN}" \
  bash -lc "cd '${REPO_ROOT}/examples/go_agent_nodes' && go run . serve"

start_agent "demo-ts" "${DATA_DIR}/demo-ts.pid" "${TS_LOG}" 8003 env \
  NODE_PATH="${REPO_ROOT}/sdk/typescript/node_modules" \
  AGENTFIELD_SERVER=http://127.0.0.1:8080 \
  TS_AGENT_ID=demo-ts-logs \
  TS_AGENT_PORT=8003 \
  TS_AGENT_BIND_HOST=0.0.0.0 \
  TS_AGENT_PUBLIC_URL=http://127.0.0.1:8003 \
  AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN="${TOKEN}" \
  node "${REPO_ROOT}/tests/functional/docker/log-demo-node/log-demo.mjs"

echo ""
echo "Open http://localhost:8080/ui/agents - expand a row -> Process logs -> Live; see tests/functional/docker/LOG_DEMO.md for the execution-observability demo path"
echo "Stop: ${REPO_ROOT}/scripts/stop-log-demo-native.sh"
