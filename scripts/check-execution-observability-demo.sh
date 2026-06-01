#!/usr/bin/env bash
# Validate the execution-observability demo docs and sample fixture without
# starting the control plane or functional test harness.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC_PATH="${REPO_ROOT}/tests/functional/docker/LOG_DEMO.md"
README_PATH="${REPO_ROOT}/tests/functional/README.md"
SAMPLE_PATH="${REPO_ROOT}/tests/functional/docker/execution-observability-sample.ndjson"

python3 - <<'PY' "${DOC_PATH}" "${README_PATH}" "${SAMPLE_PATH}"
import json
import pathlib
import sys

doc_path = pathlib.Path(sys.argv[1])
readme_path = pathlib.Path(sys.argv[2])
sample_path = pathlib.Path(sys.argv[3])

doc_text = doc_path.read_text(encoding="utf-8")
readme_text = readme_path.read_text(encoding="utf-8")

required_doc_terms = [
    "Execution observability demo stack",
    "structured execution logs",
    "advanced/debug view",
    "Process logs",
    "execution-observability-sample.ndjson",
]
for term in required_doc_terms:
    if term not in doc_text:
        raise SystemExit(f"missing doc term: {term}")

if "execution-observability-sample.ndjson" not in readme_text:
    raise SystemExit("missing README reference to execution-observability sample")

rows = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) < 4:
    raise SystemExit(f"expected at least 4 sample rows, found {len(rows)}")

required_fields = {
    "v",
    "sequence",
    "ts",
    "execution_id",
    "workflow_id",
    "run_id",
    "root_workflow_id",
    "agent_node_id",
    "level",
    "source",
    "event_type",
    "message",
    "system_generated",
    "sdk_language",
}
for row in rows:
    missing = sorted(required_fields - row.keys())
    if missing:
        raise SystemExit(f"sample row missing fields: {missing}")

sequences = [row["sequence"] for row in rows]
if sequences != sorted(sequences):
    raise SystemExit(f"sample sequences are not ordered: {sequences}")

event_types = [row["event_type"] for row in rows]
expected_event_types = [
    "execution.started",
    "log.info",
    "node.call.completed",
    "execution.completed",
]
if event_types != expected_event_types:
    raise SystemExit(f"unexpected sample event types: {event_types}")

print("Execution observability demo checks passed.")
PY
