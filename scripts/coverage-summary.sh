#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="$ROOT_DIR/test-reports/coverage"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "$1 is required. Run ./scripts/install.sh first."
    exit 1
  fi
}

require_pytest() {
  if ! python3 -m pytest --version >/dev/null 2>&1; then
    echo "python3 -m pytest is unavailable. Run ./scripts/install.sh first."
    exit 1
  fi
}

extract_go_total() {
  local module_dir="$1"
  local coverprofile="$2"
  (
    cd "$module_dir"
    go tool cover -func="$coverprofile" | awk '/^total:/ {print $3}' | tr -d '%'
  )
}

write_go_cover_report() {
  local module_dir="$1"
  local coverprofile="$2"
  local output_file="$3"
  (
    cd "$module_dir"
    go tool cover -func="$coverprofile" > "$output_file"
  )
}

rm -rf "$REPORT_DIR"
mkdir -p "$REPORT_DIR"

require_cmd go
require_cmd python3
require_pytest
require_cmd npm

echo "==> Running control plane coverage"
(
  cd "$ROOT_DIR/control-plane"
  go test -tags sqlite_fts5 -coverprofile="$REPORT_DIR/control-plane.coverprofile" ./...
)
write_go_cover_report "$ROOT_DIR/control-plane" "$REPORT_DIR/control-plane.coverprofile" "$REPORT_DIR/control-plane.cover.txt"

echo "==> Running Go SDK coverage"
(
  cd "$ROOT_DIR/sdk/go"
  go test -coverprofile="$REPORT_DIR/sdk-go.coverprofile" ./...
)
write_go_cover_report "$ROOT_DIR/sdk/go" "$REPORT_DIR/sdk-go.coverprofile" "$REPORT_DIR/sdk-go.cover.txt"

# Cobertura XML for Go surfaces — consumed by diff-cover to enforce
# per-PR patch coverage. We install gocover-cobertura on demand so CI
# and local runs work from a clean checkout.
if ! command -v gocover-cobertura >/dev/null 2>&1; then
  echo "==> Installing gocover-cobertura"
  GO111MODULE=on go install github.com/boumenot/gocover-cobertura@latest
fi
GOBIN="$(go env GOPATH)/bin"
export PATH="$GOBIN:$PATH"
(
  cd "$ROOT_DIR/control-plane"
  gocover-cobertura < "$REPORT_DIR/control-plane.coverprofile" > "$REPORT_DIR/control-plane-cobertura.xml"
)
(
  cd "$ROOT_DIR/sdk/go"
  gocover-cobertura < "$REPORT_DIR/sdk-go.coverprofile" > "$REPORT_DIR/sdk-go-cobertura.xml"
)

echo "==> Running Python SDK coverage"
(
  cd "$ROOT_DIR/sdk/python"
  python3 -m pytest \
    --cov-report=json:"$REPORT_DIR/sdk-python-coverage.json" \
    --cov-report=xml:"$REPORT_DIR/sdk-python-coverage.xml"
)

echo "==> Running TypeScript SDK coverage"
(
  cd "$ROOT_DIR/sdk/typescript"
  CI=1 npm run test:coverage:core
)
cp "$ROOT_DIR/sdk/typescript/coverage/coverage-summary.json" "$REPORT_DIR/sdk-typescript-coverage-summary.json"
if [[ -f "$ROOT_DIR/sdk/typescript/coverage/cobertura-coverage.xml" ]]; then
  cp "$ROOT_DIR/sdk/typescript/coverage/cobertura-coverage.xml" "$REPORT_DIR/sdk-typescript-cobertura.xml"
fi

echo "==> Running control plane web UI coverage"
(
  cd "$ROOT_DIR/control-plane/web/client"
  CI=1 npm run test:coverage
)
cp "$ROOT_DIR/control-plane/web/client/coverage/coverage-summary.json" "$REPORT_DIR/web-ui-coverage-summary.json"
if [[ -f "$ROOT_DIR/control-plane/web/client/coverage/cobertura-coverage.xml" ]]; then
  cp "$ROOT_DIR/control-plane/web/client/coverage/cobertura-coverage.xml" "$REPORT_DIR/web-ui-cobertura.xml"
fi
# Python already emits cobertura XML directly (see the sdk-python-coverage.xml
# path below) — no extra copy needed.

CONTROL_PLANE_TOTAL="$(extract_go_total "$ROOT_DIR/control-plane" "$REPORT_DIR/control-plane.coverprofile")"
SDK_GO_TOTAL="$(extract_go_total "$ROOT_DIR/sdk/go" "$REPORT_DIR/sdk-go.coverprofile")"

export REPORT_DIR
export CONTROL_PLANE_TOTAL
export SDK_GO_TOTAL
python3 - <<'PY'
import json
import os
from pathlib import Path

report_dir = Path(os.environ["REPORT_DIR"])

with (report_dir / "sdk-python-coverage.json").open() as fh:
    python_data = json.load(fh)

with (report_dir / "sdk-typescript-coverage-summary.json").open() as fh:
    ts_data = json.load(fh)

with (report_dir / "web-ui-coverage-summary.json").open() as fh:
    ui_data = json.load(fh)

surfaces = [
    {
        "name": "control-plane",
        "kind": "go",
        "coverage_percent": float(os.environ["CONTROL_PLANE_TOTAL"]),
        "notes": "go test -tags sqlite_fts5 -coverprofile ./...",
    },
    {
        "name": "sdk-go",
        "kind": "go",
        "coverage_percent": float(os.environ["SDK_GO_TOTAL"]),
        "notes": "go test -coverprofile ./...",
    },
    {
        "name": "sdk-python",
        "kind": "python",
        "coverage_percent": float(python_data["totals"]["percent_covered"]),
        "notes": "pytest coverage over configured tracked modules",
    },
    {
        "name": "sdk-typescript",
        "kind": "typescript",
        "coverage_percent": float(ts_data["total"]["statements"]["pct"]),
        "notes": "vitest v8 coverage over src/**/*.ts via the core suite",
    },
    {
        "name": "web-ui",
        "kind": "typescript",
        "coverage_percent": float(ui_data["total"]["statements"]["pct"]),
        "notes": "vitest v8 coverage over client/src/**/*.{ts,tsx}",
    },
]

def badge_color(pct: float) -> str:
    # Shields.io style thresholds
    if pct >= 90: return "brightgreen"
    if pct >= 80: return "green"
    if pct >= 70: return "yellowgreen"
    if pct >= 60: return "yellow"
    if pct >= 50: return "orange"
    return "red"

# Weighted average across the covered surfaces. Weights chosen to match the
# relative size (lines of source) of each surface so the aggregate isn't
# gamed by a tiny package hitting 100%. Adjust surface_weights below when
# the repo's surface sizes shift materially.
# NOTE: the repo's long-standing convention (see docs/COVERAGE.md) is to
# report per-surface numbers rather than a single blended monorepo score.
# This aggregate is intended as a single convenience signal for the README
# badge ONLY; the per-surface table below remains the source of truth.
surface_weights = {
    "control-plane": 24326,   # go statements
    "sdk-go":         1,       # placeholder, unknown statement count
    "sdk-python":     1,       # placeholder
    "sdk-typescript": 1,       # placeholder
    "web-ui":         41693,   # ts lines
}
total_w = 0.0
total_cov = 0.0
for s in surfaces:
    w = surface_weights.get(s["name"], 1)
    total_w += w
    total_cov += w * s["coverage_percent"]
aggregate_pct = (total_cov / total_w) if total_w else 0.0

summary = {
    "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat().replace("+00:00", "Z"),
    "surfaces": surfaces,
    "aggregate": {
        "coverage_percent": round(aggregate_pct, 2),
        "method": "weighted average over surfaces (control-plane + web-ui dominate)",
        "notes": (
            "Per-surface percentages remain the source of truth; see "
            "docs/COVERAGE.md. This aggregate exists for the README badge only."
        ),
    },
    "badge": {
        "schemaVersion": 1,
        "label": "coverage",
        "message": f"{aggregate_pct:.1f}%",
        "color": badge_color(aggregate_pct),
    },
    "notes": [
        "Functional tests run in a separate Docker-based workflow and are not part of these percentages.",
        "Per-surface numbers remain canonical; the aggregate is a convenience signal.",
    ],
}

(report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(report_dir / "badge.json").write_text(json.dumps(summary["badge"], indent=2) + "\n")

lines = [
    "# Coverage Summary",
    "",
    f"**Aggregate: {aggregate_pct:.2f}%** (weighted average; per-surface numbers below are canonical)",
    "",
    "| Surface | Coverage | Notes |",
    "| --- | ---: | --- |",
]

for surface in surfaces:
    lines.append(
        f"| {surface['name']} | {surface['coverage_percent']:.2f}% | {surface['notes']} |"
    )

lines.extend(
    [
        "",
        "Coverage badge endpoint data is written to `test-reports/coverage/badge.json`.",
        "Functional validation remains separate in `.github/workflows/functional-tests.yml`.",
    ]
)

(report_dir / "summary.md").write_text("\n".join(lines) + "\n")
PY

echo "Coverage artifacts written to $REPORT_DIR"
