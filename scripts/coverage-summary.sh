#!/usr/bin/env bash
# Local one-shot coverage runner.
#
# Runs every tracked surface's tests with coverage and produces the full
# aggregate report under test-reports/coverage/. Identical output to the CI
# job in .github/workflows/coverage.yml — CI just parallelizes the surfaces
# across matrix jobs, while this script runs them sequentially on your
# machine.
#
# Delegates per-surface logic to scripts/coverage-surface.sh so there is a
# single source of truth for "what commands run for surface X with
# coverage". Do not add per-surface commands here — put them in
# coverage-surface.sh so CI picks them up automatically.
#
# Usage:
#   ./scripts/coverage-summary.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="$ROOT_DIR/test-reports/coverage"

rm -rf "$REPORT_DIR"
mkdir -p "$REPORT_DIR"

SURFACES=(control-plane sdk-go sdk-python sdk-typescript web-ui)

for surface in "${SURFACES[@]}"; do
  "$ROOT_DIR/scripts/coverage-surface.sh" "$surface"
done

python3 "$ROOT_DIR/scripts/coverage-aggregate.py" --report-dir "$REPORT_DIR"

echo "Coverage artifacts written to $REPORT_DIR"
