#!/usr/bin/env python3
"""Aggregate per-surface coverage outputs into summary.json/summary.md/badge.json.

Reads ``test-reports/coverage/`` (produced by scripts/coverage-surface.sh
for each of the five tracked surfaces) and writes:

  * ``summary.json``  — full surface breakdown + aggregate + badge block
  * ``summary.md``    — human-readable table (published to the GitHub Step
                        Summary by the coverage workflow)
  * ``badge.json``    — shields.io endpoint format consumed by the README
                        coverage badge gist

Consumed downstream by:

  * scripts/coverage-gate.py        — reads ``summary.json``
  * scripts/patch-coverage-gate.sh  — reads per-surface cobertura XMLs, not
                                      this script's output
  * .github/workflows/coverage.yml  — cats ``summary.md`` into
                                      ``$GITHUB_STEP_SUMMARY``

Inputs expected in ``test-reports/coverage/``:

  * ``control-plane.total.txt``                 (plain float, Go total %)
  * ``sdk-go.total.txt``                        (plain float, Go total %)
  * ``sdk-python-coverage.json``                (pytest-cov JSON)
  * ``sdk-typescript-coverage-summary.json``    (vitest v8 summary)
  * ``web-ui-coverage-summary.json``            (vitest v8 summary)

This module was extracted from the trailing Python block of
``scripts/coverage-summary.sh`` so the aggregation step can run
independently after the CI matrix downloads per-surface artifacts into a
single directory.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any


# Weights match the relative source size of each surface (lines /
# statements), so a tiny helper package hitting 100% cannot move the
# aggregate. Kept in sync with .coverage-gate.toml:[weights]. Regenerate
# when surface sizes shift materially.
_SURFACE_WEIGHTS: dict[str, int] = {
    "control-plane": 24326,
    "sdk-go":         1,
    "sdk-python":     1,
    "sdk-typescript": 1,
    "web-ui":         41693,
}


def _badge_color(pct: float) -> str:
    """Shields.io-style threshold colors."""
    if pct >= 90: return "brightgreen"
    if pct >= 80: return "green"
    if pct >= 70: return "yellowgreen"
    if pct >= 60: return "yellow"
    if pct >= 50: return "orange"
    return "red"


def _read_total(path: Path) -> float:
    return float(path.read_text().strip())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def aggregate(report_dir: Path) -> dict[str, Any]:
    cp_total = _read_total(report_dir / "control-plane.total.txt")
    sg_total = _read_total(report_dir / "sdk-go.total.txt")
    py_data  = _read_json(report_dir / "sdk-python-coverage.json")
    ts_data  = _read_json(report_dir / "sdk-typescript-coverage-summary.json")
    ui_data  = _read_json(report_dir / "web-ui-coverage-summary.json")

    surfaces = [
        {
            "name": "control-plane",
            "kind": "go",
            "coverage_percent": cp_total,
            "notes": "go test -tags sqlite_fts5 -coverprofile ./...",
        },
        {
            "name": "sdk-go",
            "kind": "go",
            "coverage_percent": sg_total,
            "notes": "go test -coverprofile ./...",
        },
        {
            "name": "sdk-python",
            "kind": "python",
            "coverage_percent": float(py_data["totals"]["percent_covered"]),
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

    total_w = 0.0
    total_cov = 0.0
    for s in surfaces:
        w = _SURFACE_WEIGHTS.get(s["name"], 1)
        total_w += w
        total_cov += w * s["coverage_percent"]
    aggregate_pct = (total_cov / total_w) if total_w else 0.0

    now = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")

    return {
        "generated_at": now,
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
            "color": _badge_color(aggregate_pct),
        },
        "notes": [
            "Functional tests run in a separate Docker-based workflow and are not part of these percentages.",
            "Per-surface numbers remain canonical; the aggregate is a convenience signal.",
        ],
    }


def write_outputs(report_dir: Path, summary: dict[str, Any]) -> None:
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (report_dir / "badge.json").write_text(json.dumps(summary["badge"], indent=2) + "\n")

    agg = summary["aggregate"]["coverage_percent"]
    lines = [
        "# Coverage Summary",
        "",
        f"**Aggregate: {agg:.2f}%** (weighted average; per-surface numbers below are canonical)",
        "",
        "| Surface | Coverage | Notes |",
        "| --- | ---: | --- |",
    ]
    for s in summary["surfaces"]:
        lines.append(
            f"| {s['name']} | {s['coverage_percent']:.2f}% | {s['notes']} |"
        )
    lines.extend(
        [
            "",
            "Coverage badge endpoint data is written to `test-reports/coverage/badge.json`.",
            "Functional validation remains separate in `.github/workflows/functional-tests.yml`.",
        ]
    )
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("test-reports/coverage"),
        help="Directory containing per-surface coverage outputs (default: test-reports/coverage).",
    )
    args = parser.parse_args(argv)

    report_dir = args.report_dir.resolve()
    if not report_dir.is_dir():
        print(f"coverage-aggregate: not a directory: {report_dir}", file=sys.stderr)
        return 2

    try:
        summary = aggregate(report_dir)
    except FileNotFoundError as exc:
        print(f"coverage-aggregate: missing input: {exc.filename}", file=sys.stderr)
        print(
            "hint: run ./scripts/coverage-surface.sh for each of the five surfaces, "
            "or ./scripts/coverage-summary.sh to run them all sequentially.",
            file=sys.stderr,
        )
        return 2
    write_outputs(report_dir, summary)
    print(
        f"coverage-aggregate: wrote summary.json/summary.md/badge.json to {report_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
