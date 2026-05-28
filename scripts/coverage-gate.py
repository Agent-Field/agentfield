#!/usr/bin/env python3
"""Coverage gate for AgentField.

Runs after ``scripts/coverage-summary.sh`` and:

  1. Compares per-surface and aggregate coverage against a checked-in baseline
     (``coverage-baseline.json``).
  2. Applies thresholds from ``.coverage-gate.toml`` at the repo root.
  3. Writes a Markdown report (``test-reports/coverage/gate-report.md``) — this
     is what gets posted as the sticky PR comment.
  4. Writes a machine-readable JSON status (``test-reports/coverage/gate-status.json``)
     — this is the canonical source of truth for AI agents deciding what to do
     next.
  5. Exits non-zero if any rule is violated (unless ``--soft`` is passed).

Design goals:

  * Everything the gate reports has to be explainable from this file alone —
    no hidden thresholds, no magic numbers in the workflow YAML.
  * The report has to be useful to both humans *and* AI coding agents. Humans
    skim the table. Agents parse ``gate-status.json`` and follow the
    remediation plan embedded inline.
  * Failure messages must contain the exact shell command to reproduce the
    surface-level coverage locally. No "figure it out".

Usage::

    ./scripts/coverage-gate.py \\
        --summary  test-reports/coverage/summary.json \\
        --baseline coverage-baseline.json \\
        --config   .coverage-gate.toml

    # Warn-only mode (used while coverage is being brought up):
    ./scripts/coverage-gate.py ... --soft
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

EXIT_OK = 0
EXIT_VIOLATIONS = 1
EXIT_USAGE = 2

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- data helpers -----------------------------------------------------------

def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        print(f"coverage-gate: required file not found: {path}", file=sys.stderr)
        sys.exit(EXIT_USAGE)
    except json.JSONDecodeError as exc:
        print(f"coverage-gate: invalid JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(EXIT_USAGE)


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text())
    except FileNotFoundError:
        print(f"coverage-gate: required file not found: {path}", file=sys.stderr)
        sys.exit(EXIT_USAGE)
    except Exception as exc:  # pragma: no cover
        print(f"coverage-gate: invalid TOML in {path}: {exc}", file=sys.stderr)
        sys.exit(EXIT_USAGE)


def _surface_map(summary: dict[str, Any]) -> dict[str, float]:
    return {
        s["name"]: float(s["coverage_percent"])
        for s in summary.get("surfaces", [])
        if s.get("name") and s.get("coverage_percent") is not None
    }


# --- rendering helpers ------------------------------------------------------

_GLYPHS = {
    "ok":       "🟢",
    "warn":     "🟡",
    "problem":  "🟠",
    "bad":      "🔴",
    "missing":  "⚠️ ",
}


def _status_glyph(pct: float) -> str:
    if pct >= 90:
        return _GLYPHS["ok"]
    if pct >= 80:
        return _GLYPHS["warn"]
    if pct >= 70:
        return _GLYPHS["problem"]
    return _GLYPHS["bad"]


def _delta(cur: float, base: float | None) -> str:
    if base is None:
        return "—"
    d = cur - base
    arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
    return f"{arrow} {d:+.2f} pp"


# --- reproduce commands per surface ----------------------------------------
# These are the exact one-liners an agent should run locally to surface
# uncovered lines for a given regression. Keep them copy-pasteable.

_REPRO = {
    "control-plane": (
        "cd control-plane && "
        "go test -tags sqlite_fts5 -coverprofile=cover.out ./internal/... && "
        "go tool cover -func=cover.out | sort -k3 -n | head -30  # lowest-coverage funcs"
    ),
    "sdk-go": (
        "cd sdk/go && "
        "go test -coverprofile=cover.out ./... && "
        "go tool cover -func=cover.out | sort -k3 -n | head -30"
    ),
    "sdk-python": (
        "cd sdk/python && "
        "python3 -m pytest --cov=agentfield --cov-report=term-missing"
    ),
    "sdk-typescript": (
        "cd sdk/typescript && "
        "npx vitest run --config vitest.config.ts --coverage && "
        "open coverage/index.html  # or cat coverage/coverage-summary.json"
    ),
    "web-ui": (
        "cd control-plane/web/client && "
        "npx vitest run --coverage && "
        "open coverage/index.html  # or cat coverage/coverage-summary.json"
    ),
}


def _reproduce(name: str) -> str:
    return _REPRO.get(name, f"# no reproduce command registered for surface {name!r}")


# --- main -------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the AgentField coverage gate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--summary",  type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--config",   type=Path, default=REPO_ROOT / ".coverage-gate.toml")
    parser.add_argument("--soft",     action="store_true", help="report violations but exit 0")
    args = parser.parse_args(argv)

    summary  = _load_json(args.summary)
    baseline = _load_json(args.baseline)
    config   = _load_toml(args.config)

    thresholds = config.get("thresholds", {})
    min_surface      = float(thresholds.get("min_surface",      80.0))
    min_aggregate    = float(thresholds.get("min_aggregate",    85.0))
    max_surface_drop = float(thresholds.get("max_surface_drop",  1.0))
    max_aggregate_drop = float(thresholds.get("max_aggregate_drop", 0.5))

    surfaces_now  = _surface_map(summary)
    surfaces_base = {k: float(v) for k, v in baseline.get("surfaces", {}).items()}
    agg_now       = float(summary.get("aggregate", {}).get("coverage_percent", 0.0))
    agg_base      = float(baseline.get("aggregate", 0.0))

    # --- evaluate rules -----------------------------------------------------

    violations: list[dict[str, Any]] = []

    for name, pct in sorted(surfaces_now.items()):
        if pct < min_surface:
            violations.append({
                "rule": "min_surface",
                "surface": name,
                "value": pct,
                "threshold": min_surface,
                "message": (
                    f"`{name}` coverage {pct:.2f}% is below the per-surface floor "
                    f"of {min_surface:.0f}%."
                ),
                "reproduce": _reproduce(name),
            })

    if agg_now < min_aggregate:
        violations.append({
            "rule": "min_aggregate",
            "surface": "aggregate",
            "value": agg_now,
            "threshold": min_aggregate,
            "message": (
                f"weighted aggregate coverage {agg_now:.2f}% is below the floor "
                f"of {min_aggregate:.0f}%."
            ),
            "reproduce": "./scripts/coverage-summary.sh && cat test-reports/coverage/summary.md",
        })

    for name, pct in sorted(surfaces_now.items()):
        base = surfaces_base.get(name)
        if base is None:
            continue
        drop = base - pct
        if drop > max_surface_drop:
            violations.append({
                "rule": "max_surface_drop",
                "surface": name,
                "value": pct,
                "baseline": base,
                "drop": drop,
                "threshold": max_surface_drop,
                "message": (
                    f"`{name}` regressed {drop:.2f} pp against baseline "
                    f"({base:.2f}% → {pct:.2f}%). Max allowed drop is {max_surface_drop:.1f} pp."
                ),
                "reproduce": _reproduce(name),
            })

    agg_drop = agg_base - agg_now
    if agg_drop > max_aggregate_drop:
        violations.append({
            "rule": "max_aggregate_drop",
            "surface": "aggregate",
            "value": agg_now,
            "baseline": agg_base,
            "drop": agg_drop,
            "threshold": max_aggregate_drop,
            "message": (
                f"aggregate regressed {agg_drop:.2f} pp against baseline "
                f"({agg_base:.2f}% → {agg_now:.2f}%). Max allowed aggregate drop "
                f"is {max_aggregate_drop:.2f} pp."
            ),
            "reproduce": "./scripts/coverage-summary.sh && cat test-reports/coverage/summary.md",
        })

    passed = not violations

    # --- write JSON status (source of truth for agents) --------------------

    status: dict[str, Any] = {
        "schema_version": 1,
        "passed": passed,
        "generated_at": summary.get("generated_at"),
        "thresholds": {
            "min_surface": min_surface,
            "min_aggregate": min_aggregate,
            "max_surface_drop": max_surface_drop,
            "max_aggregate_drop": max_aggregate_drop,
        },
        "aggregate": {
            "current": agg_now,
            "baseline": agg_base,
            "delta_pp": round(agg_now - agg_base, 4),
        },
        "surfaces": [
            {
                "name": name,
                "current": pct,
                "baseline": surfaces_base.get(name),
                "delta_pp": round(pct - surfaces_base[name], 4) if name in surfaces_base else None,
                "reproduce": _reproduce(name),
            }
            for name, pct in sorted(surfaces_now.items())
        ],
        "violations": violations,
        "remediation": {
            "for_agents": (
                "If `passed` is false, for each object in `violations[]` run the "
                "`reproduce` command for that surface, inspect the lowest-coverage "
                "files, add tests in the same PR, and re-run the gate. Do NOT "
                "lower thresholds in .coverage-gate.toml or numbers in "
                "coverage-baseline.json to silence the gate unless "
                "the regression is intentional and you have explicit human "
                "approval in the PR description."
            ),
            "for_humans": (
                "See the Markdown report in test-reports/coverage/gate-report.md "
                "(posted as a sticky PR comment) for the table view."
            ),
        },
    }

    report_cfg = config.get("report", {})
    out_dir = args.summary.parent
    json_path = out_dir / report_cfg.get("json", "gate-status.json")
    md_path   = out_dir / report_cfg.get("markdown", "gate-report.md")

    json_path.write_text(json.dumps(status, indent=2) + "\n")

    # --- write Markdown report (PR comment + step summary) -----------------

    md: list[str] = []
    md.append("## 📊 Coverage gate")
    md.append("")
    md.append(
        f"Thresholds from [`.coverage-gate.toml`](../../.coverage-gate.toml): "
        f"per-surface ≥ **{min_surface:.0f}%**, aggregate ≥ **{min_aggregate:.0f}%**, "
        f"max per-surface regression ≤ **{max_surface_drop:.1f} pp**, "
        f"max aggregate regression ≤ **{max_aggregate_drop:.2f} pp**."
    )
    md.append("")
    md.append("| Surface | Current | Baseline | Δ | |")
    md.append("| --- | ---: | ---: | ---: | :---: |")
    all_names = sorted(set(list(surfaces_now) + list(surfaces_base)))
    for name in all_names:
        cur = surfaces_now.get(name)
        base = surfaces_base.get(name)
        if cur is None:
            md.append(f"| `{name}` | _missing_ | {base:.2f}% | — | {_GLYPHS['missing']} |")
            continue
        md.append(
            f"| `{name}` | {cur:.2f}% | "
            f"{(f'{base:.2f}%' if base is not None else '_new_')} | "
            f"{_delta(cur, base)} | {_status_glyph(cur)} |"
        )
    md.append(
        f"| **aggregate** | **{agg_now:.2f}%** | **{agg_base:.2f}%** | "
        f"**{_delta(agg_now, agg_base)}** | {_status_glyph(agg_now)} |"
    )
    md.append("")

    if passed:
        md.append("### ✅ Gate passed")
        md.append("")
        md.append("No surface regressed past the allowed threshold and the aggregate stayed above the floor.")
    else:
        md.append("### ❌ Gate failed")
        md.append("")
        md.append(f"{len(violations)} rule violation{'s' if len(violations) != 1 else ''}:")
        md.append("")
        for v in violations:
            md.append(f"- **{v['rule']}** on `{v['surface']}` — {v['message']}")
        md.append("")
        md.append("#### How to fix")
        md.append("")
        md.append("**For AI coding agents** (primary audience of this repo):")
        md.append("")
        md.append("1. Read `test-reports/coverage/gate-status.json`. It is the canonical, machine-readable source of truth for this gate.")
        md.append("2. For every entry in `violations[]`, run its `reproduce` shell command to see the lowest-coverage files on that surface.")
        md.append("3. Add tests for the uncovered code paths **in this same PR**. New tests must exercise real code; don't add no-op tests that only import the module.")
        md.append("4. Re-run `./scripts/coverage-summary.sh && ./scripts/coverage-gate.py --summary test-reports/coverage/summary.json --baseline coverage-baseline.json` locally before pushing.")
        md.append("5. **Do not** lower values in `.coverage-gate.toml` or `coverage-baseline.json` to silence the gate. Those files are the contract between this repo and every agent working in it. Only update them in a PR whose explicit purpose is to move the floor, with the reason documented in the PR description.")
        md.append("")
        md.append("**Reproduce commands by surface:**")
        md.append("")
        md.append("```bash")
        for v in violations:
            if "reproduce" in v:
                md.append(f"# {v['surface']}")
                md.append(v["reproduce"])
                md.append("")
        md.append("```")

    md_path.write_text("\n".join(md) + "\n")
    print("\n".join(md))

    gh_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_summary:
        with open(gh_summary, "a", encoding="utf-8") as fh:
            fh.write("\n".join(md) + "\n")

    if violations and not args.soft:
        return EXIT_VIOLATIONS
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
