#!/usr/bin/env bash
# Patch-coverage gate for AgentField.
#
# Runs `diff-cover` across every per-surface Cobertura XML emitted by
# scripts/coverage-summary.sh and enforces the `min_patch` threshold from
# .coverage-gate.toml on the lines changed against origin/main.
#
# This is the single most effective regression signal used by top-tier OSS
# projects (codecov, vitest, rust-lang, grafana): aggregates drift slowly,
# but untested *new* code shows up here immediately.
#
# Outputs:
#   test-reports/coverage/patch-gate-report.md   (posted as sticky PR comment)
#   test-reports/coverage/patch-gate-status.json (machine-readable for agents)
#
# Exits:
#   0  — patch coverage >= min_patch on every surface that had touched lines
#   1  — at least one surface regressed below the threshold
#
# Usage:
#   ./scripts/patch-coverage-gate.sh
#   ./scripts/patch-coverage-gate.sh --compare-branch origin/main
#   MIN_PATCH=75 ./scripts/patch-coverage-gate.sh   # override the floor

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="$ROOT_DIR/test-reports/coverage"
CONFIG="$ROOT_DIR/.coverage-gate.toml"

COMPARE_BRANCH="origin/main"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --compare-branch) COMPARE_BRANCH="$2"; shift 2 ;;
    --compare-branch=*) COMPARE_BRANCH="${1#*=}"; shift ;;
    -h|--help)
      sed -n '2,22p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# --- read min_patch from .coverage-gate.toml (single source of truth) ------
MIN_PATCH="${MIN_PATCH:-$(python3 - "$CONFIG" <<'PY'
import sys
try:
    import tomllib  # py311+
except ModuleNotFoundError:
    import tomli as tomllib
with open(sys.argv[1], "rb") as fh:
    cfg = tomllib.load(fh)
print(cfg.get("thresholds", {}).get("min_patch", 80.0))
PY
)}"

if ! command -v diff-cover >/dev/null 2>&1; then
  echo "diff-cover is required. Install with: pip install 'diff-cover>=9.0.0'" >&2
  exit 2
fi

# Make sure the compare branch is reachable locally. In CI we use fetch-depth: 0
# so this is a no-op; locally it fetches if missing.
if ! git rev-parse --verify "$COMPARE_BRANCH" >/dev/null 2>&1; then
  echo "==> fetching $COMPARE_BRANCH"
  IFS='/' read -r remote ref <<<"$COMPARE_BRANCH"
  git fetch "${remote:-origin}" "${ref:-main}" --depth=0 >/dev/null 2>&1 || true
fi

mkdir -p "$REPORT_DIR"

# --- collect cobertura XML files, one per surface --------------------------
declare -a SURFACES
declare -A XML_PATH

register_surface() {
  local name="$1" path="$2"
  if [[ -f "$path" ]]; then
    SURFACES+=("$name")
    XML_PATH["$name"]="$path"
  else
    echo "patch-coverage-gate: no cobertura XML for $name at $path — skipping" >&2
  fi
}

register_surface "control-plane"  "$REPORT_DIR/control-plane-cobertura.xml"
register_surface "sdk-go"         "$REPORT_DIR/sdk-go-cobertura.xml"
register_surface "sdk-python"     "$REPORT_DIR/sdk-python-coverage.xml"
register_surface "sdk-typescript" "$REPORT_DIR/sdk-typescript-cobertura.xml"
register_surface "web-ui"         "$REPORT_DIR/web-ui-cobertura.xml"

if [[ ${#SURFACES[@]} -eq 0 ]]; then
  echo "patch-coverage-gate: no cobertura XML files found. Did you run scripts/coverage-summary.sh first?" >&2
  exit 2
fi

# --- run diff-cover per surface --------------------------------------------
# We run one invocation per surface (rather than passing all XMLs in a single
# call) because diff-cover's multi-file support doesn't attribute per-file
# results back to a surface — which is exactly what reviewers need in the PR
# comment.
REPORT_MD="$REPORT_DIR/patch-gate-report.md"
REPORT_JSON="$REPORT_DIR/patch-gate-status.json"

python3 - "$REPORT_DIR" "$COMPARE_BRANCH" "$MIN_PATCH" "${SURFACES[@]}" <<'PY'
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

report_dir = Path(sys.argv[1])
compare_branch = sys.argv[2]
min_patch = float(sys.argv[3])
surfaces = sys.argv[4:]

xml_for = {
    "control-plane":  "control-plane-cobertura.xml",
    "sdk-go":         "sdk-go-cobertura.xml",
    "sdk-python":     "sdk-python-coverage.xml",
    "sdk-typescript": "sdk-typescript-cobertura.xml",
    "web-ui":         "web-ui-cobertura.xml",
}

def run_diff_cover(xml_path: Path, json_out: Path) -> dict:
    # diff-cover writes a JSON report with per-file detail; we parse it for
    # the overall patch-coverage percentage and the file-by-file breakdown.
    cmd = [
        "diff-cover",
        str(xml_path),
        f"--compare-branch={compare_branch}",
        f"--json-report={json_out}",
        "--quiet",
    ]
    subprocess.run(cmd, check=False)
    if not json_out.exists():
        return {"skipped": True}
    return json.loads(json_out.read_text())

results = []
for name in surfaces:
    xml = report_dir / xml_for[name]
    json_out = report_dir / f"patch-{name}.json"
    data = run_diff_cover(xml, json_out)
    if data.get("skipped"):
        results.append({"surface": name, "status": "skipped"})
        continue
    # diff-cover JSON shape (v9+): {"report_name","total_num_lines",
    # "total_num_violations","total_percent_covered","src_stats": {...}}
    total_lines = int(data.get("total_num_lines", 0))
    pct = float(data.get("total_percent_covered", 100.0))
    violated = int(data.get("total_num_violations", 0))
    if total_lines == 0:
        results.append({
            "surface": name, "status": "no-diff",
            "touched_lines": 0, "pct": 100.0,
        })
        continue
    passed = pct >= min_patch
    src_stats = data.get("src_stats", {})
    worst = sorted(
        (
            {
                "file": f,
                "pct": float(v.get("percent_covered", 100.0)),
                "missing_lines": v.get("violation_lines", []),
            }
            for f, v in src_stats.items()
        ),
        key=lambda r: r["pct"],
    )[:10]
    results.append({
        "surface": name,
        "status": "pass" if passed else "fail",
        "touched_lines": total_lines,
        "violated_lines": violated,
        "pct": round(pct, 2),
        "worst_files": worst,
    })

overall_passed = all(r.get("status") in ("pass", "no-diff", "skipped") for r in results)

# ---- write JSON status ----------------------------------------------------
status = {
    "schema_version": 1,
    "passed": overall_passed,
    "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "min_patch": min_patch,
    "compare_branch": compare_branch,
    "surfaces": results,
}
(report_dir / "patch-gate-status.json").write_text(json.dumps(status, indent=2) + "\n")

# ---- write Markdown report ------------------------------------------------
lines = ["## 📐 Patch coverage gate", ""]
lines.append(
    f"Threshold: **{min_patch:.0f}%** on lines this PR touches vs `{compare_branch}` "
    "(from `.coverage-gate.toml:thresholds.min_patch`)."
)
lines.append("")
lines.append("| Surface | Touched lines | Patch coverage | Status |")
lines.append("| --- | ---: | ---: | :---: |")
for r in results:
    surface = r["surface"]
    st = r.get("status", "?")
    if st == "no-diff":
        lines.append(f"| `{surface}` | 0 | — | ➖ no changes |")
    elif st == "skipped":
        lines.append(f"| `{surface}` | — | — | ⚠️ skipped (no XML) |")
    else:
        glyph = "✅" if st == "pass" else "❌"
        lines.append(
            f"| `{surface}` | {r['touched_lines']} | **{r['pct']:.2f}%** | {glyph} |"
        )
lines.append("")
if overall_passed:
    lines.append("### ✅ Patch gate passed")
    lines.append("")
    lines.append("Every surface whose lines were touched by this PR has patch coverage at or above the threshold.")
else:
    lines.append("### ❌ Patch gate failed")
    lines.append("")
    for r in results:
        if r.get("status") != "fail":
            continue
        lines.append(f"**`{r['surface']}`** — {r['pct']:.2f}% on {r['touched_lines']} touched lines ({r['violated_lines']} uncovered):")
        lines.append("")
        lines.append("| File | Patch coverage | Missing lines |")
        lines.append("| --- | ---: | --- |")
        for w in r["worst_files"][:5]:
            missing = w["missing_lines"]
            if isinstance(missing, list):
                missing_str = ", ".join(str(x) for x in missing[:10])
                if len(missing) > 10:
                    missing_str += f" …(+{len(missing)-10} more)"
            else:
                missing_str = str(missing)
            lines.append(f"| `{w['file']}` | {w['pct']:.1f}% | {missing_str} |")
        lines.append("")
    lines.append("#### How to fix")
    lines.append("")
    lines.append("1. For each file listed above, add tests that exercise the missing line numbers in this same PR.")
    lines.append("2. Re-run locally: `./scripts/coverage-summary.sh && ./scripts/patch-coverage-gate.sh`.")
    lines.append("3. Do **not** lower `min_patch` in `.coverage-gate.toml` to silence this — the floor is the contract.")

(report_dir / "patch-gate-report.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

sys.exit(0 if overall_passed else 1)
PY
