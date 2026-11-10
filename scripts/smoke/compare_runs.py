#!/usr/bin/env python3
"""Compare new vs baseline SWE-AF plain-build runs and enforce the smoke gate."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REQUIRED_BUILD_RESULT_KEYS = ("plan_result", "dag_state", "success", "summary")

_SECRET_PATTERNS = (
    re.compile(r"sk-or-v1-[A-Za-z0-9._-]+"),
    re.compile(r"(?i)(x-api-key\s*[:=]\s*)([^\s\"',}]+)"),
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)([^\s\"',}]+)"),
    re.compile(r"(?i)(openrouter_api_key\s*[:=]\s*)([^\s\"',}]+)"),
    re.compile(r"(?i)(agentfield_api_key\s*[:=]\s*)([^\s\"',}]+)"),
)

_LOG_ERROR_RE = re.compile(
    r"(?:"
    r"\b(?:ERROR|CRITICAL|FATAL|PANIC)\b"
    r"|\"level\"\s*:\s*\"(?:error|fatal|panic)\""
    r"|\blevel=(?:error|fatal|panic)\b"
    r"|Traceback \(most recent call last\)"
    r"|\b(?:FatalHarnessError|EmptyHarnessCompletionError|HarnessTimeoutError|ValidationError|SyntaxError)\b"
    r"|\bagent call failed:"
    r"|\bagent error \(5\d\d\)"
    r")"
)

_BENIGN_LOG_SUBSTRINGS = (
    "No config file found, using environment variables and defaults",
    "UI source path",
    "Skipping UI build",
)


def redact_secrets(text: str) -> str:
    """Scrub any secret values or token shapes from *text*."""
    if not text:
        return ""
    out = text
    for env_name in ("OPENROUTER_API_KEY", "AGENTFIELD_API_KEY"):
        secret = os.environ.get(env_name, "").strip()
        if secret and len(secret) >= 6:
            out = out.replace(secret, "[REDACTED]")
    out = _SECRET_PATTERNS[0].sub("[REDACTED_OPENROUTER_KEY]", out)
    for pat in _SECRET_PATTERNS[1:]:
        out = pat.sub(r"\1[REDACTED]", out)
    return out


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return redact_secrets(path.read_text(encoding="utf-8", errors="replace")).strip()


def normalize_error_signature(line: str) -> str:
    """Normalize dynamic IDs, timestamps, and numbers in a log error line."""
    s = redact_secrets(line.strip())
    # Strip ANSI escapes
    s = re.sub(r"\x1b\[[0-9;]*m", "", s)
    # Timestamps
    s = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?", "<TS>", s)
    s = re.sub(r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b", "<TIME>", s)
    # AgentField execution/run/workflow IDs & hex hashes
    s = re.sub(r"\b(?:exec|run|wf)_[A-Za-z0-9_-]+\b", "<AF_ID>", s)
    s = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "<UUID>", s, flags=re.I)
    s = re.sub(r"\bfeature/[0-9a-f]{6,12}-[a-z0-9-]+\b", "feature/<BUILD>-<SLUG>", s)
    s = re.sub(r"\bissue/[0-9a-f]{6,12}-[a-z0-9-]+\b", "issue/<BUILD>-<SLUG>", s)
    s = re.sub(r"\b[0-9a-f]{8,40}\b", "<HEX>", s)
    s = re.sub(r"\b\d+(?:\.\d+)?(?:ms|s|m)\b", "<DUR>", s)
    return s[:240]


def extract_error_lines(log_text: str) -> list[str]:
    """Extract sanitized error lines from a container log."""
    if not log_text:
        return []
    hits: list[str] = []
    for raw in log_text.splitlines():
        line = redact_secrets(raw.rstrip())
        if not line:
            continue
        if any(b in line for b in _BENIGN_LOG_SUBSTRINGS):
            continue
        if _LOG_ERROR_RE.search(line):
            hits.append(line[:400])
    return hits


def _extract_pr_url_or_reason(
    build_result: dict[str, Any],
    notes_payload: dict[str, Any],
    origin_refs: str,
) -> str:
    pr_url = str(build_result.get("pr_url") or "").strip()
    if pr_url:
        return pr_url

    reasons: list[str] = []
    pr_results = build_result.get("pr_results")
    if isinstance(pr_results, list):
        for entry in pr_results:
            if isinstance(entry, dict):
                url = str(entry.get("pr_url") or "").strip()
                if url:
                    return url
                err = str(entry.get("error_message") or "").strip()
                if err:
                    reasons.append(err)

    notes = notes_payload.get("notes")
    if isinstance(notes, list):
        for note in notes:
            if not isinstance(note, dict):
                continue
            tags = note.get("tags") or []
            msg = str(note.get("message") or "").strip()
            if "github_pr" in tags and msg:
                reasons.append(msg)

    pushed_branches = [
        line.split()[0]
        for line in origin_refs.splitlines()
        if line.strip() and not line.strip().startswith("main ")
    ]
    push_note = (
        f"pushed {', '.join(pushed_branches)} to local bare origin"
        if pushed_branches
        else "no feature branch pushed to local bare origin"
    )

    if reasons:
        return redact_secrets(f"none ({push_note}; {reasons[-1][:220]})")
    return redact_secrets(
        f"none ({push_note}; no GH_TOKEN configured and origin is a local bare repo)"
    )


def summarize_run(run_dir: Path, default_label: str) -> dict[str, Any]:
    meta = _read_json(run_dir / "meta.json")
    exec_resp = _read_json(run_dir / "execution_status.json")
    notes_resp = _read_json(run_dir / "execution_notes.json")

    git_log = _read_text(run_dir / "git_log.txt")
    git_diff_stat = _read_text(run_dir / "git_diff_stat.txt")
    git_status = _read_text(run_dir / "git_status.txt")
    origin_refs = _read_text(run_dir / "origin_refs.txt")
    pytest_output = _read_text(run_dir / "pytest_output.txt")
    node_log = _read_text(run_dir / "node.log")
    cp_log = _read_text(run_dir / "control-plane.log")

    registered = bool(meta.get("registered", False))
    dispatched = bool(meta.get("dispatched", False))
    timed_out_polling = bool(meta.get("timed_out_polling", False))
    container_alive = bool(meta.get("container_alive_at_end", False))
    image_built = bool(meta.get("image_built", True))

    raw_status = str(exec_resp.get("status") or meta.get("execution_status") or "").strip().lower()
    if not image_built:
        execution_status = "image_build_failed"
    elif not registered:
        execution_status = "not_registered"
    elif not dispatched:
        execution_status = "dispatch_failed"
    elif timed_out_polling and raw_status not in ("succeeded", "failed", "cancelled", "timeout"):
        execution_status = "poll_timeout"
    else:
        execution_status = raw_status or "unknown"

    execution_error = redact_secrets(str(exec_resp.get("error") or meta.get("dispatch_error") or "").strip())

    # BuildResult lives in `result` on succeeded executions, and in `result` or
    # `error_details` when SWE-AF raises ReasonerFailed(result=build_result).
    raw_result = exec_resp.get("result")
    error_details = exec_resp.get("error_details")
    build_result: dict[str, Any] = {}
    if isinstance(raw_result, dict) and any(k in raw_result for k in REQUIRED_BUILD_RESULT_KEYS):
        build_result = raw_result
    elif isinstance(error_details, dict):
        if any(k in error_details for k in REQUIRED_BUILD_RESULT_KEYS):
            build_result = error_details
        elif isinstance(error_details.get("result"), dict):
            build_result = error_details["result"]
    elif isinstance(raw_result, dict):
        build_result = raw_result

    has_valid_build_schema = all(k in build_result for k in REQUIRED_BUILD_RESULT_KEYS)
    build_success = build_result.get("success") if "success" in build_result else None
    build_summary = redact_secrets(str(build_result.get("summary") or "").strip())

    plan_result = build_result.get("plan_result") if isinstance(build_result.get("plan_result"), dict) else {}
    dag_state = build_result.get("dag_state") if isinstance(build_result.get("dag_state"), dict) else {}
    verification = build_result.get("verification") if isinstance(build_result.get("verification"), dict) else {}

    planned_issues = plan_result.get("issues") if isinstance(plan_result.get("issues"), list) else []
    completed_issues = dag_state.get("completed_issues") if isinstance(dag_state.get("completed_issues"), list) else []
    failed_issues = dag_state.get("failed_issues") if isinstance(dag_state.get("failed_issues"), list) else []
    skipped_issues = dag_state.get("skipped_issues") if isinstance(dag_state.get("skipped_issues"), list) else []
    merged_branches = dag_state.get("merged_branches") if isinstance(dag_state.get("merged_branches"), list) else []

    verification_passed = verification.get("passed") if "passed" in verification else None
    verification_summary = redact_secrets(str(verification.get("summary") or "").strip())

    pr_url_or_reason = _extract_pr_url_or_reason(build_result, notes_resp, origin_refs)

    pytest_rc = meta.get("pytest_rc")
    pytest_passed = (pytest_rc == 0)
    pytest_last_line = pytest_output.splitlines()[-1].strip() if pytest_output.splitlines() else "(no output)"

    slugify_probe = str(meta.get("slugify_probe") or "unknown").strip()
    new_commits_count = int(meta.get("new_commits_count") or 0)

    node_error_lines = extract_error_lines(node_log)
    cp_error_lines = extract_error_lines(cp_log)
    (run_dir / "error_lines.txt").write_text(
        "=== node error lines ===\n"
        + ("\n".join(node_error_lines) if node_error_lines else "(none)")
        + "\n\n=== control-plane error lines ===\n"
        + ("\n".join(cp_error_lines) if cp_error_lines else "(none)")
        + "\n",
        encoding="utf-8",
    )

    node_error_sigs = sorted({normalize_error_signature(line) for line in node_error_lines})
    cp_error_sigs = sorted({normalize_error_signature(line) for line in cp_error_lines})

    # Classify terminal failure classes per SPEC-REALBUILD.md section 4:
    # crash, refusal, schema_error, registration_error, timeout.
    failure_classes: set[str] = set()
    combined_err_blob = " ".join(
        [
            execution_error,
            json.dumps(error_details, default=str) if error_details else "",
            " ".join(
                str(i.get("error_message") or "")
                for i in failed_issues
                if isinstance(i, dict)
            ),
        ]
    ).lower()

    if not image_built:
        failure_classes.add("crash")
    if not registered or not dispatched:
        failure_classes.add("registration_error")
    if (
        timed_out_polling
        or execution_status in ("timeout", "poll_timeout")
        or "harnesstimeouterror" in combined_err_blob
        or "stale_execution_timeout" in combined_err_blob
    ):
        failure_classes.add("timeout")

    if any(
        tok in combined_err_blob
        for tok in (
            "validationerror",
            "extra_forbidden",
            "legacy config keys",
            "invalid request body",
            "failure_type=schema",
        )
    ) or (execution_status == "succeeded" and not has_valid_build_schema):
        failure_classes.add("schema_error")

    if any(
        tok in combined_err_blob
        for tok in (
            "refused",
            "refusal",
            "fence violation",
            "swe_fence",
            "permission denied by policy",
        )
    ):
        failure_classes.add("refusal")

    if (
        (registered and dispatched and not container_alive)
        or execution_status in ("failed", "cancelled", "unknown")
        or "fatalharnesserror" in combined_err_blob
        or "emptyharnesscompletionerror" in combined_err_blob
        or (execution_status == "succeeded" and len(completed_issues) == 0 and len(merged_branches) == 0)
    ):
        failure_classes.add("crash")

    return {
        "label": meta.get("label", default_label),
        "image_tag": meta.get("image_tag", ""),
        "commit_sha": meta.get("commit_sha", ""),
        "image_built": image_built,
        "registered": registered,
        "dispatched": dispatched,
        "container_alive_at_end": container_alive,
        "timed_out_polling": timed_out_polling,
        "execution_id": meta.get("execution_id", exec_resp.get("execution_id", "")),
        "run_id": meta.get("run_id", exec_resp.get("run_id", "")),
        "wall_seconds": meta.get("wall_seconds"),
        "duration_ms": exec_resp.get("duration_ms"),
        "harness_model": meta.get("harness_model", ""),
        "agentfield_sdk_version": meta.get("agentfield_sdk_version", ""),
        "command_code_version": meta.get("command_code_version", ""),
        "opencode_version": meta.get("opencode_version", ""),
        "execution_status": execution_status,
        "execution_error": execution_error,
        "has_valid_build_schema": has_valid_build_schema,
        "build_success": build_success,
        "build_summary": build_summary,
        "planned_issues_count": len(planned_issues),
        "planned_issue_names": [str(i.get("name", "?")) for i in planned_issues if isinstance(i, dict)],
        "completed_issues_count": len(completed_issues),
        "completed_issue_names": [
            f"{i.get('issue_name', '?')} ({i.get('outcome', '?')})"
            for i in completed_issues
            if isinstance(i, dict)
        ],
        "failed_issues_count": len(failed_issues),
        "failed_issue_names": [
            f"{i.get('issue_name', '?')} ({i.get('outcome', '?')})"
            for i in failed_issues
            if isinstance(i, dict)
        ],
        "skipped_issues_count": len(skipped_issues),
        "merged_branches_count": len(merged_branches),
        "merged_branches": [str(b) for b in merged_branches],
        "verification_passed": verification_passed,
        "verification_summary": verification_summary,
        "pr_url_or_reason": pr_url_or_reason,
        "checked_out_branch": meta.get("checked_out_branch", ""),
        "new_commits_count": new_commits_count,
        "git_log": git_log,
        "git_diff_stat": git_diff_stat,
        "git_status": git_status,
        "origin_refs": origin_refs,
        "pytest_rc": pytest_rc,
        "pytest_passed": pytest_passed,
        "pytest_last_line": pytest_last_line,
        "pytest_output": pytest_output,
        "slugify_probe": slugify_probe,
        "node_error_count": len(node_error_lines),
        "cp_error_count": len(cp_error_lines),
        "node_error_signatures": node_error_sigs,
        "cp_error_signatures": cp_error_sigs,
        "failure_classes": sorted(failure_classes),
    }


def evaluate_gate(new_summary: dict[str, Any], base_summary: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
    """Return (passed, failure_reasons, quality_notes) per SPEC-REALBUILD.md."""
    failures: list[str] = []
    notes: list[str] = []

    if not new_summary["image_built"]:
        failures.append("New image failed to build (`docker build` failed).")
    if not new_summary["registered"]:
        failures.append("New image failed to register `swe-planner` with `agentfield/control-plane:v0.1.57`.")
    if not new_summary["dispatched"]:
        failures.append("New image could not accept `swe-planner.build` execution dispatch.")

    new_classes = set(new_summary["failure_classes"])
    base_classes = set(base_summary["failure_classes"])
    regressions = sorted(new_classes - base_classes)
    if regressions:
        failures.append(
            "New image ended in failure class(es) absent from baseline: "
            + ", ".join(f"`{c}`" for c in regressions)
            + f" (new={sorted(new_classes)}, baseline={sorted(base_classes)})."
        )

    # Non-failing model-quality / output differences when both completed
    if not failures:
        if new_summary["build_success"] != base_summary["build_success"]:
            notes.append(
                f"Model-quality difference in `build.success`: new={new_summary['build_success']} "
                f"vs baseline={base_summary['build_success']}."
            )
        if new_summary["verification_passed"] != base_summary["verification_passed"]:
            notes.append(
                f"Model-quality difference in `verification.passed`: new={new_summary['verification_passed']} "
                f"vs baseline={base_summary['verification_passed']}."
            )
        if new_summary["pytest_passed"] != base_summary["pytest_passed"]:
            notes.append(
                f"Workspace `pytest -q` differed: new={'PASS' if new_summary['pytest_passed'] else 'FAIL'} "
                f"vs baseline={'PASS' if base_summary['pytest_passed'] else 'FAIL'}."
            )
        if new_summary["slugify_probe"] != base_summary["slugify_probe"]:
            notes.append(
                f"`slugify` behavioral probe differed: new=`{new_summary['slugify_probe']}` "
                f"vs baseline=`{base_summary['slugify_probe']}`."
            )

    return (len(failures) == 0), failures, notes


def _md_cell(value: Any) -> str:
    text = str(value if value is not None else "n/a")
    return redact_secrets(text).replace("|", "\\|").replace("\n", "<br>")


def render_markdown(
    new_s: dict[str, Any],
    base_s: dict[str, Any],
    passed: bool,
    failures: list[str],
    notes: list[str],
) -> str:
    verdict = "PASS" if passed else "FAIL"
    new_only_node_sigs = sorted(set(new_s["node_error_signatures"]) - set(base_s["node_error_signatures"]))
    new_only_cp_sigs = sorted(set(new_s["cp_error_signatures"]) - set(base_s["cp_error_signatures"]))

    rows = [
        ("Commit SHA", f"`{new_s['commit_sha'][:12]}`", f"`{base_s['commit_sha'][:12]}` ( command-code@1.88.5 sed )"),
        ("AgentField SDK / Command Code", f"`{new_s['agentfield_sdk_version']}` / `{new_s['command_code_version']}`", f"`{base_s['agentfield_sdk_version']}` / `{base_s['command_code_version']}`"),
        ("HARNESS_MODEL", f"`{new_s['harness_model']}`", f"`{base_s['harness_model']}`"),
        ("Registered & Dispatched", f"`{new_s['registered']}` / `{new_s['dispatched']}`", f"`{base_s['registered']}` / `{base_s['dispatched']}`"),
        ("Final Execution Status", f"`{new_s['execution_status']}`", f"`{base_s['execution_status']}`"),
        ("Failure Classes", f"`{new_s['failure_classes'] or ['none']}`", f"`{base_s['failure_classes'] or ['none']}`"),
        ("Wall-Clock Seconds", f"`{new_s['wall_seconds']}`", f"`{base_s['wall_seconds']}`"),
        ("Build `success`", f"`{new_s['build_success']}`", f"`{base_s['build_success']}`"),
        ("Planned Issues", f"{new_s['planned_issues_count']} (`{', '.join(new_s['planned_issue_names']) or 'none'}`)", f"{base_s['planned_issues_count']} (`{', '.join(base_s['planned_issue_names']) or 'none'}`)"),
        ("Completed Issues", f"{new_s['completed_issues_count']} (`{', '.join(new_s['completed_issue_names']) or 'none'}`)", f"{base_s['completed_issues_count']} (`{', '.join(base_s['completed_issue_names']) or 'none'}`)"),
        ("Failed / Skipped Issues", f"{new_s['failed_issues_count']} / {new_s['skipped_issues_count']}", f"{base_s['failed_issues_count']} / {base_s['skipped_issues_count']}"),
        ("Merged Branches", f"{new_s['merged_branches_count']} (`{', '.join(new_s['merged_branches']) or 'none'}`)", f"{base_s['merged_branches_count']} (`{', '.join(base_s['merged_branches']) or 'none'}`)"),
        ("Verification (`passed`)", f"`{new_s['verification_passed']}` — {new_s['verification_summary'][:140]}", f"`{base_s['verification_passed']}` — {base_s['verification_summary'][:140]}"),
        ("PR URL / Reason None", new_s["pr_url_or_reason"], base_s["pr_url_or_reason"]),
        ("Workspace Branch & New Commits", f"`{new_s['checked_out_branch']}` (+{new_s['new_commits_count']})", f"`{base_s['checked_out_branch']}` (+{base_s['new_commits_count']})"),
        ("Workspace `pytest -q`", f"{'PASS' if new_s['pytest_passed'] else 'FAIL'} (`rc={new_s['pytest_rc']}`: {new_s['pytest_last_line']})", f"{'PASS' if base_s['pytest_passed'] else 'FAIL'} (`rc={base_s['pytest_rc']}`: {base_s['pytest_last_line']})"),
        ("`slugify` Probe", f"`{new_s['slugify_probe']}`", f"`{base_s['slugify_probe']}`"),
        ("Error Lines (node / control-plane)", f"{new_s['node_error_count']} / {new_s['cp_error_count']}", f"{base_s['node_error_count']} / {base_s['cp_error_count']}"),
    ]

    lines: list[str] = [
        f"## SWE-AF #181 Plain Build Smoke Comparison: **{verdict}**",
        "",
        "Control plane: `agentfield/control-plane:v0.1.57` (local SQLite storage, throwaway `AGENTFIELD_API_KEY`)  ",
        "Target reasoner: `swe-planner.build` (`SWE_DEFAULT_RUNTIME=open_code`, image default `HARNESS_MODEL`, no `SWE_FENCE` / `SWE_STRICT_BUILD` / `GH_TOKEN`)",
        "",
    ]

    if failures:
        lines.append("### Gate Failures")
        for f in failures:
            lines.append(f"- {redact_secrets(f)}")
        lines.append("")

    if notes:
        lines.append("### Model-Quality / Output Observations (Non-Failing)")
        for n in notes:
            lines.append(f"- {redact_secrets(n)}")
        lines.append("")

    lines.extend(
        [
            "### Side-by-Side Results",
            "",
            "| Field | New (`9501e897...` / PR #181) | Baseline (`da78a77d...` + CC 1.88.5) |",
            "| --- | --- | --- |",
        ]
    )
    for field, new_val, base_val in rows:
        lines.append(f"| {_md_cell(field)} | {_md_cell(new_val)} | {_md_cell(base_val)} |")

    lines.extend(["", "### New-Only Error Signatures (Present in New, Absent in Baseline)"])
    if not new_only_node_sigs and not new_only_cp_sigs:
        lines.append("- None.")
    else:
        for sig in new_only_node_sigs:
            lines.append(f"- **[node]** `{redact_secrets(sig)}`")
        for sig in new_only_cp_sigs:
            lines.append(f"- **[control-plane]** `{redact_secrets(sig)}`")

    for title, s in (("New", new_s), ("Baseline", base_s)):
        lines.extend(
            [
                "",
                f"### {title} Workspace `git log --oneline` & `pytest -q`",
                "",
                "```text",
                redact_secrets(s["git_log"] or "(no git log)"),
                "--- origin refs ---",
                redact_secrets(s["origin_refs"] or "(no origin refs)"),
                "--- pytest -q ---",
                redact_secrets(s["pytest_output"] or "(no pytest output)"),
                "```",
            ]
        )
        if s["execution_error"]:
            lines.extend(
                [
                    "",
                    f"**{title} execution error:** `{redact_secrets(s['execution_error'])}`",
                ]
            )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-dir", required=True, type=Path)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    new_summary = summarize_run(args.new_dir, "new")
    base_summary = summarize_run(args.base_dir, "baseline")
    passed, failures, notes = evaluate_gate(new_summary, base_summary)

    markdown = render_markdown(new_summary, base_summary, passed, failures, notes)
    summary_path = args.out_dir / "summary.md"
    summary_path.write_text(markdown, encoding="utf-8")

    comparison_payload = {
        "passed": passed,
        "failures": failures,
        "notes": notes,
        "new": new_summary,
        "baseline": base_summary,
    }
    (args.out_dir / "comparison.json").write_text(
        redact_secrets(json.dumps(comparison_payload, indent=2, default=str)) + "\n",
        encoding="utf-8",
    )

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as fp:
            fp.write(markdown + "\n")

    print(markdown)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
