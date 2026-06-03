#!/usr/bin/env python3
"""Generate and insert release notes produced by git-cliff."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
CLIFF_CONFIG = REPO_ROOT / ".cliff.toml"
MARKER = "<!-- changelog:entries -->"


def run_git_cliff(version: str) -> str:
    """Invoke git-cliff and return the rendered release entry."""
    cmd = [
        "git",
        "cliff",
        "--config",
        str(CLIFF_CONFIG),
        "--unreleased",
        "--tag",
        version,
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip()
        raise SystemExit(
            f"git-cliff failed with exit code {result.returncode}: {msg}"
        )
    return result.stdout.strip()


def ensure_marker(text: str) -> tuple[str, str]:
    if MARKER not in text:
        raise SystemExit(
            f"Missing changelog marker '{MARKER}'. "
            "Make sure CHANGELOG.md follows the expected template."
        )
    before, after = text.split(MARKER, 1)
    return before, after


def remove_existing_entry(body: str, version: str) -> str:
    pattern = re.compile(
        rf"^## \[{re.escape(version)}].*?(?=^## \[|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    updated, count = pattern.subn("", body, count=1)
    if count:
        return updated.lstrip("\n")
    return body


def build_changelog(prefix: str, entry: str, suffix: str) -> str:
    suffix = suffix.lstrip("\n")
    parts = [
        prefix.rstrip(),
        "",
        MARKER,
        "",
        entry.strip(),
    ]
    if suffix:
        parts.extend(["", suffix.rstrip()])
    return "\n".join(part for part in parts if part is not None) + "\n"


def update_changelog(version: str, entry: str, dry_run: bool) -> None:
    current = CHANGELOG.read_text(encoding="utf-8")
    before, after = ensure_marker(current)
    cleaned_after = remove_existing_entry(after, version)
    next_text = build_changelog(before, entry, cleaned_after)
    if dry_run:
        sys.stdout.write(next_text)
        return
    CHANGELOG.write_text(next_text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the latest git history into CHANGELOG.md."
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Semantic version (without leading 'v') for the release entry.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the would-be changelog instead of updating the file.",
    )
    args = parser.parse_args(argv)

    try:
        rendered = run_git_cliff(args.version)
    except FileNotFoundError as exc:
        raise SystemExit(
            "git-cliff is not installed. Install it via "
            "`cargo install git-cliff` or download a release binary."
        ) from exc

    if not rendered:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        rendered = (
            f"## [{args.version}] - {today}\n\n"
            "- No notable changes were recorded for this release.\n"
        )

    update_changelog(args.version, rendered, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
