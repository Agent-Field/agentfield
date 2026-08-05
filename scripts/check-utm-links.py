#!/usr/bin/env python3
"""Keep assets/utm-links.csv in sync with the tracked links in README.md.

The CSV is the manifest of every UTM-tagged link in the README — marketing reads
it to audit campaign coverage. Nothing enforced that it stayed in sync, so it
drifted: by the time this check was written the README had 45 tracked links and
the CSV had 38 rows, one of which pointed at a link that had been deleted from
the README months earlier.

This script is the enforcement. It fails when:

  1. a README link carries a utm_id that has no row in the CSV
  2. a CSV row's utm_id no longer appears in the README (stale row)
  3. a CSV row's Target URL disagrees with the URL the README actually uses
  4. an agentfield.ai link in the README carries no UTM params at all
  5. a README link uses the www. host (splits analytics from the apex domain)

Run it locally the same way CI does:

    python3 scripts/check-utm-links.py

Stdlib only, no dependencies. Exits 0 when clean, 1 with an actionable report
otherwise — including the exact CSV row to paste for anything missing.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
CSV_PATH = REPO_ROOT / "assets" / "utm-links.csv"

# agentfield.ai links that are deliberately untagged. The install script is
# piped straight into a shell (`curl ... | sh`) — query params would end up in
# the shell command, so it stays clean.
UNTAGGED_ALLOWLIST = {
    "https://agentfield.ai/install.sh",
}

# A URL runs until markdown/HTML syntax ends it: quote, paren, angle bracket,
# square bracket, or whitespace.
URL_RE = re.compile(r'https://(?:www\.)?agentfield\.ai[^\s"\')>\]]*')
UTM_ID_RE = re.compile(r"utm_id=([A-Za-z0-9._-]+)")

CAMPAIGN_PREFIX = "https://agentfield.ai"


def target_of(url: str) -> str:
    """Strip the query string and any trailing slash, for comparison."""
    return url.split("?")[0].rstrip("/")


def read_readme_links() -> tuple[dict[str, set[str]], list[tuple[int, str]], list[tuple[int, str]]]:
    """Return (utm_id -> target URLs), untagged links, and www-host links."""
    tagged: dict[str, set[str]] = {}
    untagged: list[tuple[int, str]] = []
    www_host: list[tuple[int, str]] = []

    for lineno, line in enumerate(README.read_text(encoding="utf-8").splitlines(), start=1):
        for url in URL_RE.findall(line):
            match = UTM_ID_RE.search(url)
            if match:
                tagged.setdefault(match.group(1), set()).add(target_of(url))
                if url.startswith("https://www."):
                    www_host.append((lineno, url))
            elif url not in UNTAGGED_ALLOWLIST:
                untagged.append((lineno, url))

    return tagged, untagged, www_host


def read_csv_rows() -> tuple[dict[str, str], list[str]]:
    """Return (utm_id -> Target URL) plus any structural problems found."""
    rows: dict[str, str] = {}
    problems: list[str] = []

    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        for lineno, row in enumerate(csv.DictReader(handle), start=2):
            name = (row.get("Link Name") or "").strip()
            target = (row.get("Target URL") or "").strip()
            full = (row.get("Full UTM URL") or "").strip()

            if not (name and target and full):
                problems.append(f"line {lineno}: row is missing one of the three required columns")
                continue

            match = UTM_ID_RE.search(full)
            if not match:
                problems.append(f"line {lineno}: Full UTM URL has no utm_id ({full})")
                continue

            utm_id = match.group(1)
            if utm_id in rows:
                problems.append(f"line {lineno}: duplicate row for utm_id={utm_id}")
                continue

            if target_of(full) != target.rstrip("/"):
                problems.append(
                    f"line {lineno}: Full UTM URL does not start from Target URL\n"
                    f"    target: {target}\n"
                    f"    full  : {full}"
                )

            rows[utm_id] = target.rstrip("/")

    return rows, problems


def suggested_row(utm_id: str, target: str) -> str:
    """The CSV line a contributor should paste for a missing link."""
    query = f"utm_source=github-readme&utm_campaign=github-readme&utm_id={utm_id}"
    slug = utm_id.removeprefix("github-readme-")
    return f"Github Readme <Section> → {slug},{target},{target}?{query}"


def main() -> int:
    readme_links, untagged, www_host = read_readme_links()
    csv_rows, failures = read_csv_rows()

    missing = sorted(set(readme_links) - set(csv_rows))
    stale = sorted(set(csv_rows) - set(readme_links))

    if missing:
        lines = ["README links with no row in assets/utm-links.csv:"]
        for utm_id in missing:
            target = sorted(readme_links[utm_id])[0]
            lines.append(f"  {utm_id}\n    add: {suggested_row(utm_id, target)}")
        failures.append("\n".join(lines))

    if stale:
        lines = ["CSV rows whose utm_id no longer appears in README.md (delete them):"]
        lines += [f"  {utm_id}" for utm_id in stale]
        failures.append("\n".join(lines))

    for utm_id in sorted(set(readme_links) & set(csv_rows)):
        targets = readme_links[utm_id]
        if len(targets) > 1:
            joined = "\n      ".join(sorted(targets))
            failures.append(
                f"{utm_id}: README uses one utm_id for more than one target URL\n      {joined}"
            )
        elif next(iter(targets)) != csv_rows[utm_id]:
            failures.append(
                f"{utm_id}: CSV Target URL disagrees with README\n"
                f"    README: {next(iter(targets))}\n"
                f"    CSV   : {csv_rows[utm_id]}"
            )

    if untagged:
        lines = ["agentfield.ai links in README.md carrying no UTM params:"]
        lines += [f"  README.md:{lineno}  {url}" for lineno, url in untagged]
        lines.append("  (if a link is deliberately untagged, add it to UNTAGGED_ALLOWLIST)")
        failures.append("\n".join(lines))

    if www_host:
        lines = [f"README.md links using the www. host — use {CAMPAIGN_PREFIX} so analytics don't split:"]
        lines += [f"  README.md:{lineno}  {url}" for lineno, url in www_host]
        failures.append("\n".join(lines))

    if failures:
        print("utm-links check FAILED\n", file=sys.stderr)
        for failure in failures:
            print(f"{failure}\n", file=sys.stderr)
        return 1

    print(f"utm-links check passed — {len(readme_links)} tracked links, all present in {CSV_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
