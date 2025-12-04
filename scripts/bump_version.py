#!/usr/bin/env python3
"""Utility for bumping the project version consistently."""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path
from typing import Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = REPO_ROOT / "VERSION"
PYPROJECT_FILE = REPO_ROOT / "sdk/python/pyproject.toml"
PY_INIT_FILE = REPO_ROOT / "sdk/python/agentfield/__init__.py"
PKG_INFO_FILE = REPO_ROOT / "sdk/python/agentfield.egg-info/PKG-INFO"
TS_PACKAGE_JSON = REPO_ROOT / "sdk/typescript/package.json"
GO_TEMPLATE_FILE = REPO_ROOT / "control-plane/internal/templates/go/go.mod.tmpl"
REQUIREMENT_FILES = [
    REPO_ROOT / "examples/python_agent_nodes/hello_world_rag/requirements.txt",
    REPO_ROOT / "examples/python_agent_nodes/agentic_rag/requirements.txt",
    REPO_ROOT / "examples/python_agent_nodes/documentation_chatbot/requirements.txt",
]


SEMVER_PATTERN = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>[0-9A-Za-z.-]+))?$"
)


@dataclasses.dataclass(frozen=True)
class SemVer:
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None

    @classmethod
    def parse(cls, value: str) -> "SemVer":
        match = SEMVER_PATTERN.match(value.strip())
        if not match:
            raise ValueError(f"Invalid semantic version: {value}")
        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
            prerelease=match.group("prerelease"),
        )

    def without_prerelease(self) -> "SemVer":
        return SemVer(self.major, self.minor, self.patch, None)

    def bump(self, component: str) -> "SemVer":
        if component == "major":
            return SemVer(self.major + 1, 0, 0)
        if component == "minor":
            return SemVer(self.major, self.minor + 1, 0)
        if component == "patch":
            return SemVer(self.major, self.minor, self.patch + 1)
        raise ValueError(f"Unsupported component '{component}'")

    def with_prerelease(self, label: str, counter: int) -> "SemVer":
        if counter < 1:
            raise ValueError("Prerelease counter must be >= 1")
        return SemVer(self.major, self.minor, self.patch, f"{label}.{counter}")

    def prerelease_parts(self) -> Tuple[Optional[str], Optional[int]]:
        if not self.prerelease:
            return (None, None)
        if "." not in self.prerelease:
            return (self.prerelease, None)
        label, counter = self.prerelease.rsplit(".", 1)
        if not counter.isdigit():
            return (self.prerelease, None)
        return (label, int(counter))

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            return f"{base}-{self.prerelease}"
        return base


def determine_next_version(
    current: SemVer, channel: str, component: str, label: Optional[str]
) -> SemVer:
    base = current.without_prerelease()
    if channel == "stable":
        if current.prerelease:
            return base
        return base.bump(component)

    if channel != "prerelease":
        raise ValueError(f"Unknown channel '{channel}'")
    if not label:
        raise ValueError("Prerelease label is required for prerelease channel")

    current_label, current_counter = current.prerelease_parts()
    if current.prerelease and current_label == label and current_counter:
        return base.with_prerelease(label, current_counter + 1)

    target_base = base.bump(component)
    return target_base.with_prerelease(label, 1)


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def update_version_file(version: SemVer) -> None:
    write_file(VERSION_FILE, f"{version}\n")


def update_pyproject(version: SemVer) -> None:
    text = PYPROJECT_FILE.read_text(encoding="utf-8")
    new_text, count = re.subn(
        r'(?m)^(version\s*=\s*)"[^"]+"', rf'\1"{version}"', text, count=1
    )
    if count != 1:
        raise RuntimeError("Failed to update version in pyproject.toml")
    write_file(PYPROJECT_FILE, new_text)


def update_init(version: SemVer) -> None:
    text = PY_INIT_FILE.read_text(encoding="utf-8")
    new_text, count = re.subn(
        r'(?m)^__version__\s*=\s*"[^"]+"', f'__version__ = "{version}"', text, count=1
    )
    if count != 1:
        raise RuntimeError("Failed to update __version__ constant")
    write_file(PY_INIT_FILE, new_text)


def update_pkg_info(version: SemVer) -> None:
    if not PKG_INFO_FILE.exists():
        return
    text = PKG_INFO_FILE.read_text(encoding="utf-8")
    new_text, count = re.subn(
        r"(?m)^Version:\s+.+$", f"Version: {version}", text, count=1
    )
    if count != 1:
        raise RuntimeError("Failed to update Version field in PKG-INFO")
    write_file(PKG_INFO_FILE, new_text)


def update_requirements(version: SemVer) -> None:
    for path in REQUIREMENT_FILES:
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        replaced = False
        for idx, line in enumerate(lines):
            if line.strip().startswith("agentfield"):
                lines[idx] = f"agentfield>={version}"
                replaced = True
                break
        if not replaced:
            raise RuntimeError(f"Failed to update agentfield pin in {path}")
        write_file(path, "\n".join(lines) + "\n")


def update_go_template(version: SemVer) -> None:
    text = GO_TEMPLATE_FILE.read_text(encoding="utf-8")
    new_text, count = re.subn(
        r"(require\s+github\.com/Agent-Field/agentfield/sdk/go\s+v)[0-9A-Za-z\.-]+",
        rf"\g<1>{version}",
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Failed to update Go module template")
    write_file(GO_TEMPLATE_FILE, new_text)


def update_ts_package_json(version: SemVer) -> None:
    if not TS_PACKAGE_JSON.exists():
        return
    data = json.loads(TS_PACKAGE_JSON.read_text(encoding="utf-8"))
    data["version"] = str(version)
    write_file(TS_PACKAGE_JSON, json.dumps(data, indent=2) + "\n")


def apply_version(version: SemVer) -> None:
    update_version_file(version)
    update_pyproject(version)
    update_init(version)
    update_pkg_info(version)
    update_requirements(version)
    update_go_template(version)
    update_ts_package_json(version)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bump project version consistently.")
    parser.add_argument(
        "--channel",
        choices=["stable", "prerelease"],
        default="stable",
        help="Release channel being created.",
    )
    parser.add_argument(
        "--component",
        choices=["major", "minor", "patch"],
        default="patch",
        help="SemVer component to bump when creating a new release line.",
    )
    parser.add_argument(
        "--prerelease-label",
        dest="label",
        default=None,
        help="Label for prerelease builds (e.g., rc, beta). Required for prerelease channel.",
    )
    parser.add_argument(
        "--new-version",
        dest="new_version",
        default=None,
        help="Explicit semantic version to apply. When provided channel/component options are ignored.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the next version but do not modify any files.",
    )
    args = parser.parse_args(argv)

    if args.new_version:
        target = SemVer.parse(args.new_version)
    else:
        target = determine_next_version(
            SemVer.parse(VERSION_FILE.read_text(encoding="utf-8").strip()),
            args.channel,
            args.component,
            args.label,
        )
    if not args.dry_run:
        apply_version(target)

    version = target
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
