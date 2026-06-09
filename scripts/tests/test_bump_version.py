"""Unit tests for scripts/bump_version.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "bump_version.py"
_spec = importlib.util.spec_from_file_location("bump_version", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
bump_version = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("bump_version", bump_version)
_spec.loader.exec_module(bump_version)

SemVer = bump_version.SemVer
determine_next_version = bump_version.determine_next_version
_taken_prerelease_counters = bump_version._taken_prerelease_counters
_target_base_for_lookup = bump_version._target_base_for_lookup


def test_stable_from_stable_bumps_patch():
    assert (
        str(determine_next_version(SemVer.parse("0.1.91"), "stable", "patch", None))
        == "0.1.92"
    )


def test_stable_from_prerelease_drops_prerelease():
    assert (
        str(
            determine_next_version(
                SemVer.parse("0.1.92-rc.3"), "stable", "patch", None
            )
        )
        == "0.1.92"
    )


def test_prerelease_first_rc_from_stable_current():
    assert (
        str(
            determine_next_version(SemVer.parse("0.1.91"), "prerelease", "patch", "rc")
        )
        == "0.1.92-rc.1"
    )


def test_prerelease_increments_when_same_label():
    assert (
        str(
            determine_next_version(
                SemVer.parse("0.1.92-rc.2"), "prerelease", "patch", "rc"
            )
        )
        == "0.1.92-rc.3"
    )


def test_prerelease_advances_past_taken_counters_when_current_is_stable():
    # Regression: VERSION still says 0.1.91 stable but v0.1.92-rc.1 already exists
    # remotely from an earlier push. The legacy script returned the same name and
    # the workflow failed with "tag already exists".
    out = determine_next_version(
        SemVer.parse("0.1.91"),
        "prerelease",
        "patch",
        "rc",
        taken_counters={1, 2},
    )
    assert str(out) == "0.1.92-rc.3"


def test_prerelease_advances_past_taken_counters_when_same_label():
    out = determine_next_version(
        SemVer.parse("0.1.92-rc.5"),
        "prerelease",
        "patch",
        "rc",
        taken_counters={6, 7},
    )
    assert str(out) == "0.1.92-rc.8"


def test_taken_counters_parser_filters_unrelated_tags():
    tags = [
        "v0.1.92-rc.1",
        "v0.1.92-rc.2",
        "v0.1.91-rc.5",  # different base
        "v0.1.92-beta.1",  # different label
        "v0.1.92-rc.xx",  # non-numeric counter
        "v0.1.92",  # stable
    ]
    taken = _taken_prerelease_counters(SemVer.parse("0.1.92"), "rc", tags)
    assert taken == {1, 2}


def test_target_base_lookup_uses_current_base_when_label_matches():
    assert (
        _target_base_for_lookup(SemVer.parse("0.1.92-rc.5"), "rc", "patch")
        == SemVer.parse("0.1.92")
    )


def test_target_base_lookup_bumps_component_when_label_differs():
    assert (
        _target_base_for_lookup(SemVer.parse("0.1.91"), "rc", "patch")
        == SemVer.parse("0.1.92")
    )
    assert (
        _target_base_for_lookup(SemVer.parse("0.1.92-beta.3"), "rc", "patch")
        == SemVer.parse("0.1.93")
    )


def test_prerelease_channel_requires_label():
    with pytest.raises(ValueError):
        determine_next_version(SemVer.parse("0.1.91"), "prerelease", "patch", None)
