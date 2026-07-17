"""Regression test for issue #431: duplicate PresenceManager sweep goroutines."""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


def test_issue_431(tmp_path):
    """
    Calling PresenceManager.Start twice should not spawn a second sweep loop.

    The bug lives in a Go internal package, so this pytest injects a focused Go
    test with an overlay and runs it against the control-plane services package.
    """
    repo_root = Path(__file__).resolve().parents[1]
    control_plane_dir = repo_root / "control-plane"
    if shutil.which("go") is None:
        pytest.skip("Go toolchain is required to run the PresenceManager regression test")

    package_dir = control_plane_dir / "internal" / "services"
    injected_test_path = package_dir / "presence_manager_issue_431_test.go"
    overlay_test_path = tmp_path / "presence_manager_issue_431_test.go"
    overlay_path = tmp_path / "overlay.json"

    overlay_test_path.write_text(
        textwrap.dedent(
            r"""
            package services

            import (
                "bytes"
                "runtime/pprof"
                "strings"
                "testing"
                "time"
            )

            func issue431LoopCount(t *testing.T) int {
                t.Helper()

                var buf bytes.Buffer
                if err := pprof.Lookup("goroutine").WriteTo(&buf, 2); err != nil {
                    t.Fatalf("failed to inspect goroutines: %v", err)
                }
                return strings.Count(buf.String(), "github.com/Agent-Field/agentfield/control-plane/internal/services.(*PresenceManager).loop")
            }

            func issue431Eventually(t *testing.T, fn func() bool, msg string) {
                t.Helper()

                deadline := time.Now().Add(2 * time.Second)
                for time.Now().Before(deadline) {
                    if fn() {
                        return
                    }
                    time.Sleep(10 * time.Millisecond)
                }
                t.Fatal(msg)
            }

            func TestIssue431StartTwiceDoesNotSpawnDuplicateSweepGoroutine(t *testing.T) {
                pm := NewPresenceManager(nil, PresenceManagerConfig{
                    HeartbeatTTL:  100 * time.Millisecond,
                    SweepInterval: time.Hour,
                    HardEvictTTL:  time.Hour,
                })
                t.Cleanup(pm.Stop)

                before := issue431LoopCount(t)

                pm.Start()
                issue431Eventually(t, func() bool {
                    return issue431LoopCount(t) == before+1
                }, "first Start did not create exactly one presence sweep goroutine")

                pm.Start()
                time.Sleep(100 * time.Millisecond)

                afterSecondStart := issue431LoopCount(t)
                if afterSecondStart != before+1 {
                    t.Fatalf("Start spawned duplicate sweep goroutines: before=%d afterSecondStart=%d", before, afterSecondStart)
                }
            }
            """
        ),
        encoding="utf-8",
    )
    overlay_path.write_text(
        json.dumps({"Replace": {str(injected_test_path): str(overlay_test_path)}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "go",
            "test",
            f"-overlay={overlay_path}",
            "./internal/services",
            "-run",
            "TestIssue431StartTwiceDoesNotSpawnDuplicateSweepGoroutine",
            "-count=1",
            "-v",
        ],
        cwd=control_plane_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
