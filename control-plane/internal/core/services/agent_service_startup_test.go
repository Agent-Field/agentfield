package services

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
)

// captureStdout runs fn while capturing everything written to os.Stdout.
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	orig := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	os.Stdout = w
	done := make(chan string, 1)
	go func() {
		b, _ := io.ReadAll(r)
		done <- string(b)
	}()
	fn()
	_ = w.Close()
	os.Stdout = orig
	return <-done
}

func newStartupTestService(t *testing.T, pm *mockPortManager) *DefaultAgentService {
	t.Helper()
	return NewAgentService(
		newMockProcessManager(),
		pm,
		newMockRegistryStorage(),
		newMockAgentClient(),
		t.TempDir(),
	).(*DefaultAgentService)
}

// Contract: an automatically allocated port that encounters a strict-port
// failure triggers exactly one retry on a different port than the one that
// failed.
func TestStartWithPortRetry_RetriesOnceOnPortConflict(t *testing.T) {
	pm := newMockPortManager()
	reserved := -1
	pm.reserveFunc = func(p int) error { reserved = p; return nil }
	pm.findFreePortFunc = func(int) (int, error) { return 8002, nil }
	service := newStartupTestService(t, pm)

	var attemptPorts []int
	attempt := func(p int) (int, error, bool) {
		attemptPorts = append(attemptPorts, p)
		if len(attemptPorts) == 1 {
			// First attempt: strict-port conflict.
			return 0, errors.New("agent node failed to start: assigned port unavailable"), true
		}
		// Retry: success.
		return 777, nil, false
	}

	pid, port, err := captureRetry(t, service, 8001, true, attempt)
	if err != nil {
		t.Fatalf("expected retry to succeed, got %v", err)
	}
	if len(attemptPorts) != 2 {
		t.Fatalf("expected exactly 2 attempts, got %d: %v", len(attemptPorts), attemptPorts)
	}
	if attemptPorts[0] == attemptPorts[1] {
		t.Errorf("retry must use a different port, both were %d", attemptPorts[0])
	}
	if attemptPorts[0] != 8001 || attemptPorts[1] != 8002 {
		t.Errorf("expected attempts on 8001 then 8002, got %v", attemptPorts)
	}
	if reserved != 8001 {
		t.Errorf("failed port 8001 must be reserved before retry, reserved=%d", reserved)
	}
	if pid != 777 || port != 8002 {
		t.Errorf("expected pid=777 port=8002, got pid=%d port=%d", pid, port)
	}
}

// Contract: a non-conflict startup failure is NOT retried.
func TestStartWithPortRetry_NoRetryOnNonConflictFailure(t *testing.T) {
	pm := newMockPortManager()
	pm.findFreePortFunc = func(int) (int, error) { return 8002, nil }
	service := newStartupTestService(t, pm)

	var attempts int
	attempt := func(p int) (int, error, bool) {
		attempts++
		return 0, errors.New("boom: import error"), false
	}

	_, _, err := captureRetry(t, service, 8001, true, attempt)
	if err == nil {
		t.Fatalf("expected failure to propagate")
	}
	if attempts != 1 {
		t.Errorf("non-conflict failure must not retry, got %d attempts", attempts)
	}
}

// Contract: a first-attempt success runs exactly once.
func TestStartWithPortRetry_SuccessRunsOnce(t *testing.T) {
	pm := newMockPortManager()
	service := newStartupTestService(t, pm)

	var attempts int
	attempt := func(p int) (int, error, bool) {
		attempts++
		return 55, nil, false
	}

	pid, port, err := captureRetry(t, service, 8001, true, attempt)
	if err != nil || attempts != 1 {
		t.Fatalf("expected one successful attempt, got attempts=%d err=%v", attempts, err)
	}
	if pid != 55 || port != 8001 {
		t.Errorf("expected pid=55 port=8001, got pid=%d port=%d", pid, port)
	}
}

// Contract: when no distinct fresh port is available, the conflict failure is
// returned without a (pointless) retry on the same port.
func TestStartWithPortRetry_NoDistinctPortDoesNotRetry(t *testing.T) {
	pm := newMockPortManager()
	pm.reserveFunc = func(int) error { return nil }
	pm.findFreePortFunc = func(int) (int, error) { return 8001, nil } // same port back
	service := newStartupTestService(t, pm)

	var attempts int
	attempt := func(p int) (int, error, bool) {
		attempts++
		return 0, errors.New("assigned port unavailable"), true
	}

	_, _, err := captureRetry(t, service, 8001, true, attempt)
	if err == nil {
		t.Fatalf("expected failure to propagate")
	}
	if attempts != 1 {
		t.Errorf("must not retry when the fresh port equals the failed port, got %d attempts", attempts)
	}
}

// Contract: an internal restore/update port preference falls back to a fresh port
// when another process wins the bind race.
func TestStartWithPortRetry_PreferredPortFallsBackAfterBindConflict(t *testing.T) {
	pm := newMockPortManager()
	pm.findFreePortFunc = func(int) (int, error) { return 9124, nil }
	service := newStartupTestService(t, pm)

	var attempted []int
	attempt := func(p int) (int, error, bool) {
		attempted = append(attempted, p)
		if len(attempted) == 1 {
			return 0, errors.New("assigned port unavailable"), true
		}
		return 99, nil, false
	}

	pid, port, err := captureRetry(t, service, 9123, true, attempt)
	if err != nil || pid != 99 || port != 9124 {
		t.Fatalf("pid=%d port=%d err=%v", pid, port, err)
	}
	if fmt.Sprint(attempted) != "[9123 9124]" {
		t.Fatalf("attempted ports = %v", attempted)
	}
}

// Contract: a user-supplied port is strict and never silently retries on a
// different port after a bind conflict.
func TestStartWithPortRetry_ExplicitPortDoesNotRetry(t *testing.T) {
	service := newStartupTestService(t, newMockPortManager())
	attempts := 0
	_, port, err := captureRetry(t, service, 9123, false, func(p int) (int, error, bool) {
		attempts++
		return 0, errors.New("assigned port unavailable"), true
	})
	if err == nil || attempts != 1 || port != 9123 {
		t.Fatalf("attempts=%d port=%d err=%v", attempts, port, err)
	}
}

func TestRequestedPortIsReservedAndReleasedOrFallsBack(t *testing.T) {
	pm := newMockPortManager()
	var events []string
	pm.reserveFunc = func(port int) error {
		events = append(events, fmt.Sprintf("reserve:%d", port))
		return nil
	}
	pm.releaseFunc = func(port int) error {
		events = append(events, fmt.Sprintf("release:%d", port))
		return nil
	}
	service := newStartupTestService(t, pm)
	port, release, err := service.reserveRequestedPort(8123, false)
	if err != nil {
		t.Fatal(err)
	}
	if port != 8123 {
		t.Fatalf("reserved port = %d", port)
	}
	release()
	if fmt.Sprint(events) != "[reserve:8123 release:8123]" {
		t.Fatalf("reservation events = %v", events)
	}

	pm.reserveFunc = func(int) error { return errors.New("occupied") }
	port, release, err = service.reserveRequestedPort(8123, true)
	if err != nil {
		t.Fatal(err)
	}
	release()
	if port != 0 {
		t.Fatalf("occupied requested port = %d, want automatic allocation", port)
	}
	if _, _, err := service.reserveRequestedPort(8123, false); err == nil {
		t.Fatal("occupied explicit port did not fail")
	}
}

// Contract: runAgentGuarded reserves the parent's requested port before it
// starts dependencies, so a dependency can never be allocated that port.
func TestRunAgentGuardedReservesRequestedPortBeforeDependencies(t *testing.T) {
	home := t.TempDir()
	parentDir := filepath.Join(home, "packages", "parent")
	depDir := filepath.Join(home, "packages", "dep")
	writeManifest(t, parentDir, "name: parent\nversion: 1.0.0\nentrypoint:\n  start: missing-parent\ndependencies:\n  nodes:\n    - af://registry/dep\n")
	writeManifest(t, depDir, "name: dep\nversion: 1.0.0\nentrypoint:\n  start: missing-dep\n")
	writeRegistry(t, home, &packages.InstallationRegistry{Installed: map[string]packages.InstalledPackage{
		"parent": {Name: "parent", Path: parentDir, Status: "stopped"},
		"dep":    {Name: "dep", Path: depDir, Status: "stopped"},
	}})

	reserved := false
	pm := newMockPortManager()
	pm.reserveFunc = func(port int) error {
		if port == 8123 {
			reserved = true
		}
		return nil
	}
	pm.findFreePortFunc = func(int) (int, error) {
		if !reserved {
			t.Fatal("dependency port allocation ran before parent reservation")
		}
		return 8124, nil
	}
	processManager := newMockProcessManager()
	var assigned []string
	processManager.startFunc = func(config interfaces.ProcessConfig) (int, error) {
		for _, value := range config.Env {
			if strings.HasPrefix(value, "PORT=") {
				assigned = append(assigned, value)
			}
		}
		return 0, errors.New("stop after port assignment")
	}
	service := NewAgentService(processManager, pm, newMockRegistryStorage(), nil, home).(*DefaultAgentService)
	_, err := service.RunAgent("parent", domain.RunOptions{Port: 8123, Detach: true})
	if err == nil {
		t.Fatal("test process manager unexpectedly started the parent")
	}
	if len(assigned) < 2 || assigned[0] != "PORT=8124" || assigned[1] != "PORT=8123" {
		t.Fatalf("assigned ports = %v, want dependency 8124 then parent 8123", assigned)
	}
}

func TestRunAgentGuardedReconcilesRunningEntryWithoutPortBeforeRestore(t *testing.T) {
	home := t.TempDir()
	pkgDir := filepath.Join(home, "packages", "demo")
	writeManifest(t, pkgDir, "name: demo\nversion: 1.0.0\nentrypoint:\n  start: missing-demo\n")
	pid := os.Getpid()
	writeRegistry(t, home, &packages.InstallationRegistry{Installed: map[string]packages.InstalledPackage{
		"demo": {
			Name: "demo", Path: pkgDir, Status: "running", DesiredState: packages.DesiredStateRunning,
			Runtime: packages.RuntimeInfo{PID: &pid},
		},
	}})

	pm := newMockPortManager()
	pm.findFreePortFunc = func(int) (int, error) { return 8123, nil }
	processManager := newMockProcessManager()
	processManager.startFunc = func(interfaces.ProcessConfig) (int, error) {
		return 0, errors.New("stop after reconciliation")
	}
	service := NewAgentService(processManager, pm, newMockRegistryStorage(), nil, home).(*DefaultAgentService)
	if _, err := service.RunAgent("demo", domain.RunOptions{Detach: true}); err == nil || !strings.Contains(err.Error(), "stop after reconciliation") {
		t.Fatalf("RunAgent error=%v, want normal startup attempt after reconciliation", err)
	}
	registry, err := service.loadRegistryDirect()
	if err != nil {
		t.Fatal(err)
	}
	entry := registry.Installed["demo"]
	if entry.Status != "stopped" || entry.Runtime.PID != nil {
		t.Fatalf("malformed running entry was not reconciled before restore: %+v", entry)
	}
}

// captureRetry runs startWithPortRetry while swallowing its progress output.
func captureRetry(t *testing.T, s *DefaultAgentService, initial int, retryOnConflict bool, fn func(int) (int, error, bool)) (int, int, error) {
	t.Helper()
	var pid, port int
	var err error
	_ = captureStdout(t, func() {
		pid, port, err = s.startWithPortRetry(initial, retryOnConflict, fn)
	})
	return pid, port, err
}

func TestFreshRetryPort_ExcludesFailedPort(t *testing.T) {
	pm := newMockPortManager()
	reserved := -1
	pm.reserveFunc = func(p int) error { reserved = p; return nil }
	pm.findFreePortFunc = func(start int) (int, error) {
		if start != 8001 {
			t.Errorf("expected FindFreePort(8001), got %d", start)
		}
		return 8003, nil
	}
	service := newStartupTestService(t, pm)

	got, err := service.freshRetryPort(8001)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if reserved != 8001 {
		t.Errorf("failed port must be reserved, reserved=%d", reserved)
	}
	if got != 8003 {
		t.Errorf("expected fresh port 8003, got %d", got)
	}
}

// Contract: logIndicatesPortConflict classifies the SDK's strict-port exit as a
// conflict and other failures as not.
func TestLogIndicatesPortConflict(t *testing.T) {
	cases := []struct {
		name  string
		lines []string
		want  bool
	}{
		{"sdk log_error line", []string{"INFO boot", "AGENTFIELD_STRICT_PORT set but the assigned port 8001 is unavailable; exiting so the control plane can reallocate and retry"}, true},
		{"sdk runtime error", []string{"RuntimeError: assigned port 8005 is unavailable"}, true},
		{"unrelated traceback", []string{"Traceback (most recent call last):", "ModuleNotFoundError: No module named 'foo'"}, false},
		{"empty", nil, false},
		{"port mentioned but not unavailable", []string{"listening on assigned port 8001"}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := logIndicatesPortConflict(tc.lines); got != tc.want {
				t.Errorf("logIndicatesPortConflict(%v) = %v, want %v", tc.lines, got, tc.want)
			}
		})
	}
}

func TestReadLogTailLines(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "node.log")
	var b strings.Builder
	for i := 1; i <= 30; i++ {
		fmt.Fprintf(&b, "line-%d\n", i)
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	lines := readLogTailLines(path, 15)
	if len(lines) != 15 {
		t.Fatalf("expected 15 lines, got %d", len(lines))
	}
	if lines[0] != "line-16" || lines[14] != "line-30" {
		t.Errorf("expected tail line-16..line-30, got %s..%s", lines[0], lines[14])
	}

	// Missing file yields nil, not an error.
	if got := readLogTailLines(filepath.Join(dir, "missing.log"), 10); got != nil {
		t.Errorf("missing file should yield nil, got %v", got)
	}
	if got := readLogTailLines("", 10); got != nil {
		t.Errorf("empty path should yield nil, got %v", got)
	}
}

// Contract: the startup-failure path prints the tail of the node's log file and
// the "af logs" pointer.
func TestPrintStartupFailureDiagnostics(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "swe.log")
	var b strings.Builder
	for i := 1; i <= 20; i++ {
		fmt.Fprintf(&b, "boot-line-%d\n", i)
	}
	b.WriteString("RuntimeError: assigned port 8001 is unavailable\n")
	if err := os.WriteFile(logPath, []byte(b.String()), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	service := newStartupTestService(t, newMockPortManager())
	node := packages.InstalledPackage{
		Name:    "swe-af",
		Runtime: packages.RuntimeInfo{LogFile: logPath},
	}

	out := captureStdout(t, func() {
		service.printStartupFailureDiagnostics(node, "swe-af")
	})

	if !strings.Contains(out, "RuntimeError: assigned port 8001 is unavailable") {
		t.Errorf("diagnostics should include the failing log tail, got:\n%s", out)
	}
	if !strings.Contains(out, "Full logs: af logs swe-af") {
		t.Errorf("diagnostics should point at `af logs`, got:\n%s", out)
	}
	// Only the last ~15 lines — the earliest boot line must be trimmed.
	if strings.Contains(out, "boot-line-1\n") {
		t.Errorf("diagnostics should show only the tail, but included boot-line-1:\n%s", out)
	}
}
