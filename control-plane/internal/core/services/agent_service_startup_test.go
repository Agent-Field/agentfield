package services

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

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

// Contract: a caller-supplied port is never reassigned after a strict-port
// conflict because external configuration can depend on that exact port.
func TestStartWithPortRetry_ExplicitPortDoesNotRetry(t *testing.T) {
	pm := newMockPortManager()
	service := newStartupTestService(t, pm)

	var attempts int
	attempt := func(p int) (int, error, bool) {
		attempts++
		if p != 9123 {
			t.Errorf("attempted port = %d, want explicitly requested port 9123", p)
		}
		return 0, errors.New("assigned port unavailable"), true
	}

	_, port, err := captureRetry(t, service, 9123, false, attempt)
	if err == nil {
		t.Fatal("expected conflict failure to propagate")
	}
	if attempts != 1 {
		t.Errorf("explicit port must not retry, got %d attempts", attempts)
	}
	if port != 9123 {
		t.Errorf("returned port = %d, want explicitly requested port 9123", port)
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
