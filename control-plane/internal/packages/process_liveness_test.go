package packages

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestRuntimeProcessAliveIsBootIDAware(t *testing.T) {
	bootID := CurrentBootID()
	if bootID == "" {
		t.Skip("host does not expose a boot id")
	}
	pid := os.Getpid()
	if !RuntimeProcessAlive(RuntimeInfo{PID: &pid, BootID: bootID}) {
		t.Fatal("current PID recorded in the current boot must be live")
	}
	if RuntimeProcessAlive(RuntimeInfo{PID: &pid, BootID: "different-boot-id"}) {
		t.Fatal("a recycled PID recorded under another boot must be dead")
	}
}

func TestRuntimeProcessAliveRejectsRecycledPIDBeforeSignal(t *testing.T) {
	pid := 42
	signalled := false
	alive := runtimeProcessAlive(
		RuntimeInfo{PID: &pid, StartTime: "old-process"},
		func(int) string { return "new-process" },
		func(int) bool { signalled = true; return true },
	)
	if alive {
		t.Fatal("recycled PID must be dead")
	}
	if signalled {
		t.Fatal("identity mismatch must be rejected before signalling the PID")
	}
}

func TestRuntimeProcessAliveAcceptsMatchingProcessIdentity(t *testing.T) {
	pid := 42
	if !runtimeProcessAlive(
		RuntimeInfo{PID: &pid, StartTime: "same-process"},
		func(int) string { return "same-process" },
		func(int) bool { return true },
	) {
		t.Fatal("matching start time and live PID must be alive")
	}
}

func TestCurrentProcessStartTimeIdentifiesCurrentProcess(t *testing.T) {
	first := CurrentProcessStartTime(os.Getpid())
	if first == "" {
		if runtime.GOOS == "linux" || runtime.GOOS == "darwin" {
			t.Fatalf("process start identity is required on %s", runtime.GOOS)
		}
		t.Skip("process start identity is unsupported on this platform")
	}
	if second := CurrentProcessStartTime(os.Getpid()); second != first {
		t.Fatalf("process identity changed across reads: %q then %q", first, second)
	}

	cmd := exec.Command(os.Args[0], "-test.run=^TestProcessStartIdentityChild$")
	cmd.Env = append(os.Environ(), "AGENTFIELD_PROCESS_IDENTITY_CHILD=1")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
	})
	child := CurrentProcessStartTime(cmd.Process.Pid)
	if child == "" || child == first {
		t.Fatalf("child identity = %q, parent identity = %q", child, first)
	}
}

func TestProcessStartIdentityChild(t *testing.T) {
	if os.Getenv("AGENTFIELD_PROCESS_IDENTITY_CHILD") != "1" {
		return
	}
	time.Sleep(30 * time.Second)
}

func TestCommandProcessStartTimeIsBounded(t *testing.T) {
	t.Setenv("AGENTFIELD_PROCESS_IDENTITY_CHILD", "1")
	started := time.Now()
	got, state := commandProcessStartTimeResult(50*time.Millisecond, os.Args[0], "-test.run=^TestProcessStartIdentityChild$")
	if got != "" {
		t.Fatalf("timed out command returned %q", got)
	}
	if state != processIdentityUnavailable {
		t.Fatalf("timed out command state=%v, want identity unavailable", state)
	}
	if elapsed := time.Since(started); elapsed < 40*time.Millisecond || elapsed > 10*time.Second {
		t.Fatalf("command timeout elapsed=%s, want a bounded probe", elapsed)
	}
}

func TestRuntimeProcessStatusTreatsIdentityProbeTimeoutAsUnknown(t *testing.T) {
	pid := 42
	state := runtimeProcessStatus(
		RuntimeInfo{PID: &pid, StartTime: "recorded"},
		func(int) (string, processIdentityState) { return "", processIdentityUnavailable },
		func(int) (time.Time, bool) { return time.Time{}, false },
		func(int) bool { t.Fatal("unknown identity must not be reduced to a liveness signal"); return false },
	)
	if state != RuntimeProcessUnknown {
		t.Fatalf("process state=%v, want unknown", state)
	}
}

func TestProcessIdentityHelpersRejectInvalidProcessesAndCommands(t *testing.T) {
	if got := CurrentProcessStartTime(0); got != "" {
		t.Fatalf("zero PID identity=%q", got)
	}
	if got := CurrentProcessStartTime(99999999); got != "" {
		t.Fatalf("missing PID identity=%q", got)
	}
	if got := commandProcessStartTime("definitely-not-a-real-command"); got != "" {
		t.Fatalf("missing command identity=%q", got)
	}
	if got := commandProcessStartTime("sh", "-c", "printf '  process   start  '"); got != "process start" {
		t.Fatalf("normalized command identity=%q", got)
	}
	if linuxBootTime() == "" {
		t.Skip("Linux boot time is unavailable on this platform")
	}
}

func TestRuntimeProcessAliveRejectsMissingAndInvalidPIDs(t *testing.T) {
	if runtimeProcessAlive(RuntimeInfo{}, func(int) string { return "" }, func(int) bool { return true }) {
		t.Fatal("missing PID was alive")
	}
	pid := -1
	if runtimeProcessAlive(RuntimeInfo{PID: &pid}, func(int) string { return "" }, func(int) bool { return true }) {
		t.Fatal("negative PID was alive")
	}
	pid = 99999999
	if processExists(pid) {
		t.Fatal("missing process was alive")
	}
}

func TestInstalledPackageDesiredStateLegacyUpgrade(t *testing.T) {
	running := InstalledPackage{Status: "running"}
	if running.EffectiveDesiredState() != DesiredStateRunning || running.EnsureDesiredState() != DesiredStateRunning || running.DesiredState != DesiredStateRunning {
		t.Fatalf("running legacy entry=%+v", running)
	}
	stopped := InstalledPackage{Status: "stopped"}
	if stopped.EffectiveDesiredState() != DesiredStateStopped || stopped.EnsureDesiredState() != DesiredStateStopped {
		t.Fatalf("stopped legacy entry=%+v", stopped)
	}
	explicit := InstalledPackage{Status: "stopped", DesiredState: DesiredStateRunning}
	if explicit.EnsureDesiredState() != DesiredStateRunning {
		t.Fatalf("explicit desired state was overwritten: %+v", explicit)
	}
}

func TestCurrentBootIDMatchesProcFileWhenAvailable(t *testing.T) {
	data, err := os.ReadFile(filepath.Clean(bootIDPath))
	if err != nil {
		t.Skip("boot ID unavailable")
	}
	if got := CurrentBootID(); got != strings.TrimSpace(string(data)) {
		t.Fatalf("boot ID=%q", got)
	}
}
