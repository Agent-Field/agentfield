package packagemaint

import (
	"context"
	"errors"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages/updatecheck"
)

func restoreContractService(home string, agent AgentRunner) *Service {
	service := enabledService(home, &maintRunner{err: errors.New("unused")}, &maintJobs{}, agent, &busySequence{})
	service.enabled = func() (bool, string) { return false, "disabled" }
	service.processAlive = func(packages.RuntimeInfo) bool { return true }
	return service
}

func TestE5RestoreConfirmsARecoveredHealthProbeAndLeavesNodeAlone(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {
			Name: "demo", Status: "running", DesiredState: packages.DesiredStateRunning,
			Runtime: packages.RuntimeInfo{PID: intPtr(42), Port: intPtr(8123), StartTime: "owned"},
		},
	})
	agent := &maintAgent{}
	service := restoreContractService(home, agent)
	probes := 0
	service.healthProbe = func(context.Context, int, string) packages.HealthIdentity {
		probes++
		if probes == 1 {
			return packages.HealthIdentity{}
		}
		return packages.HealthIdentity{Healthy: true, NodeID: "demo"}
	}

	service.RunPass(context.Background())
	if probes < 2 || len(agent.stops) != 0 || len(agent.runs) != 0 {
		t.Fatalf("probes=%d stops=%v runs=%v", probes, agent.stops, agent.runs)
	}
}

func TestE6RestoreStopsAndRestartsOnceAfterAllConfirmationsFail(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {
			Name: "demo", Status: "running", DesiredState: packages.DesiredStateRunning,
			Runtime: packages.RuntimeInfo{PID: intPtr(42), Port: intPtr(8123), StartTime: "owned"},
		},
	})
	agent := &maintAgent{}
	service := restoreContractService(home, agent)
	probes := 0
	service.healthProbe = func(context.Context, int, string) packages.HealthIdentity { probes++; return packages.HealthIdentity{} }

	summary := service.RunPass(context.Background())
	if probes != 3 || len(agent.stops) != 1 || len(agent.runs) != 1 || len(summary.Restored) != 1 {
		t.Fatalf("probes=%d stops=%v runs=%v summary=%+v", probes, agent.stops, agent.runs, summary)
	}
}

type gatedRestoreAgent struct {
	mu      sync.Mutex
	calls   []string
	started chan string
	release chan struct{}
	stops   []string
}

func (a *gatedRestoreAgent) RunAgent(name string, options domain.RunOptions) (*domain.RunningAgent, error) {
	a.mu.Lock()
	a.calls = append(a.calls, name)
	a.mu.Unlock()
	a.started <- name
	<-a.release
	return &domain.RunningAgent{Name: name, Port: options.Port}, nil
}

func (a *gatedRestoreAgent) StopAgent(name string) error {
	a.mu.Lock()
	a.stops = append(a.stops, name)
	a.mu.Unlock()
	return nil
}

func (a *gatedRestoreAgent) snapshot() (calls, stops []string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]string(nil), a.calls...), append([]string(nil), a.stops...)
}

func TestE8ScheduledRunWaitsWhileManualPassOwnsTheService(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Status: "stopped", DesiredState: packages.DesiredStateRunning},
	})
	agent := &gatedRestoreAgent{started: make(chan string, 2), release: make(chan struct{})}
	ready := make(chan struct{})
	close(ready)
	var nowCalls atomic.Int64
	service := New(Config{
		AgentFieldHome: home, Agent: agent, Ready: ready,
		Enabled: func() (bool, string) { return false, "disabled" },
		Sleep:   func(context.Context, time.Duration) error { return nil },
		Now: func() time.Time {
			nowCalls.Add(1)
			return time.Now()
		},
		HostedInContainer: func() bool { return false },
	})
	if err := service.StartPass(); err != nil {
		t.Fatal(err)
	}
	<-agent.started
	service.mu.Lock()
	service.nextRunAt = time.Now().Add(-time.Second)
	service.mu.Unlock()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { service.Run(ctx); close(done) }()
	baseline := nowCalls.Load()
	time.Sleep(100 * time.Millisecond)
	if delta := nowCalls.Load() - baseline; delta > 5 {
		t.Fatalf("scheduled loop recomputed its deadline %d times while the manual pass was active", delta)
	}
	cancel()
	close(agent.release)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Run did not stop after cancellation")
	}
}

func TestE14RestoreRechecksDesiredStateBeforeAndAfterStart(t *testing.T) {
	t.Run("later package stopped during earlier restore", func(t *testing.T) {
		home := t.TempDir()
		writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
			"a": {Name: "a", Status: "stopped", DesiredState: packages.DesiredStateRunning},
			"b": {Name: "b", Status: "stopped", DesiredState: packages.DesiredStateRunning},
		})
		agent := &gatedRestoreAgent{started: make(chan string, 2), release: make(chan struct{})}
		service := restoreContractService(home, agent)
		done := make(chan Summary, 1)
		go func() { done <- service.RunPass(context.Background()) }()
		if name := <-agent.started; name != "a" {
			t.Fatalf("first restore=%q", name)
		}
		if err := packages.UpdateInstallationRegistry(filepath.Join(home, "installed.yaml"), func(registry *packages.InstallationRegistry) error {
			entry := registry.Installed["b"]
			entry.DesiredState = packages.DesiredStateStopped
			registry.Installed["b"] = entry
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		close(agent.release)
		summary := <-done
		calls, _ := agent.snapshot()
		if len(calls) != 1 || calls[0] != "a" || !containsSkip(summary.Skipped, "b", "stopped") {
			t.Fatalf("calls=%v summary=%+v", calls, summary)
		}
	})

	t.Run("package stopped during its own start", func(t *testing.T) {
		home := t.TempDir()
		writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
			"x": {Name: "x", Status: "stopped", DesiredState: packages.DesiredStateRunning},
		})
		agent := &gatedRestoreAgent{started: make(chan string, 1), release: make(chan struct{})}
		service := restoreContractService(home, agent)
		done := make(chan Summary, 1)
		go func() { done <- service.RunPass(context.Background()) }()
		<-agent.started
		if err := packages.UpdateInstallationRegistry(filepath.Join(home, "installed.yaml"), func(registry *packages.InstallationRegistry) error {
			entry := registry.Installed["x"]
			entry.DesiredState = packages.DesiredStateStopped
			registry.Installed["x"] = entry
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		close(agent.release)
		summary := <-done
		_, stops := agent.snapshot()
		if len(stops) != 1 || stops[0] != "x" || !containsSkip(summary.Skipped, "x", "stopped") {
			t.Fatalf("stops=%v summary=%+v", stops, summary)
		}
	})
}

func TestE15TimedOutRestoreRemainsTrackedUntilItsResultIsCollected(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"x": {Name: "x", Status: "stopped", DesiredState: packages.DesiredStateRunning},
	})
	agent := &gatedRestoreAgent{started: make(chan string, 1), release: make(chan struct{})}
	service := restoreContractService(home, agent)
	service.restoreTimeout = 20 * time.Millisecond
	first := service.RunPass(context.Background())
	second := service.RunPass(context.Background())
	calls, _ := agent.snapshot()
	if len(first.Errors) != 1 || !containsSkip(first.Skipped, "x", "starting") || !containsSkip(second.Skipped, "x", "starting") || len(calls) != 1 {
		t.Fatalf("calls=%v first=%+v second=%+v", calls, first, second)
	}
	close(agent.release)
	deadline := time.Now().Add(time.Second)
	for service.restorePending("x") && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	third := service.RunPass(context.Background())
	calls, _ = agent.snapshot()
	if len(calls) != 1 || len(third.Restored) != 1 || third.Restored[0] != "x" {
		t.Fatalf("calls=%v third=%+v", calls, third)
	}
}

func TestE16SuccessfulRestoreAndCheckFailureUseDifferentPhases(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"x": {
			Name: "x", Status: "stopped", DesiredState: packages.DesiredStateRunning,
			Source: "github", SourcePath: "https://github.com/acme/x", Commit: "old",
		},
	})
	agent := &maintAgent{}
	service := enabledService(home, &maintRunner{err: errors.New("remote unreachable")}, &maintJobs{}, agent, &busySequence{})
	summary := service.RunPass(context.Background())
	if len(summary.Restored) != 1 || summary.Restored[0] != "x" || len(summary.Errors) != 1 || summary.Errors[0] != "check x: remote unreachable" {
		t.Fatalf("summary=%+v", summary)
	}
}

type gateCheckRunner struct {
	started chan struct{}
	release chan struct{}
}

func (r *gateCheckRunner) Run(context.Context, ...string) ([]byte, error) {
	close(r.started)
	<-r.release
	return []byte("new\tHEAD\n"), nil
}

func TestE17BootRestoreCompletedPrecedesBootPassCompleted(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"x": {
			Name: "x", Status: "stopped", DesiredState: packages.DesiredStateStopped,
			Source: "github", SourcePath: "https://github.com/acme/x", Commit: "old",
		},
	})
	runner := &gateCheckRunner{started: make(chan struct{}), release: make(chan struct{})}
	ready := make(chan struct{})
	close(ready)
	service := New(Config{
		AgentFieldHome: home, Checker: updatecheck.NewChecker(runner), Jobs: &maintJobs{}, Ready: ready,
		Enabled: func() (bool, string) { return true, "" }, Sleep: func(context.Context, time.Duration) error { return nil },
		HostedInContainer: func() bool { return false },
	})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { service.Run(ctx); close(done) }()
	<-runner.started
	status := service.Status()
	if !status.BootRestoreCompleted || status.BootPassCompleted {
		t.Fatalf("status during update checks=%+v", status)
	}
	close(runner.release)
	cancel()
	<-done
}

func TestE22MigrationWriteFailureStillRestoresInMemoryMigration(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"x": {Name: "x", Status: "stopped"},
	})
	agent := &maintAgent{}
	service := New(Config{
		AgentFieldHome: home, Agent: agent,
		Enabled:           func() (bool, string) { return false, "disabled" },
		Sleep:             func(context.Context, time.Duration) error { return nil },
		HostedInContainer: func() bool { return true },
		UpdateRegistry: func(path string, mutate func(*packages.InstallationRegistry) error) error {
			registry, err := packages.LoadInstallationRegistry(path)
			if err != nil {
				return err
			}
			if err := mutate(registry); err != nil {
				return err
			}
			return errors.New("disk full")
		},
	})
	summary := service.RunPass(context.Background())
	if len(agent.runs) != 1 || len(summary.Restored) != 1 || summary.Restored[0] != "x" {
		t.Fatalf("runs=%v summary=%+v", agent.runs, summary)
	}
}

// E17b: on a fresh container the boot pass publishes what the restore did as
// soon as the restore loop ends, so a client does not have to wait for the
// update checks (which can take minutes) to learn which agents came back.
func TestE17bBootRestoreSnapshotIsPublishedBeforeTheChecksFinish(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"x": {
			Name: "x", Status: "stopped", DesiredState: packages.DesiredStateRunning,
			Source: "github", SourcePath: "https://github.com/acme/x", Commit: "old",
		},
	})
	runner := &gateCheckRunner{started: make(chan struct{}), release: make(chan struct{})}
	ready := make(chan struct{})
	close(ready)
	agent := &maintAgent{}
	service := New(Config{
		AgentFieldHome: home, Agent: agent, Checker: updatecheck.NewChecker(runner), Jobs: &maintJobs{}, Ready: ready,
		Enabled: func() (bool, string) { return true, "" }, Sleep: func(context.Context, time.Duration) error { return nil },
		HostedInContainer: func() bool { return false },
	})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { service.Run(ctx); close(done) }()
	<-runner.started
	status := service.Status()
	if !status.BootRestoreCompleted || status.BootPassCompleted {
		t.Fatalf("status during update checks=%+v", status)
	}
	if status.LastRun == nil || len(status.LastRun.Restored) != 1 || status.LastRun.Restored[0] != "x" {
		t.Fatalf("restore snapshot must be readable during the checks: %+v", status.LastRun)
	}
	close(runner.release)
	cancel()
	<-done
}
