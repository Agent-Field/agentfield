package packagemaint

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages/updatecheck"
	"github.com/Agent-Field/agentfield/control-plane/internal/services/packagejobs"
	"gopkg.in/yaml.v3"
)

type maintRunner struct {
	sha string
	err error
}

type sequenceMaintRunner struct {
	results []struct {
		output string
		err    error
	}
	calls int
}

func (r *sequenceMaintRunner) Run(context.Context, ...string) ([]byte, error) {
	result := r.results[r.calls]
	r.calls++
	return []byte(result.output), result.err
}

type blockingMaintRunner struct {
	started chan struct{}
	release chan struct{}
}

func (r *blockingMaintRunner) Run(context.Context, ...string) ([]byte, error) {
	close(r.started)
	<-r.release
	return []byte("new\tHEAD\n"), nil
}

func (r *maintRunner) Run(context.Context, ...string) ([]byte, error) {
	if r.err != nil {
		return nil, r.err
	}
	return []byte(r.sha + "\tHEAD\n"), nil
}

type maintJobs struct {
	calls       []string
	result      packagejobs.JobStatus
	startErrors []error
	missing     bool
	jobError    string
	activeFor   string
}

func (j *maintJobs) ActiveFor(name string) bool { return j.activeFor == name }

func (j *maintJobs) StartUpdate(name, source string) (*packagejobs.Job, error) {
	j.calls = append(j.calls, name+":"+source)
	if index := len(j.calls) - 1; index < len(j.startErrors) && j.startErrors[index] != nil {
		return nil, j.startErrors[index]
	}
	status := j.result
	if status == "" {
		status = packagejobs.StatusSucceeded
	}
	return &packagejobs.Job{ID: name + "-job", PackageName: name, Status: status}, nil
}

func (j *maintJobs) GetJob(id string) (*packagejobs.Job, bool) {
	if j.missing {
		return nil, false
	}
	status := j.result
	if status == "" {
		status = packagejobs.StatusSucceeded
	}
	return &packagejobs.Job{ID: id, Status: status, Error: j.jobError}, true
}

type maintAgent struct {
	runs []struct {
		name      string
		port      int
		preferred bool
	}
	runErrors []error
	stops     []string
	stopError error
}

type runOnlyMaintAgent struct {
	runs int
}

type alwaysFailMaintAgent struct{ err error }

func (a *alwaysFailMaintAgent) RunAgent(string, domain.RunOptions) (*domain.RunningAgent, error) {
	return nil, a.err
}

type blockingRestoreAgent struct {
	started chan struct{}
	release chan struct{}
	calls   int
}

func (a *blockingRestoreAgent) RunAgent(name string, options domain.RunOptions) (*domain.RunningAgent, error) {
	a.calls++
	if a.calls == 1 {
		close(a.started)
		<-a.release
	}
	return &domain.RunningAgent{Name: name, Port: options.Port}, nil
}

func (a *runOnlyMaintAgent) RunAgent(string, domain.RunOptions) (*domain.RunningAgent, error) {
	a.runs++
	return &domain.RunningAgent{}, nil
}

func (a *maintAgent) RunAgent(name string, options domain.RunOptions) (*domain.RunningAgent, error) {
	a.runs = append(a.runs, struct {
		name      string
		port      int
		preferred bool
	}{name: name, port: options.Port, preferred: options.PortIsPreference})
	if index := len(a.runs) - 1; index < len(a.runErrors) && a.runErrors[index] != nil {
		return nil, a.runErrors[index]
	}
	return &domain.RunningAgent{Name: name, Port: options.Port}, nil
}

func (a *maintAgent) StopAgentForUpdate(name string) error {
	a.stops = append(a.stops, name)
	return a.stopError
}

type busySequence struct {
	values []bool
	calls  int
	err    error
}

func (b *busySequence) HasActiveExecutions(context.Context, string) (bool, error) {
	if b.err != nil {
		return false, b.err
	}
	i := b.calls
	b.calls++
	if i >= len(b.values) {
		return false, nil
	}
	return b.values[i], nil
}

func writeMaintRegistry(t *testing.T, home string, entries map[string]packages.InstalledPackage) {
	t.Helper()
	data, err := yaml.Marshal(packages.InstallationRegistry{Installed: entries})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func boolPtr(value bool) *bool { return &value }
func intPtr(value int) *int    { return &value }

func enabledService(home string, runner updatecheck.Runner, jobs JobManager, agent AgentRunner, busy ActiveExecutionChecker) *Service {
	checker := updatecheck.NewChecker(runner)
	return New(Config{
		AgentFieldHome:    home,
		Checker:           checker,
		Jobs:              jobs,
		Agent:             agent,
		Executions:        busy,
		Enabled:           func() (bool, string) { return true, "" },
		ProcessAlive:      func(packages.RuntimeInfo) bool { return false },
		HealthProbe:       func(context.Context, int, string) packages.HealthIdentity { return packages.HealthIdentity{} },
		Sleep:             func(context.Context, time.Duration) error { return nil },
		HostedInContainer: func() bool { return false },
		Hosting:           func() string { return packages.HostingLocal },
	})
}

// Contract: an unpinned package with a different remote SHA and no active
// executions is updated through StartUpdate(name, "").
func TestPassUpdatesAvailablePackage(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Source: "github", SourcePath: "https://github.com/acme/demo", Commit: "old"},
	})
	jobs := &maintJobs{}
	service := enabledService(home, &maintRunner{sha: "new"}, jobs, &maintAgent{}, &busySequence{})

	summary := service.RunPass(context.Background())
	if len(jobs.calls) != 1 || jobs.calls[0] != "demo:" {
		t.Fatalf("StartUpdate calls = %v, want [demo:]", jobs.calls)
	}
	if summary.Checked != 1 || len(summary.Updated) != 1 || summary.Updated[0] != "demo" {
		t.Fatalf("summary = %+v", summary)
	}
	// E26: right after a successful unattended update the package reads as
	// current at the HEAD that was installed, not as unchecked.
	if memo := service.Checker().Cached("demo"); memo.Status != updatecheck.StatusCurrent || memo.LatestCommit != "new" {
		t.Fatalf("post-update memo = %+v, want current@new", memo)
	}
}

// Contract: pinned and explicitly paused packages are checked but never
// updated unattended, with a stable skip reason for the maintenance API.
func TestPassSkipsPinnedAndPausedPackages(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"pinned": {Name: "pinned", Source: "github", SourcePath: "https://github.com/acme/pinned@v1.2.3", Ref: "v1.2.3", Commit: "old"},
		"paused": {Name: "paused", Source: "github", SourcePath: "https://github.com/acme/paused", Commit: "old", AutoUpdate: boolPtr(false)},
	})
	jobs := &maintJobs{}
	service := enabledService(home, &maintRunner{sha: "new"}, jobs, &maintAgent{}, &busySequence{})

	summary := service.RunPass(context.Background())
	if len(jobs.calls) != 0 {
		t.Fatalf("StartUpdate calls = %v, want none", jobs.calls)
	}
	want := map[string]string{"pinned": "pinned", "paused": "auto_update_disabled"}
	for _, skip := range summary.Skipped {
		delete(want, skip.Name)
		if skip.Reason == "" {
			t.Fatalf("empty skip reason: %+v", skip)
		}
	}
	if len(want) != 0 {
		t.Fatalf("missing skips: %v; summary=%+v", want, summary)
	}
}

// Contract: active executions are retried up to three checks in a pass and
// then reported as deferred without starting an update.
func TestPassDefersAfterThreeBusyChecks(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"busy": {Name: "busy", Source: "github", SourcePath: "https://github.com/acme/busy", Commit: "old"},
	})
	jobs := &maintJobs{}
	busy := &busySequence{values: []bool{true, true, true}}
	sleeps := 0
	service := enabledService(home, &maintRunner{sha: "new"}, jobs, &maintAgent{}, busy)
	service.sleep = func(context.Context, time.Duration) error { sleeps++; return nil }

	summary := service.RunPass(context.Background())
	if busy.calls != 3 || sleeps != 2 {
		t.Fatalf("busy checks=%d sleeps=%d, want 3 checks and 2 retry waits", busy.calls, sleeps)
	}
	if len(jobs.calls) != 0 || len(summary.Skipped) != 1 || summary.Skipped[0].Reason != "deferred" {
		t.Fatalf("jobs=%v summary=%+v", jobs.calls, summary)
	}
	if cached := service.Checker().Cached("busy"); cached.Status != updatecheck.StatusDeferred {
		t.Fatalf("cached status = %q, want deferred", cached.Status)
	}
}

func TestPassRetriesBusyPackageThenUpdates(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"busy": {Name: "busy", Source: "github", SourcePath: "https://github.com/acme/busy", Commit: "old"},
	})
	jobs := &maintJobs{}
	busy := &busySequence{values: []bool{true, false}}
	service := enabledService(home, &maintRunner{sha: "new"}, jobs, &maintAgent{}, busy)
	service.RunPass(context.Background())
	if busy.calls != 2 || len(jobs.calls) != 1 {
		t.Fatalf("busy checks=%d jobs=%v", busy.calls, jobs.calls)
	}
}

func TestPassFailsClosedWhenBusyDetectionErrors(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Source: "github", SourcePath: "https://github.com/acme/demo", Commit: "old"},
	})
	jobs := &maintJobs{}
	service := enabledService(home, &maintRunner{sha: "new"}, jobs, &maintAgent{}, &busySequence{err: errors.New("manifest unavailable")})
	summary := service.RunPass(context.Background())
	if len(jobs.calls) != 0 || len(summary.Errors) != 1 || !strings.Contains(summary.Errors[0], "manifest unavailable") ||
		len(summary.Skipped) != 1 || summary.Skipped[0].Reason != "busy_check_error" {
		t.Fatalf("jobs=%v summary=%+v", jobs.calls, summary)
	}
}

// Contract: disabling updates suppresses checks and installs, but boot restore
// still restarts a registry entry that was intended to be running.
func TestDisabledPassStillRestoresDeadRunningPackageOnRecordedPort(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {
			Name: "demo", Status: "running",
			Runtime: packages.RuntimeInfo{Port: intPtr(8123), PID: intPtr(999), BootID: "old-boot"},
		},
	})
	jobs := &maintJobs{}
	agent := &maintAgent{}
	service := enabledService(home, &maintRunner{sha: "unused"}, jobs, agent, &busySequence{})
	service.enabled = func() (bool, string) { return false, "disabled by AGENTFIELD_PACKAGE_AUTO_UPDATE" }
	service.portAvailable = func(port int) bool { return port == 8123 }

	summary := service.RunPass(context.Background())
	if len(agent.runs) != 1 || agent.runs[0].name != "demo" || agent.runs[0].port != 8123 {
		t.Fatalf("restore runs = %+v", agent.runs)
	}
	if !agent.runs[0].preferred {
		t.Fatalf("restore port was not marked as an internal preference: %+v", agent.runs[0])
	}
	if len(summary.Restored) != 1 || summary.Restored[0] != "demo" || summary.Checked != 0 || len(jobs.calls) != 0 {
		t.Fatalf("summary=%+v jobs=%v", summary, jobs.calls)
	}
	status := service.Status()
	if status.Enabled || status.Reason == "" {
		t.Fatalf("maintenance status = %+v", status)
	}
}

func TestRestoreLeavesHealthyRecordedPortUntouched(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"live": {Name: "live", Status: "running", Runtime: packages.RuntimeInfo{Port: intPtr(8123), PID: intPtr(42)}},
	})
	agent := &maintAgent{}
	service := enabledService(home, &maintRunner{err: errors.New("unused")}, &maintJobs{}, agent, &busySequence{})
	service.processAlive = func(packages.RuntimeInfo) bool { return false }
	service.healthProbe = func(context.Context, int, string) packages.HealthIdentity {
		return packages.HealthIdentity{Healthy: true, NodeID: "live"}
	}
	service.RunPass(context.Background())
	if len(agent.runs) != 0 {
		t.Fatalf("live process was restarted: %+v", agent.runs)
	}
}

func TestRestoreDistinguishesForeignFromAnonymousHealthyPorts(t *testing.T) {
	for _, test := range []struct {
		name         string
		identity     packages.HealthIdentity
		wantRuns     int
		wantRestored int
	}{
		{name: "foreign node restarts on a free port", identity: packages.HealthIdentity{Healthy: true, NodeID: "somebody-else"}, wantRuns: 1, wantRestored: 1},
		{name: "anonymous Go health is already our node", identity: packages.HealthIdentity{Healthy: true}},
	} {
		t.Run(test.name, func(t *testing.T) {
			home := t.TempDir()
			writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
				"demo": {
					Name: "demo", DesiredState: packages.DesiredStateRunning,
					Runtime: packages.RuntimeInfo{Port: intPtr(8123), PID: intPtr(42), StartTime: "known"},
				},
			})
			agent := &maintAgent{}
			service := enabledService(home, &maintRunner{err: errors.New("unused")}, &maintJobs{}, agent, &busySequence{})
			service.enabled = func() (bool, string) { return false, "disabled" }
			service.processAlive = func(packages.RuntimeInfo) bool { return true }
			service.healthProbe = func(context.Context, int, string) packages.HealthIdentity { return test.identity }
			service.portAvailable = func(int) bool { return true }

			summary := service.RunPass(context.Background())
			if len(agent.stops) != 0 || len(agent.runs) != test.wantRuns || len(summary.Restored) != test.wantRestored {
				t.Fatalf("stops=%v runs=%+v summary=%+v", agent.stops, agent.runs, summary)
			}
			if test.wantRuns == 1 && agent.runs[0].port != 0 {
				t.Fatalf("foreign occupied port was reused: %+v", agent.runs)
			}
		})
	}
}

func TestRestoreAnonymousHealthyPortNeverDoubleStartsForDeadOrUnknownPID(t *testing.T) {
	for _, startTime := range []string{"recorded-but-dead", ""} {
		t.Run(fmt.Sprintf("start_time_%q", startTime), func(t *testing.T) {
			home := t.TempDir()
			writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
				"demo": {
					Name: "demo", DesiredState: packages.DesiredStateRunning,
					Runtime: packages.RuntimeInfo{Port: intPtr(8123), PID: intPtr(42), StartTime: startTime},
				},
			})
			agent := &maintAgent{}
			service := enabledService(home, &maintRunner{err: errors.New("unused")}, &maintJobs{}, agent, &busySequence{})
			service.enabled = func() (bool, string) { return false, "disabled" }
			service.processAlive = func(packages.RuntimeInfo) bool { return false }
			service.healthProbe = func(context.Context, int, string) packages.HealthIdentity {
				return packages.HealthIdentity{Healthy: true}
			}
			summary := service.RunPass(context.Background())
			if len(agent.runs) != 0 || len(summary.Restored) != 0 {
				t.Fatalf("anonymous healthy node was double-started: runs=%v summary=%+v", agent.runs, summary)
			}
		})
	}
}

func TestRestoreUsesDesiredStateAfterObservedStatusWasReconciled(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Status: "stopped", DesiredState: packages.DesiredStateRunning, Runtime: packages.RuntimeInfo{Port: intPtr(8123)}},
	})
	agent := &maintAgent{}
	service := enabledService(home, &maintRunner{err: errors.New("unused")}, &maintJobs{}, agent, &busySequence{})
	service.enabled = func() (bool, string) { return false, "disabled" }
	summary := service.RunPass(context.Background())
	if len(agent.runs) != 1 || len(summary.Restored) != 1 {
		t.Fatalf("dashboard-reconciled package was not restored: runs=%v summary=%+v", agent.runs, summary)
	}
}

func TestFailedRestoreIsRetriedByNextPass(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Status: "stopped", DesiredState: packages.DesiredStateRunning},
	})
	agent := &maintAgent{runErrors: []error{errors.New("startup restore failed"), errors.New("post-update restore failed"), nil}}
	service := enabledService(home, &maintRunner{err: errors.New("unused")}, &maintJobs{}, agent, &busySequence{})
	service.enabled = func() (bool, string) { return false, "disabled" }
	first := service.RunPass(context.Background())
	second := service.RunPass(context.Background())
	if len(first.Errors) == 0 || len(first.Restored) != 0 {
		t.Fatalf("first pass must report the failed restore without claiming success: %+v", first)
	}
	if len(second.Restored) != 1 || second.Restored[0] != "demo" {
		t.Fatalf("second pass did not restore the package: %+v", second)
	}
}

func TestRestoreRequiresLiveProcessAndHealthyPort(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Status: "running", DesiredState: packages.DesiredStateRunning, Runtime: packages.RuntimeInfo{Port: intPtr(8123), PID: intPtr(42), StartTime: "same"}},
	})
	for _, test := range []struct {
		name      string
		healthy   bool
		wantRuns  int
		wantStops int
	}{
		{name: "health answers", healthy: true},
		{name: "port silent", healthy: false, wantRuns: 1, wantStops: 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			agent := &maintAgent{}
			service := enabledService(home, &maintRunner{err: errors.New("unused")}, &maintJobs{}, agent, &busySequence{})
			service.enabled = func() (bool, string) { return false, "disabled" }
			service.processAlive = func(packages.RuntimeInfo) bool { return true }
			service.healthProbe = func(context.Context, int, string) packages.HealthIdentity {
				return packages.HealthIdentity{Healthy: test.healthy, NodeID: "demo"}
			}
			service.RunPass(context.Background())
			if len(agent.runs) != test.wantRuns || len(agent.stops) != test.wantStops {
				t.Fatalf("runs=%v stops=%v", agent.runs, agent.stops)
			}
		})
	}
}

func TestRestoreNeverStopsUnidentifiedLegacyPID(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"legacy": {Name: "legacy", DesiredState: packages.DesiredStateRunning, Runtime: packages.RuntimeInfo{Port: intPtr(8123), PID: intPtr(42)}},
	})
	agent := &maintAgent{}
	service := enabledService(home, &maintRunner{err: errors.New("unused")}, &maintJobs{}, agent, &busySequence{})
	service.enabled = func() (bool, string) { return false, "disabled" }
	service.processAlive = func(packages.RuntimeInfo) bool { return true }
	service.healthProbe = func(context.Context, int, string) packages.HealthIdentity { return packages.HealthIdentity{} }
	summary := service.RunPass(context.Background())
	if len(agent.stops) != 0 || len(agent.runs) != 0 || !containsSkip(summary.Skipped, "legacy", "starting") {
		t.Fatalf("legacy PID without identity was signalled or duplicated: stops=%v runs=%v summary=%+v", agent.stops, agent.runs, summary)
	}
}

func TestRestoreDoesNotDuplicateUnhealthyLiveProcessWithoutStopCapability(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"live": {
			Name:         "live",
			DesiredState: packages.DesiredStateRunning,
			Runtime:      packages.RuntimeInfo{Port: intPtr(8123), PID: intPtr(42), StartTime: "known"},
		},
	})
	agent := &runOnlyMaintAgent{}
	service := enabledService(home, &maintRunner{err: errors.New("unused")}, &maintJobs{}, agent, &busySequence{})
	service.enabled = func() (bool, string) { return false, "disabled" }
	service.processAlive = func(packages.RuntimeInfo) bool { return true }
	service.healthProbe = func(context.Context, int, string) packages.HealthIdentity { return packages.HealthIdentity{} }

	summary := service.RunPass(context.Background())
	if agent.runs != 0 || len(summary.Restored) != 0 || len(summary.Errors) == 0 ||
		!strings.Contains(summary.Errors[0], "cannot safely stop") {
		t.Fatalf("runs=%d summary=%+v", agent.runs, summary)
	}
}

func TestRestoreUsesManifestHealthcheckPath(t *testing.T) {
	home := t.TempDir()
	pkgDir := filepath.Join(home, "packages", "demo")
	if err := os.MkdirAll(pkgDir, 0o755); err != nil {
		t.Fatal(err)
	}
	manifest := "name: demo\nversion: 1.0.0\nentrypoint:\n  healthcheck: /readyz\n"
	if err := os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"), []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Path: pkgDir, DesiredState: packages.DesiredStateRunning, Runtime: packages.RuntimeInfo{Port: intPtr(8123)}},
	})
	agent := &maintAgent{}
	service := enabledService(home, &maintRunner{err: errors.New("unused")}, &maintJobs{}, agent, &busySequence{})
	service.enabled = func() (bool, string) { return false, "disabled" }
	var path string
	service.healthProbe = func(_ context.Context, _ int, candidate string) packages.HealthIdentity {
		path = candidate
		return packages.HealthIdentity{Healthy: true, NodeID: "demo"}
	}
	service.RunPass(context.Background())
	if path != "/readyz" || len(agent.runs) != 0 {
		t.Fatalf("health path=%q runs=%v", path, agent.runs)
	}
}

func TestRestoreFallsBackToAutomaticPortWhenRecordedPortIsOccupied(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"dead": {Name: "dead", Status: "running", Runtime: packages.RuntimeInfo{Port: intPtr(8123), PID: intPtr(42), BootID: "old"}},
	})
	agent := &maintAgent{}
	service := enabledService(home, &maintRunner{err: errors.New("unused")}, &maintJobs{}, agent, &busySequence{})
	service.portAvailable = func(port int) bool { return port != 8123 }
	service.RunPass(context.Background())
	if len(agent.runs) != 1 || agent.runs[0].port != 0 {
		t.Fatalf("restore runs = %+v, want automatic port", agent.runs)
	}
}

// Contract: a pre-provenance install (empty commit) updates exactly once; the
// successful install is expected to persist the newly resolved SHA.
func TestPassUpdatesUnknownCommit(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"legacy": {Name: "legacy", Source: "github", SourcePath: "https://github.com/acme/legacy"},
	})
	jobs := &maintJobs{}
	service := enabledService(home, &maintRunner{sha: "new"}, jobs, &maintAgent{}, &busySequence{})
	service.RunPass(context.Background())
	if len(jobs.calls) != 1 || jobs.calls[0] != "legacy:" {
		t.Fatalf("unknown-commit update calls = %v", jobs.calls)
	}
}

func TestStartPassRejectsConcurrentRun(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Source: "github", SourcePath: "https://github.com/acme/demo", Commit: "old"},
	})
	runner := &blockingMaintRunner{started: make(chan struct{}), release: make(chan struct{})}
	service := enabledService(home, runner, &maintJobs{}, &maintAgent{}, &busySequence{})
	if err := service.StartPass(); err != nil {
		t.Fatalf("first pass did not start: %v", err)
	}
	<-runner.started
	if err := service.StartPass(); !errors.Is(err, ErrPassAlreadyRunning) {
		t.Fatalf("concurrent pass error = %v, want already running", err)
	}
	close(runner.release)
	deadline := time.Now().Add(time.Second)
	for service.Status().LastRun == nil && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if service.Status().LastRun == nil {
		t.Fatal("first pass did not finish")
	}
}

func TestBusyPackageJobRetriesThenDefers(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Source: "github", SourcePath: "https://github.com/acme/demo", Commit: "old"},
	})
	jobs := &maintJobs{startErrors: []error{packagejobs.ErrBusy, packagejobs.ErrBusy, packagejobs.ErrBusy}}
	service := enabledService(home, &maintRunner{sha: "new"}, jobs, &maintAgent{}, &busySequence{})
	summary := service.RunPass(context.Background())
	if len(jobs.calls) != 3 || len(summary.Errors) != 0 || len(summary.Skipped) != 1 || summary.Skipped[0].Reason != "deferred" {
		t.Fatalf("calls=%v summary=%+v", jobs.calls, summary)
	}
}

type pendingMaintJobs struct{ started chan struct{} }

func (j *pendingMaintJobs) StartUpdate(name, _ string) (*packagejobs.Job, error) {
	close(j.started)
	return &packagejobs.Job{ID: name + "-job"}, nil
}
func (j *pendingMaintJobs) GetJob(id string) (*packagejobs.Job, bool) {
	return &packagejobs.Job{ID: id, Status: packagejobs.StatusRunning}, true
}

func TestLifecycleCancellationStopsRunNowPass(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Source: "github", SourcePath: "https://github.com/acme/demo", Commit: "old"},
	})
	jobs := &pendingMaintJobs{started: make(chan struct{})}
	service := New(Config{
		AgentFieldHome: home, Checker: updatecheck.NewChecker(&maintRunner{sha: "new"}), Jobs: jobs,
		Enabled: func() (bool, string) { return true, "" }, HealthProbe: func(context.Context, int, string) packages.HealthIdentity { return packages.HealthIdentity{} },
		Sleep: sleepContext,
	})
	lifecycle, cancel := context.WithCancel(context.Background())
	service.SetLifecycleContext(lifecycle)
	if err := service.StartPass(); err != nil {
		t.Fatalf("run-now pass did not start: %v", err)
	}
	<-jobs.started
	cancel()
	waitCtx, waitCancel := context.WithTimeout(context.Background(), time.Second)
	defer waitCancel()
	if err := service.WaitForIdle(waitCtx); err != nil {
		t.Fatalf("pass did not stop after lifecycle cancellation: %v", err)
	}
}

func TestUpdateJobWaitIsBounded(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Source: "github", SourcePath: "https://github.com/acme/demo", Commit: "old"},
	})
	service := New(Config{
		AgentFieldHome: home, Checker: updatecheck.NewChecker(&maintRunner{sha: "new"}), Jobs: &pendingMaintJobs{started: make(chan struct{})},
		Enabled: func() (bool, string) { return true, "" }, HealthProbe: func(context.Context, int, string) packages.HealthIdentity { return packages.HealthIdentity{} },
		Sleep: sleepContext, JobWaitTimeout: 10 * time.Millisecond,
	})
	summary := service.RunPass(context.Background())
	if len(summary.Errors) != 1 || !strings.Contains(summary.Errors[0], context.DeadlineExceeded.Error()) {
		t.Fatalf("summary=%+v", summary)
	}
}

func TestRegistryAccessAndAutoUpdateRoundTripPreserveSiblings(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo":    {Name: "demo", Source: "github", SourcePath: "https://github.com/acme/demo"},
		"sibling": {Name: "sibling", Version: "2.0.0", Source: "local"},
	})
	changes := 0
	service := New(Config{AgentFieldHome: home, OnRegistryChange: func() { changes++ }})
	entries, err := service.Entries()
	if err != nil || len(entries) != 1 || entries[0].ID != "demo" {
		t.Fatalf("entries=%+v err=%v", entries, err)
	}
	entry, ok, err := service.RegistryEntry("sibling")
	if err != nil || !ok || entry.Version != "2.0.0" {
		t.Fatalf("entry=%+v ok=%v err=%v", entry, ok, err)
	}
	updated, err := service.SetAutoUpdate("demo", false)
	if err != nil || updated.AutoUpdateEnabled() || changes != 1 {
		t.Fatalf("updated=%+v changes=%d err=%v", updated, changes, err)
	}
	registry, err := loadRegistryFile(filepath.Join(home, "installed.yaml"))
	if err != nil || registry.Installed["sibling"].Version != "2.0.0" {
		t.Fatalf("sibling was not preserved: registry=%+v err=%v", registry, err)
	}
	info, err := os.Stat(filepath.Join(home, "installed.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("registry mode changed after auto-update write: mode=%v", info.Mode().Perm())
	}
}

func TestSetAutoUpdateIgnoresLeftoverFixedTempFileAndPreservesMode(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo"},
	})
	registryPath := filepath.Join(home, "installed.yaml")
	if err := os.Chmod(registryPath, 0o640); err != nil {
		t.Fatal(err)
	}
	leftover := registryPath + ".tmp"
	if err := os.WriteFile(leftover, []byte("leftover"), 0o777); err != nil {
		t.Fatal(err)
	}
	service := New(Config{AgentFieldHome: home})
	if _, err := service.SetAutoUpdate("demo", false); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(registryPath)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o640 {
		t.Fatalf("registry mode=%v, want 0640", info.Mode().Perm())
	}
	contents, err := os.ReadFile(leftover)
	if err != nil || string(contents) != "leftover" {
		t.Fatalf("leftover temp file was reused: contents=%q err=%v", contents, err)
	}
}

func TestUpdatesEnabledEnvironmentMatrix(t *testing.T) {
	gitPath, gitErr := exec.LookPath("git")
	t.Setenv("AGENTFIELD_PACKAGE_AUTO_UPDATE", "off")
	if enabled, reason := updatesEnabled(); enabled || !strings.Contains(reason, "AGENTFIELD_PACKAGE_AUTO_UPDATE") {
		t.Fatalf("enabled=%v reason=%q", enabled, reason)
	}
	t.Setenv("AGENTFIELD_PACKAGE_AUTO_UPDATE", "")
	t.Setenv("PATH", t.TempDir())
	if enabled, reason := updatesEnabled(); enabled || !strings.Contains(reason, "git") {
		t.Fatalf("enabled=%v reason=%q", enabled, reason)
	}
	if gitErr == nil {
		t.Setenv("PATH", filepath.Dir(gitPath))
		if enabled, reason := updatesEnabled(); !enabled || reason != "" {
			t.Fatalf("enabled=%v reason=%q", enabled, reason)
		}
	}
}

func TestConfiguredIntervalClampsToFifteenMinutes(t *testing.T) {
	t.Setenv("AGENTFIELD_PACKAGE_UPDATE_INTERVAL", "1m")
	if got := configuredInterval(); got != 15*time.Minute {
		t.Fatalf("interval = %s, want 15m minimum", got)
	}
	t.Setenv("AGENTFIELD_PACKAGE_UPDATE_INTERVAL", "not-a-duration")
	if got := configuredInterval(); got != 6*time.Hour {
		t.Fatalf("invalid interval = %s, want 6h default", got)
	}
}

func TestConfiguredIntervalUsesDefaultAndValidConfiguredValue(t *testing.T) {
	t.Setenv("AGENTFIELD_PACKAGE_UPDATE_INTERVAL", "")
	if got := configuredInterval(); got != defaultInterval {
		t.Fatalf("default interval=%s", got)
	}
	t.Setenv("AGENTFIELD_PACKAGE_UPDATE_INTERVAL", "45m")
	if got := configuredInterval(); got != 45*time.Minute {
		t.Fatalf("configured interval=%s", got)
	}
}

func TestRunStopsOnLifecycleCancellationAfterStartupPass(t *testing.T) {
	home := t.TempDir()
	service := New(Config{
		AgentFieldHome: home, Interval: time.Millisecond,
		Enabled: func() (bool, string) { return false, "disabled" },
		Sleep:   func(context.Context, time.Duration) error { return nil },
	})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { service.Run(ctx); close(done) }()
	deadline := time.Now().Add(time.Second)
	for service.Status().LastRun == nil && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("maintenance loop ignored lifecycle cancellation")
	}
}

func TestCanceledLifecycleRejectsRunNowAndActiveContextCanReset(t *testing.T) {
	service := New(Config{AgentFieldHome: t.TempDir()})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	service.SetLifecycleContext(ctx)
	if err := service.StartPass(); !errors.Is(err, ErrShuttingDown) {
		t.Fatalf("canceled lifecycle error = %v, want shutting down", err)
	}
	service.SetLifecycleContext(context.TODO())
	if err := service.Stop(context.Background()); err != nil {
		t.Fatalf("stop service: %v", err)
	}
	if err := service.StartPass(); !errors.Is(err, ErrShuttingDown) {
		t.Fatalf("stopped service error = %v, want shutting down", err)
	}
}

func TestStopWaitIsBoundedForNonCooperativeCheck(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Source: "github", SourcePath: "https://github.com/acme/demo", Commit: "old"},
	})
	runner := &blockingMaintRunner{started: make(chan struct{}), release: make(chan struct{})}
	service := enabledService(home, runner, &maintJobs{}, &maintAgent{}, &busySequence{})
	if err := service.StartPass(); err != nil {
		t.Fatalf("pass did not start: %v", err)
	}
	<-runner.started
	waitCtx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if err := service.Stop(waitCtx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("stop err=%v", err)
	}
	close(runner.release)
	cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), time.Second)
	defer cleanupCancel()
	if err := service.WaitForIdle(cleanupCtx); err != nil {
		t.Fatal(err)
	}
}

func TestPassSummarizesRegistryAndCheckerErrorsAndIgnoresCurrent(t *testing.T) {
	home := t.TempDir()
	if err := os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed: ["), 0o600); err != nil {
		t.Fatal(err)
	}
	service := enabledService(home, &maintRunner{sha: "unused"}, &maintJobs{}, &maintAgent{}, &busySequence{})
	if summary := service.RunPass(context.Background()); len(summary.Errors) != 1 || summary.FinishedAt.IsZero() {
		t.Fatalf("invalid-registry summary=%+v", summary)
	}

	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"current": {Name: "current", Source: "github", SourcePath: "https://github.com/acme/current", Commit: "same"},
		"error":   {Name: "error", Source: "github", SourcePath: "https://github.com/acme/error", Commit: "old"},
	})
	runner := &sequenceMaintRunner{results: []struct {
		output string
		err    error
	}{{output: "same\tHEAD\n"}, {err: errors.New("remote failed")}}}
	service = enabledService(home, runner, &maintJobs{}, &maintAgent{}, &busySequence{})
	summary := service.RunPass(context.Background())
	if summary.Checked != 2 || len(summary.Updated) != 0 || len(summary.Errors) != 1 || !strings.Contains(summary.Errors[0], "remote failed") {
		t.Fatalf("summary=%+v", summary)
	}
}

func TestMaintenancePassRecoversPanicAndRecordsItInSummary(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"malformed": {
			Name: "malformed", Status: "running", DesiredState: packages.DesiredStateRunning,
			Runtime: packages.RuntimeInfo{Port: intPtr(8123), PID: intPtr(42)},
		},
	})
	service := enabledService(home, &maintRunner{sha: "unused"}, &maintJobs{}, &maintAgent{}, &busySequence{})
	service.enabled = func() (bool, string) { return false, "disabled" }
	service.healthProbe = func(context.Context, int, string) packages.HealthIdentity {
		panic("malformed runtime probe")
	}

	summary := service.RunPass(context.Background())
	if summary.FinishedAt.IsZero() || len(summary.Errors) != 1 || !strings.Contains(summary.Errors[0], "malformed runtime probe") {
		t.Fatalf("recovered summary=%+v", summary)
	}
	status := service.Status()
	if status.LastRun == nil || len(status.LastRun.Errors) != 1 {
		t.Fatalf("panic was not persisted in maintenance status: %+v", status)
	}
}

func TestBusyRetryCancellationAndJobTerminalErrors(t *testing.T) {
	canceled := errors.New("retry canceled")
	service := New(Config{
		AgentFieldHome: t.TempDir(), Executions: &busySequence{values: []bool{true}},
		Sleep: func(context.Context, time.Duration) error { return canceled },
	})
	if ready, err := service.waitUntilIdle(context.Background(), "demo"); ready || !errors.Is(err, canceled) {
		t.Fatalf("ready=%v err=%v", ready, err)
	}

	for _, test := range []struct {
		name string
		jobs JobManager
		want string
	}{
		{name: "manager unavailable", want: "unavailable"},
		{name: "job disappeared", jobs: &maintJobs{missing: true}, want: "disappeared"},
		{name: "failed without message", jobs: &maintJobs{result: packagejobs.StatusFailed}, want: "update failed"},
		{name: "failed with message", jobs: &maintJobs{result: packagejobs.StatusFailed, jobError: "build broke"}, want: "build broke"},
	} {
		t.Run(test.name, func(t *testing.T) {
			service := New(Config{AgentFieldHome: t.TempDir(), Jobs: test.jobs, Sleep: func(context.Context, time.Duration) error { return nil }})
			err := service.updateOne(context.Background(), "demo")
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("err=%v", err)
			}
		})
	}

	jobs := &maintJobs{startErrors: []error{packagejobs.ErrBusy}}
	service = New(Config{AgentFieldHome: t.TempDir(), Jobs: jobs, Sleep: func(context.Context, time.Duration) error { return canceled }})
	if err := service.updateOne(context.Background(), "demo"); !errors.Is(err, canceled) {
		t.Fatalf("busy retry err=%v", err)
	}
}

func TestRestoreReportsFailureStoppingUnhealthyLiveProcess(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", DesiredState: packages.DesiredStateRunning, Runtime: packages.RuntimeInfo{Port: intPtr(8123), PID: intPtr(42), StartTime: "known"}},
	})
	agent := &maintAgent{stopError: errors.New("cannot stop")}
	service := enabledService(home, &maintRunner{err: errors.New("unused")}, &maintJobs{}, agent, &busySequence{})
	service.enabled = func() (bool, string) { return false, "disabled" }
	service.processAlive = func(packages.RuntimeInfo) bool { return true }
	service.healthProbe = func(context.Context, int, string) packages.HealthIdentity { return packages.HealthIdentity{} }
	summary := service.RunPass(context.Background())
	if len(agent.runs) != 0 || len(summary.Errors) != 1 || !strings.Contains(summary.Errors[0], "stop unhealthy") {
		t.Fatalf("runs=%v summary=%+v", agent.runs, summary)
	}
}

func TestDefaultHealthAndPortProbes(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if request.URL.Path == "/readyz" {
			_, _ = w.Write([]byte(`{"node_id":"demo"}`))
			return
		}
		http.NotFound(w, request)
	}))
	defer server.Close()
	address := strings.TrimPrefix(server.URL, "http://")
	_, portText, err := net.SplitHostPort(address)
	if err != nil {
		t.Fatal(err)
	}
	var port int
	if _, err := fmt.Sscan(portText, &port); err != nil {
		t.Fatal(err)
	}
	if identity := packages.ProbeHealthIdentity(context.Background(), port, "/readyz"); !identity.Healthy || identity.NodeID != "demo" {
		t.Fatal("healthy loopback endpoint was not detected")
	}
	server.Close()
	if packages.ProbeHealthIdentity(context.Background(), port, "/health").Healthy {
		t.Fatal("closed health endpoint was reported healthy")
	}
	if !isPortAvailable(port) {
		t.Fatalf("port %d should be available after server close", port)
	}
	listener, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	if isPortAvailable(port) {
		t.Fatalf("occupied port %d was reported available", port)
	}
}

func TestRegistryErrorContracts(t *testing.T) {
	home := t.TempDir()
	service := New(Config{AgentFieldHome: home})
	if entries, err := service.Entries(); err != nil || len(entries) != 0 {
		t.Fatalf("missing registry entries=%v err=%v", entries, err)
	}
	if _, ok, err := service.RegistryEntry("missing"); err != nil || ok {
		t.Fatalf("missing entry ok=%v err=%v", ok, err)
	}
	if _, err := service.SetAutoUpdate("missing", false); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("set missing err=%v", err)
	}
	if err := os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed: ["), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := service.Entries(); err == nil {
		t.Fatal("invalid registry was accepted by Entries")
	}
	if _, _, err := service.RegistryEntry("demo"); err == nil {
		t.Fatal("invalid registry was accepted by RegistryEntry")
	}
	if _, err := service.SetAutoUpdate("demo", true); err == nil {
		t.Fatal("invalid registry was accepted by SetAutoUpdate")
	}
	if err := os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("{}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	registry, err := loadRegistryFile(filepath.Join(home, "installed.yaml"))
	if err != nil || registry.Installed == nil {
		t.Fatalf("empty registry=%+v err=%v", registry, err)
	}
}

func TestC6ContainerLegacyStoppedEntryMigratesToRunningAndRestores(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Status: "stopped"},
	})
	agent := &maintAgent{}
	service := enabledService(home, &maintRunner{sha: "unused"}, &maintJobs{}, agent, &busySequence{})
	service.hostedInContainer = func() bool { return true }
	service.enabled = func() (bool, string) { return false, "disabled" }
	summary := service.RunPass(context.Background())
	entry := readMaintEntry(t, home, "demo")
	if entry.DesiredState != packages.DesiredStateRunning || len(agent.runs) != 1 || len(summary.Restored) != 1 {
		t.Fatalf("entry=%+v runs=%v summary=%+v", entry, agent.runs, summary)
	}
}

func TestC7LocalLegacyStoppedEntryMigratesToStoppedWithoutRestore(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Status: "stopped"},
	})
	agent := &maintAgent{}
	service := enabledService(home, &maintRunner{sha: "unused"}, &maintJobs{}, agent, &busySequence{})
	service.enabled = func() (bool, string) { return false, "disabled" }
	service.RunPass(context.Background())
	entry := readMaintEntry(t, home, "demo")
	if entry.DesiredState != packages.DesiredStateStopped || len(agent.runs) != 0 {
		t.Fatalf("entry=%+v runs=%v", entry, agent.runs)
	}
}

func TestC8ExplicitStopAfterMigrationIsNotResurrected(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Status: "stopped"},
	})
	agent := &maintAgent{}
	service := enabledService(home, &maintRunner{sha: "unused"}, &maintJobs{}, agent, &busySequence{})
	service.hostedInContainer = func() bool { return true }
	service.enabled = func() (bool, string) { return false, "disabled" }
	service.RunPass(context.Background())
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Status: "stopped", DesiredState: packages.DesiredStateStopped},
	})
	agent.runs = nil
	service.RunPass(context.Background())
	if len(agent.runs) != 0 || readMaintEntry(t, home, "demo").DesiredState != packages.DesiredStateStopped {
		t.Fatalf("explicit stop was resurrected: runs=%v entry=%+v", agent.runs, readMaintEntry(t, home, "demo"))
	}
}

func TestC9ExistingDesiredStateIsUntouchedByContainerMigration(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Status: "running", DesiredState: packages.DesiredStateStopped},
	})
	agent := &maintAgent{}
	service := enabledService(home, &maintRunner{sha: "unused"}, &maintJobs{}, agent, &busySequence{})
	service.hostedInContainer = func() bool { return true }
	service.enabled = func() (bool, string) { return false, "disabled" }
	service.RunPass(context.Background())
	if readMaintEntry(t, home, "demo").DesiredState != packages.DesiredStateStopped || len(agent.runs) != 0 {
		t.Fatalf("entry=%+v runs=%v", readMaintEntry(t, home, "demo"), agent.runs)
	}
}

func TestC11TimedOutRestoreDoesNotBlockLaterRunNowPass(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Status: "stopped", DesiredState: packages.DesiredStateRunning},
	})
	agent := &blockingRestoreAgent{started: make(chan struct{}), release: make(chan struct{})}
	service := enabledService(home, &maintRunner{sha: "unused"}, &maintJobs{}, agent, &busySequence{})
	service.enabled = func() (bool, string) { return false, "disabled" }
	service.restoreTimeout = 10 * time.Millisecond
	summary := service.RunPass(context.Background())
	if len(summary.Errors) == 0 || !strings.Contains(summary.Errors[0], "timed out") {
		t.Fatalf("summary=%+v", summary)
	}
	close(agent.release)
	if err := service.StartPass(); err != nil {
		t.Fatalf("run-now after timeout: %v", err)
	}
	waitCtx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := service.WaitForIdle(waitCtx); err != nil {
		t.Fatalf("later pass did not finish: %v", err)
	}
}

func TestC12RestoreFailuresUseOneFiveFifteenThenIntervalBackoff(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Status: "stopped", DesiredState: packages.DesiredStateRunning},
	})
	base := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	service := enabledService(home, &maintRunner{sha: "unused"}, &maintJobs{}, &alwaysFailMaintAgent{err: errors.New("cold start")}, &busySequence{})
	service.enabled = func() (bool, string) { return false, "disabled" }
	service.now = func() time.Time { return base }
	service.interval = time.Hour
	for index, want := range []time.Duration{time.Minute, 5 * time.Minute, 15 * time.Minute, time.Hour} {
		service.RunPass(context.Background())
		if got := service.Status().NextRunAt.Sub(base); got != want {
			t.Fatalf("pass %d delay=%s want=%s", index+1, got, want)
		}
	}
}

func TestC13CleanPassResetsBackoffAndUsesConfiguredInterval(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Status: "stopped", DesiredState: packages.DesiredStateRunning},
	})
	base := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	service := enabledService(home, &maintRunner{sha: "unused"}, &maintJobs{}, &alwaysFailMaintAgent{err: errors.New("cold start")}, &busySequence{})
	service.enabled = func() (bool, string) { return false, "disabled" }
	service.now = func() time.Time { return base }
	service.interval = time.Hour
	service.RunPass(context.Background())
	service.agent = &maintAgent{}
	service.RunPass(context.Background())
	if got := service.Status().NextRunAt.Sub(base); got != time.Hour {
		t.Fatalf("clean delay=%s", got)
	}
	service.agent = &alwaysFailMaintAgent{err: errors.New("again")}
	service.RunPass(context.Background())
	if got := service.Status().NextRunAt.Sub(base); got != time.Minute {
		t.Fatalf("backoff did not reset: %s", got)
	}
}

func TestC14BootPassUsesReadinessSettleOrTwentySecondFallback(t *testing.T) {
	for _, test := range []struct {
		name  string
		ready <-chan struct{}
		want  []time.Duration
	}{
		{name: "ready", ready: closedChannel(), want: []time.Duration{bootSettleDelay}},
		{name: "fallback", ready: make(chan struct{}), want: []time.Duration{bootDelay}},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			var sleeps []time.Duration
			service := New(Config{
				AgentFieldHome: t.TempDir(), Ready: test.ready,
				Enabled: func() (bool, string) { return false, "disabled" },
				Sleep: func(_ context.Context, duration time.Duration) error {
					sleeps = append(sleeps, duration)
					return nil
				},
				HostedInContainer: func() bool { return false }, Hosting: func() string { return packages.HostingLocal },
			})
			done := make(chan struct{})
			go func() { service.Run(ctx); close(done) }()
			deadline := time.Now().Add(time.Second)
			for !service.Status().BootPassCompleted && time.Now().Before(deadline) {
				time.Sleep(time.Millisecond)
			}
			cancel()
			<-done
			if len(sleeps) != len(test.want) {
				t.Fatalf("sleeps=%v want=%v", sleeps, test.want)
			}
		})
	}
}

func TestC15BootRestoreArmsAndClearsPackageUpdateGrace(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Status: "stopped", DesiredState: packages.DesiredStateRunning},
	})
	settling := make(chan struct{})
	release := make(chan struct{})
	states := make(chan bool, 2)
	ctx, cancel := context.WithCancel(context.Background())
	service := New(Config{
		AgentFieldHome: home, Agent: &maintAgent{}, Ready: closedChannel(),
		Enabled: func() (bool, string) { return false, "disabled" },
		Sleep:   func(context.Context, time.Duration) error { close(settling); <-release; return nil },
		OnRestoreState: func(name string, active bool) {
			if name == "demo" {
				states <- active
			}
		},
		HostedInContainer: func() bool { return false }, Hosting: func() string { return packages.HostingLocal },
	})
	done := make(chan struct{})
	go func() { service.Run(ctx); close(done) }()
	<-settling
	if active := <-states; !active {
		t.Fatal("restore grace was not armed before readiness settle")
	}
	close(release)
	if active := <-states; active {
		t.Fatal("restore grace was not cleared after the attempt")
	}
	cancel()
	<-done
}

func TestC18FailedCommitIsMemoizedAndSkippedWhileHeadIsUnchanged(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Source: "github", SourcePath: "https://github.com/acme/demo", Commit: "old"},
	})
	jobs := &maintJobs{result: packagejobs.StatusFailed, jobError: "unattended update of demo would rename the package to demo-v2"}
	service := enabledService(home, &maintRunner{sha: "failed-head"}, jobs, &maintAgent{}, &busySequence{})
	service.RunPass(context.Background())
	service.RunPass(context.Background())
	cached := service.Checker().Cached("demo")
	if len(jobs.calls) != 1 || cached.Status != updatecheck.StatusFailed || cached.LatestCommit != "failed-head" || !strings.Contains(cached.Message, "rename") {
		t.Fatalf("calls=%v cached=%+v", jobs.calls, cached)
	}
}

func TestC19MovedHeadRetriesAPreviouslyFailedUpdate(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Source: "github", SourcePath: "https://github.com/acme/demo", Commit: "old"},
	})
	runner := &sequenceMaintRunner{results: []struct {
		output string
		err    error
	}{{output: "head-1\tHEAD\n"}, {output: "head-2\tHEAD\n"}}}
	jobs := &maintJobs{result: packagejobs.StatusFailed, jobError: "build broke"}
	service := enabledService(home, runner, jobs, &maintAgent{}, &busySequence{})
	service.RunPass(context.Background())
	jobs.result = packagejobs.StatusSucceeded
	service.RunPass(context.Background())
	if len(jobs.calls) != 2 {
		t.Fatalf("moved head was not retried: %v", jobs.calls)
	}
}

func TestC24RestoreSkipsPackageWithActiveUpdateJob(t *testing.T) {
	home := t.TempDir()
	writeMaintRegistry(t, home, map[string]packages.InstalledPackage{
		"demo": {Name: "demo", Status: "stopped", DesiredState: packages.DesiredStateRunning},
	})
	agent := &maintAgent{}
	service := enabledService(home, &maintRunner{sha: "unused"}, &maintJobs{activeFor: "demo"}, agent, &busySequence{})
	service.enabled = func() (bool, string) { return false, "disabled" }
	summary := service.RunPass(context.Background())
	if len(agent.runs) != 0 || !containsSkip(summary.Skipped, "demo", "updating") {
		t.Fatalf("runs=%v summary=%+v", agent.runs, summary)
	}
}

func TestC25MaintenanceStatusMarksCompletedBootPassAndPopulatesLastRun(t *testing.T) {
	ready := closedChannel()
	ctx, cancel := context.WithCancel(context.Background())
	service := New(Config{
		AgentFieldHome: t.TempDir(), Ready: ready,
		Enabled:           func() (bool, string) { return false, "disabled" },
		Sleep:             func(context.Context, time.Duration) error { return nil },
		HostedInContainer: func() bool { return false }, Hosting: func() string { return packages.HostingLocal },
	})
	if before := service.Status(); before.BootPassCompleted || before.LastRun != nil {
		t.Fatalf("before=%+v", before)
	}
	done := make(chan struct{})
	go func() { service.Run(ctx); close(done) }()
	deadline := time.Now().Add(time.Second)
	for !service.Status().BootPassCompleted && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	after := service.Status()
	cancel()
	<-done
	if !after.BootPassCompleted || after.LastRun == nil || after.Hosting != packages.HostingLocal {
		t.Fatalf("after=%+v", after)
	}
}

func readMaintEntry(t *testing.T, home, name string) packages.InstalledPackage {
	t.Helper()
	registry, err := loadRegistryFile(filepath.Join(home, "installed.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	return registry.Installed[name]
}

func closedChannel() <-chan struct{} {
	ready := make(chan struct{})
	close(ready)
	return ready
}
