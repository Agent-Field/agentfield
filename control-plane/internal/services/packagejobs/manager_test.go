package packagejobs

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
)

type stubInstaller struct {
	mu           sync.Mutex
	installErr   error
	block        <-chan struct{}
	installed    []domain.InstalledPackage
	afterInstall []domain.InstalledPackage
	calls        *[]string
	lastOptions  domain.InstallOptions
}

func (s *stubInstaller) InstallPackage(_ string, options domain.InstallOptions) error {
	if s.block != nil {
		<-s.block
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastOptions = options
	if s.calls != nil {
		*s.calls = append(*s.calls, "install")
	}
	if s.installErr == nil && s.afterInstall != nil {
		s.installed = append([]domain.InstalledPackage(nil), s.afterInstall...)
	}
	return s.installErr
}
func (s *stubInstaller) UninstallPackage(name string) error {
	if s.calls != nil {
		*s.calls = append(*s.calls, "remove:"+name)
	}
	return nil
}
func (s *stubInstaller) ListInstalledPackages() ([]domain.InstalledPackage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]domain.InstalledPackage(nil), s.installed...), nil
}
func (s *stubInstaller) GetPackageInfo(name string) (*domain.InstalledPackage, error) {
	for _, pkg := range s.installed {
		if pkg.Name == name {
			copy := pkg
			return &copy, nil
		}
	}
	return nil, os.ErrNotExist
}

type stubAgentService struct {
	running bool
	calls   *[]string
}

func (s *stubAgentService) RunAgent(name string, _ domain.RunOptions) (*domain.RunningAgent, error) {
	if s.calls != nil {
		*s.calls = append(*s.calls, "start:"+name)
	}
	return &domain.RunningAgent{Name: name}, nil
}
func (s *stubAgentService) StopAgent(name string) error {
	if s.calls != nil {
		*s.calls = append(*s.calls, "stop:"+name)
	}
	s.running = false
	return nil
}
func (s *stubAgentService) GetAgentStatus(name string) (*domain.AgentStatus, error) {
	return &domain.AgentStatus{Name: name, IsRunning: s.running}, nil
}
func (s *stubAgentService) ListRunningAgents() ([]domain.RunningAgent, error) { return nil, nil }

func waitForJob(t *testing.T, manager *Manager, id string) *Job {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		job, ok := manager.GetJob(id)
		if ok && (job.Status == StatusSucceeded || job.Status == StatusFailed) {
			return job
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("job did not finish")
	return nil
}

// Contract 1: a valid GitHub install succeeds and records its package name.
func TestInstallSucceedsAndDiscoversPackageName(t *testing.T) {
	inst := &stubInstaller{afterInstall: []domain.InstalledPackage{{Name: "demo"}}}
	manager := newManager(inst, &stubAgentService{}, t.TempDir())
	job, err := manager.StartInstall("https://github.com/owner/repo", false)
	if err != nil {
		t.Fatal(err)
	}
	got := waitForJob(t, manager, job.ID)
	if got.Status != StatusSucceeded || got.PackageName != "demo" {
		t.Fatalf("job = %#v", got)
	}
}

// Contract 2: unsafe sources are rejected before a job is created.
func TestInvalidSourcesCreateNoJobs(t *testing.T) {
	manager := newManager(&stubInstaller{}, &stubAgentService{}, t.TempDir())
	for _, source := range []string{"../local", "git@github.com:o/r.git", "https://gitlab.com/o/r"} {
		if _, err := manager.StartInstall(source, false); !errors.Is(err, ErrInvalidSource) {
			t.Errorf("source %q: err = %v", source, err)
		}
	}
	if got := len(manager.ListJobs()); got != 0 {
		t.Fatalf("created %d jobs", got)
	}
}

// Contract 3: only one package job can be active.
func TestSecondInstallIsBusy(t *testing.T) {
	release := make(chan struct{})
	manager := newManager(&stubInstaller{block: release}, &stubAgentService{}, t.TempDir())
	first, err := manager.StartInstall("https://github.com/o/one", false)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := manager.StartInstall("https://github.com/o/two", false); !errors.Is(err, ErrBusy) {
		t.Fatalf("err = %v", err)
	}
	close(release)
	if got := waitForJob(t, manager, first.ID); got.Status != StatusSucceeded {
		t.Fatalf("first status = %s", got.Status)
	}
}

// Contract 4: failures release the active-job lock.
func TestFailedInstallAllowsNextInstall(t *testing.T) {
	inst := &stubInstaller{installErr: errors.New("boom")}
	manager := newManager(inst, &stubAgentService{}, t.TempDir())
	first, _ := manager.StartInstall("https://github.com/o/one", false)
	got := waitForJob(t, manager, first.ID)
	if got.Status != StatusFailed || got.Error != "boom" {
		t.Fatalf("job = %#v", got)
	}
	inst.installErr = nil
	second, err := manager.StartInstall("https://github.com/o/two", false)
	if err != nil {
		t.Fatal(err)
	}
	waitForJob(t, manager, second.ID)
}

// Contract 6: recorded progress is free of terminal ANSI escapes.
func TestJobLinesContainNoANSI(t *testing.T) {
	manager := newManager(&stubInstaller{afterInstall: []domain.InstalledPackage{{Name: "demo"}}}, &stubAgentService{}, t.TempDir())
	job, _ := manager.StartInstall("https://github.com/o/repo", false)
	manager.appendLine(job.ID, "\x1b[31mred\x1b[0m")
	got := waitForJob(t, manager, job.ID)
	for _, line := range got.Lines {
		if strings.Contains(line, "\x1b") {
			t.Fatalf("ANSI line: %q", line)
		}
	}
}

// Contract 7: uninstall stops a running package before removing it.
func TestUninstallStopsThenRemoves(t *testing.T) {
	var calls []string
	inst := &stubInstaller{installed: []domain.InstalledPackage{{Name: "demo"}}, calls: &calls}
	manager := newManager(inst, &stubAgentService{running: true, calls: &calls}, t.TempDir())
	if err := manager.Uninstall("demo"); err != nil {
		t.Fatal(err)
	}
	if strings.Join(calls, ",") != "stop:demo,remove:demo" {
		t.Fatalf("calls = %v", calls)
	}
	if err := manager.Uninstall("missing"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing err = %v", err)
	}
}

// Contract 8: update restores the prior running state and forces installation.
func TestUpdateStopsForceInstallsAndRestarts(t *testing.T) {
	home := t.TempDir()
	if err := os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed:\n  demo:\n    source_path: https://github.com/o/repo@main\n"), 0600); err != nil {
		t.Fatal(err)
	}
	var calls []string
	inst := &stubInstaller{installed: []domain.InstalledPackage{{Name: "demo"}}, calls: &calls}
	manager := newManager(inst, &stubAgentService{running: true, calls: &calls}, home)
	job, err := manager.StartUpdate("demo")
	if err != nil {
		t.Fatal(err)
	}
	if got := waitForJob(t, manager, job.ID); got.Status != StatusSucceeded {
		t.Fatalf("job = %#v", got)
	}
	if strings.Join(calls, ",") != "stop:demo,install,start:demo" {
		t.Fatalf("calls = %v", calls)
	}
	if !inst.lastOptions.Force {
		t.Fatal("update was not forced")
	}
}

// Contract 9: progress retains only the most recent 500 lines.
func TestJobLinesAreCapped(t *testing.T) {
	release := make(chan struct{})
	manager := newManager(&stubInstaller{block: release}, &stubAgentService{}, t.TempDir())
	job, _ := manager.StartInstall("https://github.com/o/repo", false)
	for i := 0; i < maxJobLines+50; i++ {
		manager.appendLine(job.ID, "line")
	}
	got, _ := manager.GetJob(job.ID)
	if len(got.Lines) != maxJobLines {
		t.Fatalf("lines = %d", len(got.Lines))
	}
	close(release)
	waitForJob(t, manager, job.ID)
}
