package packagejobs

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	infrastorage "github.com/Agent-Field/agentfield/control-plane/internal/infrastructure/storage"
)

type stubInstaller struct {
	mu           sync.Mutex
	installErr   error
	block        <-chan struct{}
	installed    []domain.InstalledPackage
	afterInstall []domain.InstalledPackage
	calls        *[]string
	lastOptions  domain.InstallOptions
	listErr      error
	infoErr      error
	uninstallErr error
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
	return s.uninstallErr
}
func (s *stubInstaller) ListInstalledPackages() ([]domain.InstalledPackage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]domain.InstalledPackage(nil), s.installed...), s.listErr
}
func (s *stubInstaller) GetPackageInfo(name string) (*domain.InstalledPackage, error) {
	if s.infoErr != nil {
		return nil, s.infoErr
	}
	for _, pkg := range s.installed {
		if pkg.Name == name {
			copy := pkg
			return &copy, nil
		}
	}
	return nil, os.ErrNotExist
}

type stubAgentService struct {
	running   bool
	calls     *[]string
	statusErr error
	stopErr   error
	runErr    error
}

func (s *stubAgentService) RunAgent(name string, _ domain.RunOptions) (*domain.RunningAgent, error) {
	if s.calls != nil {
		*s.calls = append(*s.calls, "start:"+name)
	}
	return &domain.RunningAgent{Name: name}, s.runErr
}
func (s *stubAgentService) StopAgent(name string) error {
	if s.calls != nil {
		*s.calls = append(*s.calls, "stop:"+name)
	}
	s.running = false
	return s.stopErr
}
func (s *stubAgentService) GetAgentStatus(name string) (*domain.AgentStatus, error) {
	return &domain.AgentStatus{Name: name, IsRunning: s.running}, s.statusErr
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

// Exercises the real NewManager constructor without starting an installation.
func TestNewManagerRejectsInvalidSource(t *testing.T) {
	home := t.TempDir()
	registry := infrastorage.NewLocalRegistryStorage(
		infrastorage.NewFileSystemAdapter(),
		filepath.Join(home, "installed.json"),
	)
	manager := NewManager(registry, home, &stubAgentService{})
	if _, err := manager.StartInstall("not-a-github-url", false); !errors.Is(err, ErrInvalidSource) {
		t.Fatalf("err = %v", err)
	}
}

// Exercises validation branches for malformed repository and subdirectory components.
func TestValidateSourceEdgeCases(t *testing.T) {
	invalid := []string{
		"https://github.com/owner",
		"https://github.com/./repo",
		"https://github.com/-owner/repo",
		"https://github.com/owner/-repo",
		"https://github.com/owner/repo//",
		"https://github.com/owner/repo//../bad",
		"https://github.com/owner/repo///bad",
		"https://github.com/owner/repo//bad?query",
		"https://github.com/owner/repo//-bad",
		"https://github.com/owner/repo//bad//part",
	}
	for _, source := range invalid {
		if _, err := ValidateSource(source); !errors.Is(err, ErrInvalidSource) {
			t.Errorf("source %q: err = %v", source, err)
		}
	}
	if got, err := ValidateSource(" https://github.com/owner/repo/ "); err != nil || got != "https://github.com/owner/repo" {
		t.Fatalf("got %q, err = %v", got, err)
	}
}

// Exercises update registry not-found, read, YAML, entry, and invalid-source failures.
func TestStartUpdateRegistryFailures(t *testing.T) {
	tests := []struct {
		name    string
		content string
		setup   func(string)
		want    error
	}{
		{name: "missing", want: ErrNotFound},
		{name: "read", setup: func(home string) {
			requireDir := filepath.Join(home, "installed.yaml")
			if err := os.Mkdir(requireDir, 0o700); err != nil {
				t.Fatal(err)
			}
		}},
		{name: "yaml", content: "installed: ["},
		{name: "entry", content: "installed: {}\n", want: ErrNotFound},
		{name: "source", content: "installed:\n  demo:\n    source_path: invalid\n", want: ErrInvalidSource},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			home := t.TempDir()
			if test.setup != nil {
				test.setup(home)
			} else if test.content != "" {
				if err := os.WriteFile(filepath.Join(home, "installed.yaml"), []byte(test.content), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			manager := newManager(&stubInstaller{}, &stubAgentService{}, home)
			_, err := manager.StartUpdate("demo")
			if err == nil || (test.want != nil && !errors.Is(err, test.want)) {
				t.Fatalf("err = %v, want %v", err, test.want)
			}
		})
	}
}

// Exercises uninstall status, stop, and installer failure propagation.
func TestUninstallPropagatesOperationFailures(t *testing.T) {
	boom := errors.New("boom")
	tests := []struct {
		name  string
		inst  *stubInstaller
		agent *stubAgentService
	}{
		{"status", &stubInstaller{installed: []domain.InstalledPackage{{Name: "demo"}}}, &stubAgentService{statusErr: boom}},
		{"stop", &stubInstaller{installed: []domain.InstalledPackage{{Name: "demo"}}}, &stubAgentService{running: true, stopErr: boom}},
		{"remove", &stubInstaller{installed: []domain.InstalledPackage{{Name: "demo"}}, uninstallErr: boom}, &stubAgentService{}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := newManager(test.inst, test.agent, t.TempDir()).Uninstall("demo"); !errors.Is(err, boom) {
				t.Fatalf("err = %v", err)
			}
		})
	}
}

// Exercises nil agent service, list failures, absent append targets, and job eviction.
func TestManagerHelperEdgeCases(t *testing.T) {
	manager := newManager(&stubInstaller{listErr: errors.New("list")}, nil, t.TempDir())
	if got, err := ValidateSource("https://github.com/owner/repo//agents/demo/"); err != nil ||
		got != "https://github.com/owner/repo//agents/demo" {
		t.Fatalf("source=%q err=%v", got, err)
	}
	source, options := splitSubdir("https://github.com/owner/repo//agents/demo", true)
	if source != "https://github.com/owner/repo" || options.Path != "agents/demo" || !options.Force {
		t.Fatalf("source=%q options=%+v", source, options)
	}
	if running, err := manager.isRunning("demo"); err != nil || running {
		t.Fatalf("running=%v err=%v", running, err)
	}
	if got := manager.installedNames(); len(got) != 0 {
		t.Fatalf("names=%v", got)
	}
	if got := manager.discoverPackageName(nil); got != "" {
		t.Fatalf("name=%q", got)
	}
	manager.appendLine("missing", "ignored")
	if job, ok := manager.GetJob("missing"); ok || job != nil {
		t.Fatalf("job=%v ok=%v", job, ok)
	}

	manager.mu.Lock()
	for i := 0; i <= maxJobs; i++ {
		id := fmt.Sprintf("job-%d", i)
		manager.jobs[id] = &Job{ID: id}
		manager.order = append(manager.order, id)
	}
	manager.evictLocked()
	manager.mu.Unlock()
	if len(manager.order) != maxJobs {
		t.Fatalf("jobs=%d", len(manager.order))
	}
	if got := manager.ListJobs(); len(got) != maxJobs {
		t.Fatalf("listed jobs=%d", len(got))
	}
}
