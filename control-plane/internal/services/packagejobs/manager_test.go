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
	coreservices "github.com/Agent-Field/agentfield/control-plane/internal/core/services"
	infrastorage "github.com/Agent-Field/agentfield/control-plane/internal/infrastructure/storage"
)

// Contract: an installer that cannot report a name still installs, and the job
// falls back to inferring the name from the registry. That fallback is the
// pre-existing behaviour and has to keep working — the authoritative path is an
// upgrade, not a requirement.
func TestInstallFallsBackWhenInstallerCannotReportAName(t *testing.T) {
	inst := &stubInstaller{afterInstall: []domain.InstalledPackage{{Name: "legacy-node"}}}
	// Embedding in an anonymous struct exposes only `installer`, hiding the
	// stub's InstallPackageWithResult — so the manager sees an implementation
	// that cannot report results and must take the fallback.
	var plain installer = struct{ installer }{inst}
	manager := newManager(plain, &stubAgentService{}, t.TempDir())

	job, err := manager.StartInstall("https://github.com/owner/repo", false)
	if err != nil {
		t.Fatal(err)
	}
	got := waitForJob(t, manager, job.ID)
	if got.Status != StatusSucceeded {
		t.Fatalf("job = %#v", got)
	}
	if got.PackageName != "legacy-node" {
		t.Fatalf("package name = %q, want the name inferred from the registry", got.PackageName)
	}
}

// Contract: the installer the server actually runs reports the name it
// installed. `run` reaches that behaviour through a type assertion, which fails
// *silently* — it falls back to inferring the name from a registry diff, the
// very thing that returns nothing for an in-place `superseded_by` replacement.
// Every other test here uses a stub that satisfies the interface by
// construction, so only this one would notice a production wiring (a decorator,
// a swapped implementation) that quietly drops back to the broken path.
func TestProductionInstallerReportsTheNameItInstalled(t *testing.T) {
	var service installer = coreservices.NewPackageService(nil, infrastorage.NewFileSystemAdapter(), t.TempDir())
	if _, ok := service.(resultInstaller); !ok {
		t.Fatalf("%T cannot report what it installed — install jobs would silently "+
			"fall back to the registry diff and report no name for an in-place supersede", service)
	}
}

type stubInstaller struct {
	mu           sync.Mutex
	installErr   error
	resultName   string
	block        <-chan struct{}
	installed    []domain.InstalledPackage
	afterInstall []domain.InstalledPackage
	calls        *[]string
	lastSource   string
	lastOptions  domain.InstallOptions
	listErr      error
	infoErr      error
	uninstallErr error
}

func (s *stubInstaller) InstallPackageWithResult(source string, options domain.InstallOptions) (string, error) {
	err := s.InstallPackage(source, options)
	if err != nil {
		return "", err
	}
	return s.resultName, nil
}

func (s *stubInstaller) InstallPackage(source string, options domain.InstallOptions) error {
	if s.block != nil {
		<-s.block
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastOptions = options
	s.lastSource = source
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
	inst := &stubInstaller{resultName: "demo", afterInstall: []domain.InstalledPackage{{Name: "demo"}}}
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

func TestInstallUsesAuthoritativeNameWhenRegistrySetDoesNotChange(t *testing.T) {
	inst := &stubInstaller{
		resultName:   "shared-name",
		installed:    []domain.InstalledPackage{{Name: "shared-name"}},
		afterInstall: []domain.InstalledPackage{{Name: "shared-name"}},
	}
	manager := newManager(inst, &stubAgentService{}, t.TempDir())
	job, err := manager.StartInstall("https://github.com/owner/predecessor", false)
	if err != nil {
		t.Fatal(err)
	}
	got := waitForJob(t, manager, job.ID)
	if got.PackageName != "shared-name" {
		t.Fatalf("package name = %q, want shared-name", got.PackageName)
	}
	if got.Lines[len(got.Lines)-1] != "install completed: shared-name" {
		t.Fatalf("completion line = %q", got.Lines[len(got.Lines)-1])
	}
}

func TestInstallUsesRedirectSuccessorName(t *testing.T) {
	inst := &stubInstaller{
		resultName:   "successor",
		afterInstall: []domain.InstalledPackage{{Name: "successor"}},
	}
	manager := newManager(inst, &stubAgentService{}, t.TempDir())
	job, _ := manager.StartInstall("https://github.com/owner/predecessor", false)
	got := waitForJob(t, manager, job.ID)
	if got.PackageName != "successor" {
		t.Fatalf("package name = %q, want successor", got.PackageName)
	}
}

func TestInstallIgnoresUnrelatedConcurrentRegistryAddition(t *testing.T) {
	inst := &stubInstaller{
		resultName: "job-package",
		afterInstall: []domain.InstalledPackage{
			{Name: "unrelated"},
			{Name: "job-package"},
		},
	}
	manager := newManager(inst, &stubAgentService{}, t.TempDir())
	job, _ := manager.StartInstall("https://github.com/owner/job", false)
	got := waitForJob(t, manager, job.ID)
	if got.PackageName != "job-package" {
		t.Fatalf("package name = %q, want job-package", got.PackageName)
	}
}

func TestFailedInstallReportsNoNameOrCompletion(t *testing.T) {
	inst := &stubInstaller{resultName: "must-not-leak", installErr: errors.New("boom")}
	manager := newManager(inst, &stubAgentService{}, t.TempDir())
	job, _ := manager.StartInstall("https://github.com/owner/broken", false)
	got := waitForJob(t, manager, job.ID)
	if got.PackageName != "" {
		t.Fatalf("failed install package name = %q", got.PackageName)
	}
	for _, line := range got.Lines {
		if strings.HasPrefix(line, "install completed:") {
			t.Fatalf("failed install claimed completion: %q", line)
		}
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
	job, err := manager.StartUpdate("demo", "")
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

func TestStartUpdateSelectsOverrideOrRecordedSource(t *testing.T) {
	tests := []struct {
		name       string
		override   string
		wantSource string
		wantRepo   string
		wantPath   string
	}{
		{
			name:       "catalog override",
			override:   " https://github.com/catalog/latest//agent/ ",
			wantSource: "https://github.com/catalog/latest//agent",
			wantRepo:   "https://github.com/catalog/latest",
			wantPath:   "agent",
		},
		{
			name:       "recorded source",
			override:   " ",
			wantSource: "https://github.com/recorded/original//go",
			wantRepo:   "https://github.com/recorded/original",
			wantPath:   "go",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			home := t.TempDir()
			if err := os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed:\n  demo:\n    source_path: https://github.com/recorded/original//go@main\n"), 0o600); err != nil {
				t.Fatal(err)
			}
			inst := &stubInstaller{installed: []domain.InstalledPackage{{Name: "demo"}}}
			manager := newManager(inst, &stubAgentService{}, home)

			job, err := manager.StartUpdate("demo", test.override)
			if err != nil {
				t.Fatal(err)
			}
			if job.Source != test.wantSource {
				t.Fatalf("job source = %q, want %q", job.Source, test.wantSource)
			}
			if got := waitForJob(t, manager, job.ID); got.Status != StatusSucceeded {
				t.Fatalf("job = %#v", got)
			}
			if inst.lastSource != test.wantRepo || inst.lastOptions.Path != test.wantPath {
				t.Fatalf("installer source = %q, path = %q", inst.lastSource, inst.lastOptions.Path)
			}
		})
	}
}

func TestStartUpdateValidatesOverrideAfterRegistryLookup(t *testing.T) {
	home := t.TempDir()
	if err := os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed:\n  demo:\n    source_path: https://github.com/recorded/original\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	manager := newManager(&stubInstaller{}, &stubAgentService{}, home)
	if _, err := manager.StartUpdate("demo", "not-a-github-source"); !errors.Is(err, ErrInvalidSource) {
		t.Fatalf("invalid override error = %v", err)
	}
	if _, err := manager.StartUpdate("missing", "https://github.com/catalog/latest"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("unknown package error = %v", err)
	}
}

// Contract: updating a package whose recorded source redirects (`superseded_by`)
// to a differently-named successor follows the rename. The old package is gone
// by the time the install returns, so reporting or restarting the name that went
// in would name a node that no longer exists.
func TestUpdateFollowsASupersededRename(t *testing.T) {
	home := t.TempDir()
	if err := os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed:\n  demo:\n    source_path: https://github.com/o/repo\n"), 0600); err != nil {
		t.Fatal(err)
	}
	var calls []string
	inst := &stubInstaller{
		resultName:   "demo-v2",
		installed:    []domain.InstalledPackage{{Name: "demo"}},
		afterInstall: []domain.InstalledPackage{{Name: "demo-v2"}},
		calls:        &calls,
	}
	manager := newManager(inst, &stubAgentService{running: true, calls: &calls}, home)
	job, err := manager.StartUpdate("demo", "")
	if err != nil {
		t.Fatal(err)
	}
	got := waitForJob(t, manager, job.ID)
	if got.Status != StatusSucceeded {
		t.Fatalf("job = %#v", got)
	}
	if got.PackageName != "demo-v2" {
		t.Fatalf("package name = %q, want the successor", got.PackageName)
	}
	// Stopped under the old name (that is what was running), restarted under
	// the new one (that is what is now installed).
	if strings.Join(calls, ",") != "stop:demo,install,start:demo-v2" {
		t.Fatalf("calls = %v", calls)
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
			_, err := manager.StartUpdate("demo", "")
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

// The registry-change hook fires after successful installs and uninstalls,
// and not after failures — API reads stay consistent with API writes.
func TestRegistryChangeHook(t *testing.T) {
	inst := &stubInstaller{}
	manager := newManager(inst, &stubAgentService{}, t.TempDir())
	calls := 0
	manager.SetOnRegistryChange(func() { calls++ })

	job, err := manager.StartInstall("https://github.com/o/r", false)
	if err != nil {
		t.Fatal(err)
	}
	waitForJob(t, manager, job.ID)
	if calls != 1 {
		t.Fatalf("after install calls = %d, want 1", calls)
	}

	inst.installErr = errors.New("boom")
	job, _ = manager.StartInstall("https://github.com/o/r2", false)
	waitForJob(t, manager, job.ID)
	if calls != 1 {
		t.Fatalf("after failed install calls = %d, want 1", calls)
	}
	inst.installErr = nil

	inst.installed = []domain.InstalledPackage{{Name: "gone"}}
	if err := manager.Uninstall("gone"); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("after uninstall calls = %d, want 2", calls)
	}
}

func TestRegistryChangeHookRunsBeforeJobSucceeds(t *testing.T) {
	release := make(chan struct{})
	inst := &stubInstaller{
		block:        release,
		afterInstall: []domain.InstalledPackage{{Name: "demo"}},
	}
	manager := newManager(inst, &stubAgentService{}, t.TempDir())
	var jobID string
	var status JobStatus
	manager.SetOnRegistryChange(func() {
		job, ok := manager.GetJob(jobID)
		if !ok {
			t.Errorf("job %q not found in hook", jobID)
			return
		}
		status = job.Status
	})

	job, err := manager.StartInstall("https://github.com/o/r", false)
	if err != nil {
		t.Fatal(err)
	}
	jobID = job.ID
	close(release)
	if got := waitForJob(t, manager, job.ID); got.Status != StatusSucceeded {
		t.Fatalf("status = %s", got.Status)
	}
	if status != StatusRunning {
		t.Fatalf("status observed by hook = %s, want %s", status, StatusRunning)
	}
}

func TestUpdateRegistryChangeHook(t *testing.T) {
	tests := []struct {
		name       string
		installErr error
		wantCalls  int
	}{
		{name: "success", wantCalls: 1},
		{name: "failure", installErr: errors.New("boom"), wantCalls: 0},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			home := t.TempDir()
			if err := os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed:\n  demo:\n    source_path: owner/repo@main\n"), 0o600); err != nil {
				t.Fatal(err)
			}
			inst := &stubInstaller{
				installed:  []domain.InstalledPackage{{Name: "demo"}},
				installErr: test.installErr,
			}
			manager := newManager(inst, &stubAgentService{}, home)
			calls := 0
			manager.SetOnRegistryChange(func() { calls++ })

			job, err := manager.StartUpdate("demo", "")
			if err != nil {
				t.Fatal(err)
			}
			waitForJob(t, manager, job.ID)
			if calls != test.wantCalls {
				t.Fatalf("hook calls = %d, want %d", calls, test.wantCalls)
			}
		})
	}
}

func TestSourceFromRegistry(t *testing.T) {
	tests := []struct {
		name    string
		source  string
		want    string
		wantErr error
	}{
		{name: "resolved ref", source: "owner/repo@sha", want: "https://github.com/owner/repo"},
		{name: "bare repository", source: "owner/repo", want: "https://github.com/owner/repo"},
		{name: "subdirectory ref", source: "owner/repo//subdir@ref", want: "https://github.com/owner/repo//subdir"},
		{name: "at before final subdirectory segment", source: "owner/repo//sub@dir/agent", wantErr: ErrInvalidSource},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := sourceFromRegistry(test.source)
			if !errors.Is(err, test.wantErr) {
				t.Fatalf("err = %v, want %v", err, test.wantErr)
			}
			if got != test.want {
				t.Fatalf("source = %q, want %q", got, test.want)
			}
		})
	}
}

func TestUninstallIsBusyDuringInstallAndSucceedsAfterward(t *testing.T) {
	release := make(chan struct{})
	var calls []string
	inst := &stubInstaller{
		block:        release,
		installed:    []domain.InstalledPackage{{Name: "demo"}},
		afterInstall: []domain.InstalledPackage{{Name: "demo"}},
		calls:        &calls,
	}
	manager := newManager(inst, &stubAgentService{}, t.TempDir())
	job, err := manager.StartInstall("https://github.com/o/r", false)
	if err != nil {
		t.Fatal(err)
	}

	if err := manager.Uninstall("demo"); !errors.Is(err, ErrBusy) {
		t.Fatalf("uninstall during install err = %v, want %v", err, ErrBusy)
	}
	if len(calls) != 0 {
		t.Fatalf("calls during install = %v, want none", calls)
	}

	close(release)
	if got := waitForJob(t, manager, job.ID); got.Status != StatusSucceeded {
		t.Fatalf("install status = %s", got.Status)
	}
	if err := manager.Uninstall("demo"); err != nil {
		t.Fatalf("uninstall after install: %v", err)
	}
	if got := strings.Join(calls, ","); got != "install,remove:demo" {
		t.Fatalf("calls = %s", got)
	}
}

// A catalog-source override changes WHERE the update fetches from and nothing
// else: it is still an update job — forced reinstall in place, the running
// node stopped first and restarted after — not a plain install of a new source.
func TestStartUpdateOverrideKeepsUpdateSemantics(t *testing.T) {
	home := t.TempDir()
	if err := os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed:\n  demo:\n    source_path: https://github.com/recorded/original//go\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	var calls []string
	inst := &stubInstaller{installed: []domain.InstalledPackage{{Name: "demo"}}, calls: &calls}
	manager := newManager(inst, &stubAgentService{running: true, calls: &calls}, home)

	job, err := manager.StartUpdate("demo", "https://github.com/catalog/latest")
	if err != nil {
		t.Fatal(err)
	}
	if job.Kind != JobUpdate {
		t.Fatalf("job kind = %q, want %q", job.Kind, JobUpdate)
	}
	if got := waitForJob(t, manager, job.ID); got.Status != StatusSucceeded {
		t.Fatalf("job = %#v", got)
	}
	if strings.Join(calls, ",") != "stop:demo,install,start:demo" {
		t.Fatalf("calls = %v", calls)
	}
	if !inst.lastOptions.Force {
		t.Fatal("override update was not forced")
	}
	if inst.lastSource != "https://github.com/catalog/latest" {
		t.Fatalf("installer source = %q", inst.lastSource)
	}
}
