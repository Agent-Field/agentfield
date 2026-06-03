package commands

import (
	"bytes"
	"errors"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/cli/framework"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
)

type fakePackageService struct {
	installErr   error
	listErr      error
	installed    []domain.InstalledPackage
	lastSource   string
	lastOptions  domain.InstallOptions
	installCalls int
	listCalls    int
}

func (f *fakePackageService) InstallPackage(source string, options domain.InstallOptions) error {
	f.installCalls++
	f.lastSource = source
	f.lastOptions = options
	return f.installErr
}

func (f *fakePackageService) UninstallPackage(name string) error {
	return nil
}

func (f *fakePackageService) ListInstalledPackages() ([]domain.InstalledPackage, error) {
	f.listCalls++
	return f.installed, f.listErr
}

func (f *fakePackageService) GetPackageInfo(name string) (*domain.InstalledPackage, error) {
	return nil, nil
}

type fakeAgentService struct {
	runErr       error
	listErr      error
	runningAgent *domain.RunningAgent
	running      []domain.RunningAgent
	lastName     string
	lastOptions  domain.RunOptions
	runCalls     int
	listCalls    int
}

func (f *fakeAgentService) RunAgent(name string, options domain.RunOptions) (*domain.RunningAgent, error) {
	f.runCalls++
	f.lastName = name
	f.lastOptions = options
	return f.runningAgent, f.runErr
}

func (f *fakeAgentService) StopAgent(name string) error {
	return nil
}

func (f *fakeAgentService) GetAgentStatus(name string) (*domain.AgentStatus, error) {
	return nil, nil
}

func (f *fakeAgentService) ListRunningAgents() ([]domain.RunningAgent, error) {
	f.listCalls++
	return f.running, f.listErr
}

type fakeDevService struct {
	runErr      error
	lastPath    string
	lastOptions domain.DevOptions
	runCalls    int
}

func (f *fakeDevService) RunInDevMode(path string, options domain.DevOptions) error {
	f.runCalls++
	f.lastPath = path
	f.lastOptions = options
	return f.runErr
}

func (f *fakeDevService) StopDevMode(path string) error {
	return nil
}

func (f *fakeDevService) GetDevStatus(path string) (*domain.DevStatus, error) {
	return nil, nil
}

func captureStdout(t *testing.T, fn func()) string {
	t.Helper()

	oldStdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe() error = %v", err)
	}
	os.Stdout = w

	defer func() {
		os.Stdout = oldStdout
	}()

	fn()

	if err := w.Close(); err != nil {
		t.Fatalf("w.Close() error = %v", err)
	}

	var buf bytes.Buffer
	if _, err := io.Copy(&buf, r); err != nil {
		t.Fatalf("io.Copy() error = %v", err)
	}
	if err := r.Close(); err != nil {
		t.Fatalf("r.Close() error = %v", err)
	}

	return buf.String()
}

func TestInstallCommand_MetadataAndExecute(t *testing.T) {
	now := time.Now()
	tests := []struct {
		name           string
		args           []string
		wantSource     string
		wantOptions    domain.InstallOptions
		installErr     error
		listErr        error
		installed      []domain.InstalledPackage
		wantErrIs      error
		wantListCalls  int
		wantInstallRun int
	}{
		{
			name:           "success without verbose",
			args:           []string{"pkg-a"},
			wantSource:     "pkg-a",
			wantOptions:    domain.InstallOptions{Force: false, Verbose: false},
			installed:      []domain.InstalledPackage{{Name: "pkg-a", InstalledAt: now}},
			wantListCalls:  0,
			wantInstallRun: 1,
		},
		{
			name:           "success with verbose and force",
			args:           []string{"pkg-b", "--force", "--verbose"},
			wantSource:     "pkg-b",
			wantOptions:    domain.InstallOptions{Force: true, Verbose: true},
			installed:      []domain.InstalledPackage{{Name: "pkg-b", InstalledAt: now}, {Name: "pkg-c", InstalledAt: now}},
			wantListCalls:  1,
			wantInstallRun: 1,
		},
		{
			name:           "verbose continues when listing packages fails",
			args:           []string{"pkg-c", "--verbose"},
			wantSource:     "pkg-c",
			wantOptions:    domain.InstallOptions{Force: false, Verbose: true},
			listErr:        errors.New("list failed"),
			wantListCalls:  1,
			wantInstallRun: 1,
		},
		{
			name:           "install failure returns service error",
			args:           []string{"pkg-d", "--force"},
			wantSource:     "pkg-d",
			wantOptions:    domain.InstallOptions{Force: true, Verbose: false},
			installErr:     errors.New("install failed"),
			wantErrIs:      errors.New("install failed"),
			wantListCalls:  0,
			wantInstallRun: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pkgSvc := &fakePackageService{
				installErr: tt.installErr,
				listErr:    tt.listErr,
				installed:  tt.installed,
			}

			command := NewInstallCommand(&framework.ServiceContainer{
				PackageService: pkgSvc,
			})
			if got := command.GetName(); got != "install" {
				t.Fatalf("GetName() = %q, want %q", got, "install")
			}
			if got := command.GetDescription(); !strings.Contains(got, "Install") {
				t.Fatalf("GetDescription() = %q, want description containing Install", got)
			}

			cobraCmd := command.BuildCobraCommand()
			cobraCmd.SetArgs(tt.args)
			cobraCmd.SilenceUsage = true
			cobraCmd.SilenceErrors = true

			err := func() error {
				var cmdErr error
				captureStdout(t, func() {
					cmdErr = cobraCmd.Execute()
				})
				return cmdErr
			}()

			if tt.wantErrIs != nil {
				if err == nil || err.Error() != tt.wantErrIs.Error() {
					t.Fatalf("Execute() error = %v, want %v", err, tt.wantErrIs)
				}
			} else if err != nil {
				t.Fatalf("Execute() error = %v", err)
			}

			if pkgSvc.installCalls != tt.wantInstallRun {
				t.Fatalf("InstallPackage() calls = %d, want %d", pkgSvc.installCalls, tt.wantInstallRun)
			}
			if pkgSvc.lastSource != tt.wantSource {
				t.Fatalf("InstallPackage() source = %q, want %q", pkgSvc.lastSource, tt.wantSource)
			}
			if pkgSvc.lastOptions != tt.wantOptions {
				t.Fatalf("InstallPackage() options = %+v, want %+v", pkgSvc.lastOptions, tt.wantOptions)
			}
			if pkgSvc.listCalls != tt.wantListCalls {
				t.Fatalf("ListInstalledPackages() calls = %d, want %d", pkgSvc.listCalls, tt.wantListCalls)
			}
		})
	}
}

func TestRunCommand_MetadataAndExecute(t *testing.T) {
	startedAt := time.Date(2026, 4, 8, 12, 0, 0, 0, time.UTC)
	tests := []struct {
		name          string
		args          []string
		runErr        error
		listErr       error
		runningAgent  *domain.RunningAgent
		runningAgents []domain.RunningAgent
		wantErr       string
		wantName      string
		wantOptions   domain.RunOptions
		wantListCalls int
	}{
		{
			name:          "run failure returns service error",
			args:          []string{"agent-a", "--port", "8123"},
			runErr:        errors.New("run failed"),
			wantErr:       "run failed",
			wantName:      "agent-a",
			wantOptions:   domain.RunOptions{Port: 8123, Detach: true},
			wantListCalls: 0,
		},
		{
			name:          "success without verbose or detach list",
			args:          []string{"agent-b", "--detach=false"},
			runningAgent:  &domain.RunningAgent{Name: "agent-b", PID: 111, Port: 7001, Status: "running", StartedAt: startedAt},
			wantName:      "agent-b",
			wantOptions:   domain.RunOptions{Port: 0, Detach: false},
			wantListCalls: 0,
		},
		{
			name:          "verbose success lists running agents",
			args:          []string{"agent-c", "--verbose", "--port", "9000"},
			runningAgent:  &domain.RunningAgent{Name: "agent-c", PID: 222, Port: 9000, Status: "running", StartedAt: startedAt, LogFile: "/tmp/agent-c.log"},
			runningAgents: []domain.RunningAgent{{Name: "agent-c"}, {Name: "agent-d"}},
			wantName:      "agent-c",
			wantOptions:   domain.RunOptions{Port: 9000, Detach: true},
			wantListCalls: 1,
		},
		{
			name:          "verbose handles list error",
			args:          []string{"agent-d", "--verbose"},
			runningAgent:  &domain.RunningAgent{Name: "agent-d", PID: 333, Port: 7002, Status: "running", StartedAt: startedAt},
			listErr:       errors.New("list failed"),
			wantName:      "agent-d",
			wantOptions:   domain.RunOptions{Port: 0, Detach: true},
			wantListCalls: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agentSvc := &fakeAgentService{
				runErr:       tt.runErr,
				listErr:      tt.listErr,
				runningAgent: tt.runningAgent,
				running:      tt.runningAgents,
			}

			command := NewRunCommand(&framework.ServiceContainer{
				AgentService: agentSvc,
			})
			if got := command.GetName(); got != "run" {
				t.Fatalf("GetName() = %q, want %q", got, "run")
			}
			if got := command.GetDescription(); !strings.Contains(got, "Run") {
				t.Fatalf("GetDescription() = %q, want description containing Run", got)
			}

			cobraCmd := command.BuildCobraCommand()
			cobraCmd.SetArgs(tt.args)
			cobraCmd.SilenceUsage = true
			cobraCmd.SilenceErrors = true

			var err error
			captureStdout(t, func() {
				err = cobraCmd.Execute()
			})

			if tt.wantErr != "" {
				if err == nil || err.Error() != tt.wantErr {
					t.Fatalf("Execute() error = %v, want %q", err, tt.wantErr)
				}
			} else if err != nil {
				t.Fatalf("Execute() error = %v", err)
			}

			if agentSvc.runCalls != 1 {
				t.Fatalf("RunAgent() calls = %d, want 1", agentSvc.runCalls)
			}
			if agentSvc.lastName != tt.wantName {
				t.Fatalf("RunAgent() name = %q, want %q", agentSvc.lastName, tt.wantName)
			}
			if agentSvc.lastOptions != tt.wantOptions {
				t.Fatalf("RunAgent() options = %+v, want %+v", agentSvc.lastOptions, tt.wantOptions)
			}
			if agentSvc.listCalls != tt.wantListCalls {
				t.Fatalf("ListRunningAgents() calls = %d, want %d", agentSvc.listCalls, tt.wantListCalls)
			}
		})
	}
}

func TestDevCommand_MetadataAndExecute(t *testing.T) {
	tests := []struct {
		name        string
		args        []string
		runErr      error
		wantErr     string
		wantPath    string
		wantOptions domain.DevOptions
	}{
		{
			name:        "defaults to current directory",
			args:        nil,
			wantPath:    ".",
			wantOptions: domain.DevOptions{Port: 0, WatchFiles: false, Verbose: false},
		},
		{
			name:        "uses provided path and flags",
			args:        []string{"./agent", "--port", "7007", "--watch", "--verbose"},
			wantPath:    "./agent",
			wantOptions: domain.DevOptions{Port: 7007, WatchFiles: true, Verbose: true},
		},
		{
			name:        "returns dev mode error",
			args:        []string{"./broken"},
			runErr:      errors.New("dev failed"),
			wantErr:     "dev failed",
			wantPath:    "./broken",
			wantOptions: domain.DevOptions{Port: 0, WatchFiles: false, Verbose: false},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			devSvc := &fakeDevService{runErr: tt.runErr}
			command := NewDevCommand(&framework.ServiceContainer{
				DevService: devSvc,
			})

			if got := command.GetName(); got != "dev" {
				t.Fatalf("GetName() = %q, want %q", got, "dev")
			}
			if got := command.GetDescription(); !strings.Contains(got, "development mode") {
				t.Fatalf("GetDescription() = %q, want description containing development mode", got)
			}

			cobraCmd := command.BuildCobraCommand()
			cobraCmd.SetArgs(tt.args)
			cobraCmd.SilenceUsage = true
			cobraCmd.SilenceErrors = true

			var err error
			captureStdout(t, func() {
				err = cobraCmd.Execute()
			})

			if tt.wantErr != "" {
				if err == nil || err.Error() != tt.wantErr {
					t.Fatalf("Execute() error = %v, want %q", err, tt.wantErr)
				}
			} else if err != nil {
				t.Fatalf("Execute() error = %v", err)
			}

			if devSvc.runCalls != 1 {
				t.Fatalf("RunInDevMode() calls = %d, want 1", devSvc.runCalls)
			}
			if devSvc.lastPath != tt.wantPath {
				t.Fatalf("RunInDevMode() path = %q, want %q", devSvc.lastPath, tt.wantPath)
			}
			if devSvc.lastOptions != tt.wantOptions {
				t.Fatalf("RunInDevMode() options = %+v, want %+v", devSvc.lastOptions, tt.wantOptions)
			}
		})
	}
}
