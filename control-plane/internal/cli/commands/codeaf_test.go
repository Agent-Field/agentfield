package commands

import (
	"errors"
	"io"
	"os"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/cli/framework"
	"github.com/Agent-Field/agentfield/control-plane/internal/codeaf"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
)

func TestMain(m *testing.M) {
	_ = os.Setenv("AGENTFIELD_SKIP_CODEAF", "1")
	os.Exit(m.Run())
}

func TestCodeafInstallAndRunHooks(t *testing.T) {
	old := ensureCodeaf
	t.Cleanup(func() { ensureCodeaf = old })
	for _, jsonOutput := range []bool{false, true} {
		for _, succeeds := range []bool{false, true} {
			install := &fakePackageService{}
			run := &fakeAgentService{runningAgent: &domain.RunningAgent{PID: 1, Port: 1234}}
			if !succeeds {
				install.installErr = errors.New("install failed")
				run.runErr = errors.New("run failed")
			}
			services := &framework.ServiceContainer{PackageService: install, AgentService: run}
			calls := 0
			ensureCodeaf = func(codeaf.Options, io.Writer) error {
				calls++
				if install.installCalls != 1 {
					t.Error("ensure before successful install")
				}
				return nil
			}
			cmd := NewInstallCommand(services).BuildCobraCommand()
			args := []string{"fixture"}
			if jsonOutput {
				args = append(args, "--json")
			}
			cmd.SetArgs(args)
			captureStdout(t, func() { _ = cmd.Execute() })
			want := 0
			if succeeds {
				want = 1
			}
			if calls != want {
				t.Fatalf("install hook calls=%d want %d", calls, want)
			}
			calls = 0
			ensureCodeaf = func(codeaf.Options, io.Writer) error {
				calls++
				if run.runCalls != 0 {
					t.Error("ensure after node started")
				}
				return nil
			}
			cmd = NewRunCommand(services).BuildCobraCommand()
			cmd.SetArgs(args)
			captureStdout(t, func() { _ = cmd.Execute() })
			if calls != 1 {
				t.Fatalf("run hook calls=%d", calls)
			}
		}
	}
}
