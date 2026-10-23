package commands

import (
	"context"
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
	old, oldBackground := ensureCodeaf, ensureCodeafBackground
	t.Cleanup(func() { ensureCodeaf = old; ensureCodeafBackground = oldBackground })
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
			ctx := context.WithValue(context.Background(), struct{}{}, "install-context")
			ensureCodeaf = func(opts codeaf.Options, _ io.Writer) error {
				if opts.Context != ctx {
					t.Error("command context not forwarded")
				}
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
			cmd.SetContext(ctx)
			captureStdout(t, func() { _ = cmd.Execute() })
			want := 0
			if succeeds {
				want = 1
			}
			if calls != want {
				t.Fatalf("install hook calls=%d want %d", calls, want)
			}
			calls = 0
			ensureCodeafBackground = func() error {
				calls++
				if run.runCalls != 0 {
					t.Error("ensure after node started")
				}
				return errors.New("background installer unavailable")
			}
			cmd = NewRunCommand(services).BuildCobraCommand()
			cmd.SetArgs(args)
			captureStdout(t, func() {
				if err := cmd.Execute(); err != run.runErr {
					t.Errorf("background error changed node result: %v", err)
				}
			})
			if calls != 1 {
				t.Fatalf("run hook calls=%d", calls)
			}
		}
	}
}
