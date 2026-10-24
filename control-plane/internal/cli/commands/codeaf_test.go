package commands

import (
	"context"
	"errors"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/cli/framework"
	"github.com/Agent-Field/agentfield/control-plane/internal/codeaf"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
)

func TestMain(m *testing.M) {
	_ = os.Setenv("AGENTFIELD_SKIP_CODEAF", "1")
	os.Exit(m.Run())
}

func TestCodeafInstallHook(t *testing.T) {
	old := ensureCodeaf
	t.Cleanup(func() { ensureCodeaf = old })
	for _, jsonOutput := range []bool{false, true} {
		for _, succeeds := range []bool{false, true} {
			install := &fakePackageService{}
			if !succeeds {
				install.installErr = errors.New("install failed")
			}
			services := &framework.ServiceContainer{PackageService: install}
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
		}
	}
}

func TestRunDoesNotProvisionCodeaf(t *testing.T) {
	home, temporary := t.TempDir(), t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("TMPDIR", temporary)
	t.Setenv("AGENTFIELD_SKIP_CODEAF", "0")
	old := ensureCodeaf
	t.Cleanup(func() { ensureCodeaf = old })
	ensureCodeaf = func(codeaf.Options, io.Writer) error {
		t.Fatal("af run attempted CodeAF provisioning")
		return nil
	}
	run := &fakeAgentService{runningAgent: &domain.RunningAgent{PID: 1, Port: 1234}}
	cmd := NewRunCommand(&framework.ServiceContainer{AgentService: run}).BuildCobraCommand()
	cmd.SetArgs([]string{"fixture"})
	captureStdout(t, func() {
		if err := cmd.Execute(); err != nil {
			t.Fatal(err)
		}
	})
	if run.runCalls != 1 {
		t.Fatalf("run calls=%d", run.runCalls)
	}
	for _, dir := range []string{home, temporary} {
		entries, err := os.ReadDir(dir)
		if err != nil || len(entries) != 0 {
			t.Fatalf("unexpected CodeAF files in %s: %v %v", dir, entries, err)
		}
	}
	source, err := os.ReadFile("run.go")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(strings.ToLower(string(source)), "codeaf") {
		t.Fatal("af run still references CodeAF")
	}
}
