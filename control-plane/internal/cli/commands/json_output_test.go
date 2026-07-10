package commands

import (
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/cli/framework"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/stretchr/testify/require"
)

func decodeJSONEnvelope(t *testing.T, output string) jsonEnvelope {
	t.Helper()

	var env jsonEnvelope
	require.NoError(t, json.Unmarshal([]byte(output), &env), "stdout must be a single JSON envelope, got: %s", output)
	return env
}

// Contract: `af install <src> --json` emits {ok:true,data:{source,status}} on
// stdout and nothing else.
func TestInstallCommandJSON(t *testing.T) {
	pkgSvc := &fakePackageService{}
	command := NewInstallCommand(&framework.ServiceContainer{PackageService: pkgSvc})

	cobraCmd := command.BuildCobraCommand()
	cobraCmd.SetArgs([]string{"./my-agent", "--json", "--force"})
	cobraCmd.SilenceUsage = true
	cobraCmd.SilenceErrors = true

	var execErr error
	output := captureStdout(t, func() {
		execErr = cobraCmd.Execute()
	})

	require.NoError(t, execErr)
	require.Equal(t, 1, pkgSvc.installCalls)
	require.Equal(t, "./my-agent", pkgSvc.lastSource)
	require.True(t, pkgSvc.lastOptions.Force)

	env := decodeJSONEnvelope(t, output)
	require.True(t, env.OK)
	data := env.Data.(map[string]interface{})
	require.Equal(t, "./my-agent", data["source"])
	require.Equal(t, "installed", data["status"])
}

// Contract: install failure emits {ok:false,error:{code:install_failed}} and
// returns the error (non-zero exit).
func TestInstallCommandJSONFailure(t *testing.T) {
	pkgSvc := &fakePackageService{installErr: errors.New("no such package")}
	command := NewInstallCommand(&framework.ServiceContainer{PackageService: pkgSvc})

	cobraCmd := command.BuildCobraCommand()
	cobraCmd.SetArgs([]string{"ghost", "--json"})
	cobraCmd.SilenceUsage = true
	cobraCmd.SilenceErrors = true

	var execErr error
	output := captureStdout(t, func() {
		execErr = cobraCmd.Execute()
	})

	require.Error(t, execErr)

	env := decodeJSONEnvelope(t, output)
	require.False(t, env.OK)
	require.NotNil(t, env.Error)
	require.Equal(t, "install_failed", env.Error.Code)
	require.Contains(t, env.Error.Message, "no such package")
}

// Contract: `af run <name> --json` emits {ok:true,data:{node,pid,port,status,
// log_file,started_at,detach}}.
func TestRunCommandJSON(t *testing.T) {
	started := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	agentSvc := &fakeAgentService{
		runningAgent: &domain.RunningAgent{
			Name:      "demo",
			PID:       4242,
			Port:      8005,
			Status:    "running",
			StartedAt: started,
			LogFile:   "/tmp/demo.log",
		},
	}
	command := NewRunCommand(&framework.ServiceContainer{AgentService: agentSvc})

	cobraCmd := command.BuildCobraCommand()
	cobraCmd.SetArgs([]string{"demo", "--json", "--port", "8005"})
	cobraCmd.SilenceUsage = true
	cobraCmd.SilenceErrors = true

	var execErr error
	output := captureStdout(t, func() {
		execErr = cobraCmd.Execute()
	})

	require.NoError(t, execErr)
	require.Equal(t, 1, agentSvc.runCalls)
	require.Equal(t, "demo", agentSvc.lastName)
	require.Equal(t, 8005, agentSvc.lastOptions.Port)
	require.True(t, agentSvc.lastOptions.Detach)

	env := decodeJSONEnvelope(t, output)
	require.True(t, env.OK)
	data := env.Data.(map[string]interface{})
	require.Equal(t, "demo", data["node"])
	require.Equal(t, float64(4242), data["pid"])
	require.Equal(t, float64(8005), data["port"])
	require.Equal(t, "running", data["status"])
	require.Equal(t, "/tmp/demo.log", data["log_file"])
	require.Equal(t, "2026-07-01T12:00:00Z", data["started_at"])
	require.Equal(t, true, data["detach"])
}

// Contract: run failure emits {ok:false,error:{code:run_failed}} and returns
// the error.
func TestRunCommandJSONFailure(t *testing.T) {
	agentSvc := &fakeAgentService{runErr: errors.New("port in use")}
	command := NewRunCommand(&framework.ServiceContainer{AgentService: agentSvc})

	cobraCmd := command.BuildCobraCommand()
	cobraCmd.SetArgs([]string{"demo", "--json"})
	cobraCmd.SilenceUsage = true
	cobraCmd.SilenceErrors = true

	var execErr error
	output := captureStdout(t, func() {
		execErr = cobraCmd.Execute()
	})

	require.Error(t, execErr)

	env := decodeJSONEnvelope(t, output)
	require.False(t, env.OK)
	require.Equal(t, "run_failed", env.Error.Code)
	require.Contains(t, env.Error.Message, "port in use")
}
