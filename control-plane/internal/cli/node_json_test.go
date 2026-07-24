package cli

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

// resetLifecycleFlags restores the package-level --json flag state so tests
// that call run*Command functions directly (without re-registering flags via
// New*Command) are unaffected by ordering.
func resetLifecycleFlags(t *testing.T) {
	t.Helper()
	t.Cleanup(func() {
		listJSON = false
		stopJSON = false
		logsJSON = false
		logsFollow = false
		logsTail = 50
	})
}

func writeRegistry(t *testing.T, home string, registry *packages.InstallationRegistry) {
	t.Helper()

	data, err := yaml.Marshal(registry)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), data, 0o644))
}

func decodeNodeEnvelope(t *testing.T, output string) AgentResponse {
	t.Helper()

	var resp AgentResponse
	require.NoError(t, json.Unmarshal([]byte(output), &resp), "stdout must be a single JSON envelope, got: %s", output)
	return resp
}

// Contract: `af list --json` emits {ok:true,data:{nodes:[...],total}} with
// port only for running nodes, and nothing but the envelope on stdout.
func TestListCommandJSON(t *testing.T) {
	resetLifecycleFlags(t)
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)

	port := 8005
	writeRegistry(t, home, &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"beta-node": {
				Name:        "beta-node",
				Version:     "0.2.0",
				Description: "second node",
				Status:      "stopped",
			},
			"alpha-node": {
				Name:        "alpha-node",
				Version:     "1.0.0",
				Description: "first node",
				Status:      "running",
				Runtime:     packages.RuntimeInfo{Port: &port},
			},
		},
	})

	output := captureOutput(t, func() {
		cmd := NewListCommand()
		cmd.SetArgs([]string{"--json"})
		require.NoError(t, cmd.Execute())
	})

	resp := decodeNodeEnvelope(t, output)
	require.True(t, resp.OK)

	data := resp.Data.(map[string]interface{})
	require.Equal(t, float64(2), data["total"])
	nodes := data["nodes"].([]interface{})
	require.Len(t, nodes, 2)

	first := nodes[0].(map[string]interface{})
	require.Equal(t, "alpha-node", first["name"])
	require.Equal(t, "1.0.0", first["version"])
	require.Equal(t, "running", first["status"])
	require.Equal(t, float64(8005), first["port"])

	second := nodes[1].(map[string]interface{})
	require.Equal(t, "beta-node", second["name"])
	require.Equal(t, "stopped", second["status"])
	_, hasPort := second["port"]
	require.False(t, hasPort)
}

// Contract: an empty registry is a success with zero nodes, not an error.
func TestListCommandJSONEmpty(t *testing.T) {
	resetLifecycleFlags(t)
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)

	output := captureOutput(t, func() {
		cmd := NewListCommand()
		cmd.SetArgs([]string{"--json"})
		require.NoError(t, cmd.Execute())
	})

	resp := decodeNodeEnvelope(t, output)
	require.True(t, resp.OK)
	data := resp.Data.(map[string]interface{})
	require.Equal(t, float64(0), data["total"])
}

// Contract: a corrupt registry is {ok:false,error:{code:registry_error}} with
// a non-zero exit (cliExitError).
func TestListCommandJSONCorruptRegistry(t *testing.T) {
	resetLifecycleFlags(t)
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed: ["), 0o644))

	var execErr error
	output := captureOutput(t, func() {
		cmd := NewListCommand()
		cmd.SetArgs([]string{"--json"})
		cmd.SilenceErrors = true
		execErr = cmd.Execute()
	})

	require.Error(t, execErr)
	require.True(t, IsCLIExitError(execErr))
	require.Equal(t, 1, ExitCode(execErr))

	resp := decodeNodeEnvelope(t, output)
	require.False(t, resp.OK)
	require.NotNil(t, resp.Error)
	require.Equal(t, "registry_error", resp.Error.Code)
}

// Contract: stopping a node that is not running is a no-op success reporting
// status not_running, with no human progress text on stdout.
func TestStopCommandJSONNotRunning(t *testing.T) {
	resetLifecycleFlags(t)
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)

	writeRegistry(t, home, &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"demo": {Name: "demo", Version: "1.0.0", Status: "stopped"},
		},
	})

	output := captureOutput(t, func() {
		cmd := NewStopCommand()
		cmd.SetArgs([]string{"demo", "--json"})
		require.NoError(t, cmd.Execute())
	})

	resp := decodeNodeEnvelope(t, output)
	require.True(t, resp.OK)
	data := resp.Data.(map[string]interface{})
	require.Equal(t, "demo", data["node"])
	require.Equal(t, "not_running", data["status"])
	require.NotContains(t, output, "is not running") // no human progress line
}

// Contract: stopping a node that is not installed is a structured error with
// non-zero exit.
func TestStopCommandJSONNotInstalled(t *testing.T) {
	resetLifecycleFlags(t)
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)

	var execErr error
	output := captureOutput(t, func() {
		cmd := NewStopCommand()
		cmd.SetArgs([]string{"ghost", "--json"})
		cmd.SilenceErrors = true
		execErr = cmd.Execute()
	})

	require.Error(t, execErr)
	require.True(t, IsCLIExitError(execErr))

	resp := decodeNodeEnvelope(t, output)
	require.False(t, resp.OK)
	require.Equal(t, "stop_failed", resp.Error.Code)
	require.Contains(t, resp.Error.Message, "not installed")
}

// Contract: default (non-json) stop output for a not-running node is
// unchanged.
func TestStopCommandDefaultOutputUnchanged(t *testing.T) {
	resetLifecycleFlags(t)
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)

	writeRegistry(t, home, &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"demo": {Name: "demo", Version: "1.0.0", Status: "stopped"},
		},
	})

	output := captureOutput(t, func() {
		cmd := NewStopCommand()
		cmd.SetArgs([]string{"demo"})
		require.NoError(t, cmd.Execute())
	})
	require.Equal(t, "⚠️  Agent node demo is not running\n", output)
}

// Contract: `af logs --json` returns {node, log_path, lines} honoring --tail.
func TestLogsCommandJSON(t *testing.T) {
	resetLifecycleFlags(t)
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)

	require.NoError(t, os.MkdirAll(filepath.Join(home, "logs"), 0o755))
	logFile := filepath.Join(home, "logs", "demo.log")
	require.NoError(t, os.WriteFile(logFile, []byte("one\ntwo\nthree\nfour\n"), 0o644))

	writeRegistry(t, home, &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"demo": {
				Name:    "demo",
				Version: "1.0.0",
				Status:  "running",
				Runtime: packages.RuntimeInfo{LogFile: logFile},
			},
		},
	})

	output := captureOutput(t, func() {
		cmd := NewLogsCommand()
		cmd.SetArgs([]string{"demo", "--json", "--tail", "2"})
		require.NoError(t, cmd.Execute())
	})

	resp := decodeNodeEnvelope(t, output)
	require.True(t, resp.OK)
	data := resp.Data.(map[string]interface{})
	require.Equal(t, "demo", data["node"])
	require.Equal(t, logFile, data["log_path"])
	lines := data["lines"].([]interface{})
	require.Equal(t, []interface{}{"three", "four"}, lines)
}

// Contract: a node that has never run (no log file) yields empty lines, not an
// error — mirroring the human command.
func TestLogsCommandJSONMissingLogFile(t *testing.T) {
	resetLifecycleFlags(t)
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)

	writeRegistry(t, home, &packages.InstallationRegistry{
		Installed: map[string]packages.InstalledPackage{
			"demo": {
				Name:    "demo",
				Version: "1.0.0",
				Status:  "stopped",
				Runtime: packages.RuntimeInfo{LogFile: filepath.Join(home, "logs", "never.log")},
			},
		},
	})

	output := captureOutput(t, func() {
		cmd := NewLogsCommand()
		cmd.SetArgs([]string{"demo", "--json"})
		require.NoError(t, cmd.Execute())
	})

	resp := decodeNodeEnvelope(t, output)
	require.True(t, resp.OK)
	data := resp.Data.(map[string]interface{})
	require.Empty(t, data["lines"])
}

// Contract: --follow and --json are mutually exclusive.
func TestLogsCommandJSONFollowRejected(t *testing.T) {
	resetLifecycleFlags(t)
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)

	var execErr error
	output := captureOutput(t, func() {
		cmd := NewLogsCommand()
		cmd.SetArgs([]string{"demo", "--json", "--follow"})
		cmd.SilenceErrors = true
		execErr = cmd.Execute()
	})

	require.Error(t, execErr)
	resp := decodeNodeEnvelope(t, output)
	require.False(t, resp.OK)
	require.Equal(t, "invalid_flags", resp.Error.Code)
}

// Contract: logs for an uninstalled node is a structured error.
func TestLogsCommandJSONNotInstalled(t *testing.T) {
	resetLifecycleFlags(t)
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)

	var execErr error
	output := captureOutput(t, func() {
		cmd := NewLogsCommand()
		cmd.SetArgs([]string{"ghost", "--json"})
		cmd.SilenceErrors = true
		execErr = cmd.Execute()
	})

	require.Error(t, execErr)
	resp := decodeNodeEnvelope(t, output)
	require.False(t, resp.OK)
	require.Equal(t, "logs_failed", resp.Error.Code)
	require.Contains(t, resp.Error.Message, "not installed")
}

func TestTailFileLines(t *testing.T) {
	dir := t.TempDir()

	path := filepath.Join(dir, "log.txt")
	require.NoError(t, os.WriteFile(path, []byte("a\nb\nc\n"), 0o644))

	lines, err := tailFileLines(path, 2)
	require.NoError(t, err)
	require.Equal(t, []string{"b", "c"}, lines)

	lines, err = tailFileLines(path, 10)
	require.NoError(t, err)
	require.Equal(t, []string{"a", "b", "c"}, lines)

	empty := filepath.Join(dir, "empty.txt")
	require.NoError(t, os.WriteFile(empty, nil, 0o644))
	lines, err = tailFileLines(empty, 5)
	require.NoError(t, err)
	require.Empty(t, lines)

	_, err = tailFileLines(filepath.Join(dir, "missing.txt"), 5)
	require.Error(t, err)
}

// Guard: the JSON envelope helpers must not leak table/panel text into stdout.
func TestListCommandJSONStdoutIsPureJSON(t *testing.T) {
	resetLifecycleFlags(t)
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)

	output := captureOutput(t, func() {
		cmd := NewListCommand()
		cmd.SetArgs([]string{"--json"})
		require.NoError(t, cmd.Execute())
	})

	trimmed := strings.TrimSpace(output)
	require.True(t, strings.HasPrefix(trimmed, "{"), "stdout must start with JSON, got: %q", trimmed)
	require.True(t, strings.HasSuffix(trimmed, "}"), "stdout must end with JSON, got: %q", trimmed)
}
