package agent

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestAgent(t *testing.T) *Agent {
	t.Helper()
	a, err := New(Config{
		NodeID:  "node-1",
		Version: "1.0.0",
		Logger:  log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	return a
}

func captureOutput(t *testing.T, fn func() error) (string, string, error) {
	t.Helper()
	stdoutR, stdoutW, _ := os.Pipe()
	stderrR, stderrW, _ := os.Pipe()

	origStdout := os.Stdout
	origStderr := os.Stderr
	os.Stdout = stdoutW
	os.Stderr = stderrW

	err := fn()

	stdoutW.Close()
	stderrW.Close()
	os.Stdout = origStdout
	os.Stderr = origStderr

	outBytes, _ := io.ReadAll(stdoutR)
	errBytes, _ := io.ReadAll(stderrR)

	return string(outBytes), string(errBytes), err
}

func TestParseCLIArgs_MergePriority(t *testing.T) {
	a := newTestAgent(t)

	tmpFile, err := os.CreateTemp(t.TempDir(), "input-*.json")
	require.NoError(t, err)
	_, err = tmpFile.WriteString(`{"source":"file","shared":2}`)
	require.NoError(t, err)
	tmpFile.Close()

	origStdin := os.Stdin
	stdinR, stdinW, _ := os.Pipe()
	os.Stdin = stdinR
	_, _ = stdinW.WriteString(`{"source":"stdin","shared":1}`)
	stdinW.Close()
	t.Cleanup(func() { os.Stdin = origStdin })

	inv, err := a.parseCLIArgs([]string{
		"--input-file", tmpFile.Name(),
		"--input", `{"source":"flag","shared":3}`,
		"--set", "shared=4",
	})
	require.NoError(t, err)

	require.NotNil(t, inv.input)
	assert.Equal(t, "flag", inv.input["source"])
	assert.Equal(t, float64(4), inv.input["shared"])
}

func TestRunCLI_ExecutesDefaultReasoner(t *testing.T) {
	a := newTestAgent(t)

	a.RegisterReasoner("greet", func(ctx context.Context, input map[string]any) (any, error) {
		assert.True(t, IsCLIMode(ctx))
		args := GetCLIArgs(ctx)
		assert.Equal(t, "Bob", args["name"])
		return fmt.Sprintf("Hello, %s", input["name"]), nil
	}, WithCLI(), WithDefaultCLI(), WithDescription("Greets a user"))

	stdout, stderr, err := captureOutput(t, func() error {
		return a.runCLI(context.Background(), []string{"--set", "name=Bob", "--output", "json"})
	})

	require.NoError(t, err)
	assert.Contains(t, stdout, "Hello, Bob")
	assert.Equal(t, "", strings.TrimSpace(stderr))
}

func TestCLIHelpersAndFormatter(t *testing.T) {
	assert.Equal(t, "", (&CLIError{}).Error())
	err := &CLIError{Code: 2, Err: errors.New("bad input")}
	assert.Equal(t, "bad input", err.Error())
	assert.ErrorIs(t, err.Unwrap(), err.Err)
	assert.Equal(t, 2, err.ExitCode())
	assert.Nil(t, (*CLIError)(nil).Unwrap())
	assert.Equal(t, 0, (*CLIError)(nil).ExitCode())
	assert.Equal(t, "text", colorText(false, ansiBold, "text"))
	assert.Contains(t, colorText(true, ansiBold, "text"), ansiBold)

	stdout, stderr, runErr := captureOutput(t, func() error {
		formatter := defaultFormatter("json", false)
		formatter(context.Background(), map[string]any{"ok": true}, nil)
		defaultFormatter("pretty", false)(context.Background(), map[string]any{"ok": true}, nil)
		defaultFormatter("yaml", false)(context.Background(), map[string]any{"ok": true}, nil)
		defaultFormatter("bogus", false)(context.Background(), map[string]any{"ok": true}, nil)
		defaultFormatter("json", false)(context.Background(), nil, nil)
		defaultFormatter("json", false)(context.Background(), nil, errors.New("boom"))
		defaultFormatter("json", false)(context.Background(), map[string]any{"bad": make(chan int)}, nil)
		return nil
	})

	require.NoError(t, runErr)
	assert.Contains(t, stdout, `{"ok":true}`)
	assert.Contains(t, stdout, "ok: true")
	assert.Contains(t, stderr, "Unknown output format bogus")
	assert.Contains(t, stderr, "Error: boom")
	assert.Contains(t, stderr, "Error encoding JSON")
}

func TestPrintListHelpAndVersion(t *testing.T) {
	a := newTestAgent(t)
	a.cfg.CLIConfig = &CLIConfig{
		AppName:             "af-demo",
		AppDescription:      "Demo CLI",
		HelpPreamble:        "Before you begin",
		HelpEpilog:          "More help later",
		EnvironmentVars:     []string{"AGENTFIELD_TOKEN=secret"},
		DefaultOutputFormat: "json",
	}
	a.RegisterReasoner("beta", func(context.Context, map[string]any) (any, error) {
		return "ok", nil
	}, WithCLI(), WithDescription("Beta handler"))
	a.RegisterReasoner("alpha", func(context.Context, map[string]any) (any, error) {
		return "ok", nil
	}, WithDefaultCLI(), WithDescription("Alpha handler"))

	stdout, _, err := captureOutput(t, func() error {
		a.printList(false)
		a.printHelp("", false)
		a.printHelp("alpha", false)
		a.printHelp("missing", false)
		a.printVersion()
		return nil
	})

	require.NoError(t, err)
	assert.Contains(t, stdout, "Available reasoners:")
	assert.Contains(t, stdout, "alpha (default) - Alpha handler")
	assert.Contains(t, stdout, "af-demo - Demo CLI")
	assert.Contains(t, stdout, "Before you begin")
	assert.Contains(t, stdout, "Environment Variables:")
	assert.Contains(t, stdout, "Reasoner: alpha")
	assert.Contains(t, stdout, `Unknown reasoner "missing"`)
	assert.Contains(t, stdout, "AgentField SDK: v")
	assert.Contains(t, stdout, "Agent: node-1 v1.0.0")

	empty := newTestAgent(t)
	emptyOut, _, err := captureOutput(t, func() error {
		empty.printList(false)
		return nil
	})
	require.NoError(t, err)
	assert.Contains(t, emptyOut, "No CLI reasoners registered.")
}

func TestRunCLI_CommandsAndErrors(t *testing.T) {
	t.Run("returns CLI error when no reasoners are CLI enabled", func(t *testing.T) {
		a := newTestAgent(t)
		err := a.runCLI(context.Background(), nil)
		var cliErr *CLIError
		require.ErrorAs(t, err, &cliErr)
		assert.Equal(t, 2, cliErr.ExitCode())
		assert.Contains(t, cliErr.Error(), "no CLI reasoners registered")
	})

	t.Run("supports version list and help commands", func(t *testing.T) {
		a := newTestAgent(t)
		a.RegisterReasoner("alpha", func(context.Context, map[string]any) (any, error) {
			return "ok", nil
		}, WithCLI(), WithDefaultCLI(), WithDescription("Alpha handler"))

		stdout, stderr, err := captureOutput(t, func() error {
			require.NoError(t, a.runCLI(context.Background(), []string{"version"}))
			require.NoError(t, a.runCLI(context.Background(), []string{"list"}))
			require.NoError(t, a.runCLI(context.Background(), []string{"help", "alpha"}))
			require.NoError(t, a.runCLI(context.Background(), []string{"--help", "alpha"}))
			return nil
		})

		require.NoError(t, err)
		assert.Contains(t, stdout, "AgentField SDK: v")
		assert.Contains(t, stdout, "Available reasoners:")
		assert.Contains(t, stdout, "Reasoner: alpha")
		assert.Empty(t, strings.TrimSpace(stderr))
	})

	t.Run("requires a default reasoner when command is omitted", func(t *testing.T) {
		a := newTestAgent(t)
		a.RegisterReasoner("alpha", func(context.Context, map[string]any) (any, error) {
			return "ok", nil
		}, WithCLI())

		_, _, err := captureOutput(t, func() error {
			return a.runCLI(context.Background(), nil)
		})

		var cliErr *CLIError
		require.ErrorAs(t, err, &cliErr)
		assert.Equal(t, 2, cliErr.ExitCode())
		assert.Contains(t, cliErr.Error(), "no default CLI reasoner configured")
	})

	t.Run("rejects unavailable reasoners", func(t *testing.T) {
		a := newTestAgent(t)
		a.RegisterReasoner("beta", func(context.Context, map[string]any) (any, error) {
			return "ok", nil
		}, WithCLI(), WithDefaultCLI())
		a.RegisterReasoner("alpha", func(context.Context, map[string]any) (any, error) {
			return "ok", nil
		})

		err := a.runCLI(context.Background(), []string{"alpha"})
		var cliErr *CLIError
		require.ErrorAs(t, err, &cliErr)
		assert.Equal(t, 2, cliErr.ExitCode())
		assert.Contains(t, cliErr.Error(), `reasoner "alpha" is not available for CLI use`)
	})

	t.Run("formats execution errors as exit code 1", func(t *testing.T) {
		a := newTestAgent(t)
		a.RegisterReasoner("alpha", func(context.Context, map[string]any) (any, error) {
			return nil, errors.New("boom")
		}, WithCLI(), WithDefaultCLI())

		stdout, stderr, err := captureOutput(t, func() error {
			return a.runCLI(context.Background(), nil)
		})

		assert.Contains(t, stdout, "reasoner.invoke.failed")
		assert.Contains(t, stderr, "Error: boom")
		var cliErr *CLIError
		require.ErrorAs(t, err, &cliErr)
		assert.Equal(t, 1, cliErr.ExitCode())
		assert.EqualError(t, cliErr.Err, "boom")
	})
}

func TestCLIParsingAndHelperErrors(t *testing.T) {
	a := newTestAgent(t)
	origStdin := os.Stdin
	t.Cleanup(func() { os.Stdin = origStdin })

	t.Run("parseCLIArgs rejects bad flags and formats", func(t *testing.T) {
		stdinR, stdinW, _ := os.Pipe()
		os.Stdin = stdinR
		stdinW.Close()

		_, err := a.parseCLIArgs([]string{"--output", "xml"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), `unsupported output format "xml"`)

		_, err = a.parseCLIArgs([]string{"--nope"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "unknown flag --nope")

		_, err = a.parseCLIArgs([]string{"alpha", "beta"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "unexpected argument beta")
	})

	t.Run("parseCLIArgs surfaces stdin and file JSON errors", func(t *testing.T) {
		stdinR, stdinW, _ := os.Pipe()
		os.Stdin = stdinR
		_, _ = stdinW.WriteString(`{"broken":`)
		stdinW.Close()

		_, err := a.parseCLIArgs(nil)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "parse JSON input")

		stdinR, stdinW, _ = os.Pipe()
		os.Stdin = stdinR
		stdinW.Close()

		_, err = a.parseCLIArgs([]string{"--input-file", filepath.Join(t.TempDir(), "missing.json")})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "read input file")
	})

	t.Run("helper functions cover remaining branches", func(t *testing.T) {
		assert.True(t, isSupportedOutput("json"))
		assert.False(t, isSupportedOutput("toml"))

		assert.Equal(t, "", parseScalar(""))
		assert.Equal(t, true, parseScalar("true"))
		assert.Equal(t, "value", parseScalar("value"))

		err := applySet(map[string]string{}, "")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "empty --set value")

		err = applySet(map[string]string{}, "missingequals")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "expected key=value")

		err = applySet(map[string]string{}, " =value")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "missing key")

		result, err := parseJSONFromFile("")
		require.NoError(t, err)
		assert.Nil(t, result)

		result, err = decodeJSONInput("")
		require.NoError(t, err)
		assert.Nil(t, result)

		result, err = decodeJSONInput(`{"value":1}`)
		require.NoError(t, err)
		assert.Equal(t, float64(1), result["value"])

		args := buildCLIArgMap(cliInvocation{command: "alpha", outputFormat: "yaml", useColor: false})
		assert.Equal(t, "alpha", args["__command"])
		assert.Equal(t, "yaml", args["__output"])
		assert.Equal(t, "false", args["__color"])
	})
}
