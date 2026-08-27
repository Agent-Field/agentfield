package cli

import (
	"encoding/json"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/spf13/cobra"
	"github.com/stretchr/testify/require"
)

// runRealServerCommand executes the *real* CLI tree — the same
// NewRootCommand the shipped binaries build — and returns the *cobra.Command
// the `server` subcommand ran with. Using the real tree means these tests also
// fail if --verbose stops being a persistent flag (and so stops being visible
// from `server`), rather than only checking a command shape the test invented.
func runRealServerCommand(t *testing.T, args ...string) *cobra.Command {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	resetCLIStateForTest()

	var executed *cobra.Command
	root := NewRootCommand(func(cmd *cobra.Command, _ []string) { executed = cmd }, VersionInfo{})
	root.SetOut(io.Discard)
	root.SetErr(io.Discard)
	root.SetArgs(args)
	require.NoError(t, root.Execute())
	require.NotNil(t, executed, "the server command did not run")
	return executed
}

// snapshotLogger restores the global logger after a test reinitializes it.
func snapshotLogger(t *testing.T) {
	t.Helper()
	prev := logger.Logger
	t.Cleanup(func() { logger.Logger = prev })
}

// TestConfiguredLogLevelSilencesInfoLines covers contract C1: a configured
// level of warn (from AGENTFIELD_LOG_LEVEL or logging.level in the YAML) stops
// info lines from being emitted while warnings and errors still are.
func TestConfiguredLogLevelSilencesInfoLines(t *testing.T) {
	snapshotLogger(t)

	cmd := runRealServerCommand(t, "server")
	require.True(t, logger.Logger.Info().Enabled(), "precondition: the CLI starts at info")

	ApplyConfiguredLogLevel(cmd, "warn")

	require.False(t, logger.Logger.Info().Enabled(), "info lines must be suppressed at level warn")
	require.True(t, logger.Logger.Warn().Enabled(), "warnings must survive at level warn")
	require.True(t, logger.Logger.Error().Enabled(), "errors must survive at level warn")
}

// TestConfiguredLogLevelEnablesDebug covers C1 in the other direction: a
// configured level of debug turns debug lines on for the shipped binary.
func TestConfiguredLogLevelEnablesDebug(t *testing.T) {
	snapshotLogger(t)

	cmd := runRealServerCommand(t, "server")
	require.False(t, logger.Logger.Debug().Enabled(), "precondition: debug is off by default")

	ApplyConfiguredLogLevel(cmd, "debug")

	require.True(t, logger.Logger.Debug().Enabled(), "debug lines must be emitted at level debug")
}

// TestVerboseFlagBeatsConfiguredLevel covers contract C2: --verbose wins over
// the env/YAML level, so a debugging session is never silenced by config. It
// runs against the real command tree, so it also fails if --verbose is moved
// off the root's persistent flags and stops reaching the server subcommand.
func TestVerboseFlagBeatsConfiguredLevel(t *testing.T) {
	snapshotLogger(t)

	cmd := runRealServerCommand(t, "--verbose", "server")
	require.True(t, logger.Logger.Debug().Enabled(), "precondition: --verbose installs a debug logger")

	ApplyConfiguredLogLevel(cmd, "warn")

	require.True(t, logger.Logger.Debug().Enabled(), "--verbose must keep debug logging on")
}

// TestEmptyConfiguredLevelLeavesLoggerAlone guards the defensive branch for an
// unset level: callers that have no level to apply must not reset the logger
// the root command already installed.
func TestEmptyConfiguredLevelLeavesLoggerAlone(t *testing.T) {
	snapshotLogger(t)

	cmd := runRealServerCommand(t, "--verbose", "server")

	ApplyConfiguredLogLevel(cmd, "   ")

	require.True(t, logger.Logger.Debug().Enabled(), "an unset level must not change the logger")
}

// TestFlagChangedHandlesMissingFlagAndCommand guards the helper's edge cases:
// a nil command and a command without the flag both report "not set" rather
// than panicking.
func TestFlagChangedHandlesMissingFlagAndCommand(t *testing.T) {
	require.False(t, FlagChanged(nil, "verbose"))
	require.False(t, FlagChanged(&cobra.Command{Use: "bare"}, "verbose"))
}

// captureStderr swaps os.Stderr for a pipe while fn runs and returns whatever
// was written to it. The logger writes to os.Stderr, so this is what an
// operator actually sees in their terminal or container log.
func captureStderr(t *testing.T, fn func()) string {
	t.Helper()
	prev := os.Stderr
	reader, writer, err := os.Pipe()
	require.NoError(t, err)
	os.Stderr = writer
	defer func() { os.Stderr = prev }()

	fn()

	require.NoError(t, writer.Close())
	out, err := io.ReadAll(reader)
	require.NoError(t, err)
	require.NoError(t, reader.Close())
	return string(out)
}

// jsonLines splits captured output into decoded log events.
func jsonLines(t *testing.T, out string) []map[string]interface{} {
	t.Helper()
	events := []map[string]interface{}{}
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var event map[string]interface{}
		require.NoError(t, json.Unmarshal([]byte(line), &event), "log line is not JSON: %s", line)
		events = append(events, event)
	}
	return events
}

// TestUnrecognizedConfiguredLevelIsReported covers contract V5: a log level
// that is not a level ("warm") must not be swallowed. The operator gets one
// warn line naming what they set and what is accepted, and the server runs at
// info — the fallback that has always happened, now visible.
func TestUnrecognizedConfiguredLevelIsReported(t *testing.T) {
	snapshotLogger(t)

	cmd := runRealServerCommand(t, "server")
	out := captureStderr(t, func() { ApplyConfiguredLogLevel(cmd, "warm") })

	events := jsonLines(t, out)
	require.Len(t, events, 1, "expected exactly one line about the bad level, got: %q", out)
	require.Equal(t, "warn", events[0]["level"], "the report must be at warn so it survives the info fallback")
	require.Equal(t, "warm", events[0]["configured_level"], "the offending value must be named")
	require.Equal(t, []interface{}{"debug", "info", "warn", "error"}, events[0]["accepted_levels"])
	require.Contains(t, events[0]["message"], "log level")

	require.True(t, logger.Logger.Info().Enabled(), "an unrecognized level must fall back to info")
	require.False(t, logger.Logger.Debug().Enabled(), "the fallback is info, not debug")
}

// TestRecognizedConfiguredLevelsAreSilent covers contract V6: every level the
// parser accepts — canonical names and aliases alike — applies without
// complaint, so the new report cannot become background noise.
func TestRecognizedConfiguredLevelsAreSilent(t *testing.T) {
	for _, level := range []string{"debug", "info", "warn", "error", "warning", "verbose", "trace", "err", "WARN", " warn "} {
		t.Run(level, func(t *testing.T) {
			snapshotLogger(t)

			cmd := runRealServerCommand(t, "server")
			out := captureStderr(t, func() { ApplyConfiguredLogLevel(cmd, level) })

			require.Empty(t, jsonLines(t, out), "a valid level must not be reported as invalid")
		})
	}
}

// TestVerboseFlagSuppressesTheUnrecognizedLevelReport documents the one case
// where a bad value stays quiet: --verbose short-circuits before the config
// level is consulted at all, so nothing falls back and there is nothing to
// report.
func TestVerboseFlagSuppressesTheUnrecognizedLevelReport(t *testing.T) {
	snapshotLogger(t)

	cmd := runRealServerCommand(t, "--verbose", "server")
	out := captureStderr(t, func() { ApplyConfiguredLogLevel(cmd, "warm") })

	require.Empty(t, jsonLines(t, out))
	require.True(t, logger.Logger.Debug().Enabled(), "--verbose still wins")
}
