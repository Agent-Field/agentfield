package cli

import (
	"io"
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
