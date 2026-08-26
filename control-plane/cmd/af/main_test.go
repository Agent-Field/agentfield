package main

import (
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/spf13/cobra"
	"github.com/stretchr/testify/require"
)

// newServerCommand builds a root command shaped like the real CLI: a
// persistent --verbose flag on the root and a `server` subcommand that
// inherits it.
func newServerCommand(t *testing.T, args ...string) *cobra.Command {
	t.Helper()

	var verbose bool
	root := &cobra.Command{Use: "af"}
	root.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "Enable verbose logging")

	var executed *cobra.Command
	server := &cobra.Command{
		Use: "server",
		Run: func(cmd *cobra.Command, _ []string) { executed = cmd },
	}
	root.AddCommand(server)
	root.SetArgs(args)
	root.SetOut(nil)
	require.NoError(t, root.Execute())
	require.NotNil(t, executed, "server command did not run")
	return executed
}

// restoreLogger snapshots the global logger so a test can freely reinitialize it.
func restoreLogger(t *testing.T) {
	t.Helper()
	prev := logger.Logger
	t.Cleanup(func() { logger.Logger = prev })
}

// TestConfiguredLogLevelSilencesInfoLines covers contract C1: a configured
// level of warn (from AGENTFIELD_LOG_LEVEL or logging.level in the YAML) stops
// info lines from being emitted while warnings and errors still are.
func TestConfiguredLogLevelSilencesInfoLines(t *testing.T) {
	restoreLogger(t)
	logger.InitLogger(false)
	require.True(t, logger.Logger.Info().Enabled(), "precondition: info is on by default")

	applyConfiguredLogLevel(newServerCommand(t, "server"), "warn")

	require.False(t, logger.Logger.Info().Enabled(), "info lines must be suppressed at level warn")
	require.True(t, logger.Logger.Warn().Enabled(), "warnings must survive at level warn")
	require.True(t, logger.Logger.Error().Enabled(), "errors must survive at level warn")
}

// TestConfiguredLogLevelEnablesDebug covers C1 in the other direction: a
// configured level of debug turns debug lines on for the shipped binary.
func TestConfiguredLogLevelEnablesDebug(t *testing.T) {
	restoreLogger(t)
	logger.InitLogger(false)
	require.False(t, logger.Logger.Debug().Enabled(), "precondition: debug is off by default")

	applyConfiguredLogLevel(newServerCommand(t, "server"), "debug")

	require.True(t, logger.Logger.Debug().Enabled(), "debug lines must be emitted at level debug")
}

// TestVerboseFlagBeatsConfiguredLevel covers contract C2: --verbose wins over
// the env/YAML level, so a debugging session is never silenced by config.
func TestVerboseFlagBeatsConfiguredLevel(t *testing.T) {
	restoreLogger(t)
	logger.InitLogger(true)

	applyConfiguredLogLevel(newServerCommand(t, "--verbose", "server"), "warn")

	require.True(t, logger.Logger.Debug().Enabled(), "--verbose must keep debug logging on")
}

// TestEmptyConfiguredLevelLeavesLoggerAlone covers contract C3.
func TestEmptyConfiguredLevelLeavesLoggerAlone(t *testing.T) {
	restoreLogger(t)
	logger.InitLogger(true)

	applyConfiguredLogLevel(newServerCommand(t, "server"), "   ")

	require.True(t, logger.Logger.Debug().Enabled(), "an unset level must not change the logger")
}

// TestFlagChangedHandlesMissingFlagAndCommand guards the helper's edge cases:
// a nil command and a command without the flag both report "not set" rather
// than panicking.
func TestFlagChangedHandlesMissingFlagAndCommand(t *testing.T) {
	require.False(t, flagChanged(nil, "verbose"))
	require.False(t, flagChanged(&cobra.Command{Use: "bare"}, "verbose"))
}
