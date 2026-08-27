package main

import (
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/cli"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/require"
)

// serverCommandWithConfig writes an agentfield.yaml, runs the real CLI tree
// (the one main() builds) with `--config <file> server`, and returns the
// *cobra.Command the server subcommand ran with. The precedence helper itself
// is covered in internal/cli; these tests bind the af server's *use* of it.
func serverCommandWithConfig(t *testing.T, yaml string, extraArgs ...string) *cobra.Command {
	t.Helper()

	// The global logger and viper's global config are both process-wide.
	prevLogger := logger.Logger
	t.Cleanup(func() { logger.Logger = prevLogger })
	viper.Reset()
	t.Cleanup(viper.Reset)

	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("AGENTFIELD_HOME", home)

	configPath := filepath.Join(t.TempDir(), "agentfield.yaml")
	require.NoError(t, os.WriteFile(configPath, []byte(yaml), 0o644))

	args := append(append([]string{}, extraArgs...), "--config", configPath, "server")

	var executed *cobra.Command
	root := cli.NewRootCommand(func(cmd *cobra.Command, _ []string) { executed = cmd }, cli.VersionInfo{})
	root.SetOut(io.Discard)
	root.SetErr(io.Discard)
	root.SetArgs(args)
	require.NoError(t, root.Execute())
	require.NotNil(t, executed, "the server command did not run")
	return executed
}

// TestServerConfigLoadSilencesInfoLines covers contract C1 at the seam the
// shipped binary actually uses: the config the af server runs on cannot be
// obtained without the level configured in it having taken effect. Deleting
// the level wiring from loadServerConfig fails this test, and loadServerConfig
// cannot be dropped from runServer without losing the config itself.
func TestServerConfigLoadSilencesInfoLines(t *testing.T) {
	cmd := serverCommandWithConfig(t, "logging:\n  level: warn\n")
	require.True(t, logger.Logger.Info().Enabled(), "precondition: the CLI starts at info")

	cfg, err := loadServerConfig(cmd)
	require.NoError(t, err)

	require.Equal(t, "warn", cfg.Logging.Level)
	require.False(t, logger.Logger.Info().Enabled(), "a configured level of warn must stop info lines")
	require.True(t, logger.Logger.Warn().Enabled(), "warnings must survive at level warn")
}

// TestServerConfigLoadEnablesDebug covers C1 in the other direction through the
// real config-loading path (viper + ApplyDefaults + ApplyEnvOverrides).
func TestServerConfigLoadEnablesDebug(t *testing.T) {
	cmd := serverCommandWithConfig(t, "logging:\n  level: debug\n")
	require.False(t, logger.Logger.Debug().Enabled(), "precondition: debug is off by default")

	_, err := loadServerConfig(cmd)
	require.NoError(t, err)

	require.True(t, logger.Logger.Debug().Enabled(), "a configured level of debug must turn debug lines on")
}

// TestServerConfigLoadKeepsVerboseWinning covers contract C2 end to end: a
// `--verbose` run of the real CLI is not silenced by logging.level in the YAML.
func TestServerConfigLoadKeepsVerboseWinning(t *testing.T) {
	cmd := serverCommandWithConfig(t, "logging:\n  level: warn\n", "--verbose")
	require.True(t, logger.Logger.Debug().Enabled(), "precondition: --verbose installs a debug logger")

	_, err := loadServerConfig(cmd)
	require.NoError(t, err)

	require.True(t, logger.Logger.Debug().Enabled(), "--verbose must beat the configured level")
}

// TestServerConfigLoadWithoutLoggingSectionStaysAtInfo checks the upgrade path
// for operators who have configured nothing: the default output is unchanged.
func TestServerConfigLoadWithoutLoggingSectionStaysAtInfo(t *testing.T) {
	cmd := serverCommandWithConfig(t, "agentfield:\n  port: 7001\n")

	cfg, err := loadServerConfig(cmd)
	require.NoError(t, err)

	require.Equal(t, 7001, cfg.AgentField.Port, "the config file must actually have been read")
	require.True(t, logger.Logger.Info().Enabled(), "info stays on when nothing is configured")
	require.False(t, logger.Logger.Debug().Enabled(), "debug stays off when nothing is configured")
}
