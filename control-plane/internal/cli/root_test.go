package cli

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/spf13/cobra"
	"github.com/stretchr/testify/require"
)

func resetCLIStateForTest() {
	cfgFile = ""
	verbose = false
	openBrowserFlag = true
	uiDevFlag = false
	backendOnlyFlag = false
	portFlag = 0
	noVCExecution = false
}

func TestRootCommandDisplaysHelp(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	resetCLIStateForTest()

	cmd := NewRootCommand(func(cmd *cobra.Command, args []string) {}, VersionInfo{
		Version: "test",
		Commit:  "test",
		Date:    "test",
	})
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)
	cmd.SetArgs([]string{"--help"})

	require.NoError(t, cmd.Execute())
}

// Contract: when a subcommand fails at runtime, the CLI must NOT print the
// usage/help block — usage is for mis-invocation, not runtime errors.
func TestRootCommand_RuntimeErrorDoesNotPrintUsage(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	resetCLIStateForTest()

	cmd := NewRootCommand(func(cmd *cobra.Command, args []string) {}, VersionInfo{})
	cmd.AddCommand(&cobra.Command{
		Use: "boom",
		RunE: func(*cobra.Command, []string) error {
			return errors.New("kaboom")
		},
	})

	var buf bytes.Buffer
	cmd.SetOut(&buf)
	cmd.SetErr(&buf)
	cmd.SetArgs([]string{"boom"})

	err := cmd.Execute()
	require.Error(t, err) // the error still propagates to main.go
	require.NotContains(t, buf.String(), "Usage:", "runtime error must not dump the usage block")
}

func TestRootCommandServerFlags(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	resetCLIStateForTest()

	invoked := false
	cmd := NewRootCommand(func(cmd *cobra.Command, args []string) {
		invoked = true
		require.False(t, GetOpenBrowserFlag())
		require.True(t, GetBackendOnlyFlag())
		require.True(t, GetUIDevFlag())
		require.Equal(t, 9090, GetPortFlag())
		require.True(t, noVCExecution)
	}, VersionInfo{
		Version: "test",
		Commit:  "test",
		Date:    "test",
	})
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)
	cmd.SetArgs([]string{
		"server",
		"--open=false",
		"--backend-only",
		"--ui-dev",
		"--port=9090",
		"--no-vc-execution",
	})

	require.NoError(t, cmd.Execute())
	require.True(t, invoked)
}

func TestRootCommandHonorsConfigFlag(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	resetCLIStateForTest()

	configDir := t.TempDir()
	configPath := filepath.Join(configDir, "agentfield.yaml")
	require.NoError(t, os.WriteFile(configPath, []byte("agentfield:\n  port: 7000\n"), 0o644))

	var received string
	cmd := NewRootCommand(func(cmd *cobra.Command, args []string) {
		received = GetConfigFilePath()
	}, VersionInfo{
		Version: "test",
		Commit:  "test",
		Date:    "test",
	})
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)
	cmd.SetArgs([]string{"--config", configPath, "server", "--open=false"})

	require.NoError(t, cmd.Execute())
	require.Equal(t, configPath, received)
}
