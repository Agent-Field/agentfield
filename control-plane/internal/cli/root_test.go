package cli

import (
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

	cmd := NewRootCommand(func(cmd *cobra.Command, args []string) {})
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)
	cmd.SetArgs([]string{"--help"})

	require.NoError(t, cmd.Execute())
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
	configPath := filepath.Join(configDir, "haxen.yaml")
	require.NoError(t, os.WriteFile(configPath, []byte("haxen:\n  port: 7000\n"), 0o644))

	var received string
	cmd := NewRootCommand(func(cmd *cobra.Command, args []string) {
		received = GetConfigFilePath()
	})
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)
	cmd.SetArgs([]string{"--config", configPath, "server", "--open=false"})

	require.NoError(t, cmd.Execute())
	require.Equal(t, configPath, received)
}
