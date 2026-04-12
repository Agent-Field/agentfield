package agent

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseCLIArgs_AdditionalBranches(t *testing.T) {
	a := newTestAgent(t)
	a.cfg.CLIConfig = &CLIConfig{
		DefaultOutputFormat: " YAML ",
		DisableColors:       true,
	}

	origStdin := os.Stdin
	t.Cleanup(func() { os.Stdin = origStdin })

	t.Run("captures help target and CLI config defaults", func(t *testing.T) {
		stdinR, stdinW, _ := os.Pipe()
		os.Stdin = stdinR
		stdinW.Close()

		inv, err := a.parseCLIArgs([]string{"help", "alpha"})
		require.NoError(t, err)
		assert.Equal(t, "help", inv.command)
		assert.Equal(t, "alpha", inv.helpTarget)
		assert.Equal(t, "yaml", inv.outputFormat)
		assert.False(t, inv.useColor)
	})

	t.Run("falls back to pretty when output is blank", func(t *testing.T) {
		stdinR, stdinW, _ := os.Pipe()
		os.Stdin = stdinR
		stdinW.Close()

		inv, err := a.parseCLIArgs([]string{"--output", "   "})
		require.NoError(t, err)
		assert.Equal(t, "pretty", inv.outputFormat)
	})

	t.Run("rejects missing flag values and invalid inline JSON", func(t *testing.T) {
		stdinR, stdinW, _ := os.Pipe()
		os.Stdin = stdinR
		stdinW.Close()

		_, err := a.parseCLIArgs([]string{"--set"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "missing key=value after --set")

		_, err = a.parseCLIArgs([]string{"--input"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "missing value for --input")

		_, err = a.parseCLIArgs([]string{"--input-file"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "missing value for --input-file")

		_, err = a.parseCLIArgs([]string{"--output"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "missing value for --output")

		_, err = a.parseCLIArgs([]string{"--input", `{"broken":`})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "parse JSON input")
	})
}
