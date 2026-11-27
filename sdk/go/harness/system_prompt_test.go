package harness

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSystemPromptIsPrependedToPrompt(t *testing.T) {
	const want = "Be terse.\n\nTask:\nhello"
	opts := Options{SystemPrompt: "Be terse."}

	t.Run("codex", func(t *testing.T) {
		var got string
		p := NewCodexProvider("codex")
		p.runCLI = func(_ context.Context, _ []string, _ map[string]string, _ string, _ int, stdin []byte) (*CLIResult, error) {
			got = string(stdin)
			return &CLIResult{}, nil
		}
		_, err := p.Execute(context.Background(), "hello", opts)
		require.NoError(t, err)
		assert.Equal(t, want, got)
	})

	t.Run("cursor", func(t *testing.T) {
		var got string
		p := NewCursorProvider("agent")
		p.runCLI = func(_ context.Context, cmd []string, _ map[string]string, _ string, _ int) (*CLIResult, error) {
			got = cmd[len(cmd)-1]
			return &CLIResult{}, nil
		}
		_, err := p.Execute(context.Background(), "hello", opts)
		require.NoError(t, err)
		assert.Equal(t, want, got)
	})

	t.Run("gemini", func(t *testing.T) {
		script := writeTestScript(t, t.TempDir(), "gemini", "#!/bin/sh\nprintf '%s' \"$2\"\n")
		raw, err := NewGeminiProvider(script).Execute(context.Background(), "hello", opts)
		require.NoError(t, err)
		assert.Equal(t, want, raw.Result)
	})
}
