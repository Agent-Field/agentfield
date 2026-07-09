//go:build integration

package harness

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestClaudeCodeProvider_Integration_StdinWithTools drives the REAL claude CLI
// with an allowed-tools list set, which places the variadic --allowedTools flag
// last in the arg vector. Before the stdin fix the trailing positional prompt
// was swallowed by --allowedTools and the CLI exited non-zero with
// "Input must be provided ... when using --print". Delivering the prompt on
// stdin (no trailing positional) resolves it.
//
// Requires an authenticated claude CLI. Its path is taken from
// AGENTFIELD_CLAUDE_BIN (the package TestMain shadows a bare "claude" on PATH
// with a stub, so an explicit path is required to reach the real binary). Opt-in:
//
//	AGENTFIELD_CLAUDE_BIN="$(command -v claude)" \
//	  go test -tags integration -run TestClaudeCodeProvider_Integration ./harness/
func TestClaudeCodeProvider_Integration_StdinWithTools(t *testing.T) {
	binPath := os.Getenv("AGENTFIELD_CLAUDE_BIN")
	if binPath == "" {
		t.Skip("set AGENTFIELD_CLAUDE_BIN to the real claude binary to run this test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	p := NewClaudeCodeProvider(binPath)
	raw, err := p.Execute(ctx, "Reply with exactly: HELLO_AGENTFIELD", Options{
		Model:   "haiku",
		Tools:   []string{"Read", "Write"},
		Timeout: 120,
	})
	require.NoError(t, err)

	t.Logf("IsError: %v", raw.IsError)
	t.Logf("ErrorMessage: %s", raw.ErrorMessage)
	t.Logf("Result: %s", raw.Result)
	t.Logf("ReturnCode: %d", raw.ReturnCode)

	assert.False(t, raw.IsError, "expected no error, got: %s", raw.ErrorMessage)
	assert.Contains(t, raw.Result, "HELLO_AGENTFIELD")
}
