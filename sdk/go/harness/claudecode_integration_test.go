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
// It also proves the stream-json output path end-to-end: the provider invokes
// the CLI with `--output-format stream-json --verbose` (so the idle watchdog
// sees per-message progress instead of total silence), and the assertions on
// SessionID/CostUSD/NumTurns only pass if the terminal "result" event of the
// real stream was parsed.
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
	t.Logf("SessionID: %s", raw.Metrics.SessionID)
	t.Logf("NumTurns: %d", raw.Metrics.NumTurns)
	if raw.Metrics.CostUSD != nil {
		t.Logf("CostUSD: %f", *raw.Metrics.CostUSD)
	}

	assert.False(t, raw.IsError, "expected no error, got: %s", raw.ErrorMessage)
	assert.Contains(t, raw.Result, "HELLO_AGENTFIELD")

	// Stream-json parsing proof: these fields only exist on the terminal
	// "result" event of the JSONL stream.
	assert.NotEmpty(t, raw.Metrics.SessionID, "session_id must be parsed from the stream's result event")
	require.NotNil(t, raw.Metrics.CostUSD, "total_cost_usd must be parsed from the stream's result event")
	assert.Greater(t, *raw.Metrics.CostUSD, 0.0)
	assert.GreaterOrEqual(t, raw.Metrics.NumTurns, 1)
	assert.Greater(t, len(raw.Messages), 1, "stream-json must yield multiple events, not one json blob")
}
