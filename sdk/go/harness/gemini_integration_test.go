//go:build integration

package harness

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGeminiProvider_Integration(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	p := NewGeminiProvider("")
	raw, err := p.Execute(ctx, "Reply with exactly: HELLO_AGENTFIELD", Options{
		Timeout: 120,
	})
	require.NoError(t, err)

	t.Logf("IsError: %v", raw.IsError)
	t.Logf("ErrorMessage: %s", raw.ErrorMessage)
	t.Logf("Result: %s", raw.Result)
	t.Logf("NumTurns: %d", raw.Metrics.NumTurns)
	t.Logf("ReturnCode: %d", raw.ReturnCode)

	assert.False(t, raw.IsError, "expected no error, got: %s", raw.ErrorMessage)
	assert.Contains(t, raw.Result, "HELLO_AGENTFIELD")
}
