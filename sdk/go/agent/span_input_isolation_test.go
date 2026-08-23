package agent

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestSpanInputSnapshotIsolatedFromCallerMutation pins two behaviors at once:
// the traced input is a snapshot taken at Span entry (later mutations by fn
// must not leak into the emitted events), and the async start-event goroutine
// must never serialize caller-owned memory — mutating the input map inside fn
// is safe. Run with -race to enforce the second half.
func TestSpanInputSnapshotIsolatedFromCallerMutation(t *testing.T) {
	ag, eventCh, closeServer := newSpanEventHarness(t)
	defer closeServer()

	input := map[string]any{"k": "initial"}
	_, err := ag.Span(spanParentContext(), "mutating", input, func(ctx context.Context) (any, error) {
		for j := 0; j < 1000; j++ {
			input["k"] = j
		}
		return "done", nil
	})
	require.NoError(t, err)

	received := collectSpanEvents(t, eventCh, 2)
	for _, evt := range received {
		require.NotNil(t, evt.InputData, "event %q lost its input", evt.Status)
		assert.Equal(t, "initial", evt.InputData["k"],
			"event %q must carry the input as of Span entry, not fn's mutations", evt.Status)
	}
}
