package agent

import (
	"context"
	"io"
	"log"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRegisterReasoner_DefaultCLIConflictAndRealtimeValidation(t *testing.T) {
	a, err := New(Config{
		NodeID:  "node-1",
		Version: "1.0.0",
		Logger:  log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)

	a.RegisterReasoner("first", func(context.Context, map[string]any) (any, error) {
		return nil, nil
	}, WithDefaultCLI())

	// A second default CLI registration should keep the original default.
	a.RegisterReasoner("second", func(context.Context, map[string]any) (any, error) {
		return nil, nil
	}, WithDefaultCLI(), WithRequireRealtimeValidation())

	first := a.reasoners["first"]
	second := a.reasoners["second"]

	require.NotNil(t, first)
	require.NotNil(t, second)
	assert.True(t, first.DefaultCLI)
	assert.False(t, second.DefaultCLI)
	assert.Equal(t, "first", a.defaultCLIReasoner)

	_, marked := a.realtimeValidationFunctions["second"]
	assert.True(t, marked)
}
