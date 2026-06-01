package types

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDecodeStoredExecutionPayload_Envelope(t *testing.T) {
	raw, err := json.Marshal(map[string]interface{}{
		"input": map[string]interface{}{"ticker": "NVDA"},
		"context": map[string]interface{}{
			"analysis_group": "summary.short_form",
		},
	})
	require.NoError(t, err)

	payload := DecodeStoredExecutionPayload(raw)

	require.Equal(t, map[string]interface{}{"ticker": "NVDA"}, payload.Input)
	require.Equal(t, map[string]interface{}{"analysis_group": "summary.short_form"}, payload.Context)
}

func TestDecodeStoredExecutionPayload_LegacyInputOnly(t *testing.T) {
	raw, err := json.Marshal(map[string]interface{}{
		"ticker": "NVDA",
		"limit":  5,
	})
	require.NoError(t, err)

	payload := DecodeStoredExecutionPayload(raw)

	require.Equal(t, map[string]interface{}{"ticker": "NVDA", "limit": float64(5)}, payload.Input)
	require.Nil(t, payload.Context)
}
