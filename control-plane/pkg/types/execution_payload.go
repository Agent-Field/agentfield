package types

import "encoding/json"

// StoredExecutionPayload is the envelope persisted for execution requests.
// It preserves the original input plus optional caller-defined context.
type StoredExecutionPayload struct {
	Input   map[string]interface{} `json:"input,omitempty"`
	Context map[string]interface{} `json:"context,omitempty"`
}

// DecodeStoredExecutionPayload extracts the persisted execution input/context shape.
// Older records may not use the envelope, so object payloads fall back to input-only.
func DecodeStoredExecutionPayload(raw json.RawMessage) StoredExecutionPayload {
	if len(raw) == 0 {
		return StoredExecutionPayload{}
	}

	var envelope StoredExecutionPayload
	if err := json.Unmarshal(raw, &envelope); err == nil && (envelope.Input != nil || envelope.Context != nil) {
		return envelope
	}

	var generic map[string]interface{}
	if err := json.Unmarshal(raw, &generic); err == nil {
		return StoredExecutionPayload{Input: generic}
	}

	return StoredExecutionPayload{}
}
