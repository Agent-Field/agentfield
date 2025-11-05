package ui

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/stretchr/testify/require"
)

type testPayloadStore struct {
	data map[string][]byte
}

func newTestPayloadStore() *testPayloadStore {
	return &testPayloadStore{data: make(map[string][]byte)}
}

func (s *testPayloadStore) SaveFromReader(ctx context.Context, r io.Reader) (*services.PayloadRecord, error) {
	return nil, errors.New("not implemented")
}

func (s *testPayloadStore) SaveBytes(ctx context.Context, data []byte) (*services.PayloadRecord, error) {
	return nil, errors.New("not implemented")
}

func (s *testPayloadStore) Open(ctx context.Context, uri string) (io.ReadCloser, error) {
	payload, ok := s.data[uri]
	if !ok {
		return nil, errors.New("payload not found")
	}
	return io.NopCloser(bytes.NewReader(payload)), nil
}

func (s *testPayloadStore) Remove(ctx context.Context, uri string) error {
	delete(s.data, uri)
	return nil
}

func TestHasMeaningfulDataDetectsCorruptedPlaceholder(t *testing.T) {
	placeholder := map[string]interface{}{
		"error":   corruptedJSONSentinel,
		"preview": "truncated",
	}

	require.False(t, hasMeaningfulData(placeholder))
}

func TestHasMeaningfulDataAllowsValidMap(t *testing.T) {
	payload := map[string]interface{}{"foo": "bar"}
	require.True(t, hasMeaningfulData(payload))
}

func TestResolveExecutionDataFallsBackForCorruptedPreview(t *testing.T) {
	store := newTestPayloadStore()
	handler := &ExecutionHandler{payloads: store}

	raw := []byte(`{"error":"corrupted_json_data","preview":"partial"}`)
	uri := "payload://test"
	store.data[uri] = []byte(`{"full":true}`)

	data, size := handler.resolveExecutionData(context.Background(), raw, &uri)

	require.Equal(t, len(store.data[uri]), size)

	marshalled, err := json.Marshal(data)
	require.NoError(t, err)
	require.JSONEq(t, `{"full":true}`, string(marshalled))
}
