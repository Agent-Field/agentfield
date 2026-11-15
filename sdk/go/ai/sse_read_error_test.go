package ai

import (
	"context"
	"io"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"
)

// A Reader may return both data and an error, then EOF on subsequent reads.
type sseReadErrorReader struct {
	data  string
	err   error
	reads int
}

func (r *sseReadErrorReader) Read(p []byte) (int, error) {
	r.reads++
	if r.data == "" {
		return 0, io.EOF
	}
	n := copy(p, r.data)
	r.data = r.data[n:]
	if r.data == "" {
		return n, r.err
	}
	return n, nil
}

func TestSSEDecoder_PreservesReadErrorAfterData(t *testing.T) {
	for _, tc := range []struct {
		name    string
		data    string
		ids     []string
		err     error
		wantErr error
	}{
		{"complete events", "data: {\"id\":\"first\"}\n\ndata: {\"id\":\"second\"}\n\n", []string{"first", "second"}, io.ErrUnexpectedEOF, io.ErrUnexpectedEOF},
		{"partial event", "data: {\"id\":\"partial\"}", nil, io.ErrUnexpectedEOF, io.ErrUnexpectedEOF},
		{"ignored events", ": keepalive\n\ndata: invalid\n\n", nil, io.ErrUnexpectedEOF, io.ErrUnexpectedEOF},
		{"complete then partial", "data: {\"id\":\"first\"}\n\ndata: {\"id\":\"partial\"}", []string{"first"}, io.ErrUnexpectedEOF, io.ErrUnexpectedEOF},
		{"complete then ignored", "data: {\"id\":\"first\"}\n\n: keepalive\n\ndata: invalid\n\n", []string{"first"}, io.ErrUnexpectedEOF, io.ErrUnexpectedEOF},
		{"DONE before read error", "data: [DONE]\n\n", nil, io.ErrUnexpectedEOF, io.EOF},
		{"clean EOF", "data: {\"id\":\"last\"}\n\n", []string{"last"}, io.EOF, io.EOF},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reader := &sseReadErrorReader{data: tc.data, err: tc.err}
			decoder := NewSSEDecoder(reader)
			for _, id := range tc.ids {
				chunk, err := decoder.Decode()
				require.NoError(t, err)
				require.Equal(t, id, chunk.ID)
			}
			_, err := decoder.Decode()
			require.ErrorIs(t, err, tc.wantErr)
			require.Equal(t, 1, reader.reads, "do not read again after a terminal error")
		})
	}
}

func TestStreamComplete_PreservesReadErrorAfterData(t *testing.T) {
	reader := &sseReadErrorReader{data: "data: {\"id\":\"first\"}\n\n", err: io.ErrUnexpectedEOF}
	client, err := NewClient(&Config{
		APIKey: "test-key", BaseURL: "https://api.example.com/v1", Model: "test-model",
	}, WithTransport(&roundTripFunc{fn: func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(reader)}, nil
	}}))
	require.NoError(t, err)
	chunks, errs := client.StreamComplete(context.Background(), "hello")
	got, streamErr := collectStream(chunks, errs)
	require.Len(t, got, 1)
	require.Equal(t, "first", got[0].ID)
	require.ErrorIs(t, streamErr, io.ErrUnexpectedEOF)
}
