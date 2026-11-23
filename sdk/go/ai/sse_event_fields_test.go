package ai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"testing/iotest"
	"time"

	"github.com/stretchr/testify/require"
)

func TestStreamCompleteWithSSEFields(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: message\nid: 1\n: keepalive\ndata: {\"id\":\"first\",\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"+
			"data: {\"id\":\"second\",\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\nid: 2\n\n"+
			"event: message\ndata: [DONE]\n\n"+
			"data: {\"id\":\"after-done\"}\n\n")
	}))
	defer server.Close()

	client, err := NewClient(&Config{APIKey: "test-key", BaseURL: server.URL, Model: "test-model"})
	require.NoError(t, err)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	chunks, errs := client.StreamComplete(ctx, "hello")
	got, streamErr := collectStream(chunks, errs)
	require.NoError(t, streamErr)
	require.Len(t, got, 2)
	require.Equal(t, "first", got[0].ID)
	require.Equal(t, "hello", got[0].Choices[0].Delta.Content)
	require.Equal(t, "second", got[1].ID)
	require.Equal(t, " world", got[1].Choices[0].Delta.Content)
}

func TestSSEDecoderEventFields(t *testing.T) {
	for _, tc := range []struct {
		name  string
		input string
	}{
		{"data only", "data: {\"id\":\"a\"}\n\n"},
		{"metadata before data", "event: message\nid: 1\nretry: 100\ndata: {\"id\":\"a\"}\n\n"},
		{"metadata after data", "data: {\"id\":\"a\"}\nid: 1\nevent: message\n\n"},
		{"comment before data", ": keepalive\ndata: {\"id\":\"a\"}\n\n"},
		{"multiline data", "data: {\ndata: \"id\":\ndata: \"a\"}\n\n"},
		{"interleaved fields", "data: {\n: keepalive\nid: 1\ndata: \"id\":\"a\"}\nretry: 100\n\n"},
		{"CRLF with metadata", "event: message\r\ndata:{\"id\":\"a\"}\r\nid: 1\r\n\r\n"},
		{"empty data line", "data\ndata: {\"id\":\"a\"}\ndata:\n\n"},
		{"ignored blocks", ": keepalive\n\nevent: message\nid: 1\n\ndata: \n\nData: {\"id\":\"wrong\"}\n\ndatax: {\"id\":\"wrong\"}\n\ndata: {\"id\":\"a\"}\n\n"},
		{"malformed JSON recovery", "event: message\ndata: invalid\n\ndata: {\"id\":\"a\"}\n\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, fragmented := range []bool{false, true} {
				var reader io.Reader = strings.NewReader(tc.input)
				if fragmented {
					reader = iotest.OneByteReader(reader)
				}
				decoder := NewSSEDecoder(reader)
				chunk, err := decoder.Decode()
				require.NoError(t, err, "fragmented=%v", fragmented)
				require.Equal(t, "a", chunk.ID, "fragmented=%v", fragmented)
				_, err = decoder.Decode()
				require.ErrorIs(t, err, io.EOF)
			}
		})
	}
}
