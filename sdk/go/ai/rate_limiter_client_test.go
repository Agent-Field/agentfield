package ai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func successChatResponse() Response {
	return Response{
		ID:     "chatcmpl-1",
		Object: "chat.completion",
		Model:  "gpt-4o",
		Choices: []Choice{
			{
				Index: 0,
				Message: Message{
					Role:    "assistant",
					Content: []ContentPart{{Type: "text", Text: "ok"}},
				},
				FinishReason: "stop",
			},
		},
	}
}

func TestClientRetriesOn429ThenSucceeds(t *testing.T) {
	var calls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&calls, 1)
		if n < 3 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"message":"rate limit exceeded","type":"rate_limit_error"}}`))
			return
		}
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(successChatResponse())
	}))
	defer server.Close()

	config := &Config{
		APIKey:              "test-key",
		BaseURL:             server.URL,
		Model:               "gpt-4o",
		Timeout:             5 * time.Second,
		RateLimitMaxRetries: 5,
		RateLimitBaseDelay:  time.Millisecond,
		RateLimitMaxDelay:   10 * time.Millisecond,
	}
	client, err := NewClient(config)
	require.NoError(t, err)
	require.NotNil(t, client.rateLimiter)

	resp, err := client.Complete(context.Background(), "hi")
	require.NoError(t, err)
	assert.Equal(t, "ok", resp.Text())
	assert.Equal(t, int32(3), atomic.LoadInt32(&calls))
}

func TestClientReturnsAPIErrorAfterRetriesExhausted(t *testing.T) {
	var calls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limit exceeded"}}`))
	}))
	defer server.Close()

	config := &Config{
		APIKey:              "test-key",
		BaseURL:             server.URL,
		Model:               "gpt-4o",
		Timeout:             5 * time.Second,
		RateLimitMaxRetries: 2,
		RateLimitBaseDelay:  time.Millisecond,
		RateLimitMaxDelay:   10 * time.Millisecond,
	}
	client, err := NewClient(config)
	require.NoError(t, err)

	_, err = client.Complete(context.Background(), "hi")
	require.Error(t, err)
	assert.ErrorIs(t, err, ErrMaxRetriesExceeded)
	assert.Equal(t, int32(3), atomic.LoadInt32(&calls)) // initial + 2 retries
}

func TestClientDoesNotRetryNon429(t *testing.T) {
	var calls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":{"message":"boom"}}`))
	}))
	defer server.Close()

	config := &Config{
		APIKey:              "test-key",
		BaseURL:             server.URL,
		Model:               "gpt-4o",
		Timeout:             5 * time.Second,
		RateLimitMaxRetries: 5,
		RateLimitBaseDelay:  time.Millisecond,
	}
	client, err := NewClient(config)
	require.NoError(t, err)

	_, err = client.Complete(context.Background(), "hi")
	require.Error(t, err)
	var apiErr *APIError
	require.ErrorAs(t, err, &apiErr)
	assert.Equal(t, http.StatusInternalServerError, apiErr.StatusCode)
	assert.Equal(t, int32(1), atomic.LoadInt32(&calls)) // no retries on 500
}

func TestClientNoRateLimiterWhenDisabled(t *testing.T) {
	config := &Config{
		APIKey:  "test-key",
		BaseURL: "https://example.com/v1",
		Model:   "gpt-4o",
	}
	client, err := NewClient(config)
	require.NoError(t, err)
	assert.Nil(t, client.rateLimiter)
}
