package ai

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestAPIErrorCapsUnparseableBody(t *testing.T) {
	body := bytes.Repeat([]byte("x"), maxAPIErrorBody+1)
	err := newAPIError(502, body)

	var apiErr *APIError
	require.ErrorAs(t, err, &apiErr)
	require.Len(t, apiErr.Body, maxAPIErrorBody)
	require.True(t, errors.Is(err, &APIError{StatusCode: 502}))
}

func TestAPIErrorCarriesRetryAfterHeader(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "7")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"slow down","type":"rate_limit_error"}}`))
	}))
	defer server.Close()

	client, err := NewClient(&Config{APIKey: "k", BaseURL: server.URL, Model: "m"})
	require.NoError(t, err)

	_, err = client.Complete(context.Background(), "hi")
	var apiErr *APIError
	require.ErrorAs(t, err, &apiErr)
	require.Equal(t, 7*time.Second, apiErr.RetryAfter)
	require.Equal(t, 7*time.Second, retryAfterFromError(err))
}

func TestParseRetryAfter(t *testing.T) {
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	require.Equal(t, time.Duration(0), parseRetryAfter("", now))
	require.Equal(t, time.Duration(0), parseRetryAfter("soon", now))
	require.Equal(t, time.Duration(0), parseRetryAfter("-3", now))
	require.Equal(t, 30*time.Second, parseRetryAfter("30", now))
	require.Equal(t, 90*time.Second,
		parseRetryAfter(now.Add(90*time.Second).Format(http.TimeFormat), now))
	require.Equal(t, time.Duration(0),
		parseRetryAfter(now.Add(-time.Minute).Format(http.TimeFormat), now))
}
