package agent

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newStatusCallbackAgent(client *http.Client) *Agent {
	return &Agent{
		cfg:        Config{Token: "token-123"},
		httpClient: client,
		logger:     log.New(io.Discard, "", 0),
	}
}

func TestPostExecutionStatusDoesNotRetryNonRetryableClientErrors(t *testing.T) {
	// The control plane answers 400 for a bad payload, 404 for an unknown
	// execution and 409 for a conflicting terminal status. Sending the same
	// callback again cannot succeed, so it must fail after one attempt instead
	// of retrying for about 15s.
	for _, code := range []int{
		http.StatusBadRequest,
		http.StatusUnauthorized,
		http.StatusForbidden,
		http.StatusNotFound,
		http.StatusConflict,
	} {
		t.Run(fmt.Sprint(code), func(t *testing.T) {
			var attempts atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				attempts.Add(1)
				http.Error(w, http.StatusText(code), code)
			}))
			defer server.Close()

			start := time.Now()
			err := newStatusCallbackAgent(server.Client()).postExecutionStatus(context.Background(), server.URL, []byte(`{"status":"succeeded"}`))

			require.Error(t, err)
			assert.Contains(t, err.Error(), fmt.Sprint(code))
			assert.Equal(t, int32(1), attempts.Load())
			assert.Less(t, time.Since(start), 500*time.Millisecond)
		})
	}
}

func TestPostExecutionStatusRetriesTransientClientErrors(t *testing.T) {
	for _, code := range []int{http.StatusRequestTimeout, http.StatusTooManyRequests} {
		t.Run(fmt.Sprint(code), func(t *testing.T) {
			var attempts atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if attempts.Add(1) == 1 {
					http.Error(w, http.StatusText(code), code)
					return
				}
				w.WriteHeader(http.StatusNoContent)
			}))
			defer server.Close()

			err := newStatusCallbackAgent(server.Client()).postExecutionStatus(context.Background(), server.URL, []byte(`{"status":"succeeded"}`))

			require.NoError(t, err)
			assert.Equal(t, int32(2), attempts.Load())
		})
	}
}
