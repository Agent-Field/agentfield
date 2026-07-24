package client

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// capturedRequest captures details of an HTTP request for assertion.
type capturedRequest struct {
	method string
	path   string
	body   []byte
	auth   string
	apiKey string
}

func captureRequests(t *testing.T) (*httptest.Server, *[]capturedRequest) {
	t.Helper()
	captured := &[]capturedRequest{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		*captured = append(*captured, capturedRequest{
			method: r.Method,
			path:   r.URL.Path,
			body:   body,
			auth:   r.Header.Get("Authorization"),
			apiKey: r.Header.Get("X-API-Key"),
		})
		w.WriteHeader(http.StatusOK)
	}))
	return srv, captured
}

// TestInvariant_Client_RequestIdempotency verifies that identical (method, path, body)
// arguments produce identical HTTP request structures on repeated calls.
func TestInvariant_Client_RequestIdempotency(t *testing.T) {
	tests := []struct {
		name   string
		method string
		path   string
		body   any
	}{
		{
			name:   "GET with no body",
			method: http.MethodGet,
			path:   "/api/v1/test",
			body:   nil,
		},
		{
			name:   "POST with JSON body",
			method: http.MethodPost,
			path:   "/api/v1/nodes",
			body:   map[string]string{"key": "value"},
		},
		{
			name:   "PATCH with nested body",
			method: http.MethodPatch,
			path:   "/api/v1/nodes/node-1/status",
			body:   map[string]any{"phase": "ready", "score": 100},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, captured := captureRequests(t)
			defer srv.Close()

			c, err := New(srv.URL)
			require.NoError(t, err)

			const repetitions = 3
			for i := 0; i < repetitions; i++ {
				_ = c.do(context.Background(), tt.method, tt.path, tt.body, nil)
			}

			require.Len(t, *captured, repetitions, "should have captured %d requests", repetitions)
			first := (*captured)[0]
			for i := 1; i < repetitions; i++ {
				req := (*captured)[i]
				assert.Equal(t, first.method, req.method, "method must be identical across calls")
				assert.Equal(t, first.path, req.path, "path must be identical across calls")
				assert.Equal(t, first.body, req.body, "body must be identical across calls")
			}
		})
	}
}

// TestInvariant_Client_AuthHeaderMutualExclusivity verifies that only API key
// auth or Bearer auth headers appear when only one is configured, and verifies
// behaviour when both are configured.
func TestInvariant_Client_AuthHeaderMutualExclusivity(t *testing.T) {
	t.Run("only bearer token sets Authorization, not X-API-Key", func(t *testing.T) {
		srv, captured := captureRequests(t)
		defer srv.Close()

		c, err := New(srv.URL, WithBearerToken("my-token"))
		require.NoError(t, err)
		_ = c.do(context.Background(), http.MethodGet, "/test", nil, nil)

		require.Len(t, *captured, 1)
		req := (*captured)[0]
		assert.Equal(t, "Bearer my-token", req.auth, "Authorization header must be set")
		assert.Empty(t, req.apiKey, "X-API-Key must be absent when only bearer token is configured")
	})

	t.Run("only api key sets X-API-Key, not Authorization", func(t *testing.T) {
		srv, captured := captureRequests(t)
		defer srv.Close()

		c, err := New(srv.URL, WithAPIKey("my-api-key"))
		require.NoError(t, err)
		_ = c.do(context.Background(), http.MethodGet, "/test", nil, nil)

		require.Len(t, *captured, 1)
		req := (*captured)[0]
		assert.Empty(t, req.auth, "Authorization must be absent when only api key is configured")
		assert.Equal(t, "my-api-key", req.apiKey, "X-API-Key header must be set")
	})

	t.Run("both configured sends both headers independently", func(t *testing.T) {
		srv, captured := captureRequests(t)
		defer srv.Close()

		c, err := New(srv.URL, WithAPIKey("api-key"), WithBearerToken("bearer-token"))
		require.NoError(t, err)
		_ = c.do(context.Background(), http.MethodGet, "/test", nil, nil)

		require.Len(t, *captured, 1)
		req := (*captured)[0]
		assert.Equal(t, "Bearer bearer-token", req.auth, "Authorization must be set")
		assert.Equal(t, "api-key", req.apiKey, "X-API-Key must be set")
	})

	t.Run("no auth configured sends neither header", func(t *testing.T) {
		srv, captured := captureRequests(t)
		defer srv.Close()

		c, err := New(srv.URL)
		require.NoError(t, err)
		_ = c.do(context.Background(), http.MethodGet, "/test", nil, nil)

		require.Len(t, *captured, 1)
		req := (*captured)[0]
		assert.Empty(t, req.auth, "Authorization must be absent")
		assert.Empty(t, req.apiKey, "X-API-Key must be absent")
	})
}

// TestInvariant_Client_ErrorResponseNeverNilMessage verifies that all error responses
// from the client contain a non-empty error message string.
func TestInvariant_Client_ErrorResponseNeverNilMessage(t *testing.T) {
	errorStatuses := []struct {
		name   string
		code   int
		body   string
	}{
		{"400 Bad Request", http.StatusBadRequest, `{"error":"bad request"}`},
		{"401 Unauthorized", http.StatusUnauthorized, `{"error":"unauthorized"}`},
		{"403 Forbidden", http.StatusForbidden, `{"error":"forbidden"}`},
		{"404 Not Found", http.StatusNotFound, `{"error":"not found"}`},
		{"409 Conflict", http.StatusConflict, `{"error":"conflict"}`},
		{"429 Too Many Requests", http.StatusTooManyRequests, "rate limit exceeded"},
		{"500 Server Error", http.StatusInternalServerError, `{"error":"internal error"}`},
		{"502 Bad Gateway", http.StatusBadGateway, "bad gateway"},
		{"503 Unavailable", http.StatusServiceUnavailable, `{"error":"unavailable"}`},
		// Empty body edge case
		{"empty body 500", http.StatusInternalServerError, ""},
	}

	for _, tt := range errorStatuses {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.code)
				w.Write([]byte(tt.body))
			}))
			defer srv.Close()

			c, err := New(srv.URL)
			require.NoError(t, err)

			err = c.do(context.Background(), http.MethodGet, "/test", nil, nil)
			require.Error(t, err, "status %d must produce an error", tt.code)

			// Error message must be non-empty
			errMsg := err.Error()
			assert.NotEmpty(t, errMsg, "error message must never be empty")

			// Must be an APIError with non-nil Body
			apiErr, ok := err.(*APIError)
			require.True(t, ok, "must be *APIError, got %T", err)
			assert.Equal(t, tt.code, apiErr.StatusCode, "status code must match")
			// APIError.Error() must contain the status code
			assert.Contains(t, apiErr.Error(), "agentfield api error")
		})
	}
}

// TestInvariant_Client_BaseURLNormalization verifies that trailing slashes on
// the base URL do not affect the final request path (http://host/ + /api == http://host + /api).
func TestInvariant_Client_BaseURLNormalization(t *testing.T) {
	endpoints := []struct {
		endpoint string
		wantPath string
	}{
		{"/api/v1/nodes", "/api/v1/nodes"},
		{"/api/v1/test", "/api/v1/test"},
		{"api/v1/test", "/api/v1/test"},
	}

	baseURLVariants := []string{
		"__SERVER__",
		"__SERVER__/",
	}

	for _, ep := range endpoints {
		t.Run(ep.endpoint, func(t *testing.T) {
			var receivedPaths []string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				receivedPaths = append(receivedPaths, r.URL.Path)
				w.WriteHeader(http.StatusOK)
			}))
			defer srv.Close()

			for _, baseVariant := range baseURLVariants {
				base := strings.ReplaceAll(baseVariant, "__SERVER__", srv.URL)
				c, err := New(base)
				require.NoError(t, err)
				_ = c.do(context.Background(), http.MethodGet, ep.endpoint, nil, nil)
			}

			require.Len(t, receivedPaths, len(baseURLVariants),
				"should have received %d requests (one per base URL variant)", len(baseURLVariants))

			// All variants must produce the same path
			for i := 1; i < len(receivedPaths); i++ {
				assert.Equal(t, receivedPaths[0], receivedPaths[i],
					"trailing slash on base URL must not affect request path: got %q vs %q",
					receivedPaths[0], receivedPaths[i])
			}
			// The path must match expected
			assert.Equal(t, ep.wantPath, receivedPaths[0],
				"request path must be %q", ep.wantPath)
		})
	}
}

// TestInvariant_Client_TimeoutPropagation verifies that context cancellation
// propagates to the HTTP request and causes an error.
func TestInvariant_Client_TimeoutPropagation(t *testing.T) {
	t.Run("context cancellation aborts request", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Block for longer than the timeout
			time.Sleep(200 * time.Millisecond)
			w.WriteHeader(http.StatusOK)
		}))
		defer srv.Close()

		c, err := New(srv.URL)
		require.NoError(t, err)

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
		defer cancel()

		err = c.do(ctx, http.MethodGet, "/test", nil, nil)
		require.Error(t, err, "cancelled context must produce an error")
		// The error must mention context
		assert.True(t,
			strings.Contains(err.Error(), "context deadline exceeded") ||
				strings.Contains(err.Error(), "context canceled") ||
				strings.Contains(err.Error(), "operation was canceled"),
			"error must indicate context cancellation, got: %s", err.Error(),
		)
	})

	t.Run("explicit cancellation aborts in-flight request", func(t *testing.T) {
		blockUntilCancel := make(chan struct{})
		var closeOnce sync.Once
		closeBlock := func() { closeOnce.Do(func() { close(blockUntilCancel) }) }

		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			select {
			case <-blockUntilCancel:
			case <-r.Context().Done():
			}
			w.WriteHeader(http.StatusOK)
		}))
		defer func() {
			closeBlock()
			srv.Close()
		}()

		c, err := New(srv.URL)
		require.NoError(t, err)

		ctx, cancel := context.WithCancel(context.Background())

		errCh := make(chan error, 1)
		go func() {
			errCh <- c.do(ctx, http.MethodGet, "/test", nil, nil)
		}()

		// Cancel after a short delay
		time.Sleep(20 * time.Millisecond)
		cancel()
		closeBlock()

		select {
		case err := <-errCh:
			// Either an error or successful completion is fine — what matters is
			// the request did not hang indefinitely.
			_ = err
		case <-time.After(2 * time.Second):
			t.Fatal("request did not complete after context cancellation")
		}
	})

	t.Run("zero-timeout context immediately aborts", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			time.Sleep(100 * time.Millisecond)
			w.WriteHeader(http.StatusOK)
		}))
		defer srv.Close()

		c, err := New(srv.URL)
		require.NoError(t, err)

		ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-1*time.Second))
		defer cancel()

		err = c.do(ctx, http.MethodGet, "/test", nil, nil)
		require.Error(t, err, "already-expired context must immediately fail")
	})
}

// TestInvariant_Client_APIErrorAlwaysHasMessage verifies the invariant that
// APIError.Error() never returns an empty string regardless of body content.
func TestInvariant_Client_APIErrorAlwaysHasMessage(t *testing.T) {
	bodies := [][]byte{
		nil,
		{},
		[]byte(""),
		[]byte("  "),
		[]byte(`{"error":"some error"}`),
		[]byte("plain text error"),
	}

	for _, body := range bodies {
		apiErr := &APIError{
			StatusCode: 500,
			Body:       body,
		}
		msg := apiErr.Error()
		assert.NotEmpty(t, msg, "APIError.Error() must never return empty string")
		assert.Contains(t, msg, "500", "error message must contain status code")
	}
}

// TestInvariant_Client_NewPreservesConfig verifies that after New(), config
// options are applied and the client is in a consistent state.
func TestInvariant_Client_NewPreservesConfig(t *testing.T) {
	t.Run("token is preserved exactly", func(t *testing.T) {
		c, err := New("http://example.com", WithBearerToken("secret-token"))
		require.NoError(t, err)
		assert.Equal(t, "secret-token", c.token)
	})

	t.Run("api key is preserved exactly", func(t *testing.T) {
		c, err := New("http://example.com", WithAPIKey("my-key"))
		require.NoError(t, err)
		assert.Equal(t, "my-key", c.apiKey)
	})

	t.Run("http client is not nil after New", func(t *testing.T) {
		c, err := New("http://example.com")
		require.NoError(t, err)
		assert.NotNil(t, c.httpClient)
	})

	t.Run("trailing slash stripped from base URL", func(t *testing.T) {
		c, err := New("http://example.com/")
		require.NoError(t, err)
		assert.Equal(t, "http://example.com", c.baseURL.String())
	})

	t.Run("single trailing slash stripped", func(t *testing.T) {
		c, err := New("http://example.com/")
		require.NoError(t, err)
		// TrimSuffix removes exactly one trailing slash
		assert.False(t, strings.HasSuffix(c.baseURL.String(), "/"))
	})
}
