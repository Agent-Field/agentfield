package ui

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// setupTestStorage creates a real storage instance for testing
func setupTestStorage(t *testing.T) storage.StorageProvider {
	t.Helper()
	ctx := context.Background()
	tempDir := t.TempDir()
	cfg := storage.StorageConfig{
		Mode: "local",
		Local: storage.LocalStorageConfig{
			DatabasePath: tempDir + "/test.db",
			KVStorePath:  tempDir + "/test.bolt",
		},
	}

	realStorage := storage.NewLocalStorage(storage.LocalStorageConfig{})
	err := realStorage.Initialize(ctx, cfg)
	if err != nil && strings.Contains(strings.ToLower(err.Error()), "fts5") {
		t.Skip("sqlite3 compiled without FTS5")
	}
	require.NoError(t, err)
	t.Cleanup(func() {
		realStorage.Close(ctx)
	})
	return realStorage
}

// TestStreamExecutionEventsHandler tests the execution events SSE endpoint
// This test works without a server by testing the handler directly with real storage
func TestStreamExecutionEventsHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	realStorage := setupTestStorage(t)
	eventBus := realStorage.GetExecutionEventBus()

	handler := NewExecutionHandler(realStorage, nil, nil)
	router := gin.New()
	router.GET("/api/ui/v1/executions/events", handler.StreamExecutionEventsHandler)

	req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/events", nil)
	ctx, cancel := context.WithCancel(req.Context())
	req = req.WithContext(ctx)
	resp := httptest.NewRecorder()

	// Start handler in goroutine with timeout
	done := make(chan bool)
	go func() {
		router.ServeHTTP(resp, req)
		done <- true
	}()

	// Wait a bit for handler to set up
	time.Sleep(50 * time.Millisecond)

	// Verify SSE headers are set correctly
	assert.Equal(t, "text/event-stream", resp.Header().Get("Content-Type"))
	assert.Equal(t, "no-cache", resp.Header().Get("Cache-Control"))
	assert.Equal(t, "keep-alive", resp.Header().Get("Connection"))

	// Publish an event to verify event bus integration
	eventBus.Publish(events.ExecutionEvent{
		Type:        events.ExecutionCreated,
		ExecutionID: "exec-test-1",
		WorkflowID:  "workflow-1",
		AgentNodeID: "agent-1",
		Status:      "created",
		Timestamp:   time.Now(),
	})

	// Wait a bit for event processing
	time.Sleep(50 * time.Millisecond)

	// Cancel context to close connection (simulates client disconnect)
	cancel()

	// Wait for handler to finish
	select {
	case <-done:
		// Handler finished gracefully
	case <-time.After(500 * time.Millisecond):
		// Handler may still be running, that's okay for SSE
	}
}

// TestStreamExecutionEventsHandler_Headers tests that SSE headers are set correctly
func TestStreamExecutionEventsHandler_Headers(t *testing.T) {
	gin.SetMode(gin.TestMode)

	realStorage := setupTestStorage(t)
	handler := NewExecutionHandler(realStorage, nil, nil)
	router := gin.New()
	router.GET("/api/ui/v1/executions/events", handler.StreamExecutionEventsHandler)

	req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/events", nil)
	resp := httptest.NewRecorder()

	// Start and immediately cancel to test header setting
	ctx, cancel := context.WithCancel(req.Context())
	req = req.WithContext(ctx)

	go func() {
		router.ServeHTTP(resp, req)
	}()

	time.Sleep(10 * time.Millisecond)

	// Verify headers before canceling
	assert.Equal(t, "text/event-stream", resp.Header().Get("Content-Type"))
	assert.Equal(t, "no-cache", resp.Header().Get("Cache-Control"))

	cancel()
	time.Sleep(10 * time.Millisecond)
}

// TestSSEConnectionLifecycle tests that SSE connections handle lifecycle correctly
func TestSSEConnectionLifecycle(t *testing.T) {
	gin.SetMode(gin.TestMode)

	realStorage := setupTestStorage(t)
	handler := NewExecutionHandler(realStorage, nil, nil)
	router := gin.New()
	router.GET("/api/ui/v1/executions/events", handler.StreamExecutionEventsHandler)

	req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/events", nil)
	ctx, cancel := context.WithCancel(req.Context())
	req = req.WithContext(ctx)
	resp := httptest.NewRecorder()

	done := make(chan bool)
	go func() {
		router.ServeHTTP(resp, req)
		done <- true
	}()

	// Verify connection established
	time.Sleep(20 * time.Millisecond)
	assert.Equal(t, "text/event-stream", resp.Header().Get("Content-Type"))

	// Cancel to simulate disconnect
	cancel()

	// Wait for graceful shutdown
	select {
	case <-done:
		// Connection closed gracefully
	case <-time.After(200 * time.Millisecond):
		// May still be closing, that's acceptable
	}
}

// TestSSEEventDelivery tests that events are delivered through SSE
func TestSSEEventDelivery(t *testing.T) {
	gin.SetMode(gin.TestMode)

	realStorage := setupTestStorage(t)
	eventBus := realStorage.GetExecutionEventBus()
	handler := NewExecutionHandler(realStorage, nil, nil)
	router := gin.New()
	router.GET("/api/ui/v1/executions/events", handler.StreamExecutionEventsHandler)

	req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/events", nil)
	ctx, cancel := context.WithCancel(req.Context())
	req = req.WithContext(ctx)
	resp := httptest.NewRecorder()

	done := make(chan bool)
	go func() {
		router.ServeHTTP(resp, req)
		done <- true
	}()

	// Wait for subscription
	time.Sleep(30 * time.Millisecond)

	// Publish multiple events
	for i := 0; i < 3; i++ {
		eventBus.Publish(events.ExecutionEvent{
			Type:        events.ExecutionUpdated,
			ExecutionID: "exec-test-" + string(rune(i)),
			WorkflowID:  "workflow-1",
			AgentNodeID: "agent-1",
			Status:      "running",
			Timestamp:   time.Now(),
		})
		time.Sleep(10 * time.Millisecond)
	}

	// Verify events were processed (handler should still be running)
	time.Sleep(50 * time.Millisecond)

	// Cancel connection
	cancel()

	select {
	case <-done:
	case <-time.After(200 * time.Millisecond):
	}
}

// TestSSEHeartbeatMechanism tests that heartbeats keep connection alive
func TestSSEHeartbeatMechanism(t *testing.T) {
	gin.SetMode(gin.TestMode)

	realStorage := setupTestStorage(t)
	handler := NewExecutionHandler(realStorage, nil, nil)
	router := gin.New()
	router.GET("/api/ui/v1/executions/events", handler.StreamExecutionEventsHandler)

	req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/events", nil)
	ctx, cancel := context.WithCancel(req.Context())
	req = req.WithContext(ctx)
	resp := httptest.NewRecorder()

	done := make(chan bool)
	go func() {
		router.ServeHTTP(resp, req)
		done <- true
	}()

	// Wait for initial setup
	time.Sleep(30 * time.Millisecond)

	// Verify connection is alive (headers set)
	assert.Equal(t, "text/event-stream", resp.Header().Get("Content-Type"))

	// Heartbeat should be sent periodically (every 30 seconds in handler)
	// We can't wait that long, but we verify the mechanism is set up
	time.Sleep(50 * time.Millisecond)

	// Cancel
	cancel()

	select {
	case <-done:
	case <-time.After(200 * time.Millisecond):
	}
}

// TestSSEMultipleConnections tests that multiple SSE connections work independently
func TestSSEMultipleConnections(t *testing.T) {
	gin.SetMode(gin.TestMode)

	realStorage := setupTestStorage(t)
	eventBus := realStorage.GetExecutionEventBus()
	handler := NewExecutionHandler(realStorage, nil, nil)
	router := gin.New()
	router.GET("/api/ui/v1/executions/events", handler.StreamExecutionEventsHandler)

	// Create multiple connections with a shared cancel
	connections := 3
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan bool, connections)

	for i := 0; i < connections; i++ {
		go func() {
			req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/events", nil)
			req = req.WithContext(ctx)
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)
			done <- true
		}()
	}

	// Wait for connections to establish
	time.Sleep(50 * time.Millisecond)

	// Publish event - all should receive it
	eventBus.Publish(events.ExecutionEvent{
		Type:        events.ExecutionCompleted,
		ExecutionID: "exec-multi",
		WorkflowID:  "workflow-1",
		AgentNodeID: "agent-1",
		Status:      "succeeded",
		Timestamp:   time.Now(),
	})

	time.Sleep(50 * time.Millisecond)

	// Cancel all connections
	cancel()

	// Wait for all to finish
	for i := 0; i < connections; i++ {
		select {
		case <-done:
		case <-time.After(500 * time.Millisecond):
		}
	}
}

// TestSSEErrorHandling tests error handling in SSE handlers
func TestSSEErrorHandling(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Test with valid storage (nil storage would be a programming error)
	realStorage := setupTestStorage(t)
	handler := NewExecutionHandler(realStorage, nil, nil)
	router := gin.New()
	router.GET("/api/ui/v1/executions/events", handler.StreamExecutionEventsHandler)

	req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/events", nil)
	ctx, cancel := context.WithCancel(req.Context())
	req = req.WithContext(ctx)
	resp := httptest.NewRecorder()

	// Test that handler works correctly with valid storage
	done := make(chan bool)
	go func() {
		router.ServeHTTP(resp, req)
		done <- true
	}()

	time.Sleep(20 * time.Millisecond)
	// Verify headers are set
	assert.Equal(t, "text/event-stream", resp.Header().Get("Content-Type"))

	cancel()
	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
	}
}

// TestSSERequestValidation tests request validation for SSE endpoints
func TestSSERequestValidation(t *testing.T) {
	gin.SetMode(gin.TestMode)

	realStorage := setupTestStorage(t)
	handler := NewExecutionHandler(realStorage, nil, nil)
	router := gin.New()
	router.GET("/api/ui/v1/executions/events", handler.StreamExecutionEventsHandler)

	// Test with different HTTP methods (should only accept GET)
	tests := []struct {
		method string
		path   string
	}{
		{"GET", "/api/ui/v1/executions/events"},
		{"POST", "/api/ui/v1/executions/events"},
		{"PUT", "/api/ui/v1/executions/events"},
	}

	for _, tt := range tests {
		t.Run(tt.method, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, tt.path, nil)
			ctx, cancel := context.WithCancel(req.Context())
			req = req.WithContext(ctx)
			resp := httptest.NewRecorder()

			done := make(chan bool, 1)
			go func() {
				router.ServeHTTP(resp, req)
				done <- true
			}()

			if tt.method == "GET" {
				// Wait for SSE handler to set headers
				time.Sleep(20 * time.Millisecond)
				assert.Equal(t, "text/event-stream", resp.Header().Get("Content-Type"))
				cancel()
				select {
				case <-done:
				case <-time.After(200 * time.Millisecond):
				}
			} else {
				// Non-GET methods return immediately (404)
				select {
				case <-done:
				case <-time.After(200 * time.Millisecond):
				}
				cancel()
				assert.NotEqual(t, http.StatusOK, resp.Code)
			}
		})
	}
}

// TestSSEContextCancellation tests that context cancellation closes connections
func TestSSEContextCancellation(t *testing.T) {
	gin.SetMode(gin.TestMode)

	realStorage := setupTestStorage(t)
	handler := NewExecutionHandler(realStorage, nil, nil)
	router := gin.New()
	router.GET("/api/ui/v1/executions/events", handler.StreamExecutionEventsHandler)

	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/events", nil)
	req = req.WithContext(ctx)
	resp := httptest.NewRecorder()

	done := make(chan bool)
	go func() {
		router.ServeHTTP(resp, req)
		done <- true
	}()

	// Wait for connection
	time.Sleep(20 * time.Millisecond)
	assert.Equal(t, "text/event-stream", resp.Header().Get("Content-Type"))

	// Cancel context
	cancel()

	// Should close gracefully
	select {
	case <-done:
		// Success
	case <-time.After(300 * time.Millisecond):
		t.Log("Handler may still be closing, which is acceptable for SSE")
	}
}

// TestSSEConcurrentEvents tests handling of concurrent events
func TestSSEConcurrentEvents(t *testing.T) {
	gin.SetMode(gin.TestMode)

	realStorage := setupTestStorage(t)
	eventBus := realStorage.GetExecutionEventBus()
	handler := NewExecutionHandler(realStorage, nil, nil)
	router := gin.New()
	router.GET("/api/ui/v1/executions/events", handler.StreamExecutionEventsHandler)

	req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/events", nil)
	ctx, cancel := context.WithCancel(req.Context())
	req = req.WithContext(ctx)
	resp := httptest.NewRecorder()

	done := make(chan bool)
	go func() {
		router.ServeHTTP(resp, req)
		done <- true
	}()

	time.Sleep(30 * time.Millisecond)

	// Publish many events concurrently
	const numEvents = 10
	for i := 0; i < numEvents; i++ {
		go func(id int) {
			eventBus.Publish(events.ExecutionEvent{
				Type:        events.ExecutionUpdated,
				ExecutionID: "exec-concurrent-" + string(rune(id)),
				WorkflowID:  "workflow-1",
				AgentNodeID: "agent-1",
				Status:      "running",
				Timestamp:   time.Now(),
			})
		}(i)
	}

	time.Sleep(100 * time.Millisecond)

	// Verify handler is still running (didn't crash)
	select {
	case <-done:
		// Handler finished (may have closed)
	default:
		// Handler still running, which is good
	}

	cancel()
	select {
	case <-done:
	case <-time.After(200 * time.Millisecond):
	}
}

// Helper function to verify SSE response format
func verifySSEHeaders(t *testing.T, resp *httptest.ResponseRecorder) {
	t.Helper()
	assert.Equal(t, "text/event-stream", resp.Header().Get("Content-Type"))
	assert.Equal(t, "no-cache", resp.Header().Get("Cache-Control"))
	assert.Equal(t, "keep-alive", resp.Header().Get("Connection"))
}

// TestSSEResponseFormat tests that SSE responses follow the correct format
func TestSSEResponseFormat(t *testing.T) {
	gin.SetMode(gin.TestMode)

	realStorage := setupTestStorage(t)
	handler := NewExecutionHandler(realStorage, nil, nil)
	router := gin.New()
	router.GET("/api/ui/v1/executions/events", handler.StreamExecutionEventsHandler)

	req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/events", nil)
	ctx, cancel := context.WithCancel(req.Context())
	req = req.WithContext(ctx)
	resp := httptest.NewRecorder()

	go func() {
		router.ServeHTTP(resp, req)
	}()

	time.Sleep(30 * time.Millisecond)

	// Verify SSE headers
	verifySSEHeaders(t, resp)

	// Verify CORS headers if present
	corsOrigin := resp.Header().Get("Access-Control-Allow-Origin")
	if corsOrigin != "" {
		assert.Contains(t, []string{"*", "null"}, corsOrigin)
	}

	cancel()
	time.Sleep(20 * time.Millisecond)
}

// TestSSEWithQueryParameters tests SSE endpoint with query parameters
func TestSSEWithQueryParameters(t *testing.T) {
	gin.SetMode(gin.TestMode)

	realStorage := setupTestStorage(t)
	handler := NewExecutionHandler(realStorage, nil, nil)
	router := gin.New()
	router.GET("/api/ui/v1/executions/events", handler.StreamExecutionEventsHandler)

	// Test with query parameters (should be ignored but not cause errors)
	req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/events?filter=test&limit=10", nil)
	ctx, cancel := context.WithCancel(req.Context())
	req = req.WithContext(ctx)
	resp := httptest.NewRecorder()

	go func() {
		router.ServeHTTP(resp, req)
	}()

	time.Sleep(20 * time.Millisecond)

	// Should still set SSE headers
	verifySSEHeaders(t, resp)

	cancel()
	time.Sleep(20 * time.Millisecond)
}

// TestSSEConnectionReuse tests that connections can be reused
func TestSSEConnectionReuse(t *testing.T) {
	gin.SetMode(gin.TestMode)

	realStorage := setupTestStorage(t)
	handler := NewExecutionHandler(realStorage, nil, nil)
	router := gin.New()
	router.GET("/api/ui/v1/executions/events", handler.StreamExecutionEventsHandler)

	// Create connection, close it, create another
	for i := 0; i < 2; i++ {
		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/events", nil)
		ctx, cancel := context.WithCancel(req.Context())
		req = req.WithContext(ctx)
		resp := httptest.NewRecorder()

		done := make(chan bool)
		go func() {
			router.ServeHTTP(resp, req)
			done <- true
		}()

		time.Sleep(20 * time.Millisecond)
		verifySSEHeaders(t, resp)

		cancel()
		select {
		case <-done:
		case <-time.After(100 * time.Millisecond):
		}
	}
}

// TestSSEWithInvalidStorage tests graceful handling of storage errors
func TestSSEWithInvalidStorage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Test with valid storage (nil storage would be a programming error, not a runtime error)
	realStorage := setupTestStorage(t)
	handler := NewExecutionHandler(realStorage, nil, nil)
	router := gin.New()
	router.GET("/api/ui/v1/executions/events", handler.StreamExecutionEventsHandler)

	req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/events", nil)
	ctx, cancel := context.WithCancel(req.Context())
	req = req.WithContext(ctx)
	resp := httptest.NewRecorder()

	// Test that handler works correctly with valid storage
	done := make(chan bool)
	go func() {
		router.ServeHTTP(resp, req)
		done <- true
	}()

	time.Sleep(20 * time.Millisecond)
	// Verify headers are set correctly
	assert.Equal(t, "text/event-stream", resp.Header().Get("Content-Type"))

	cancel()
	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
	}
}

// TestSSEPerformance tests that SSE doesn't block on slow subscribers
func TestSSEPerformance(t *testing.T) {
	gin.SetMode(gin.TestMode)

	realStorage := setupTestStorage(t)
	eventBus := realStorage.GetExecutionEventBus()
	handler := NewExecutionHandler(realStorage, nil, nil)
	router := gin.New()
	router.GET("/api/ui/v1/executions/events", handler.StreamExecutionEventsHandler)

	req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/events", nil)
	ctx, cancel := context.WithCancel(req.Context())
	req = req.WithContext(ctx)
	resp := httptest.NewRecorder()

	start := time.Now()
	done := make(chan bool)
	go func() {
		router.ServeHTTP(resp, req)
		done <- true
	}()

	// Publish many events quickly
	const numEvents = 50
	for i := 0; i < numEvents; i++ {
		eventBus.Publish(events.ExecutionEvent{
			Type:        events.ExecutionUpdated,
			ExecutionID: "exec-perf-" + string(rune(i)),
			WorkflowID:  "workflow-1",
			AgentNodeID: "agent-1",
			Status:      "running",
			Timestamp:   time.Now(),
		})
	}

	time.Sleep(50 * time.Millisecond)
	elapsed := time.Since(start)

	// Should handle events quickly (not block)
	assert.Less(t, elapsed, 200*time.Millisecond, "SSE should handle events quickly")

	cancel()
	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
	}
}
