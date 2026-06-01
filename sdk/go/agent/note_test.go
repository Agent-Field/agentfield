package agent

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNote_Basic(t *testing.T) {
	var receivedPayload notePayload
	var receivedHeaders http.Header
	var mu sync.Mutex
	requestReceived := make(chan struct{})

	// Create a test server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()

		// Parse the payload
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &receivedPayload)
		receivedHeaders = r.Header.Clone()

		w.WriteHeader(http.StatusOK)
		close(requestReceived)
	}))
	defer server.Close()

	cfg := Config{
		NodeID:        "test-node",
		Version:       "1.0.0",
		AgentFieldURL: server.URL + "/api/v1", // Will be converted to /api/ui/v1
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	ctx := contextWithExecution(context.Background(), ExecutionContext{
		RunID:       "run-123",
		ExecutionID: "exec-456",
		SessionID:   "session-789",
		ActorID:     "actor-abc",
		WorkflowID:  "workflow-xyz",
	})

	agent.Note(ctx, "Test message", "tag1", "tag2")

	// Wait for the note to be sent
	select {
	case <-requestReceived:
		// Success
	case <-time.After(2 * time.Second):
		t.Fatal("Timeout waiting for note request")
	}

	mu.Lock()
	defer mu.Unlock()

	// Verify payload
	assert.Equal(t, "Test message", receivedPayload.Message)
	assert.Equal(t, []string{"tag1", "tag2"}, receivedPayload.Tags)
	assert.Equal(t, "test-node", receivedPayload.AgentNodeID)
	assert.Greater(t, receivedPayload.Timestamp, float64(0))

	// Verify headers
	assert.Equal(t, "run-123", receivedHeaders.Get("X-Run-ID"))
	assert.Equal(t, "exec-456", receivedHeaders.Get("X-Execution-ID"))
	assert.Equal(t, "session-789", receivedHeaders.Get("X-Session-ID"))
	assert.Equal(t, "actor-abc", receivedHeaders.Get("X-Actor-ID"))
	assert.Equal(t, "workflow-xyz", receivedHeaders.Get("X-Workflow-ID"))
	assert.Equal(t, "test-node", receivedHeaders.Get("X-Agent-Node-ID"))
}

func TestNotef_Formatted(t *testing.T) {
	var receivedPayload notePayload
	requestReceived := make(chan struct{})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &receivedPayload)
		w.WriteHeader(http.StatusOK)
		close(requestReceived)
	}))
	defer server.Close()

	cfg := Config{
		NodeID:        "test-node",
		Version:       "1.0.0",
		AgentFieldURL: server.URL + "/api/v1",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	ctx := contextWithExecution(context.Background(), ExecutionContext{
		RunID: "run-123",
	})

	agent.Notef(ctx, "Processing %d items with status %s", 42, "active")

	select {
	case <-requestReceived:
	case <-time.After(2 * time.Second):
		t.Fatal("Timeout waiting for note request")
	}

	assert.Equal(t, "Processing 42 items with status active", receivedPayload.Message)
}

func TestNote_NoTags(t *testing.T) {
	var receivedPayload notePayload
	requestReceived := make(chan struct{})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &receivedPayload)
		w.WriteHeader(http.StatusOK)
		close(requestReceived)
	}))
	defer server.Close()

	cfg := Config{
		NodeID:        "test-node",
		Version:       "1.0.0",
		AgentFieldURL: server.URL + "/api/v1",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	ctx := contextWithExecution(context.Background(), ExecutionContext{
		RunID: "run-123",
	})

	// Call Note without tags
	agent.Note(ctx, "No tags message")

	select {
	case <-requestReceived:
	case <-time.After(2 * time.Second):
		t.Fatal("Timeout waiting for note request")
	}

	assert.Equal(t, "No tags message", receivedPayload.Message)
	assert.Equal(t, []string{}, receivedPayload.Tags)
}

func TestNote_NoAgentFieldURL(t *testing.T) {
	// Agent without AgentFieldURL should not send notes
	cfg := Config{
		NodeID:  "test-node",
		Version: "1.0.0",
		Logger:  log.New(io.Discard, "", 0),
		// No AgentFieldURL
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	ctx := contextWithExecution(context.Background(), ExecutionContext{
		RunID: "run-123",
	})

	// This should not panic or block
	agent.Note(ctx, "This note goes nowhere")

	// Give it a moment to ensure no panic
	time.Sleep(100 * time.Millisecond)
}

func TestNote_ServerError(t *testing.T) {
	// Server that returns an error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	cfg := Config{
		NodeID:        "test-node",
		Version:       "1.0.0",
		AgentFieldURL: server.URL + "/api/v1",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	ctx := contextWithExecution(context.Background(), ExecutionContext{
		RunID: "run-123",
	})

	// Should not panic even with server error
	agent.Note(ctx, "Test message")

	// Give it time to complete
	time.Sleep(200 * time.Millisecond)
}

func TestNote_URLConversion(t *testing.T) {
	var requestPath string
	requestReceived := make(chan struct{})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestPath = r.URL.Path
		w.WriteHeader(http.StatusOK)
		close(requestReceived)
	}))
	defer server.Close()

	tests := []struct {
		name           string
		agentFieldURL  string
		expectedPath   string
	}{
		{
			name:          "Standard /api/v1 URL",
			agentFieldURL: server.URL + "/api/v1",
			expectedPath:  "/api/ui/v1/executions/note",
		},
		{
			name:          "URL without /api/v1",
			agentFieldURL: server.URL,
			expectedPath:  "/api/ui/v1/executions/note",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requestReceived = make(chan struct{})
			requestPath = ""

			cfg := Config{
				NodeID:        "test-node",
				Version:       "1.0.0",
				AgentFieldURL: tt.agentFieldURL,
				Logger:        log.New(io.Discard, "", 0),
			}

			agent, err := New(cfg)
			require.NoError(t, err)

			ctx := contextWithExecution(context.Background(), ExecutionContext{
				RunID: "run-123",
			})

			agent.Note(ctx, "Test")

			select {
			case <-requestReceived:
				assert.Equal(t, tt.expectedPath, requestPath)
			case <-time.After(2 * time.Second):
				t.Fatal("Timeout waiting for note request")
			}
		})
	}
}

func TestNote_WithToken(t *testing.T) {
	var receivedAuthHeader string
	requestReceived := make(chan struct{})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuthHeader = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
		close(requestReceived)
	}))
	defer server.Close()

	cfg := Config{
		NodeID:        "test-node",
		Version:       "1.0.0",
		AgentFieldURL: server.URL + "/api/v1",
		Token:         "test-token-123",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	ctx := contextWithExecution(context.Background(), ExecutionContext{
		RunID: "run-123",
	})

	agent.Note(ctx, "Test message")

	select {
	case <-requestReceived:
	case <-time.After(2 * time.Second):
		t.Fatal("Timeout waiting for note request")
	}

	assert.Equal(t, "Bearer test-token-123", receivedAuthHeader)
}

func TestNote_FireAndForget(t *testing.T) {
	// Test that Note doesn't block even if server is slow
	slowServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Second) // Slow response
		w.WriteHeader(http.StatusOK)
	}))
	defer slowServer.Close()

	cfg := Config{
		NodeID:        "test-node",
		Version:       "1.0.0",
		AgentFieldURL: slowServer.URL + "/api/v1",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	ctx := contextWithExecution(context.Background(), ExecutionContext{
		RunID: "run-123",
	})

	start := time.Now()
	agent.Note(ctx, "Test message")
	elapsed := time.Since(start)

	// Note should return immediately (< 100ms), not wait for server
	assert.Less(t, elapsed, 100*time.Millisecond, "Note should not block")
}

func TestNote_MultipleNotes(t *testing.T) {
	var mu sync.Mutex
	noteCount := 0

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		noteCount++
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	cfg := Config{
		NodeID:        "test-node",
		Version:       "1.0.0",
		AgentFieldURL: server.URL + "/api/v1",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	ctx := contextWithExecution(context.Background(), ExecutionContext{
		RunID: "run-123",
	})

	// Send multiple notes
	for i := 0; i < 5; i++ {
		agent.Notef(ctx, "Note %d", i)
	}

	// Wait for all notes to be sent
	time.Sleep(500 * time.Millisecond)

	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, 5, noteCount)
}
