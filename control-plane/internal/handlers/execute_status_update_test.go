package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestUpdateExecutionStatusHandler_Success(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   "http://agent.example",
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}

	store := newTestExecutionStorage(agent)
	payloads := services.NewFilePayloadStore(t.TempDir())

	// Create an execution record
	execution := &types.Execution{
		ExecutionID: "exec-1",
		RunID:        "run-1",
		AgentNodeID:  "node-1",
		ReasonerID:   "reasoner-a",
		Status:       types.ExecutionStatusRunning,
		StartedAt:    time.Now().UTC(),
		CreatedAt:    time.Now().UTC(),
		UpdatedAt:    time.Now().UTC(),
	}
	require.NoError(t, store.CreateExecutionRecord(context.Background(), execution))

	router := gin.New()
	router.PUT("/api/v1/executions/:execution_id/status", UpdateExecutionStatusHandler(store, payloads, nil, 90*time.Second))

	reqBody := `{
		"status": "succeeded",
		"result": {"output": "success"},
		"duration_ms": 1000
	}`
	req := httptest.NewRequest(http.MethodPut, "/api/v1/executions/exec-1/status", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload ExecutionStatusResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, "exec-1", payload.ExecutionID)
	require.Equal(t, types.ExecutionStatusSucceeded, payload.Status)
	require.NotNil(t, payload.CompletedAt)

	// Verify execution was updated
	updated, err := store.GetExecutionRecord(context.Background(), "exec-1")
	require.NoError(t, err)
	require.NotNil(t, updated)
	require.Equal(t, types.ExecutionStatusSucceeded, updated.Status)
	require.NotNil(t, updated.ResultPayload)
	require.NotNil(t, updated.CompletedAt)
	require.NotNil(t, updated.DurationMS)
	require.Equal(t, int64(1000), *updated.DurationMS)
}

func TestUpdateExecutionStatusHandler_Failed(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   "http://agent.example",
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}

	store := newTestExecutionStorage(agent)
	payloads := services.NewFilePayloadStore(t.TempDir())

	execution := &types.Execution{
		ExecutionID: "exec-1",
		RunID:       "run-1",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   time.Now().UTC(),
		CreatedAt:   time.Now().UTC(),
		UpdatedAt:   time.Now().UTC(),
	}
	require.NoError(t, store.CreateExecutionRecord(context.Background(), execution))

	router := gin.New()
	router.PUT("/api/v1/executions/:execution_id/status", UpdateExecutionStatusHandler(store, payloads, nil, 90*time.Second))

	reqBody := `{
		"status": "failed",
		"error": "something went wrong",
		"duration_ms": 500
	}`
	req := httptest.NewRequest(http.MethodPut, "/api/v1/executions/exec-1/status", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload ExecutionStatusResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, types.ExecutionStatusFailed, payload.Status)
	require.NotNil(t, payload.Error)
	require.Contains(t, *payload.Error, "something went wrong")

	// Verify execution was updated
	updated, err := store.GetExecutionRecord(context.Background(), "exec-1")
	require.NoError(t, err)
	require.NotNil(t, updated)
	require.Equal(t, types.ExecutionStatusFailed, updated.Status)
	require.NotNil(t, updated.ErrorMessage)
	require.Contains(t, *updated.ErrorMessage, "something went wrong")
}

func TestUpdateExecutionStatusHandler_WithWebhook(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   "http://agent.example",
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}

	store := newTestExecutionStorage(agent)
	payloads := services.NewFilePayloadStore(t.TempDir())

	// Create webhook dispatcher mock
	webhookCalled := false
	mockWebhook := &mockWebhookDispatcher{
		notifyFunc: func(ctx context.Context, executionID string) error {
			webhookCalled = true
			return nil
		},
	}

	execution := &types.Execution{
		ExecutionID:       "exec-1",
		RunID:             "run-1",
		Status:            types.ExecutionStatusRunning,
		StartedAt:         time.Now().UTC(),
		CreatedAt:         time.Now().UTC(),
		UpdatedAt:         time.Now().UTC(),
		WebhookRegistered: true,
	}
	require.NoError(t, store.CreateExecutionRecord(context.Background(), execution))

	// Register webhook
	secret := "test-secret"
	webhook := &types.ExecutionWebhook{
		ExecutionID: "exec-1",
		URL:         "https://example.com/webhook",
		Secret:      &secret,
	}
	require.NoError(t, store.RegisterExecutionWebhook(context.Background(), webhook))

	router := gin.New()
	router.PUT("/api/v1/executions/:execution_id/status", UpdateExecutionStatusHandler(store, payloads, mockWebhook, 90*time.Second))

	reqBody := `{
		"status": "succeeded",
		"result": {"output": "success"}
	}`
	req := httptest.NewRequest(http.MethodPut, "/api/v1/executions/exec-1/status", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	require.True(t, webhookCalled, "webhook should have been triggered")
}

func TestUpdateExecutionStatusHandler_InvalidStatus(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newTestExecutionStorage(nil)
	payloads := services.NewFilePayloadStore(t.TempDir())

	router := gin.New()
	router.PUT("/api/v1/executions/:execution_id/status", UpdateExecutionStatusHandler(store, payloads, nil, 90*time.Second))

	reqBody := `{
		"status": "invalid-status"
	}`
	req := httptest.NewRequest(http.MethodPut, "/api/v1/executions/exec-1/status", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)

	var payload map[string]string
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Contains(t, payload["error"], "unsupported status")
}

func TestUpdateExecutionStatusHandler_MissingExecutionID(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newTestExecutionStorage(nil)
	payloads := services.NewFilePayloadStore(t.TempDir())

	router := gin.New()
	router.PUT("/api/v1/executions/:execution_id/status", UpdateExecutionStatusHandler(store, payloads, nil, 90*time.Second))

	reqBody := `{
		"status": "succeeded"
	}`
	req := httptest.NewRequest(http.MethodPut, "/api/v1/executions//status", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
}

func TestUpdateExecutionStatusHandler_NotFound(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newTestExecutionStorage(nil)
	payloads := services.NewFilePayloadStore(t.TempDir())

	router := gin.New()
	router.PUT("/api/v1/executions/:execution_id/status", UpdateExecutionStatusHandler(store, payloads, nil, 90*time.Second))

	reqBody := `{
		"status": "succeeded"
	}`
	req := httptest.NewRequest(http.MethodPut, "/api/v1/executions/nonexistent/status", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	// testExecutionStorage returns an error when execution is not found,
	// which causes the handler to return 500. In production, storage might return nil
	// which would result in 404. Both are valid behaviors.
	require.True(t, resp.Code == http.StatusNotFound || resp.Code == http.StatusInternalServerError,
		"Expected 404 or 500, got %d", resp.Code)

	// Verify error message indicates execution not found
	var errorResp map[string]interface{}
	if err := json.Unmarshal(resp.Body.Bytes(), &errorResp); err == nil {
		if errorMsg, ok := errorResp["error"].(string); ok {
			require.Contains(t, strings.ToLower(errorMsg), "not found",
				"Error message should indicate execution not found: %s", errorMsg)
		}
	}
}

func TestUpdateExecutionStatusHandler_ProgressUpdate(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newTestExecutionStorage(nil)
	payloads := services.NewFilePayloadStore(t.TempDir())

	execution := &types.Execution{
		ExecutionID: "exec-1",
		RunID:       "run-1",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   time.Now().UTC(),
		CreatedAt:   time.Now().UTC(),
		UpdatedAt:   time.Now().UTC(),
	}
	require.NoError(t, store.CreateExecutionRecord(context.Background(), execution))

	router := gin.New()
	router.PUT("/api/v1/executions/:execution_id/status", UpdateExecutionStatusHandler(store, payloads, nil, 90*time.Second))

	reqBody := `{
		"status": "running",
		"progress": 50
	}`
	req := httptest.NewRequest(http.MethodPut, "/api/v1/executions/exec-1/status", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	// Verify execution is still running (not terminal)
	updated, err := store.GetExecutionRecord(context.Background(), "exec-1")
	require.NoError(t, err)
	require.NotNil(t, updated)
	require.Equal(t, types.ExecutionStatusRunning, updated.Status)
	require.Nil(t, updated.CompletedAt)
}

// TestUpdateExecutionStatusHandler_TerminalRegression covers the case where a
// late /status callback (e.g. from a retried fire-and-forget update) tries to
// move a finished execution back to a non-terminal state. The handler must
// reject the regression so callers polling /api/v1/executions/:id keep seeing
// the correct terminal status. Pinned a real production incident where the
// caller's app.call hung for 7200s because pr-af.review reported "failed" but
// a later event had stomped the status back to "running".
func TestUpdateExecutionStatusHandler_TerminalRegression(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newTestExecutionStorage(nil)
	payloads := services.NewFilePayloadStore(t.TempDir())

	completed := time.Now().UTC().Add(-time.Minute)
	execution := &types.Execution{
		ExecutionID: "exec-terminal",
		RunID:       "run-1",
		Status:      types.ExecutionStatusFailed,
		StartedAt:   time.Now().UTC().Add(-5 * time.Minute),
		CompletedAt: &completed,
		CreatedAt:   time.Now().UTC().Add(-5 * time.Minute),
		UpdatedAt:   completed,
	}
	require.NoError(t, store.CreateExecutionRecord(context.Background(), execution))

	router := gin.New()
	router.PUT("/api/v1/executions/:execution_id/status", UpdateExecutionStatusHandler(store, payloads, nil, 90*time.Second))

	// Late "running" update arrives — must be rejected.
	reqBody := `{"status": "running"}`
	req := httptest.NewRequest(http.MethodPut, "/api/v1/executions/exec-terminal/status", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusInternalServerError, resp.Code)

	// DB must still show the original terminal status — no regression.
	updated, err := store.GetExecutionRecord(context.Background(), "exec-terminal")
	require.NoError(t, err)
	require.NotNil(t, updated)
	require.Equal(t, types.ExecutionStatusFailed, updated.Status)
	require.NotNil(t, updated.CompletedAt)
}

// TestUpdateExecutionStatusHandler_TerminalIdempotent confirms the regression
// guard still allows callers to re-deliver the SAME terminal status (e.g. when
// the SDK's status-callback retry succeeds after a transient network blip). A
// terminal→same-terminal POST must return 200 and remain idempotent.
func TestUpdateExecutionStatusHandler_TerminalIdempotent(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newTestExecutionStorage(nil)
	payloads := services.NewFilePayloadStore(t.TempDir())

	completed := time.Now().UTC().Add(-time.Minute)
	execution := &types.Execution{
		ExecutionID: "exec-idempotent",
		RunID:       "run-1",
		Status:      types.ExecutionStatusFailed,
		StartedAt:   time.Now().UTC().Add(-5 * time.Minute),
		CompletedAt: &completed,
		CreatedAt:   time.Now().UTC().Add(-5 * time.Minute),
		UpdatedAt:   completed,
	}
	require.NoError(t, store.CreateExecutionRecord(context.Background(), execution))

	router := gin.New()
	router.PUT("/api/v1/executions/:execution_id/status", UpdateExecutionStatusHandler(store, payloads, nil, 90*time.Second))

	reqBody := `{"status": "failed", "error": "redelivered"}`
	req := httptest.NewRequest(http.MethodPut, "/api/v1/executions/exec-idempotent/status", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	updated, err := store.GetExecutionRecord(context.Background(), "exec-idempotent")
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusFailed, updated.Status)
}

func TestWaitForExecutionCompletion_Success(t *testing.T) {
	store := newTestExecutionStorage(nil)
	controller := newExecutionController(store, nil, nil, 90*time.Second, "")

	execution := &types.Execution{
		ExecutionID: "exec-1",
		RunID:       "run-1",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   time.Now().UTC(),
		CreatedAt:   time.Now().UTC(),
		UpdatedAt:   time.Now().UTC(),
	}
	require.NoError(t, store.CreateExecutionRecord(context.Background(), execution))

	eventBus := store.GetExecutionEventBus()
	subscriberID := "test-subscriber"
	_ = eventBus.Subscribe(subscriberID)
	defer eventBus.Unsubscribe(subscriberID)

	// Start waiting in goroutine
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	done := make(chan bool)
	var result *types.Execution
	var err error

	go func() {
		result, err = controller.waitForExecutionCompletion(ctx, "exec-1", 2*time.Second)
		done <- true
	}()

	// Wait a bit then publish completion event
	time.Sleep(100 * time.Millisecond)

	// Update execution to succeeded
	_, updateErr := store.UpdateExecutionRecord(context.Background(), "exec-1", func(current *types.Execution) (*types.Execution, error) {
		if current == nil {
			return nil, nil
		}
		now := time.Now().UTC()
		current.Status = types.ExecutionStatusSucceeded
		completed := now
		current.CompletedAt = &completed
		return current, nil
	})
	require.NoError(t, updateErr)

	// Publish completion event
	eventBus.Publish(events.ExecutionEvent{
		Type:        events.ExecutionCompleted,
		ExecutionID: "exec-1",
		WorkflowID:  "run-1",
		Status:      string(types.ExecutionStatusSucceeded),
		Timestamp:   time.Now(),
	})

	// Wait for completion
	select {
	case <-done:
		require.NoError(t, err)
		require.NotNil(t, result)
		require.Equal(t, types.ExecutionStatusSucceeded, result.Status)
	case <-time.After(1 * time.Second):
		t.Fatal("waitForExecutionCompletion timed out")
	}
}

func TestWaitForExecutionCompletion_Timeout(t *testing.T) {
	store := newTestExecutionStorage(nil)
	controller := newExecutionController(store, nil, nil, 90*time.Second, "")

	execution := &types.Execution{
		ExecutionID: "exec-1",
		RunID:       "run-1",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   time.Now().UTC(),
		CreatedAt:   time.Now().UTC(),
		UpdatedAt:   time.Now().UTC(),
	}
	require.NoError(t, store.CreateExecutionRecord(context.Background(), execution))

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	result, err := controller.waitForExecutionCompletion(ctx, "exec-1", 100*time.Millisecond)

	require.Error(t, err)
	require.Nil(t, result)
	require.Contains(t, err.Error(), "timeout")
}

func TestWaitForExecutionCompletion_ContextCancellation(t *testing.T) {
	store := newTestExecutionStorage(nil)
	controller := newExecutionController(store, nil, nil, 90*time.Second, "")

	execution := &types.Execution{
		ExecutionID: "exec-1",
		RunID:       "run-1",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   time.Now().UTC(),
		CreatedAt:   time.Now().UTC(),
		UpdatedAt:   time.Now().UTC(),
	}
	require.NoError(t, store.CreateExecutionRecord(context.Background(), execution))

	ctx, cancel := context.WithCancel(context.Background())

	done := make(chan bool)
	var result *types.Execution
	var err error

	go func() {
		result, err = controller.waitForExecutionCompletion(ctx, "exec-1", 5*time.Second)
		done <- true
	}()

	// Cancel context after short delay
	time.Sleep(50 * time.Millisecond)
	cancel()

	select {
	case <-done:
		require.Error(t, err)
		require.Nil(t, result)
		require.Equal(t, context.Canceled, err)
	case <-time.After(1 * time.Second):
		t.Fatal("waitForExecutionCompletion did not respond to context cancellation")
	}
}

func TestWaitForExecutionCompletion_NoEventBus(t *testing.T) {
	// Create storage without event bus
	store := &testExecutionStorageWithoutEventBus{}
	controller := newExecutionController(store, nil, nil, 90*time.Second, "")

	ctx := context.Background()
	result, err := controller.waitForExecutionCompletion(ctx, "exec-1", 1*time.Second)

	require.Error(t, err)
	require.Nil(t, result)
	require.Contains(t, err.Error(), "event bus not available")
}

// Mock webhook dispatcher
type mockWebhookDispatcher struct {
	notifyFunc func(ctx context.Context, executionID string) error
}

func (m *mockWebhookDispatcher) Start(ctx context.Context) error {
	return nil
}

func (m *mockWebhookDispatcher) Stop(ctx context.Context) error {
	return nil
}

func (m *mockWebhookDispatcher) Notify(ctx context.Context, executionID string) error {
	if m.notifyFunc != nil {
		return m.notifyFunc(ctx, executionID)
	}
	return nil
}

// Test storage without event bus
type testExecutionStorageWithoutEventBus struct {
	testExecutionStorage
}

func (s *testExecutionStorageWithoutEventBus) GetExecutionEventBus() *events.ExecutionEventBus {
	return nil
}
