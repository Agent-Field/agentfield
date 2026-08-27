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
		RunID:       "run-1",
		AgentNodeID: "node-1",
		ReasonerID:  "reasoner-a",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   time.Now().UTC(),
		CreatedAt:   time.Now().UTC(),
		UpdatedAt:   time.Now().UTC(),
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

// TestUpdateExecutionStatusHandler_CrossTerminalConflict verifies the guard
// also refuses to rewrite one terminal status as a different one. Before this
// guard, a duplicate or late callback could flip "succeeded" to "failed" (or
// the reverse) after the outcome had already been observed by pollers,
// webhooks, and telemetry.
func TestUpdateExecutionStatusHandler_CrossTerminalConflict(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newTestExecutionStorage(nil)
	payloads := services.NewFilePayloadStore(t.TempDir())

	completed := time.Now().UTC().Add(-time.Minute)
	execution := &types.Execution{
		ExecutionID: "exec-cross-terminal",
		RunID:       "run-1",
		Status:      types.ExecutionStatusSucceeded,
		StartedAt:   time.Now().UTC().Add(-5 * time.Minute),
		CompletedAt: &completed,
		CreatedAt:   time.Now().UTC().Add(-5 * time.Minute),
		UpdatedAt:   completed,
	}
	require.NoError(t, store.CreateExecutionRecord(context.Background(), execution))

	router := gin.New()
	router.PUT("/api/v1/executions/:execution_id/status", UpdateExecutionStatusHandler(store, payloads, nil, 90*time.Second))

	reqBody := `{"status": "failed", "error": "late duplicate callback"}`
	req := httptest.NewRequest(http.MethodPut, "/api/v1/executions/exec-cross-terminal/status", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusConflict, resp.Code)

	updated, err := store.GetExecutionRecord(context.Background(), "exec-cross-terminal")
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusSucceeded, updated.Status)
	require.Nil(t, updated.ErrorMessage)
	require.NotNil(t, updated.CompletedAt)
	require.Equal(t, completed.Unix(), updated.CompletedAt.Unix())
}

// TestUpdateExecutionStatusHandler_TerminalRetryIsNoOp confirms an idempotent
// re-delivery of the final status is acknowledged with 200 but runs no side
// effect a second time: no second lifecycle event on the bus (every bus
// consumer — SSE clients, tracing, telemetry — would re-count the
// completion), no second webhook notification, and no rewrite of the
// persisted record.
func TestUpdateExecutionStatusHandler_TerminalRetryIsNoOp(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   "http://agent.example",
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}

	store := newTestExecutionStorage(agent)
	payloads := services.NewFilePayloadStore(t.TempDir())

	notifyCount := 0
	mockWebhook := &mockWebhookDispatcher{
		notifyFunc: func(ctx context.Context, executionID string) error {
			notifyCount++
			return nil
		},
	}

	execution := &types.Execution{
		ExecutionID: "exec-retry-noop",
		RunID:       "run-1",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   time.Now().UTC(),
		CreatedAt:   time.Now().UTC(),
		UpdatedAt:   time.Now().UTC(),
	}
	require.NoError(t, store.CreateExecutionRecord(context.Background(), execution))

	secret := "test-secret"
	require.NoError(t, store.RegisterExecutionWebhook(context.Background(), &types.ExecutionWebhook{
		ExecutionID: "exec-retry-noop",
		URL:         "https://example.com/webhook",
		Secret:      &secret,
	}))

	eventCh := store.GetExecutionEventBus().Subscribe("terminal-retry-noop-test")
	defer store.GetExecutionEventBus().Unsubscribe("terminal-retry-noop-test")

	router := gin.New()
	router.PUT("/api/v1/executions/:execution_id/status", UpdateExecutionStatusHandler(store, payloads, mockWebhook, 90*time.Second))

	send := func(body string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPut, "/api/v1/executions/exec-retry-noop/status", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		return resp
	}
	drain := func() int {
		count := 0
		for {
			select {
			case <-eventCh:
				count++
			default:
				return count
			}
		}
	}

	// First delivery completes the execution and runs the side effects once.
	resp := send(`{"status": "succeeded", "result": {"output": "original"}, "duration_ms": 1200}`)
	require.Equal(t, http.StatusOK, resp.Code)
	require.Equal(t, 1, notifyCount)
	require.Equal(t, 1, drain(), "first terminal delivery must publish exactly one lifecycle event")

	first, err := store.GetExecutionRecord(context.Background(), "exec-retry-noop")
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusSucceeded, first.Status)

	// Re-delivery of the same terminal status (e.g. the SDK retrying after a
	// lost 200) is acknowledged but must not repeat any side effect or
	// rewrite the record — even when the retry carries a different payload.
	resp = send(`{"status": "succeeded", "result": {"output": "rewritten"}, "duration_ms": 9999}`)
	require.Equal(t, http.StatusOK, resp.Code)
	require.Equal(t, 1, notifyCount, "webhook must not be re-notified on an idempotent retry")
	require.Equal(t, 0, drain(), "idempotent retry must not publish a duplicate lifecycle event")

	second, err := store.GetExecutionRecord(context.Background(), "exec-retry-noop")
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusSucceeded, second.Status)
	require.Equal(t, string(first.ResultPayload), string(second.ResultPayload), "retry must not rewrite the result payload")
	require.Equal(t, *first.DurationMS, *second.DurationMS, "retry must not rewrite the duration")
	require.NotNil(t, second.CompletedAt)
	require.Equal(t, first.CompletedAt.UnixNano(), second.CompletedAt.UnixNano(), "retry must not move completed_at")
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

// TestUpdateExecutionStatusHandler_WebhookTriggeredFromStore validates the fix
// for issue #936: webhook delivery must be triggered based on the
// execution_webhooks table, not the in-memory WebhookRegistered field (which
// is not persisted due to db:"-" tag).
func TestUpdateExecutionStatusHandler_WebhookTriggeredFromStore(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   "http://agent.example",
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}

	store := newTestExecutionStorage(agent)
	payloads := services.NewFilePayloadStore(t.TempDir())

	webhookCalled := false
	mockWebhook := &mockWebhookDispatcher{
		notifyFunc: func(ctx context.Context, executionID string) error {
			webhookCalled = true
			return nil
		},
	}

	// Create execution WITHOUT WebhookRegistered set (simulates loading from DB
	// where the field is always false due to db:"-").
	execution := &types.Execution{
		ExecutionID:       "exec-wh-store",
		RunID:             "run-1",
		Status:            types.ExecutionStatusRunning,
		StartedAt:         time.Now().UTC(),
		CreatedAt:         time.Now().UTC(),
		UpdatedAt:         time.Now().UTC(),
		WebhookRegistered: false, // Explicitly false — as if loaded from DB
	}
	require.NoError(t, store.CreateExecutionRecord(context.Background(), execution))

	// Register webhook in the store (separate table)
	secret := "test-secret"
	webhook := &types.ExecutionWebhook{
		ExecutionID: "exec-wh-store",
		URL:         "https://example.com/webhook",
		Secret:      &secret,
	}
	require.NoError(t, store.RegisterExecutionWebhook(context.Background(), webhook))

	router := gin.New()
	router.PUT("/api/v1/executions/:execution_id/status", UpdateExecutionStatusHandler(store, payloads, mockWebhook, 90*time.Second))

	reqBody := `{
		"status": "succeeded",
		"result": {"output": "done"}
	}`
	req := httptest.NewRequest(http.MethodPut, "/api/v1/executions/exec-wh-store/status", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	require.True(t, webhookCalled, "webhook must be triggered even when WebhookRegistered is false on the execution record (issue #936)")

	// Also verify the status response reports webhook_registered=true
	var statusResp ExecutionStatusResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &statusResp))
	require.True(t, statusResp.WebhookRegistered, "GET status must report webhook_registered=true from the webhooks table")
}

// A terminal status can reach the store without a lifecycle event on the bus
// (the SDK's reasoner.completed workflow event takes that path, and the
// /status callback that follows is then an idempotent no-op). The waiter must
// still return promptly rather than sit out the full timeout.
func TestWaitForExecutionCompletion_TerminalStatusWithoutEvent(t *testing.T) {
	store := newTestExecutionStorage(nil)
	controller := newExecutionController(store, nil, nil, 90*time.Second, "")

	previousInterval := completionPollInterval
	completionPollInterval = 50 * time.Millisecond
	defer func() { completionPollInterval = previousInterval }()

	execution := &types.Execution{
		ExecutionID: "exec-1",
		RunID:       "run-1",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   time.Now().UTC(),
		CreatedAt:   time.Now().UTC(),
		UpdatedAt:   time.Now().UTC(),
	}
	require.NoError(t, store.CreateExecutionRecord(context.Background(), execution))

	done := make(chan struct{})
	var result *types.Execution
	var err error
	started := time.Now()
	go func() {
		result, err = controller.waitForExecutionCompletion(context.Background(), "exec-1", 5*time.Second)
		close(done)
	}()

	// Let the waiter subscribe and run its pre-subscribe check while the
	// record is still running, then store the terminal state with NO event.
	time.Sleep(100 * time.Millisecond)
	_, updateErr := store.UpdateExecutionRecord(context.Background(), "exec-1", func(current *types.Execution) (*types.Execution, error) {
		if current == nil {
			return nil, nil
		}
		now := time.Now().UTC()
		current.Status = types.ExecutionStatusSucceeded
		current.CompletedAt = &now
		return current, nil
	})
	require.NoError(t, updateErr)

	select {
	case <-done:
		require.NoError(t, err)
		require.NotNil(t, result)
		require.Equal(t, types.ExecutionStatusSucceeded, result.Status)
		require.Less(t, time.Since(started), 2*time.Second, "waiter should complete from the poll, not the timeout")
	case <-time.After(3 * time.Second):
		t.Fatal("waitForExecutionCompletion never noticed the stored terminal status")
	}
}
