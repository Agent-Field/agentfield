package ui

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// mockForwarder implements services.ObservabilityForwarder for testing.
type mockForwarder struct {
	status      types.ObservabilityForwarderStatus
	reloadErr   error
	redriveResp types.ObservabilityRedriveResponse
}

func (m *mockForwarder) Start(ctx context.Context) error {
	return nil
}

func (m *mockForwarder) Stop(ctx context.Context) error {
	return nil
}

func (m *mockForwarder) ReloadConfig(ctx context.Context) error {
	return m.reloadErr
}

func (m *mockForwarder) GetStatus() types.ObservabilityForwarderStatus {
	return m.status
}

func (m *mockForwarder) Redrive(ctx context.Context) types.ObservabilityRedriveResponse {
	return m.redriveResp
}

// setupTestEnvironment creates test storage and handler for observability webhook tests.
func setupTestEnvironment(t *testing.T) (*storage.LocalStorage, *mockForwarder, *ObservabilityWebhookHandler, *gin.Engine) {
	t.Helper()

	gin.SetMode(gin.TestMode)

	ctx := context.Background()
	tempDir := t.TempDir()

	cfg := storage.StorageConfig{
		Mode: "local",
		Local: storage.LocalStorageConfig{
			DatabasePath: filepath.Join(tempDir, "test.db"),
			KVStorePath:  filepath.Join(tempDir, "test.bolt"),
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

	mockFwd := &mockForwarder{
		status: types.ObservabilityForwarderStatus{
			Enabled:         true,
			WebhookURL:      "https://example.com/webhook",
			QueueDepth:      5,
			EventsForwarded: 100,
			EventsDropped:   2,
			DeadLetterCount: 3,
		},
	}

	handler := NewObservabilityWebhookHandler(realStorage, mockFwd)
	router := gin.New()

	// Register routes
	router.GET("/api/v1/settings/observability-webhook", handler.GetWebhookHandler)
	router.POST("/api/v1/settings/observability-webhook", handler.SetWebhookHandler)
	router.DELETE("/api/v1/settings/observability-webhook", handler.DeleteWebhookHandler)
	router.GET("/api/v1/settings/observability-webhook/status", handler.GetStatusHandler)
	router.POST("/api/v1/settings/observability-webhook/redrive", handler.RedriveHandler)
	router.GET("/api/v1/settings/observability-webhook/dlq", handler.GetDeadLetterQueueHandler)
	router.DELETE("/api/v1/settings/observability-webhook/dlq", handler.ClearDeadLetterQueueHandler)

	return realStorage, mockFwd, handler, router
}

// Test GET /api/v1/settings/observability-webhook - not configured
func TestGetWebhookHandler_NotConfigured(t *testing.T) {
	_, _, _, router := setupTestEnvironment(t)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/settings/observability-webhook", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result types.ObservabilityWebhookConfigResponse
	err := json.Unmarshal(resp.Body.Bytes(), &result)
	require.NoError(t, err)
	require.False(t, result.Configured)
	require.Nil(t, result.Config)
}

// Test GET /api/v1/settings/observability-webhook - configured
func TestGetWebhookHandler_Configured(t *testing.T) {
	store, _, _, router := setupTestEnvironment(t)

	// Set up a webhook config
	secret := "test-secret"
	config := &types.ObservabilityWebhookConfig{
		ID:  "global",
		URL: "https://example.com/webhook",
		Secret: &secret,
		Headers: map[string]string{
			"X-Custom": "value",
		},
		Enabled: true,
	}
	err := store.SetObservabilityWebhook(context.Background(), config)
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/settings/observability-webhook", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result types.ObservabilityWebhookConfigResponse
	err = json.Unmarshal(resp.Body.Bytes(), &result)
	require.NoError(t, err)
	require.True(t, result.Configured)
	require.NotNil(t, result.Config)
	require.Equal(t, "https://example.com/webhook", result.Config.URL)
	require.True(t, result.Config.HasSecret)
	require.Equal(t, "value", result.Config.Headers["X-Custom"])
	require.True(t, result.Config.Enabled)
}

// Test POST /api/v1/settings/observability-webhook - create new config
func TestSetWebhookHandler_Create(t *testing.T) {
	_, _, _, router := setupTestEnvironment(t)

	reqBody := types.ObservabilityWebhookConfigRequest{
		URL:     "https://webhook.example.com/events",
		Secret:  stringPtr("my-secret"),
		Headers: map[string]string{"Authorization": "Bearer token"},
		Enabled: boolPtr(true),
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/settings/observability-webhook", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result map[string]interface{}
	err := json.Unmarshal(resp.Body.Bytes(), &result)
	require.NoError(t, err)
	require.Equal(t, true, result["success"])
	require.Contains(t, result["message"].(string), "configured successfully")
}

// Test POST /api/v1/settings/observability-webhook - missing URL
func TestSetWebhookHandler_MissingURL(t *testing.T) {
	_, _, _, router := setupTestEnvironment(t)

	reqBody := types.ObservabilityWebhookConfigRequest{
		URL: "",
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/settings/observability-webhook", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)

	var result ErrorResponse
	err := json.Unmarshal(resp.Body.Bytes(), &result)
	require.NoError(t, err)
	require.Contains(t, strings.ToLower(result.Error), "url")
}

// Test POST /api/v1/settings/observability-webhook - invalid URL
func TestSetWebhookHandler_InvalidURL(t *testing.T) {
	_, _, _, router := setupTestEnvironment(t)

	reqBody := map[string]interface{}{
		"url": "not-a-valid-url",
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/settings/observability-webhook", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
}

// Test POST /api/v1/settings/observability-webhook - FTP URL rejected
func TestSetWebhookHandler_NonHTTPURL(t *testing.T) {
	_, _, _, router := setupTestEnvironment(t)

	reqBody := map[string]interface{}{
		"url": "ftp://files.example.com/webhook",
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/settings/observability-webhook", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)

	var result ErrorResponse
	err := json.Unmarshal(resp.Body.Bytes(), &result)
	require.NoError(t, err)
	require.Contains(t, strings.ToLower(result.Error), "http")
}

// Test POST /api/v1/settings/observability-webhook - defaults enabled to true
func TestSetWebhookHandler_DefaultsEnabled(t *testing.T) {
	store, _, _, router := setupTestEnvironment(t)

	reqBody := map[string]interface{}{
		"url": "https://webhook.example.com/events",
		// enabled not specified
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/settings/observability-webhook", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	// Verify stored config has enabled = true
	config, err := store.GetObservabilityWebhook(context.Background())
	require.NoError(t, err)
	require.NotNil(t, config)
	require.True(t, config.Enabled)
}

// Test DELETE /api/v1/settings/observability-webhook
func TestDeleteWebhookHandler(t *testing.T) {
	store, _, _, router := setupTestEnvironment(t)

	// Create config first
	config := &types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     "https://example.com/webhook",
		Enabled: true,
	}
	err := store.SetObservabilityWebhook(context.Background(), config)
	require.NoError(t, err)

	// Delete
	req := httptest.NewRequest(http.MethodDelete, "/api/v1/settings/observability-webhook", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result map[string]interface{}
	err = json.Unmarshal(resp.Body.Bytes(), &result)
	require.NoError(t, err)
	require.Equal(t, true, result["success"])

	// Verify deleted
	retrieved, err := store.GetObservabilityWebhook(context.Background())
	require.NoError(t, err)
	require.Nil(t, retrieved)
}

// Test GET /api/v1/settings/observability-webhook/status
func TestGetStatusHandler(t *testing.T) {
	_, mockFwd, _, router := setupTestEnvironment(t)

	now := time.Now().UTC()
	lastErr := "connection timeout"
	mockFwd.status = types.ObservabilityForwarderStatus{
		Enabled:          true,
		WebhookURL:       "https://example.com/webhook",
		QueueDepth:       10,
		EventsForwarded:  500,
		EventsDropped:    5,
		DeadLetterCount:  15,
		LastForwardedAt:  &now,
		LastError:        &lastErr,
	}

	req := httptest.NewRequest(http.MethodGet, "/api/v1/settings/observability-webhook/status", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result types.ObservabilityForwarderStatus
	err := json.Unmarshal(resp.Body.Bytes(), &result)
	require.NoError(t, err)
	require.True(t, result.Enabled)
	require.Equal(t, "https://example.com/webhook", result.WebhookURL)
	require.Equal(t, 10, result.QueueDepth)
	require.Equal(t, int64(500), result.EventsForwarded)
	require.Equal(t, int64(5), result.EventsDropped)
	require.Equal(t, int64(15), result.DeadLetterCount)
	require.NotNil(t, result.LastForwardedAt)
	require.NotNil(t, result.LastError)
	require.Equal(t, "connection timeout", *result.LastError)
}

// Test GET /api/v1/settings/observability-webhook/status - no forwarder
func TestGetStatusHandler_NoForwarder(t *testing.T) {
	gin.SetMode(gin.TestMode)

	ctx := context.Background()
	tempDir := t.TempDir()
	cfg := storage.StorageConfig{
		Mode: "local",
		Local: storage.LocalStorageConfig{
			DatabasePath: filepath.Join(tempDir, "test.db"),
			KVStorePath:  filepath.Join(tempDir, "test.bolt"),
		},
	}

	realStorage := storage.NewLocalStorage(storage.LocalStorageConfig{})
	err := realStorage.Initialize(ctx, cfg)
	if err != nil && strings.Contains(strings.ToLower(err.Error()), "fts5") {
		t.Skip("sqlite3 compiled without FTS5")
	}
	require.NoError(t, err)
	defer realStorage.Close(ctx)

	// Create handler with nil forwarder
	handler := NewObservabilityWebhookHandler(realStorage, nil)
	router := gin.New()
	router.GET("/api/v1/settings/observability-webhook/status", handler.GetStatusHandler)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/settings/observability-webhook/status", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result types.ObservabilityForwarderStatus
	err = json.Unmarshal(resp.Body.Bytes(), &result)
	require.NoError(t, err)
	require.False(t, result.Enabled)
}

// Test POST /api/v1/settings/observability-webhook/redrive - success
func TestRedriveHandler_Success(t *testing.T) {
	_, mockFwd, _, router := setupTestEnvironment(t)

	mockFwd.redriveResp = types.ObservabilityRedriveResponse{
		Success:   true,
		Message:   "redrove 10 events",
		Processed: 10,
		Failed:    0,
	}

	req := httptest.NewRequest(http.MethodPost, "/api/v1/settings/observability-webhook/redrive", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result types.ObservabilityRedriveResponse
	err := json.Unmarshal(resp.Body.Bytes(), &result)
	require.NoError(t, err)
	require.True(t, result.Success)
	require.Equal(t, 10, result.Processed)
	require.Equal(t, 0, result.Failed)
}

// Test POST /api/v1/settings/observability-webhook/redrive - partial failure
func TestRedriveHandler_PartialFailure(t *testing.T) {
	_, mockFwd, _, router := setupTestEnvironment(t)

	mockFwd.redriveResp = types.ObservabilityRedriveResponse{
		Success:   false,
		Message:   "redrove 5 events, 3 failed",
		Processed: 5,
		Failed:    3,
	}

	req := httptest.NewRequest(http.MethodPost, "/api/v1/settings/observability-webhook/redrive", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code) // Still 200 as operation completed

	var result types.ObservabilityRedriveResponse
	err := json.Unmarshal(resp.Body.Bytes(), &result)
	require.NoError(t, err)
	require.False(t, result.Success)
	require.Equal(t, 5, result.Processed)
	require.Equal(t, 3, result.Failed)
}

// Test POST /api/v1/settings/observability-webhook/redrive - no forwarder
func TestRedriveHandler_NoForwarder(t *testing.T) {
	gin.SetMode(gin.TestMode)

	ctx := context.Background()
	tempDir := t.TempDir()
	cfg := storage.StorageConfig{
		Mode: "local",
		Local: storage.LocalStorageConfig{
			DatabasePath: filepath.Join(tempDir, "test.db"),
			KVStorePath:  filepath.Join(tempDir, "test.bolt"),
		},
	}

	realStorage := storage.NewLocalStorage(storage.LocalStorageConfig{})
	err := realStorage.Initialize(ctx, cfg)
	if err != nil && strings.Contains(strings.ToLower(err.Error()), "fts5") {
		t.Skip("sqlite3 compiled without FTS5")
	}
	require.NoError(t, err)
	defer realStorage.Close(ctx)

	handler := NewObservabilityWebhookHandler(realStorage, nil)
	router := gin.New()
	router.POST("/api/v1/settings/observability-webhook/redrive", handler.RedriveHandler)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/settings/observability-webhook/redrive", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusServiceUnavailable, resp.Code)
}

// Test GET /api/v1/settings/observability-webhook/dlq
func TestGetDeadLetterQueueHandler(t *testing.T) {
	store, _, _, router := setupTestEnvironment(t)

	// Add some DLQ entries
	for i := 0; i < 5; i++ {
		event := &types.ObservabilityEvent{
			EventType:   "execution_failed",
			EventSource: "execution",
			Timestamp:   time.Now().UTC().Format(time.RFC3339),
			Data:        map[string]interface{}{"index": i},
		}
		err := store.AddToDeadLetterQueue(context.Background(), event, "webhook unavailable", 3)
		require.NoError(t, err)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/v1/settings/observability-webhook/dlq", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result types.ObservabilityDeadLetterListResponse
	err := json.Unmarshal(resp.Body.Bytes(), &result)
	require.NoError(t, err)
	require.Equal(t, int64(5), result.TotalCount)
	require.Len(t, result.Entries, 5)
}

// Test GET /api/v1/settings/observability-webhook/dlq - with pagination
func TestGetDeadLetterQueueHandler_Pagination(t *testing.T) {
	store, _, _, router := setupTestEnvironment(t)

	// Add DLQ entries
	for i := 0; i < 10; i++ {
		event := &types.ObservabilityEvent{
			EventType:   "test_event",
			EventSource: "test",
			Timestamp:   time.Now().UTC().Format(time.RFC3339),
			Data:        map[string]interface{}{"index": i},
		}
		err := store.AddToDeadLetterQueue(context.Background(), event, "test error", 3)
		require.NoError(t, err)
	}

	// First page
	req := httptest.NewRequest(http.MethodGet, "/api/v1/settings/observability-webhook/dlq?limit=3&offset=0", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result types.ObservabilityDeadLetterListResponse
	err := json.Unmarshal(resp.Body.Bytes(), &result)
	require.NoError(t, err)
	require.Equal(t, int64(10), result.TotalCount)
	require.Len(t, result.Entries, 3)

	// Second page
	req = httptest.NewRequest(http.MethodGet, "/api/v1/settings/observability-webhook/dlq?limit=3&offset=3", nil)
	resp = httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result2 types.ObservabilityDeadLetterListResponse
	err = json.Unmarshal(resp.Body.Bytes(), &result2)
	require.NoError(t, err)
	require.Len(t, result2.Entries, 3)

	// Verify no overlap
	for _, e1 := range result.Entries {
		for _, e2 := range result2.Entries {
			require.NotEqual(t, e1.ID, e2.ID)
		}
	}
}

// Test GET /api/v1/settings/observability-webhook/dlq - empty
func TestGetDeadLetterQueueHandler_Empty(t *testing.T) {
	_, _, _, router := setupTestEnvironment(t)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/settings/observability-webhook/dlq", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result types.ObservabilityDeadLetterListResponse
	err := json.Unmarshal(resp.Body.Bytes(), &result)
	require.NoError(t, err)
	require.Equal(t, int64(0), result.TotalCount)
	require.Empty(t, result.Entries)
}

// Test GET /api/v1/settings/observability-webhook/dlq - limit capping
func TestGetDeadLetterQueueHandler_LimitCap(t *testing.T) {
	store, _, _, router := setupTestEnvironment(t)

	// Add many entries
	for i := 0; i < 50; i++ {
		event := &types.ObservabilityEvent{
			EventType:   "test_event",
			EventSource: "test",
			Timestamp:   time.Now().UTC().Format(time.RFC3339),
			Data:        map[string]interface{}{"index": i},
		}
		err := store.AddToDeadLetterQueue(context.Background(), event, "test error", 3)
		require.NoError(t, err)
	}

	// Request with limit > 1000 (should be capped)
	req := httptest.NewRequest(http.MethodGet, "/api/v1/settings/observability-webhook/dlq?limit=2000", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result types.ObservabilityDeadLetterListResponse
	err := json.Unmarshal(resp.Body.Bytes(), &result)
	require.NoError(t, err)
	// Should have all 50 entries (capped limit is 1000, but we only have 50)
	require.Len(t, result.Entries, 50)
}

// Test DELETE /api/v1/settings/observability-webhook/dlq
func TestClearDeadLetterQueueHandler(t *testing.T) {
	store, _, _, router := setupTestEnvironment(t)

	// Add DLQ entries
	for i := 0; i < 5; i++ {
		event := &types.ObservabilityEvent{
			EventType:   "test_event",
			EventSource: "test",
			Timestamp:   time.Now().UTC().Format(time.RFC3339),
			Data:        map[string]interface{}{"index": i},
		}
		err := store.AddToDeadLetterQueue(context.Background(), event, "test error", 3)
		require.NoError(t, err)
	}

	// Verify entries exist
	count, err := store.GetDeadLetterQueueCount(context.Background())
	require.NoError(t, err)
	require.Equal(t, int64(5), count)

	// Clear DLQ
	req := httptest.NewRequest(http.MethodDelete, "/api/v1/settings/observability-webhook/dlq", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result map[string]interface{}
	err = json.Unmarshal(resp.Body.Bytes(), &result)
	require.NoError(t, err)
	require.Equal(t, true, result["success"])

	// Verify empty
	count, err = store.GetDeadLetterQueueCount(context.Background())
	require.NoError(t, err)
	require.Equal(t, int64(0), count)
}

// Test DELETE /api/v1/settings/observability-webhook/dlq - already empty
func TestClearDeadLetterQueueHandler_Empty(t *testing.T) {
	_, _, _, router := setupTestEnvironment(t)

	req := httptest.NewRequest(http.MethodDelete, "/api/v1/settings/observability-webhook/dlq", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result map[string]interface{}
	err := json.Unmarshal(resp.Body.Bytes(), &result)
	require.NoError(t, err)
	require.Equal(t, true, result["success"])
}

// Test parseIntParam helper
func TestParseIntParam(t *testing.T) {
	tests := []struct {
		input    string
		expected int
		hasError bool
	}{
		{"10", 10, false},
		{"0", 0, false},
		{"-5", -5, false},
		{"100", 100, false},
		{"abc", 0, true},
		{"", 0, true},
		{"10.5", 10, false}, // Note: Sscanf parses until first non-digit
	}

	for _, tt := range tests {
		result, err := parseIntParam(tt.input)
		if tt.hasError {
			require.Error(t, err, "expected error for input: %s", tt.input)
		} else {
			require.NoError(t, err, "unexpected error for input: %s", tt.input)
			require.Equal(t, tt.expected, result, "wrong result for input: %s", tt.input)
		}
	}
}

// Helper functions
func stringPtr(s string) *string {
	return &s
}

func boolPtr(b bool) *bool {
	return &b
}
