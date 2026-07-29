package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

var errBatchStorageBoom = errors.New("storage unavailable")

// batchCountingStorage wraps testExecutionStorage and counts how many times
// the batch fetch method is invoked, so tests can assert the N+1 fix.
type batchCountingStorage struct {
	*testExecutionStorage
	batchCalls                  int
	singleGetCalls              int
	getExecutionRecordsBatchErr error
	getExecutionRecordErrs      map[string]error
}

func newBatchCountingStorage() *batchCountingStorage {
	return &batchCountingStorage{
		testExecutionStorage: newTestExecutionStorage(nil),
	}
}

// Override the single-record fetch to count it; the handler should no longer
// touch this path for batch status.
func (s *batchCountingStorage) GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error) {
	s.singleGetCalls++
	if err := s.getExecutionRecordErrs[executionID]; err != nil {
		return nil, err
	}
	return s.testExecutionStorage.GetExecutionRecord(ctx, executionID)
}

func (s *batchCountingStorage) GetExecutionRecordsBatch(ctx context.Context, executionIDs []string) (map[string]*types.Execution, error) {
	s.batchCalls++
	if s.getExecutionRecordsBatchErr != nil {
		return nil, s.getExecutionRecordsBatchErr
	}
	return s.testExecutionStorage.GetExecutionRecordsBatch(ctx, executionIDs)
}

func seedBatchExecution(t *testing.T, store *batchCountingStorage, id, status string) {
	t.Helper()
	now := time.Now().UTC()
	exec := &types.Execution{
		ExecutionID: id,
		RunID:       "run-batch",
		AgentNodeID: "agent-1",
		ReasonerID:  "reasoner-" + id,
		NodeID:      "node-1",
		Status:      status,
		StartedAt:   now,
		CreatedAt:   now,
		UpdatedAt:   now,
	}
	require.NoError(t, store.CreateExecutionRecord(context.Background(), exec))
}

func TestHandleBatchStatus_SingleFetchForTenIDs(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newBatchCountingStorage()

	// Seed 10 existing executions.
	for i := 0; i < 10; i++ {
		seedBatchExecution(t, store, "exec-"+string(rune('a'+i)), string(types.ExecutionStatusSucceeded))
	}

	// Request 10 existing + 2 missing.
	ids := []string{
		"exec-a", "exec-b", "exec-c", "exec-d", "exec-e",
		"exec-f", "exec-g", "exec-h", "exec-i", "exec-j",
		"missing-1", "missing-2",
	}
	body, _ := json.Marshal(BatchStatusRequest{ExecutionIDs: ids})

	router := gin.New()
	router.POST("/batch", BatchExecutionStatusHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/batch", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.Equal(t, 1, store.batchCalls, "handleBatchStatus must make exactly one storage fetch")
	require.Equal(t, 0, store.singleGetCalls, "handleBatchStatus must not call GetExecutionRecord per ID")

	var response BatchStatusResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))
	require.Len(t, response, len(ids))

	// Existing IDs get the rendered status response contract.
	for _, id := range ids[:10] {
		entry, ok := response[id]
		require.True(t, ok, "missing entry for %s", id)
		require.Equal(t, id, entry.ExecutionID)
		require.Equal(t, "run-batch", entry.RunID)
		require.Equal(t, string(types.ExecutionStatusSucceeded), entry.Status)
		require.NotEmpty(t, entry.StartedAt)
	}

	// Missing IDs preserve the prior per-ID response behavior: not_found.
	for _, id := range ids[10:] {
		entry, ok := response[id]
		require.True(t, ok, "missing entry for %s", id)
		require.Equal(t, id, entry.ExecutionID)
		require.Equal(t, "not_found", entry.Status)
	}
}

func TestHandleBatchStatus_RejectsOversizedBatch(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newBatchCountingStorage()

	ids := make([]string, 501)
	for i := range ids {
		ids[i] = "exec-" + string(rune('a'+i%26))
	}
	body, _ := json.Marshal(BatchStatusRequest{ExecutionIDs: ids})

	router := gin.New()
	router.POST("/batch", BatchExecutionStatusHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/batch", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	require.Equal(t, http.StatusBadRequest, w.Code)
	require.Equal(t, 0, store.batchCalls, "oversized batch must not hit storage")
}

func TestHandleBatchStatus_BatchStorageErrorPreservesPerIDResults(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newBatchCountingStorage()
	store.getExecutionRecordsBatchErr = errBatchStorageBoom
	store.getExecutionRecordErrs = map[string]error{"bad": errBatchStorageBoom}
	seedBatchExecution(t, store, "good", string(types.ExecutionStatusSucceeded))

	body, _ := json.Marshal(BatchStatusRequest{ExecutionIDs: []string{"bad", "good"}})
	router := gin.New()
	router.POST("/batch", BatchExecutionStatusHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/batch", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.Equal(t, 1, store.batchCalls)
	require.Equal(t, 2, store.singleGetCalls)

	var response BatchStatusResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))
	require.Equal(t, "error", response["bad"].Status)
	require.Contains(t, *response["bad"].Error, "load execution: storage unavailable")
	require.Equal(t, string(types.ExecutionStatusSucceeded), response["good"].Status)
}

func TestHandleBatchStatus_BatchStorageErrorWithCanceledContextReturnsPerIDErrors(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newBatchCountingStorage()
	store.getExecutionRecordsBatchErr = context.Canceled
	store.getExecutionRecordErrs = map[string]error{
		"exec-a": context.Canceled,
		"exec-b": context.Canceled,
	}
	body, _ := json.Marshal(BatchStatusRequest{ExecutionIDs: []string{"exec-a", "exec-b"}})
	router := gin.New()
	router.POST("/batch", BatchExecutionStatusHandler(store))

	reqCtx, cancel := context.WithCancel(context.Background())
	cancel()
	req := httptest.NewRequest(http.MethodPost, "/batch", bytes.NewReader(body)).WithContext(reqCtx)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.Equal(t, 1, store.batchCalls)
	require.Equal(t, 2, store.singleGetCalls)

	var response BatchStatusResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &response))
	for _, id := range []string{"exec-a", "exec-b"} {
		require.Equal(t, "error", response[id].Status)
		require.Contains(t, *response[id].Error, "load execution: context canceled")
	}
}
