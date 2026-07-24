package agentic

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
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// eventsTestStorage overrides the two storage calls queryEvents composes.
type eventsTestStorage struct {
	*mockStatusStorage
	queryExecutionRecordsFn       func(context.Context, types.ExecutionFilter) ([]*types.Execution, error)
	listWorkflowExecutionEventsFn func(context.Context, string, *int64, int) ([]*types.WorkflowExecutionEvent, error)
}

func (s *eventsTestStorage) QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
	if s.queryExecutionRecordsFn != nil {
		return s.queryExecutionRecordsFn(ctx, filter)
	}
	return s.mockStatusStorage.QueryExecutionRecords(ctx, filter)
}

func (s *eventsTestStorage) ListWorkflowExecutionEvents(ctx context.Context, executionID string, afterSeq *int64, limit int) ([]*types.WorkflowExecutionEvent, error) {
	if s.listWorkflowExecutionEventsFn != nil {
		return s.listWorkflowExecutionEventsFn(ctx, executionID, afterSeq, limit)
	}
	return s.mockStatusStorage.ListWorkflowExecutionEvents(ctx, executionID, afterSeq, limit)
}

func postEventsQuery(t *testing.T, store *eventsTestStorage, body map[string]interface{}) *httptest.ResponseRecorder {
	t.Helper()

	router := gin.New()
	router.POST("/query", QueryHandler(store))

	encoded, err := json.Marshal(body)
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewReader(encoded))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec
}

func eventAt(executionID string, seq int64, eventType string, emittedAt time.Time) *types.WorkflowExecutionEvent {
	return &types.WorkflowExecutionEvent{
		ExecutionID: executionID,
		WorkflowID:  "wf-1",
		Sequence:    seq,
		EventType:   eventType,
		EmittedAt:   emittedAt,
	}
}

// Contract: an events query without run_id or execution_id is a 400 with a
// structured error, not a full-table scan.
func TestQueryEvents_RequiresRunOrExecutionFilter(t *testing.T) {
	store := &eventsTestStorage{mockStatusStorage: &mockStatusStorage{}}

	rec := postEventsQuery(t, store, map[string]interface{}{"resource": "events"})

	require.Equal(t, http.StatusBadRequest, rec.Code)
	resp := decodeEnvelope(t, rec.Body)
	require.False(t, resp.OK)
	require.NotNil(t, resp.Error)
	assert.Equal(t, "missing_filter", resp.Error.Code)
}

// Contract: filtering by execution_id returns that execution's persisted
// events sorted by emitted_at ascending, with since/until bounds applied.
func TestQueryEvents_ByExecutionID(t *testing.T) {
	base := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	store := &eventsTestStorage{
		mockStatusStorage: &mockStatusStorage{},
		listWorkflowExecutionEventsFn: func(_ context.Context, executionID string, afterSeq *int64, limit int) ([]*types.WorkflowExecutionEvent, error) {
			require.Equal(t, "exec-1", executionID)
			require.Nil(t, afterSeq)
			require.Zero(t, limit)
			return []*types.WorkflowExecutionEvent{
				eventAt("exec-1", 1, "execution.started", base),
				eventAt("exec-1", 2, "execution.waiting", base.Add(2*time.Minute)),
				eventAt("exec-1", 3, "execution.completed", base.Add(10*time.Minute)),
			}, nil
		},
	}

	rec := postEventsQuery(t, store, map[string]interface{}{
		"resource": "events",
		"filters": map[string]interface{}{
			"execution_id": "exec-1",
			"since":        base.Add(time.Minute).Format(time.RFC3339),
			"until":        base.Add(5 * time.Minute).Format(time.RFC3339),
		},
	})

	require.Equal(t, http.StatusOK, rec.Code)
	resp := decodeEnvelope(t, rec.Body)
	require.True(t, resp.OK)

	data := resp.Data.(map[string]interface{})
	assert.Equal(t, "events", data["resource"])
	results := data["results"].([]interface{})
	require.Len(t, results, 1)
	first := results[0].(map[string]interface{})
	assert.Equal(t, "execution.waiting", first["event_type"])
	assert.Equal(t, float64(1), data["total"])
}

// Contract: filtering by run_id fans out over the run's executions and merges
// their events into one emitted_at-ascending list (ties broken by sequence).
func TestQueryEvents_ByRunIDMergesAndSorts(t *testing.T) {
	base := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)

	eventsByExecution := map[string][]*types.WorkflowExecutionEvent{
		"exec-a": {
			eventAt("exec-a", 1, "execution.started", base),
			eventAt("exec-a", 2, "execution.completed", base.Add(4*time.Minute)),
		},
		"exec-b": {
			eventAt("exec-b", 1, "execution.started", base.Add(2*time.Minute)),
			// Same timestamp as exec-a seq 2 but higher sequence — sorts after.
			eventAt("exec-b", 5, "execution.completed", base.Add(4*time.Minute)),
		},
	}

	store := &eventsTestStorage{
		mockStatusStorage: &mockStatusStorage{},
		queryExecutionRecordsFn: func(_ context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
			require.NotNil(t, filter.RunID)
			require.Equal(t, "run-1", *filter.RunID)
			return []*types.Execution{
				{ExecutionID: "exec-a", RunID: "run-1"},
				{ExecutionID: "exec-b", RunID: "run-1"},
			}, nil
		},
		listWorkflowExecutionEventsFn: func(_ context.Context, executionID string, _ *int64, _ int) ([]*types.WorkflowExecutionEvent, error) {
			return eventsByExecution[executionID], nil
		},
	}

	rec := postEventsQuery(t, store, map[string]interface{}{
		"resource": "events",
		"filters":  map[string]interface{}{"run_id": "run-1"},
	})

	require.Equal(t, http.StatusOK, rec.Code)
	resp := decodeEnvelope(t, rec.Body)
	require.True(t, resp.OK)

	data := resp.Data.(map[string]interface{})
	results := data["results"].([]interface{})
	require.Len(t, results, 4)
	assert.Equal(t, float64(4), data["total"])

	var order []string
	for _, raw := range results {
		evt := raw.(map[string]interface{})
		order = append(order, evt["execution_id"].(string))
	}
	assert.Equal(t, []string{"exec-a", "exec-b", "exec-a", "exec-b"}, order)
}

// Contract: limit and offset paginate after filtering; total reflects the
// pre-pagination count.
func TestQueryEvents_LimitAndOffset(t *testing.T) {
	base := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	all := []*types.WorkflowExecutionEvent{
		eventAt("exec-1", 1, "e1", base),
		eventAt("exec-1", 2, "e2", base.Add(time.Minute)),
		eventAt("exec-1", 3, "e3", base.Add(2*time.Minute)),
		eventAt("exec-1", 4, "e4", base.Add(3*time.Minute)),
	}
	store := &eventsTestStorage{
		mockStatusStorage: &mockStatusStorage{},
		listWorkflowExecutionEventsFn: func(_ context.Context, _ string, _ *int64, _ int) ([]*types.WorkflowExecutionEvent, error) {
			return all, nil
		},
	}

	rec := postEventsQuery(t, store, map[string]interface{}{
		"resource": "events",
		"filters":  map[string]interface{}{"execution_id": "exec-1"},
		"limit":    2,
		"offset":   1,
	})

	require.Equal(t, http.StatusOK, rec.Code)
	resp := decodeEnvelope(t, rec.Body)
	data := resp.Data.(map[string]interface{})
	results := data["results"].([]interface{})
	require.Len(t, results, 2)
	assert.Equal(t, "e2", results[0].(map[string]interface{})["event_type"])
	assert.Equal(t, "e3", results[1].(map[string]interface{})["event_type"])
	assert.Equal(t, float64(4), data["total"])

	// Offset past the end yields an empty page, not an error.
	rec = postEventsQuery(t, store, map[string]interface{}{
		"resource": "events",
		"filters":  map[string]interface{}{"execution_id": "exec-1"},
		"limit":    2,
		"offset":   10,
	})
	require.Equal(t, http.StatusOK, rec.Code)
	resp = decodeEnvelope(t, rec.Body)
	data = resp.Data.(map[string]interface{})
	assert.Empty(t, data["results"])
	assert.Equal(t, float64(4), data["total"])
}

// Contract: storage failures surface as 500 query_failed envelopes.
func TestQueryEvents_StorageErrors(t *testing.T) {
	t.Run("event listing fails", func(t *testing.T) {
		store := &eventsTestStorage{
			mockStatusStorage: &mockStatusStorage{},
			listWorkflowExecutionEventsFn: func(_ context.Context, _ string, _ *int64, _ int) ([]*types.WorkflowExecutionEvent, error) {
				return nil, errors.New("db down")
			},
		}
		rec := postEventsQuery(t, store, map[string]interface{}{
			"resource": "events",
			"filters":  map[string]interface{}{"execution_id": "exec-1"},
		})
		require.Equal(t, http.StatusInternalServerError, rec.Code)
		resp := decodeEnvelope(t, rec.Body)
		require.NotNil(t, resp.Error)
		assert.Equal(t, "query_failed", resp.Error.Code)
	})

	t.Run("run fan-out fails", func(t *testing.T) {
		store := &eventsTestStorage{
			mockStatusStorage: &mockStatusStorage{},
			queryExecutionRecordsFn: func(_ context.Context, _ types.ExecutionFilter) ([]*types.Execution, error) {
				return nil, errors.New("db down")
			},
		}
		rec := postEventsQuery(t, store, map[string]interface{}{
			"resource": "events",
			"filters":  map[string]interface{}{"run_id": "run-1"},
		})
		require.Equal(t, http.StatusInternalServerError, rec.Code)
		resp := decodeEnvelope(t, rec.Body)
		require.NotNil(t, resp.Error)
		assert.Equal(t, "query_failed", resp.Error.Code)
	})
}
