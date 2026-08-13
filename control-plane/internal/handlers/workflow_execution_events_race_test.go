package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// raceLosingStorage simulates losing an insert race on the events endpoint:
// the first CreateExecutionRecord call plants the concurrent writer's row
// (a "running" event that won the race) and then fails with a unique
// constraint error, exactly as SQLite/Postgres would for the losing INSERT.
type raceLosingStorage struct {
	*testExecutionStorage
	raced bool
}

func (s *raceLosingStorage) CreateExecutionRecord(ctx context.Context, execution *types.Execution) error {
	if !s.raced {
		s.raced = true
		winner := *execution
		winner.Status = string(types.ExecutionStatusRunning)
		winner.ResultPayload = nil
		winner.CompletedAt = nil
		winner.DurationMS = nil
		if err := s.testExecutionStorage.CreateExecutionRecord(ctx, &winner); err != nil {
			return err
		}
		return fmt.Errorf("insert execution: UNIQUE constraint failed: executions.execution_id")
	}
	return s.testExecutionStorage.CreateExecutionRecord(ctx, execution)
}

// TestWorkflowExecutionEventHandler_InsertRaceLoserMergesTerminalEvent pins
// the fix for the async-start/sync-terminal race: when a terminal event
// loses the insert race to its own "running" event, it must merge via the
// update path instead of being dropped — a dropped terminal event leaves the
// execution dangling in "running" forever.
func TestWorkflowExecutionEventHandler_InsertRaceLoserMergesTerminalEvent(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := &raceLosingStorage{testExecutionStorage: newTestExecutionStorage(&types.AgentNode{ID: "deep_research"})}
	handler := WorkflowExecutionEventHandler(storage)

	duration := int64(42)
	payload := WorkflowExecutionEventRequest{
		ExecutionID: "exec_raced",
		RunID:       "run_race",
		ReasonerID:  "span_stage",
		AgentNodeID: "deep_research",
		Status:      "succeeded",
		Result:      map[string]string{"result": "ok"},
		DurationMS:  &duration,
	}

	body, err := json.Marshal(payload)
	require.NoError(t, err)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/workflow/executions/events", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	c.Request = req

	handler(c)

	require.Equal(t, http.StatusOK, w.Code, "race loser must not surface a 500: %s", w.Body.String())

	exec, err := storage.GetExecutionRecord(context.Background(), "exec_raced")
	require.NoError(t, err)
	require.NotNil(t, exec)
	assert.Equal(t, string(types.ExecutionStatusSucceeded), exec.Status, "terminal event must win the merge, not be dropped")
	require.NotNil(t, exec.CompletedAt)
	assert.WithinDuration(t, time.Now(), *exec.CompletedAt, time.Second)
	require.NotNil(t, exec.ResultPayload)
	assert.Contains(t, string(exec.ResultPayload), "result")
}

// TestWorkflowExecutionEventHandler_TerminalOnlyNodeBackdatesStart pins the
// timeline shape of a node created from its terminal event alone (start
// event shed or lost): StartedAt must be backdated by the reported duration
// instead of collapsing to a zero-width bar at arrival time.
func TestWorkflowExecutionEventHandler_TerminalOnlyNodeBackdatesStart(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := newTestExecutionStorage(&types.AgentNode{ID: "deep_research"})
	handler := WorkflowExecutionEventHandler(storage)

	duration := int64(5000)
	payload := WorkflowExecutionEventRequest{
		ExecutionID: "exec_terminal_only",
		RunID:       "run_backdate",
		ReasonerID:  "span_stage",
		AgentNodeID: "deep_research",
		Status:      "succeeded",
		DurationMS:  &duration,
	}

	body, err := json.Marshal(payload)
	require.NoError(t, err)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/workflow/executions/events", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	c.Request = req

	handler(c)

	require.Equal(t, http.StatusOK, w.Code)

	exec, err := storage.GetExecutionRecord(context.Background(), "exec_terminal_only")
	require.NoError(t, err)
	require.NotNil(t, exec)
	require.NotNil(t, exec.CompletedAt)
	assert.Equal(t, 5*time.Second, exec.CompletedAt.Sub(exec.StartedAt),
		"StartedAt must be backdated by duration_ms, not stamped at arrival")
}

// vanishingRowStorage simulates a row deleted between the handler's read and
// its update (e.g. by cleanup): the first GetExecutionRecord reports a
// phantom row, and UpdateExecutionRecord mirrors the real LocalStorage
// semantics for a missing row — updater(nil), silent no-op.
type vanishingRowStorage struct {
	*testExecutionStorage
	phantomServed bool
}

func (s *vanishingRowStorage) GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error) {
	if !s.phantomServed {
		s.phantomServed = true
		return &types.Execution{ExecutionID: executionID, Status: string(types.ExecutionStatusRunning)}, nil
	}
	return s.testExecutionStorage.GetExecutionRecord(ctx, executionID)
}

func (s *vanishingRowStorage) UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error) {
	if exec, _ := s.testExecutionStorage.GetExecutionRecord(ctx, executionID); exec == nil {
		_, err := update(nil)
		return nil, err
	}
	return s.testExecutionStorage.UpdateExecutionRecord(ctx, executionID, update)
}

// TestWorkflowExecutionEventHandler_RowVanishedBetweenReadAndUpdate pins the
// re-create path: if the row disappears after the handler saw it, the event
// must still be persisted rather than silently no-oped.
func TestWorkflowExecutionEventHandler_RowVanishedBetweenReadAndUpdate(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := &vanishingRowStorage{testExecutionStorage: newTestExecutionStorage(&types.AgentNode{ID: "deep_research"})}
	handler := WorkflowExecutionEventHandler(storage)

	payload := WorkflowExecutionEventRequest{
		ExecutionID: "exec_vanished",
		RunID:       "run_vanish",
		ReasonerID:  "span_stage",
		AgentNodeID: "deep_research",
		Status:      "succeeded",
	}

	body, err := json.Marshal(payload)
	require.NoError(t, err)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/workflow/executions/events", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	c.Request = req

	handler(c)

	require.Equal(t, http.StatusOK, w.Code, "vanished row must be re-created: %s", w.Body.String())

	exec, err := storage.testExecutionStorage.GetExecutionRecord(context.Background(), "exec_vanished")
	require.NoError(t, err)
	require.NotNil(t, exec, "event must be persisted after the row vanished")
	assert.Equal(t, string(types.ExecutionStatusSucceeded), exec.Status)
}
