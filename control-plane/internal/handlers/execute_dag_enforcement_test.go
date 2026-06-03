package handlers

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

func seedExecutionStatus(t *testing.T, store *testExecutionStorage, executionID, runID, status string) {
	t.Helper()
	now := time.Now().UTC()
	require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID: executionID,
		RunID:       runID,
		Status:      status,
		StartedAt:   now,
		CreatedAt:   now,
		UpdatedAt:   now,
	}))
}

func testPreparedExecution(agent *types.AgentNode, executionID string) *preparedExecution {
	return &preparedExecution{
		exec: &types.Execution{
			ExecutionID: executionID,
			RunID:       "run-1",
		},
		requestBody: []byte(`{"input":{"foo":"bar"}}`),
		agent:       agent,
		target: &parsedTarget{
			NodeID:     agent.ID,
			TargetName: "reasoner-a",
		},
	}
}

func TestCallAgent_ReturnsEarlyWhenExecutionCancelled(t *testing.T) {
	var requestCount int32
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&requestCount, 1)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer agentServer.Close()

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   agentServer.URL,
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}

	store := newTestExecutionStorage(agent)
	seedExecutionStatus(t, store, "exec-cancelled", "run-1", types.ExecutionStatusCancelled)

	controller := newExecutionController(store, nil, nil, 90*time.Second, "")
	plan := testPreparedExecution(agent, "exec-cancelled")

	body, elapsed, asyncAccepted, err := controller.callAgent(context.Background(), plan)
	require.Error(t, err)
	require.Contains(t, err.Error(), "execution cancelled")
	require.Nil(t, body)
	require.Equal(t, time.Duration(0), elapsed)
	require.False(t, asyncAccepted)
	require.Equal(t, int32(0), atomic.LoadInt32(&requestCount))
}

func TestWaitForResume_UnblocksOnExecutionResumedEvent(t *testing.T) {
	store := newTestExecutionStorage(nil)
	controller := newExecutionController(store, nil, nil, 90*time.Second, "")
	seedExecutionStatus(t, store, "exec-paused", "run-1", types.ExecutionStatusPaused)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	done := make(chan error, 1)
	go func() {
		done <- controller.waitForResume(ctx, "exec-paused")
	}()

	time.Sleep(100 * time.Millisecond)
	_, updateErr := store.UpdateExecutionRecord(context.Background(), "exec-paused", func(current *types.Execution) (*types.Execution, error) {
		if current == nil {
			return nil, nil
		}
		current.Status = types.ExecutionStatusRunning
		current.UpdatedAt = time.Now().UTC()
		return current, nil
	})
	require.NoError(t, updateErr)
	store.GetExecutionEventBus().Publish(events.ExecutionEvent{
		Type:        events.ExecutionResumed,
		ExecutionID: "exec-paused",
		WorkflowID:  "run-1",
		Status:      types.ExecutionStatusRunning,
		Timestamp:   time.Now(),
	})

	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(time.Second):
		t.Fatal("waitForResume did not unblock on ExecutionResumed")
	}
}

func TestWaitForResume_ReturnsErrorOnExecutionCancelledEvent(t *testing.T) {
	store := newTestExecutionStorage(nil)
	controller := newExecutionController(store, nil, nil, 90*time.Second, "")
	seedExecutionStatus(t, store, "exec-paused-cancel", "run-1", types.ExecutionStatusPaused)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	done := make(chan error, 1)
	go func() {
		done <- controller.waitForResume(ctx, "exec-paused-cancel")
	}()

	time.Sleep(100 * time.Millisecond)
	_, updateErr := store.UpdateExecutionRecord(context.Background(), "exec-paused-cancel", func(current *types.Execution) (*types.Execution, error) {
		if current == nil {
			return nil, nil
		}
		current.Status = types.ExecutionStatusCancelled
		current.UpdatedAt = time.Now().UTC()
		return current, nil
	})
	require.NoError(t, updateErr)
	store.GetExecutionEventBus().Publish(events.ExecutionEvent{
		Type:        events.ExecutionCancelledEvent,
		ExecutionID: "exec-paused-cancel",
		WorkflowID:  "run-1",
		Status:      types.ExecutionStatusCancelled,
		Timestamp:   time.Now(),
	})

	select {
	case err := <-done:
		require.Error(t, err)
		require.Contains(t, err.Error(), "execution cancelled")
	case <-time.After(time.Second):
		t.Fatal("waitForResume did not unblock on ExecutionCancelledEvent")
	}
}

func TestCallAgent_WaitsWhenPausedThenContinuesAfterResume(t *testing.T) {
	var requestCount int32
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&requestCount, 1)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer agentServer.Close()

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   agentServer.URL,
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}

	store := newTestExecutionStorage(agent)
	seedExecutionStatus(t, store, "exec-paused", "run-1", types.ExecutionStatusPaused)

	controller := newExecutionController(store, nil, nil, 90*time.Second, "")
	plan := testPreparedExecution(agent, "exec-paused")

	type callResult struct {
		body          []byte
		elapsed       time.Duration
		asyncAccepted bool
		err           error
	}
	resultCh := make(chan callResult, 1)

	go func() {
		body, elapsed, asyncAccepted, err := controller.callAgent(context.Background(), plan)
		resultCh <- callResult{body: body, elapsed: elapsed, asyncAccepted: asyncAccepted, err: err}
	}()

	time.Sleep(100 * time.Millisecond)
	require.Equal(t, int32(0), atomic.LoadInt32(&requestCount))

	_, updateErr := store.UpdateExecutionRecord(context.Background(), "exec-paused", func(current *types.Execution) (*types.Execution, error) {
		if current == nil {
			return nil, nil
		}
		current.Status = types.ExecutionStatusRunning
		current.UpdatedAt = time.Now().UTC()
		return current, nil
	})
	require.NoError(t, updateErr)
	store.GetExecutionEventBus().Publish(events.ExecutionEvent{
		Type:        events.ExecutionResumed,
		ExecutionID: "exec-paused",
		WorkflowID:  "run-1",
		Status:      types.ExecutionStatusRunning,
		Timestamp:   time.Now(),
	})

	select {
	case res := <-resultCh:
		require.NoError(t, res.err)
		require.False(t, res.asyncAccepted)
		require.NotNil(t, res.body)
		require.Greater(t, res.elapsed, time.Duration(0))
	case <-time.After(2 * time.Second):
		t.Fatal("callAgent did not resume after ExecutionResumed")
	}

	require.Equal(t, int32(1), atomic.LoadInt32(&requestCount))
}

func TestAsyncExecutionJobProcess_SkipsWhenExecutionCancelled(t *testing.T) {
	var requestCount int32
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&requestCount, 1)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer agentServer.Close()

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   agentServer.URL,
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}

	store := newTestExecutionStorage(agent)
	seedExecutionStatus(t, store, "exec-job-cancelled", "run-1", types.ExecutionStatusCancelled)

	controller := newExecutionController(store, nil, nil, 90*time.Second, "")
	job := asyncExecutionJob{
		controller: controller,
		plan:       *testPreparedExecution(agent, "exec-job-cancelled"),
	}

	job.process()
	require.Equal(t, int32(0), atomic.LoadInt32(&requestCount))
}
