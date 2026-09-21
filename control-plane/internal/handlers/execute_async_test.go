package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func useAsyncPoolForTest(t *testing.T, pool *asyncWorkerPool) {
	t.Helper()
	oldPool := asyncPool
	asyncPool = pool
	asyncPoolOnce = sync.Once{}
	asyncPoolOnce.Do(func() {})
	t.Cleanup(func() {
		asyncPool = oldPool
		asyncPoolOnce = sync.Once{}
		if oldPool != nil {
			asyncPoolOnce.Do(func() {})
		}
	})
}

func TestAsyncAdmissionPersistsQueuedUntilWorkerDispatch(t *testing.T) {
	gin.SetMode(gin.TestMode)
	pool := newAsyncWorkerPool(0, 1)
	useAsyncPoolForTest(t, pool)
	store := newTestExecutionStorage(testRestartAgent("http://agent.example"))
	eventCh := store.GetExecutionEventBus().Subscribe("queued-admission")
	t.Cleanup(func() { store.GetExecutionEventBus().Unsubscribe("queued-admission") })

	router := gin.New()
	router.POST("/api/v1/execute/async/:target", ExecuteAsyncHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
	router.GET("/api/v1/executions/:execution_id", GetExecutionStatusHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/async/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	require.Equal(t, http.StatusAccepted, resp.Code, resp.Body.String())

	var accepted AsyncExecuteResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &accepted))
	require.Equal(t, string(types.ExecutionStatusQueued), accepted.Status)

	statusResp := httptest.NewRecorder()
	router.ServeHTTP(statusResp, httptest.NewRequest(http.MethodGet, "/api/v1/executions/"+accepted.ExecutionID, nil))
	require.Equal(t, http.StatusOK, statusResp.Code, statusResp.Body.String())
	var status ExecutionStatusResponse
	require.NoError(t, json.Unmarshal(statusResp.Body.Bytes(), &status))
	require.Equal(t, string(types.ExecutionStatusQueued), status.Status)

	workflow, err := store.GetWorkflowExecution(context.Background(), accepted.ExecutionID)
	require.NoError(t, err)
	require.NotNil(t, workflow)
	require.Equal(t, string(types.ExecutionStatusQueued), workflow.Status)
	select {
	case event := <-eventCh:
		require.Equal(t, events.ExecutionUpdated, event.Type)
		require.Equal(t, string(types.ExecutionStatusQueued), event.Status)
		data, ok := event.Data.(map[string]interface{})
		require.True(t, ok)
		require.Equal(t, "async", data["execution_mode"])
		require.Equal(t, "reasoner", data["target_type"])
	case <-time.After(time.Second):
		t.Fatal("queued admission event was not emitted")
	}

	stopCtx, cancel := context.WithCancel(context.Background())
	cancel()
	pool.Stop(stopCtx)
}

func TestAsyncQueueKeepsSecondExecutionQueuedUntilWorkerIsFree(t *testing.T) {
	gin.SetMode(gin.TestMode)
	pool := newAsyncWorkerPool(1, 2)
	useAsyncPoolForTest(t, pool)

	firstDispatched := make(chan struct{})
	secondDispatched := make(chan struct{})
	releaseFirst := make(chan struct{})
	releaseSecond := make(chan struct{})
	var releaseFirstOnce sync.Once
	var releaseSecondOnce sync.Once
	releaseFirstJob := func() { releaseFirstOnce.Do(func() { close(releaseFirst) }) }
	releaseSecondJob := func() { releaseSecondOnce.Do(func() { close(releaseSecond) }) }
	defer releaseFirstJob()
	defer releaseSecondJob()

	var calls atomic.Int32
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		switch calls.Add(1) {
		case 1:
			close(firstDispatched)
			<-releaseFirst
		case 2:
			close(secondDispatched)
			<-releaseSecond
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer agentServer.Close()

	store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	router := gin.New()
	router.POST("/api/v1/execute/async/:target", ExecuteAsyncHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
	router.GET("/api/v1/executions/:execution_id", GetExecutionStatusHandler(store))

	submit := func() AsyncExecuteResponse {
		req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/async/node-1.reasoner-a", strings.NewReader(`{"input":{}}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusAccepted, resp.Code, resp.Body.String())
		var accepted AsyncExecuteResponse
		require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &accepted))
		return accepted
	}
	readStatus := func(executionID string) string {
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/api/v1/executions/"+executionID, nil))
		require.Equal(t, http.StatusOK, resp.Code, resp.Body.String())
		var status ExecutionStatusResponse
		require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &status))
		return status.Status
	}

	first := submit()
	select {
	case <-firstDispatched:
	case <-time.After(time.Second):
		t.Fatal("first execution was not dispatched")
	}

	second := submit()
	require.Equal(t, string(types.ExecutionStatusQueued), readStatus(second.ExecutionID))
	secondWorkflow, err := store.GetWorkflowExecution(context.Background(), second.ExecutionID)
	require.NoError(t, err)
	require.NotNil(t, secondWorkflow)
	require.Equal(t, string(types.ExecutionStatusQueued), secondWorkflow.Status)
	select {
	case <-secondDispatched:
		t.Fatal("second execution dispatched while the only worker was busy")
	default:
	}

	releaseFirstJob()
	select {
	case <-secondDispatched:
	case <-time.After(2 * time.Second):
		t.Fatal("second execution was not dispatched after the worker became free")
	}
	require.Equal(t, string(types.ExecutionStatusRunning), readStatus(second.ExecutionID))
	secondWorkflow, err = store.GetWorkflowExecution(context.Background(), second.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, string(types.ExecutionStatusRunning), secondWorkflow.Status)

	releaseSecondJob()
	require.Eventually(t, func() bool {
		firstRecord, firstErr := store.GetExecutionRecord(context.Background(), first.ExecutionID)
		secondRecord, secondErr := store.GetExecutionRecord(context.Background(), second.ExecutionID)
		return firstErr == nil && secondErr == nil && firstRecord != nil && secondRecord != nil &&
			firstRecord.Status == types.ExecutionStatusSucceeded && secondRecord.Status == types.ExecutionStatusSucceeded
	}, 2*time.Second, 10*time.Millisecond)
	stopCtx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	pool.Stop(stopCtx)
}

func TestAsyncDispatchTransitionsQueuedRowsToRunningThenSucceeded(t *testing.T) {
	gin.SetMode(gin.TestMode)
	pool := newAsyncWorkerPool(1, 1)
	useAsyncPoolForTest(t, pool)
	dispatched := make(chan struct{})
	releaseAgent := make(chan struct{})
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(releaseAgent) }) }
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		close(dispatched)
		<-releaseAgent
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer agentServer.Close()
	defer release()

	store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	eventCh := store.GetExecutionEventBus().Subscribe("dispatch-transition")
	t.Cleanup(func() { store.GetExecutionEventBus().Unsubscribe("dispatch-transition") })
	router := gin.New()
	router.POST("/api/v1/execute/async/:target", ExecuteAsyncHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/async/node-1.reasoner-a", strings.NewReader(`{"input":{}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	require.Equal(t, http.StatusAccepted, resp.Code, resp.Body.String())
	var accepted AsyncExecuteResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &accepted))

	select {
	case <-dispatched:
	case <-time.After(time.Second):
		t.Fatal("agent was not called")
	}
	record, err := store.GetExecutionRecord(context.Background(), accepted.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusRunning, record.Status)
	workflow, err := store.GetWorkflowExecution(context.Background(), accepted.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, string(types.ExecutionStatusRunning), workflow.Status)
	queuedEventSeen := false
	startedEventSeen := false
	deadline := time.After(time.Second)
	for !queuedEventSeen || !startedEventSeen {
		select {
		case event := <-eventCh:
			switch event.Type {
			case events.ExecutionUpdated:
				if event.Status == string(types.ExecutionStatusQueued) {
					queuedEventSeen = true
				}
			case events.ExecutionStarted:
				require.Equal(t, string(types.ExecutionStatusRunning), event.Status)
				startedEventSeen = true
			}
		case <-deadline:
			t.Fatalf("expected queued admission and dispatch-time started events; queued=%t started=%t", queuedEventSeen, startedEventSeen)
		}
	}

	release()
	require.Eventually(t, func() bool {
		record, recordErr := store.GetExecutionRecord(context.Background(), accepted.ExecutionID)
		workflow, workflowErr := store.GetWorkflowExecution(context.Background(), accepted.ExecutionID)
		return recordErr == nil && workflowErr == nil && record != nil && workflow != nil &&
			record.Status == types.ExecutionStatusSucceeded && workflow.Status == string(types.ExecutionStatusSucceeded)
	}, 2*time.Second, 10*time.Millisecond)
	stopCtx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	pool.Stop(stopCtx)
}

func TestSyncAdmissionRemainsRunningWhileAgentCallIsInFlight(t *testing.T) {
	gin.SetMode(gin.TestMode)
	dispatched := make(chan struct{})
	releaseAgent := make(chan struct{})
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(releaseAgent) }) }
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		close(dispatched)
		<-releaseAgent
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer agentServer.Close()
	defer release()

	store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	router := gin.New()
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.reasoner-a", strings.NewReader(`{"input":{}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		router.ServeHTTP(resp, req)
		close(done)
	}()

	select {
	case <-dispatched:
	case <-time.After(time.Second):
		t.Fatal("agent was not called")
	}
	records, err := store.QueryExecutionRecords(context.Background(), types.ExecutionFilter{})
	require.NoError(t, err)
	require.Len(t, records, 1)
	require.Equal(t, types.ExecutionStatusRunning, records[0].Status)
	workflow, err := store.GetWorkflowExecution(context.Background(), records[0].ExecutionID)
	require.NoError(t, err)
	require.Equal(t, string(types.ExecutionStatusRunning), workflow.Status)

	release()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("sync execution did not complete")
	}
	require.Equal(t, http.StatusOK, resp.Code, resp.Body.String())
}

func TestCancelledQueuedExecutionIsSkippedByWorker(t *testing.T) {
	gin.SetMode(gin.TestMode)
	var agentCalls atomic.Int32
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		agentCalls.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer agentServer.Close()

	store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	now := time.Now().UTC()
	exec := &types.Execution{
		ExecutionID: "cancel-queued", RunID: "cancel-queued", NodeID: "node-1", AgentNodeID: "node-1",
		ReasonerID: "reasoner-a", Status: types.ExecutionStatusQueued, CreatedAt: now, StartedAt: now, UpdatedAt: now,
	}
	require.NoError(t, store.CreateExecutionRecord(context.Background(), exec))
	require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID: exec.ExecutionID, WorkflowID: exec.RunID, RunID: &exec.RunID,
		AgentNodeID: "node-1", ReasonerID: "reasoner-a", Status: string(types.ExecutionStatusQueued),
		CreatedAt: now, StartedAt: now, UpdatedAt: now,
	}))

	router := gin.New()
	router.POST("/api/v1/executions/:execution_id/cancel", CancelExecutionHandler(store))
	cancelResp := httptest.NewRecorder()
	router.ServeHTTP(cancelResp, httptest.NewRequest(http.MethodPost, "/api/v1/executions/cancel-queued/cancel", nil))
	require.Equal(t, http.StatusOK, cancelResp.Code, cancelResp.Body.String())

	target, err := parseTarget("node-1.reasoner-a")
	require.NoError(t, err)
	job := asyncExecutionJob{
		controller: newExecutionController(store, nil, nil, time.Second, ""),
		plan:       preparedExecution{exec: exec, target: target, agent: store.agent, requestBody: []byte(`{}`)},
	}
	job.processWithContext(context.Background())
	require.Zero(t, agentCalls.Load())
	stored, err := store.GetExecutionRecord(context.Background(), exec.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusCancelled, stored.Status)
	workflow, err := store.GetWorkflowExecution(context.Background(), exec.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, string(types.ExecutionStatusCancelled), workflow.Status)
}

type failQueuedTransitionStorage struct {
	*testExecutionStorage
	failNext   atomic.Bool
	mutatorRan atomic.Bool
}

func (s *failQueuedTransitionStorage) UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error) {
	if !s.failNext.CompareAndSwap(true, false) {
		return s.testExecutionStorage.UpdateExecutionRecord(ctx, executionID, update)
	}
	current, err := s.testExecutionStorage.GetExecutionRecord(ctx, executionID)
	if err != nil {
		return nil, err
	}
	if _, err := update(current); err != nil {
		return nil, err
	}
	s.mutatorRan.Store(true)
	return nil, errors.New("forced queued-to-running persistence failure")
}

type dispatchRaceStorage struct {
	*testExecutionStorage
	statusOnFirstUpdate string
	firstUpdateDone     chan struct{}
	updateCalls         atomic.Int32
}

func (s *dispatchRaceStorage) UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error) {
	if s.updateCalls.Add(1) != 1 {
		return s.testExecutionStorage.UpdateExecutionRecord(ctx, executionID, update)
	}
	if s.firstUpdateDone != nil {
		defer close(s.firstUpdateDone)
	}

	_, err := s.testExecutionStorage.UpdateExecutionRecord(ctx, executionID, func(current *types.Execution) (*types.Execution, error) {
		current.Status = s.statusOnFirstUpdate
		current.UpdatedAt = time.Now().UTC()
		return current, nil
	})
	if err != nil {
		return nil, err
	}
	if err := s.testExecutionStorage.UpdateWorkflowExecution(ctx, executionID, func(current *types.WorkflowExecution) (*types.WorkflowExecution, error) {
		current.Status = s.statusOnFirstUpdate
		current.UpdatedAt = time.Now().UTC()
		return current, nil
	}); err != nil {
		return nil, err
	}

	return s.testExecutionStorage.UpdateExecutionRecord(ctx, executionID, update)
}

func queuedAsyncJobForTest(t *testing.T, store ExecutionStore, baseStore *testExecutionStorage, executionID string) asyncExecutionJob {
	t.Helper()

	now := time.Now().UTC()
	exec := &types.Execution{
		ExecutionID: executionID, RunID: executionID, NodeID: "node-1", AgentNodeID: "node-1",
		ReasonerID: "reasoner-a", Status: types.ExecutionStatusQueued, CreatedAt: now, StartedAt: now, UpdatedAt: now,
	}
	require.NoError(t, baseStore.CreateExecutionRecord(context.Background(), exec))
	require.NoError(t, baseStore.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID: exec.ExecutionID, WorkflowID: exec.RunID, RunID: &exec.RunID,
		AgentNodeID: "node-1", ReasonerID: "reasoner-a", Status: string(types.ExecutionStatusQueued),
		CreatedAt: now, StartedAt: now, UpdatedAt: now,
	}))
	target, err := parseTarget("node-1.reasoner-a")
	require.NoError(t, err)
	return asyncExecutionJob{
		controller: newExecutionController(store, nil, nil, time.Second, ""),
		plan: preparedExecution{
			exec: exec, target: target, agent: baseStore.agent, requestBody: []byte(`{}`), executionMode: "async",
		},
	}
}

func TestQueuedTransitionFailureDoesNotPublishStartedOrChangePlanStatus(t *testing.T) {
	gin.SetMode(gin.TestMode)
	var agentCalls atomic.Int32
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		agentCalls.Add(1)
		w.WriteHeader(http.StatusAccepted)
	}))
	defer agentServer.Close()

	baseStore := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	store := &failQueuedTransitionStorage{testExecutionStorage: baseStore}
	store.failNext.Store(true)
	eventCh := store.GetExecutionEventBus().Subscribe("failed-dispatch-transition")
	t.Cleanup(func() { store.GetExecutionEventBus().Unsubscribe("failed-dispatch-transition") })

	now := time.Now().UTC()
	exec := &types.Execution{
		ExecutionID: "queued-transition-fails", RunID: "queued-transition-fails", NodeID: "node-1", AgentNodeID: "node-1",
		ReasonerID: "reasoner-a", Status: types.ExecutionStatusQueued, CreatedAt: now, StartedAt: now, UpdatedAt: now,
	}
	require.NoError(t, baseStore.CreateExecutionRecord(context.Background(), exec))
	require.NoError(t, baseStore.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID: exec.ExecutionID, WorkflowID: exec.RunID, RunID: &exec.RunID,
		AgentNodeID: "node-1", ReasonerID: "reasoner-a", Status: string(types.ExecutionStatusQueued),
		CreatedAt: now, StartedAt: now, UpdatedAt: now,
	}))
	target, err := parseTarget("node-1.reasoner-a")
	require.NoError(t, err)
	job := asyncExecutionJob{
		controller: newExecutionController(store, nil, nil, time.Second, ""),
		plan: preparedExecution{
			exec: exec, target: target, agent: baseStore.agent, requestBody: []byte(`{}`), executionMode: "async",
		},
	}

	job.processWithContext(context.Background())

	require.True(t, store.mutatorRan.Load(), "the failing store must invoke the update mutator")
	require.EqualValues(t, 1, agentCalls.Load(), "admitted work still dispatches after a transient storage failure")
	require.Equal(t, types.ExecutionStatusQueued, exec.Status, "the in-memory plan must not claim an unpersisted transition")
	stored, err := baseStore.GetExecutionRecord(context.Background(), exec.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusQueued, stored.Status)
	select {
	case event := <-eventCh:
		require.NotEqual(t, events.ExecutionStarted, event.Type)
	default:
	}
}

func TestTimedOutQueuedExecutionIsSkippedByWorker(t *testing.T) {
	gin.SetMode(gin.TestMode)
	var agentCalls atomic.Int32
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		agentCalls.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer agentServer.Close()

	store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	now := time.Now().UTC()
	exec := &types.Execution{
		ExecutionID: "queued-reaped", RunID: "queued-reaped", NodeID: "node-1", AgentNodeID: "node-1",
		ReasonerID: "reasoner-a", Status: types.ExecutionStatusQueued, CreatedAt: now, StartedAt: now, UpdatedAt: now,
	}
	require.NoError(t, store.CreateExecutionRecord(context.Background(), exec))
	require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID: exec.ExecutionID, WorkflowID: exec.RunID, RunID: &exec.RunID,
		AgentNodeID: "node-1", ReasonerID: "reasoner-a", Status: string(types.ExecutionStatusQueued),
		CreatedAt: now, StartedAt: now, UpdatedAt: now,
	}))
	_, err := store.UpdateExecutionRecord(context.Background(), exec.ExecutionID, func(current *types.Execution) (*types.Execution, error) {
		current.Status = types.ExecutionStatusTimeout
		return current, nil
	})
	require.NoError(t, err)
	require.NoError(t, store.UpdateWorkflowExecution(context.Background(), exec.ExecutionID, func(current *types.WorkflowExecution) (*types.WorkflowExecution, error) {
		current.Status = string(types.ExecutionStatusTimeout)
		return current, nil
	}))

	target, err := parseTarget("node-1.reasoner-a")
	require.NoError(t, err)
	job := asyncExecutionJob{
		controller: newExecutionController(store, nil, nil, time.Second, ""),
		plan:       preparedExecution{exec: exec, target: target, agent: store.agent, requestBody: []byte(`{}`)},
	}
	job.processWithContext(context.Background())

	require.Zero(t, agentCalls.Load())
	stored, err := store.GetExecutionRecord(context.Background(), exec.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusTimeout, stored.Status)
}

func TestQueuedExecutionThatBecomesTerminalDuringDispatchIsSkipped(t *testing.T) {
	var agentCalls atomic.Int32
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		agentCalls.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer agentServer.Close()

	baseStore := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	store := &dispatchRaceStorage{
		testExecutionStorage: baseStore,
		statusOnFirstUpdate:  types.ExecutionStatusTimeout,
	}
	job := queuedAsyncJobForTest(t, store, baseStore, "terminal-during-dispatch")
	eventCh := store.GetExecutionEventBus().Subscribe("terminal-during-dispatch")
	t.Cleanup(func() { store.GetExecutionEventBus().Unsubscribe("terminal-during-dispatch") })

	job.processWithContext(context.Background())

	require.EqualValues(t, 1, store.updateCalls.Load(), "the worker should stop after observing the terminal update")
	require.Zero(t, agentCalls.Load())
	stored, err := baseStore.GetExecutionRecord(context.Background(), job.plan.exec.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusTimeout, stored.Status)
	workflow, err := baseStore.GetWorkflowExecution(context.Background(), job.plan.exec.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, string(types.ExecutionStatusTimeout), workflow.Status)
	select {
	case event := <-eventCh:
		t.Fatalf("terminal execution emitted an unexpected event before dispatch: %s", event.Type)
	default:
	}
}

func TestQueuedExecutionPausedDuringDispatchStopsWhenResumeWaitFails(t *testing.T) {
	var agentCalls atomic.Int32
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		agentCalls.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer agentServer.Close()

	baseStore := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	store := &dispatchRaceStorage{
		testExecutionStorage: baseStore,
		statusOnFirstUpdate:  types.ExecutionStatusPaused,
	}
	job := queuedAsyncJobForTest(t, store, baseStore, "paused-during-dispatch-cancelled-wait")
	workerCtx, cancel := context.WithCancel(context.Background())
	cancel()

	job.processWithContext(workerCtx)

	require.EqualValues(t, 1, store.updateCalls.Load(), "a failed resume wait must not retry dispatch")
	require.Zero(t, agentCalls.Load())
	stored, err := baseStore.GetExecutionRecord(context.Background(), job.plan.exec.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusPaused, stored.Status)
	workflow, err := baseStore.GetWorkflowExecution(context.Background(), job.plan.exec.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, string(types.ExecutionStatusPaused), workflow.Status)
}

func TestQueuedExecutionPausedDuringDispatchResumesAndDispatchesOnce(t *testing.T) {
	var agentCalls atomic.Int32
	var updateCallsAtAgent atomic.Int32
	var store *dispatchRaceStorage
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		agentCalls.Add(1)
		updateCallsAtAgent.Store(store.updateCalls.Load())
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer agentServer.Close()

	baseStore := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	firstUpdateDone := make(chan struct{})
	store = &dispatchRaceStorage{
		testExecutionStorage: baseStore,
		statusOnFirstUpdate:  types.ExecutionStatusPaused,
		firstUpdateDone:      firstUpdateDone,
	}
	job := queuedAsyncJobForTest(t, store, baseStore, "paused-during-dispatch-resumed")
	processDone := make(chan struct{})
	go func() {
		job.processWithContext(context.Background())
		close(processDone)
	}()

	select {
	case <-firstUpdateDone:
	case <-time.After(time.Second):
		t.Fatal("worker did not observe the dispatch-time pause")
	}
	require.Eventually(t, func() bool {
		return baseStore.GetExecutionEventBus().GetSubscriberCount() == 1
	}, time.Second, time.Millisecond, "worker did not begin waiting for resume")

	_, err := baseStore.UpdateExecutionRecord(context.Background(), job.plan.exec.ExecutionID, func(current *types.Execution) (*types.Execution, error) {
		current.Status = types.ExecutionStatusRunning
		current.UpdatedAt = time.Now().UTC()
		return current, nil
	})
	require.NoError(t, err)
	require.NoError(t, baseStore.UpdateWorkflowExecution(context.Background(), job.plan.exec.ExecutionID, func(current *types.WorkflowExecution) (*types.WorkflowExecution, error) {
		current.Status = string(types.ExecutionStatusRunning)
		current.UpdatedAt = time.Now().UTC()
		return current, nil
	}))
	baseStore.GetExecutionEventBus().Publish(events.ExecutionEvent{
		Type: events.ExecutionResumed, ExecutionID: job.plan.exec.ExecutionID,
		WorkflowID: job.plan.exec.RunID, Status: types.ExecutionStatusRunning, Timestamp: time.Now().UTC(),
	})

	select {
	case <-processDone:
	case <-time.After(2 * time.Second):
		t.Fatal("worker did not dispatch after the execution resumed")
	}
	require.EqualValues(t, 1, agentCalls.Load())
	require.EqualValues(t, 2, updateCallsAtAgent.Load(), "the dispatch loop should retry exactly once before calling the agent")
	require.Eventually(t, func() bool {
		stored, recordErr := baseStore.GetExecutionRecord(context.Background(), job.plan.exec.ExecutionID)
		return recordErr == nil && stored != nil && stored.Status == types.ExecutionStatusSucceeded
	}, 2*time.Second, time.Millisecond, "the resumed execution should persist its successful completion")
}

func TestExecuteAsyncHandler_PoolStoppedTerminatesPersistedRow(t *testing.T) {
	gin.SetMode(gin.TestMode)
	pool := newAsyncWorkerPool(0, 4)
	useAsyncPoolForTest(t, pool)
	oldLimiter := concurrencyLimiter
	concurrencyLimiter = &AgentConcurrencyLimiter{maxPerAgent: 2}
	defer func() { concurrencyLimiter = oldLimiter }()

	agent := testRestartAgent("http://agent.example")
	baseStore := newTestExecutionStorage(agent)
	store := &stopPoolOnCreateStorage{testExecutionStorage: baseStore, pool: pool}
	router := gin.New()
	router.POST("/api/v1/execute/async/:target", ExecuteAsyncHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/async/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusServiceUnavailable, resp.Code)
	require.Contains(t, resp.Body.String(), "async execution queue stopped")
	require.Equal(t, "1", resp.Header().Get("Retry-After"))
	var body map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
	require.Equal(t, string(ErrorCategoryControlPlaneShutdown), body["error_category"])
	require.Equal(t, float64(1), body["retry_after"])
	records, err := baseStore.QueryExecutionRecords(context.Background(), types.ExecutionFilter{})
	require.NoError(t, err)
	require.Len(t, records, 1)
	require.Equal(t, types.ExecutionStatusFailed, records[0].Status)
	require.NotNil(t, records[0].StatusReason)
	require.Equal(t, "control_plane_shutdown", *records[0].StatusReason)
	workflows, err := baseStore.QueryWorkflowExecutions(context.Background(), types.WorkflowExecutionFilters{})
	require.NoError(t, err)
	require.Len(t, workflows, 1)
	require.NotNil(t, workflows[0].StatusReason)
	require.Equal(t, "control_plane_shutdown", *workflows[0].StatusReason)
	require.Equal(t, string(records[0].Status), workflows[0].Status)
	require.Zero(t, concurrencyLimiter.GetRunningCount("node-1"))
}

func TestExecuteAsyncHandler_AlreadyStoppedPoolReturnsShutdownWithoutPersistence(t *testing.T) {
	pool := newAsyncWorkerPool(0, 2)
	pool.mu.Lock()
	pool.stopped = true
	pool.mu.Unlock()
	useAsyncPoolForTest(t, pool)
	store := newTestExecutionStorage(testRestartAgent("http://agent.example"))
	router := gin.New()
	router.POST("/api/v1/execute/async/:target", ExecuteAsyncHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/async/node-1.reasoner-a", strings.NewReader(`{"input":{}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusServiceUnavailable, resp.Code)
	require.Equal(t, "1", resp.Header().Get("Retry-After"))
	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
	require.Equal(t, string(ErrorCategoryControlPlaneShutdown), body["error_category"])
	records, err := store.QueryExecutionRecords(context.Background(), types.ExecutionFilter{})
	require.NoError(t, err)
	require.Empty(t, records)
}

type stopPoolOnCreateStorage struct {
	*testExecutionStorage
	pool   *asyncWorkerPool
	cancel context.CancelFunc
}

type cancelBeforeShutdownUpdateStore struct {
	*testExecutionStorage
	once          sync.Once
	interleaveErr error
}

func (s *cancelBeforeShutdownUpdateStore) UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error) {
	s.once.Do(func() {
		reason := "cancelled_by_user"
		_, s.interleaveErr = s.testExecutionStorage.UpdateExecutionRecord(ctx, executionID, func(current *types.Execution) (*types.Execution, error) {
			current.Status = types.ExecutionStatusCancelled
			current.StatusReason = &reason
			return current, nil
		})
		if s.interleaveErr != nil {
			return
		}
		s.interleaveErr = s.testExecutionStorage.UpdateWorkflowExecution(ctx, executionID, func(current *types.WorkflowExecution) (*types.WorkflowExecution, error) {
			current.Status = string(types.ExecutionStatusCancelled)
			current.StatusReason = &reason
			return current, nil
		})
	})
	if s.interleaveErr != nil {
		return nil, s.interleaveErr
	}
	return s.testExecutionStorage.UpdateExecutionRecord(ctx, executionID, update)
}

func (s *stopPoolOnCreateStorage) CreateExecutionRecord(ctx context.Context, execution *types.Execution) error {
	s.pool.mu.Lock()
	s.pool.stopped = true
	s.pool.mu.Unlock()
	err := s.testExecutionStorage.CreateExecutionRecord(ctx, execution)
	if s.cancel != nil {
		s.cancel()
	}
	return err
}

func TestAsyncShutdownTerminalizationPreservesCancellationInterleaving(t *testing.T) {
	oldLimiter := concurrencyLimiter
	concurrencyLimiter = &AgentConcurrencyLimiter{maxPerAgent: 2}
	t.Cleanup(func() { concurrencyLimiter = oldLimiter })
	require.NoError(t, concurrencyLimiter.Acquire("node-1"))

	base := newTestExecutionStorage(testRestartAgent("http://agent.example"))
	now := time.Now().UTC()
	exec := &types.Execution{
		ExecutionID: "exec-cancel-race", RunID: "run-cancel-race", AgentNodeID: "node-1", NodeID: "node-1",
		ReasonerID: "reasoner-a", Status: types.ExecutionStatusRunning,
		CreatedAt: now, StartedAt: now, UpdatedAt: now,
	}
	require.NoError(t, base.CreateExecutionRecord(context.Background(), exec))
	require.NoError(t, base.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID: exec.ExecutionID, WorkflowID: exec.RunID, RunID: &exec.RunID,
		AgentNodeID: "node-1", ReasonerID: "reasoner-a", Status: string(types.ExecutionStatusRunning),
		CreatedAt: now, StartedAt: now, UpdatedAt: now,
	}))
	store := &cancelBeforeShutdownUpdateStore{testExecutionStorage: base}
	target, err := parseTarget("node-1.reasoner-a")
	require.NoError(t, err)
	job := asyncExecutionJob{
		controller: newExecutionController(store, nil, nil, time.Second, ""),
		plan: preparedExecution{
			exec: exec, target: target, slotHeld: true,
		},
	}

	job.terminateForControlPlaneShutdown(newControlPlaneShutdownError("control plane stopped"))

	stored, err := base.GetExecutionRecord(context.Background(), exec.ExecutionID)
	require.NoError(t, err)
	workflow, err := base.GetWorkflowExecution(context.Background(), exec.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusCancelled, stored.Status)
	require.Equal(t, string(types.ExecutionStatusCancelled), workflow.Status)
	require.Equal(t, "cancelled_by_user", *stored.StatusReason)
	require.Equal(t, "cancelled_by_user", *workflow.StatusReason)
	require.Zero(t, concurrencyLimiter.GetRunningCount("node-1"))
}

func TestAsyncForcedWorkerCancellationReleasesOnlyOwnedSlot(t *testing.T) {
	oldLimiter := concurrencyLimiter
	concurrencyLimiter = &AgentConcurrencyLimiter{maxPerAgent: 2}
	t.Cleanup(func() { concurrencyLimiter = oldLimiter })
	require.NoError(t, concurrencyLimiter.Acquire("node-1"))
	require.NoError(t, concurrencyLimiter.Acquire("node-1"))

	agent := testRestartAgent("http://agent.example")
	store := newTestExecutionStorage(agent)
	now := time.Now().UTC()
	exec := &types.Execution{
		ExecutionID: "exec-forced-stop", RunID: "run-forced-stop", AgentNodeID: "node-1", NodeID: "node-1",
		ReasonerID: "reasoner-a", Status: types.ExecutionStatusRunning,
		CreatedAt: now, StartedAt: now, UpdatedAt: now,
	}
	require.NoError(t, store.CreateExecutionRecord(context.Background(), exec))
	require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID: exec.ExecutionID, WorkflowID: exec.RunID, RunID: &exec.RunID,
		AgentNodeID: "node-1", ReasonerID: "reasoner-a", Status: string(types.ExecutionStatusRunning),
		CreatedAt: now, StartedAt: now, UpdatedAt: now,
	}))
	target, err := parseTarget("node-1.reasoner-a")
	require.NoError(t, err)
	job := asyncExecutionJob{
		controller: newExecutionController(store, nil, nil, time.Second, ""),
		plan: preparedExecution{
			exec: exec, target: target, agent: agent, requestBody: []byte(`{}`), slotHeld: true,
		},
	}
	workerCtx, cancel := context.WithCancel(context.Background())
	cancel()
	job.processWithContext(workerCtx)

	// This job owned one of two live slots. Forced shutdown must not consume the
	// other execution's count through failForControlPlaneShutdown plus defer.
	require.EqualValues(t, 1, concurrencyLimiter.GetRunningCount("node-1"))
	ReleaseExecutionSlot("node-1")
	require.Zero(t, concurrencyLimiter.GetRunningCount("node-1"))
}

func TestExecuteAsyncHandler_QueueSaturation(t *testing.T) {
	gin.SetMode(gin.TestMode)
	useAsyncPoolForTest(t, newAsyncWorkerPool(1, 1))

	workerStarted := make(chan struct{})
	releaseWorker := make(chan struct{})
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		select {
		case <-workerStarted:
		default:
			close(workerStarted)
		}
		<-releaseWorker
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"result":{}}`))
	}))
	defer agentServer.Close()

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   agentServer.URL,
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}
	store := newTestExecutionStorage(agent)
	payloads := services.NewFilePayloadStore(t.TempDir())
	router := gin.New()
	const burstSize = 10
	var ready sync.WaitGroup
	ready.Add(burstSize)
	start := make(chan struct{})
	router.Use(func(c *gin.Context) {
		ready.Done()
		<-start
	})
	router.POST("/api/v1/execute/async/:target", ExecuteAsyncHandler(store, payloads, nil, 90*time.Second, ""))

	request := func() *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/async/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		return resp
	}
	responses := make(chan *httptest.ResponseRecorder, burstSize)
	for i := 0; i < burstSize; i++ {
		go func() { responses <- request() }()
	}
	ready.Wait()
	close(start)

	accepted := 0
	rejected := 0
	for i := 0; i < burstSize; i++ {
		resp := <-responses
		switch resp.Code {
		case http.StatusAccepted:
			accepted++
		case http.StatusServiceUnavailable:
			rejected++
			require.Contains(t, resp.Body.String(), "async execution queue is full")
		default:
			t.Fatalf("unexpected async response status %d: %s", resp.Code, resp.Body.String())
		}
	}
	require.Equal(t, 2, accepted, "workers + queue capacity must be admitted")
	require.Equal(t, burstSize-2, rejected)

	records, err := store.QueryExecutionRecords(context.Background(), types.ExecutionFilter{})
	require.NoError(t, err)
	require.Len(t, records, 2, "rejected requests must not persist execution rows")
	workflows, err := store.QueryWorkflowExecutions(context.Background(), types.WorkflowExecutionFilters{})
	require.NoError(t, err)
	require.Len(t, workflows, 2, "rejected requests must not persist workflow rows")
	close(releaseWorker)
	<-workerStarted
	stopCtx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	asyncPool.Stop(stopCtx)
	require.Eventually(t, func() bool {
		completed, queryErr := store.QueryExecutionRecords(context.Background(), types.ExecutionFilter{})
		if queryErr != nil || len(completed) != 2 {
			return false
		}
		for _, record := range completed {
			if record.Status == types.ExecutionStatusRunning {
				return false
			}
		}
		return true
	}, time.Second, 10*time.Millisecond)
}

func TestExecuteAsyncHandler_ConcurrencyRejectionHasNoPersistence(t *testing.T) {
	gin.SetMode(gin.TestMode)
	oldLimiter := concurrencyLimiter
	concurrencyLimiter = &AgentConcurrencyLimiter{maxPerAgent: 1}
	require.NoError(t, concurrencyLimiter.Acquire("node-1"))
	defer func() { concurrencyLimiter = oldLimiter }()

	useAsyncPoolForTest(t, newAsyncWorkerPool(1, 2))

	agent := &types.AgentNode{ID: "node-1", BaseURL: "http://agent.example", Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}}}
	store := newTestExecutionStorage(agent)
	payloadDir := t.TempDir()
	router := gin.New()
	router.POST("/api/v1/execute/async/:target", ExecuteAsyncHandler(store, services.NewFilePayloadStore(payloadDir), nil, time.Second, ""))
	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/async/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusTooManyRequests, resp.Code)
	require.Equal(t, "1", resp.Header().Get("Retry-After"))
	var body map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
	require.Equal(t, "concurrency_limit", body["error_category"])
	require.Equal(t, float64(1), body["retry_after"])
	records, err := store.QueryExecutionRecords(context.Background(), types.ExecutionFilter{})
	require.NoError(t, err)
	require.Empty(t, records)
	workflows, err := store.QueryWorkflowExecutions(context.Background(), types.WorkflowExecutionFilters{})
	require.NoError(t, err)
	require.Empty(t, workflows)
	files, err := filepath.Glob(filepath.Join(payloadDir, "*"))
	require.NoError(t, err)
	require.Empty(t, files)
}

func TestExecuteAsyncHandler_ChunkedOversizeBodyHasNoPersistence(t *testing.T) {
	gin.SetMode(gin.TestMode)
	useAsyncPoolForTest(t, newAsyncWorkerPool(0, 1))

	agent := &types.AgentNode{ID: "node-1", BaseURL: "http://agent.example", Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}}}
	store := newTestExecutionStorage(agent)
	router := gin.New()
	router.POST("/api/v1/execute/async/:target", ExecuteAsyncHandler(store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/async/node-1.reasoner-a", strings.NewReader(`{"input":{"value":"oversize"}}`))
	req.ContentLength = -1
	req.TransferEncoding = []string{"chunked"}
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	req.Body = http.MaxBytesReader(resp, req.Body, 8)
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusRequestEntityTooLarge, resp.Code)
	require.JSONEq(t, `{"error":"request body too large"}`, resp.Body.String())
	records, err := store.QueryExecutionRecords(context.Background(), types.ExecutionFilter{})
	require.NoError(t, err)
	require.Empty(t, records)
	workflows, err := store.QueryWorkflowExecutions(context.Background(), types.WorkflowExecutionFilters{})
	require.NoError(t, err)
	require.Empty(t, workflows)
}

func TestExecuteAsyncHandler_QueueFullHasNoPersistence(t *testing.T) {
	gin.SetMode(gin.TestMode)
	pool := newAsyncWorkerPool(0, 1)
	useAsyncPoolForTest(t, pool)
	require.True(t, asyncPool.reserve())

	agent := &types.AgentNode{ID: "node-1", BaseURL: "http://agent.example", Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}}}
	store := newTestExecutionStorage(agent)
	payloadDir := t.TempDir()
	router := gin.New()
	router.POST("/api/v1/execute/async/:target", ExecuteAsyncHandler(store, services.NewFilePayloadStore(payloadDir), nil, time.Second, ""))
	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/async/node-1.reasoner-a", strings.NewReader(`{"input":{}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	require.Equal(t, http.StatusServiceUnavailable, resp.Code)
	require.Equal(t, "1", resp.Header().Get("Retry-After"))
	var body map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
	require.Equal(t, "concurrency_limit", body["error_category"])
	require.Equal(t, float64(1), body["retry_after"])
	records, err := store.QueryExecutionRecords(context.Background(), types.ExecutionFilter{})
	require.NoError(t, err)
	require.Empty(t, records)
	workflows, err := store.QueryWorkflowExecutions(context.Background(), types.WorkflowExecutionFilters{})
	require.NoError(t, err)
	require.Empty(t, workflows)
	files, err := filepath.Glob(filepath.Join(payloadDir, "*"))
	require.NoError(t, err)
	require.Empty(t, files)
}

func TestWriteExecutionError_ConcurrencyLimitIncludesRetryAfter(t *testing.T) {
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	writeExecutionError(ctx, &executionPreconditionError{code: http.StatusTooManyRequests, message: "busy", category: ErrorCategoryConcurrencyLimit})
	require.Equal(t, "1", recorder.Header().Get("Retry-After"))
	require.JSONEq(t, `{"error":"busy","error_category":"concurrency_limit","retry_after":1}`, recorder.Body.String())
}

func TestWriteExecutionError_NodeUnavailableIncludesRetryAfter(t *testing.T) {
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	writeExecutionError(ctx, &executionPreconditionError{
		code: http.StatusServiceUnavailable, message: "offline", category: ErrorCategoryNodeUnavailable, errorCode: "node_unavailable",
	})
	require.Equal(t, "1", recorder.Header().Get("Retry-After"))
	require.JSONEq(t, `{"error":"node_unavailable","message":"offline","error_category":"node_unavailable","retry_after":1}`, recorder.Body.String())
}

func TestAsyncWorkerPoolStopFailsQueuedJobsAndRejectsSubmissions(t *testing.T) {
	workerStarted := make(chan struct{})
	releaseWorker := make(chan struct{})
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(workerStarted)
		select {
		case <-r.Context().Done():
		case <-releaseWorker:
		}
	}))
	defer agentServer.Close()
	agent := &types.AgentNode{ID: "node-1", BaseURL: agentServer.URL}
	store := newTestExecutionStorage(agent)
	now := time.Now().UTC()
	target, err := parseTarget("node-1.reasoner-a")
	require.NoError(t, err)
	pool := newAsyncWorkerPool(1, 2)
	for _, id := range []string{"running-1", "queued-1"} {
		status := types.ExecutionStatusRunning
		if id == "queued-1" {
			status = types.ExecutionStatusQueued
		}
		exec := &types.Execution{ExecutionID: id, RunID: id, NodeID: "node-1", AgentNodeID: "node-1", ReasonerID: "reasoner-a", Status: status, CreatedAt: now, StartedAt: now, UpdatedAt: now}
		require.NoError(t, store.CreateExecutionRecord(context.Background(), exec))
		require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{ExecutionID: id, WorkflowID: id, RunID: &id, AgentNodeID: "node-1", ReasonerID: "reasoner-a", Status: string(status), StartedAt: now, CreatedAt: now, UpdatedAt: now}))
		require.True(t, pool.submit(asyncExecutionJob{controller: newExecutionController(store, nil, nil, time.Second, ""), plan: preparedExecution{exec: exec, target: target, agent: agent, requestBody: []byte(`{"input":{}}`)}}))
	}
	<-workerStarted

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	pool.Stop(ctx)
	close(releaseWorker)
	require.False(t, pool.submit(asyncExecutionJob{}))
	for _, id := range []string{"running-1", "queued-1"} {
		stored, getErr := store.GetExecutionRecord(context.Background(), id)
		require.NoError(t, getErr)
		require.Equal(t, types.ExecutionStatusFailed, stored.Status)
		require.Equal(t, "control_plane_shutdown", *stored.StatusReason)
		require.NotNil(t, stored.ErrorMessage)
		require.Contains(t, *stored.ErrorMessage, "control plane shut down")
		workflow, workflowErr := store.GetWorkflowExecution(context.Background(), id)
		require.NoError(t, workflowErr)
		require.Equal(t, types.ExecutionStatusFailed, workflow.Status)
		require.Equal(t, "control_plane_shutdown", *workflow.StatusReason)
	}
}

func TestAsyncWorkerPoolStopDoesNotStartQueuedJobsAfterReturn(t *testing.T) {
	var starts atomic.Int32
	firstStarted := make(chan struct{})
	releaseFirst := make(chan struct{})
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if starts.Add(1) == 1 {
			close(firstStarted)
			<-releaseFirst
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"result":{}}`))
	}))
	defer agentServer.Close()

	agent := &types.AgentNode{ID: "node-1", BaseURL: agentServer.URL, Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}}}
	store := newTestExecutionStorage(agent)
	target, err := parseTarget("node-1.reasoner-a")
	require.NoError(t, err)
	pool := newAsyncWorkerPool(1, 8)
	now := time.Now().UTC()
	for i := 0; i < 4; i++ {
		id := fmt.Sprintf("stop-start-%d", i)
		exec := &types.Execution{ExecutionID: id, RunID: id, NodeID: "node-1", AgentNodeID: "node-1", ReasonerID: "reasoner-a", Status: types.ExecutionStatusRunning, CreatedAt: now, StartedAt: now, UpdatedAt: now}
		require.NoError(t, store.CreateExecutionRecord(context.Background(), exec))
		require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{ExecutionID: id, WorkflowID: id, RunID: &id, AgentNodeID: "node-1", ReasonerID: "reasoner-a", Status: types.ExecutionStatusRunning, StartedAt: now, CreatedAt: now, UpdatedAt: now}))
		require.True(t, pool.submit(asyncExecutionJob{controller: newExecutionController(store, nil, nil, time.Second, ""), plan: preparedExecution{exec: exec, target: target, agent: agent, requestBody: []byte(`{"input":{}}`)}}))
	}
	<-firstStarted
	stopCtx, cancel := context.WithCancel(context.Background())
	cancel()
	pool.Stop(stopCtx)
	require.Equal(t, int32(1), starts.Load())
	close(releaseFirst)
	require.Eventually(t, func() bool { return starts.Load() == 1 }, 100*time.Millisecond, 10*time.Millisecond)
}

func TestExecuteAsyncHandler_WithWebhook(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var requestCount int32
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&requestCount, 1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer agentServer.Close()

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   agentServer.URL,
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}

	store := newTestExecutionStorage(agent)
	payloads := services.NewFilePayloadStore(t.TempDir())

	router := gin.New()
	router.POST("/api/v1/execute/async/:target", ExecuteAsyncHandler(store, payloads, nil, 90*time.Second, ""))

	reqBody := `{
		"input": {"foo": "bar"},
		"webhook": {
			"url": "https://example.com/webhook",
			"secret": "test-secret"
		}
	}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/async/node-1.reasoner-a", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusAccepted, resp.Code)

	var payload AsyncExecuteResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.NotEmpty(t, payload.ExecutionID)
	require.True(t, payload.WebhookRegistered)

	// Wait for async execution to complete
	require.Eventually(t, func() bool {
		record, err := store.GetExecutionRecord(context.Background(), payload.ExecutionID)
		if err != nil || record == nil {
			return false
		}
		return record.Status == types.ExecutionStatusSucceeded
	}, 2*time.Second, 50*time.Millisecond)

	require.Eventually(t, func() bool {
		return atomic.LoadInt32(&requestCount) > 0
	}, time.Second, 50*time.Millisecond)
}

func TestExecuteAsyncHandler_InvalidWebhook(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   "http://agent.example",
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}

	store := newTestExecutionStorage(agent)
	payloads := services.NewFilePayloadStore(t.TempDir())

	router := gin.New()
	router.POST("/api/v1/execute/async/:target", ExecuteAsyncHandler(store, payloads, nil, 90*time.Second, ""))

	// Webhook with invalid URL (too long)
	longURL := strings.Repeat("a", 4097)
	reqBody := `{
		"input": {"foo": "bar"},
		"webhook": {
			"url": "` + longURL + `"
		}
	}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/async/node-1.reasoner-a", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusAccepted, resp.Code)

	var payload AsyncExecuteResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.NotEmpty(t, payload.ExecutionID)
	require.False(t, payload.WebhookRegistered)
	require.NotNil(t, payload.WebhookError)
}

func TestHandleSync_AsyncAcknowledgment(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var requestCount int32
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&requestCount, 1)
		// Return HTTP 202 Accepted
		w.WriteHeader(http.StatusAccepted)
	}))
	defer agentServer.Close()

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   agentServer.URL,
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}

	store := newTestExecutionStorage(agent)
	payloads := services.NewFilePayloadStore(t.TempDir())

	router := gin.New()
	router.POST("/api/v1/execute/:target", ExecuteHandler(store, payloads, nil, 90*time.Second, ""))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	// Start request in goroutine since it will wait for completion
	done := make(chan bool)
	go func() {
		router.ServeHTTP(resp, req)
		done <- true
	}()

	// Simulate status update callback after a short delay
	time.Sleep(100 * time.Millisecond)
	executionID := ""
	records, _ := store.QueryExecutionRecords(context.Background(), types.ExecutionFilter{})
	if len(records) > 0 {
		executionID = records[0].ExecutionID
	}

	if executionID != "" {
		// Update execution to completed state
		_, err := store.UpdateExecutionRecord(context.Background(), executionID, func(current *types.Execution) (*types.Execution, error) {
			if current == nil {
				return nil, nil
			}
			now := time.Now().UTC()
			current.Status = types.ExecutionStatusSucceeded
			result := json.RawMessage(`{"result":"success"}`)
			current.ResultPayload = result
			completed := now
			current.CompletedAt = &completed
			duration := int64(100)
			current.DurationMS = &duration
			return current, nil
		})
		if err == nil {
			// Publish completion event
			eventBus := store.GetExecutionEventBus()
			if eventBus != nil {
				eventBus.Publish(events.ExecutionEvent{
					Type:        events.ExecutionCompleted,
					ExecutionID: executionID,
					WorkflowID:  "test-run",
					Status:      string(types.ExecutionStatusSucceeded),
					Timestamp:   time.Now(),
				})
			}
		}
	}

	// Wait for response or timeout
	select {
	case <-done:
		// Response completed
	case <-time.After(2 * time.Second):
		t.Fatal("Request timed out waiting for async completion")
	}

	// Note: In a real scenario, the sync handler would wait for the callback
	// This test verifies the async acknowledgment path exists
	require.Equal(t, int32(1), atomic.LoadInt32(&requestCount))
}

func TestCallAgent_HTTP202Response(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Return HTTP 202 Accepted
		w.WriteHeader(http.StatusAccepted)
	}))
	defer agentServer.Close()

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   agentServer.URL,
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}

	store := newTestExecutionStorage(agent)
	controller := newExecutionController(store, nil, nil, 90*time.Second, "")

	plan := &preparedExecution{
		exec: &types.Execution{
			ExecutionID: "test-exec",
			RunID:       "test-run",
		},
		requestBody: []byte(`{"input":{"foo":"bar"}}`),
		agent:       agent,
		target: &parsedTarget{
			NodeID:     "node-1",
			TargetName: "reasoner-a",
		},
	}

	body, elapsed, asyncAccepted, err := controller.callAgent(context.Background(), plan)

	require.NoError(t, err)
	require.True(t, asyncAccepted)
	require.Nil(t, body)
	require.Greater(t, elapsed, time.Duration(0))
}

func TestCallAgent_ErrorResponse(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"internal server error"}`))
	}))
	defer agentServer.Close()

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   agentServer.URL,
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}

	store := newTestExecutionStorage(agent)
	controller := newExecutionController(store, nil, nil, 90*time.Second, "")

	plan := &preparedExecution{
		exec: &types.Execution{
			ExecutionID: "test-exec",
			RunID:       "test-run",
		},
		requestBody: []byte(`{"input":{"foo":"bar"}}`),
		agent:       agent,
		target: &parsedTarget{
			NodeID:     "node-1",
			TargetName: "reasoner-a",
		},
	}

	body, elapsed, asyncAccepted, err := controller.callAgent(context.Background(), plan)

	require.Error(t, err)
	require.False(t, asyncAccepted)
	require.Contains(t, err.Error(), "agent error (500)")
	require.NotNil(t, body)
	require.Greater(t, elapsed, time.Duration(0))
}

func TestCallAgent_Timeout(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Server that delays response beyond timeout
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second)
		w.WriteHeader(http.StatusOK)
	}))
	defer agentServer.Close()

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   agentServer.URL,
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}

	store := newTestExecutionStorage(agent)
	controller := newExecutionController(store, nil, nil, 90*time.Second, "")
	// Set shorter timeout for test
	controller.httpClient.Timeout = 100 * time.Millisecond

	plan := &preparedExecution{
		exec: &types.Execution{
			ExecutionID: "test-exec",
			RunID:       "test-run",
		},
		requestBody: []byte(`{"input":{"foo":"bar"}}`),
		agent:       agent,
		target: &parsedTarget{
			NodeID:     "node-1",
			TargetName: "reasoner-a",
		},
	}

	body, elapsed, asyncAccepted, err := controller.callAgent(context.Background(), plan)

	require.Error(t, err)
	require.False(t, asyncAccepted)
	// Error message may vary but should indicate timeout or deadline exceeded
	errorMsg := err.Error()
	require.True(t,
		strings.Contains(strings.ToLower(errorMsg), "timeout") ||
			strings.Contains(strings.ToLower(errorMsg), "deadline exceeded") ||
			strings.Contains(strings.ToLower(errorMsg), "context deadline"),
		"Expected timeout-related error, got: %s", errorMsg)
	require.Nil(t, body)
	require.Greater(t, elapsed, time.Duration(0))
}

func TestCallAgent_ReadResponseError(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Close connection immediately to cause read error
		hj, ok := w.(http.Hijacker)
		if ok {
			conn, _, _ := hj.Hijack()
			conn.Close()
		}
	}))
	defer agentServer.Close()

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   agentServer.URL,
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}

	store := newTestExecutionStorage(agent)
	controller := newExecutionController(store, nil, nil, 90*time.Second, "")

	plan := &preparedExecution{
		exec: &types.Execution{
			ExecutionID: "test-exec",
			RunID:       "test-run",
		},
		requestBody: []byte(`{"input":{"foo":"bar"}}`),
		agent:       agent,
		target: &parsedTarget{
			NodeID:     "node-1",
			TargetName: "reasoner-a",
		},
	}

	body, elapsed, asyncAccepted, err := controller.callAgent(context.Background(), plan)

	require.Error(t, err)
	require.False(t, asyncAccepted)
	require.Contains(t, err.Error(), "agent call failed")
	require.Nil(t, body)
	require.Greater(t, elapsed, time.Duration(0))
}

func TestCallAgent_HeaderPropagation(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var receivedHeaders http.Header
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedHeaders = r.Header
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer agentServer.Close()

	agent := &types.AgentNode{
		ID:        "node-1",
		BaseURL:   agentServer.URL,
		Reasoners: []types.ReasonerDefinition{{ID: "reasoner-a"}},
	}

	store := newTestExecutionStorage(agent)
	controller := newExecutionController(store, nil, nil, 90*time.Second, "")

	parentID := "parent-exec-123"
	sessionID := "session-456"
	actorID := "actor-789"

	plan := &preparedExecution{
		exec: &types.Execution{
			ExecutionID:       "test-exec",
			RunID:             "test-run",
			ParentExecutionID: &parentID,
			SessionID:         &sessionID,
			ActorID:           &actorID,
		},
		requestBody: []byte(`{"input":{"foo":"bar"}}`),
		agent:       agent,
		target: &parsedTarget{
			NodeID:     "node-1",
			TargetName: "reasoner-a",
		},
	}

	_, _, _, err := controller.callAgent(context.Background(), plan)
	require.NoError(t, err)

	require.Equal(t, "test-run", receivedHeaders.Get("X-Run-ID"))
	require.Equal(t, "test-exec", receivedHeaders.Get("X-Execution-ID"))
	require.Equal(t, parentID, receivedHeaders.Get("X-Parent-Execution-ID"))
	require.Equal(t, sessionID, receivedHeaders.Get("X-Session-ID"))
	require.Equal(t, actorID, receivedHeaders.Get("X-Actor-ID"))
}
