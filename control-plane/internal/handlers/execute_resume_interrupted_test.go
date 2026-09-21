package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func withResumeInterruptedSettings(t *testing.T, enabled bool, attempts, limit int, window, delay time.Duration) {
	t.Helper()
	oldEnabled := ResumeInterruptedRuns()
	oldAttempts := ResumeInterruptedMaxAttempts()
	oldLimit := ResumeInterruptedLimit()
	oldWindow := ResumeInterruptedWindow()
	oldDelay := ResumeInterruptedDelay()
	SetResumeInterruptedRuns(enabled)
	SetResumeInterruptedMaxAttempts(attempts)
	SetResumeInterruptedLimit(limit)
	SetResumeInterruptedWindow(window)
	SetResumeInterruptedDelay(delay)
	t.Cleanup(func() {
		SetResumeInterruptedRuns(oldEnabled)
		SetResumeInterruptedMaxAttempts(oldAttempts)
		SetResumeInterruptedLimit(oldLimit)
		SetResumeInterruptedWindow(oldWindow)
		SetResumeInterruptedDelay(oldDelay)
	})
}

func TestStatusCallbackMarksOnlyGracefulShutdownCancellation(t *testing.T) {
	withResumeInterruptedSettings(t, false, 1, 25, time.Hour, 0)
	store := newTestExecutionStorage(nil)
	now := time.Now().UTC()
	for _, id := range []string{"graceful-cancel", "user-cancel"} {
		require.NoError(t, store.CreateExecutionRecord(t.Context(), &types.Execution{
			ExecutionID: id, RunID: "run-" + id, Status: types.ExecutionStatusRunning,
			StartedAt: now, CreatedAt: now, UpdatedAt: now,
		}))
	}
	router := gin.New()
	router.PUT("/executions/:execution_id/status", UpdateExecutionStatusHandler(store, services.NopPayloadStore{}, nil, time.Second))

	graceful := httptest.NewRequest(http.MethodPut, "/executions/graceful-cancel/status", strings.NewReader(`{"status":"cancelled","error":"cancelled during graceful shutdown"}`))
	graceful.Header.Set("Content-Type", "application/json")
	gracefulResponse := httptest.NewRecorder()
	router.ServeHTTP(gracefulResponse, graceful)
	require.Equal(t, http.StatusOK, gracefulResponse.Code)
	storedGraceful, err := store.GetExecutionRecord(t.Context(), "graceful-cancel")
	require.NoError(t, err)
	require.NotNil(t, storedGraceful.StatusReason)
	require.Equal(t, types.ExecutionReasonAgentShutdownCancelled, *storedGraceful.StatusReason)
	require.Nil(t, storedGraceful.RestartedAsExecutionID)

	user := httptest.NewRequest(http.MethodPut, "/executions/user-cancel/status", strings.NewReader(`{"status":"cancelled","error":"cancelled_by_control_plane"}`))
	user.Header.Set("Content-Type", "application/json")
	userResponse := httptest.NewRecorder()
	router.ServeHTTP(userResponse, user)
	require.Equal(t, http.StatusOK, userResponse.Code)
	storedUser, err := store.GetExecutionRecord(t.Context(), "user-cancel")
	require.NoError(t, err)
	require.Nil(t, storedUser.StatusReason)
}

func TestHandOffInterruptedRunCreatesFollowableWorkflowRestart(t *testing.T) {
	withResumeInterruptedSettings(t, true, 1, 25, time.Hour, 0)
	withTestAsyncPool(t)

	agentCalled := make(chan struct{}, 1)
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		select {
		case agentCalled <- struct{}{}:
		default:
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer agentServer.Close()

	store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	now := time.Now().UTC()
	reason := types.ExecutionReasonAgentShutdownCancelled
	root := &types.Execution{
		ExecutionID: "interrupted-root-followable", RunID: "interrupted-run-followable",
		AgentNodeID: "node-1", NodeID: "node-1", ReasonerID: "reasoner-a",
		Status: types.ExecutionStatusCancelled, StatusReason: &reason,
		InputPayload: json.RawMessage(`{"input":{"value":"stored"},"context":{"trace":"kept"}}`),
		StartedAt:    now.Add(-time.Minute), CreatedAt: now.Add(-time.Minute), UpdatedAt: now,
	}
	seedExecutionRecord(t, store, root)
	staleRootSnapshot := *root
	controller := newExecutionController(store, services.NopPayloadStore{}, nil, time.Second, "")

	successorID, ok := controller.handOffInterruptedRun(t.Context(), root)
	require.True(t, ok)
	require.NotEmpty(t, successorID)
	require.NotEqual(t, root.ExecutionID, successorID)
	successor, err := store.GetExecutionRecord(t.Context(), successorID)
	require.NoError(t, err)
	require.NotNil(t, successor)
	require.NotEqual(t, root.RunID, successor.RunID)
	require.JSONEq(t, string(root.InputPayload), string(successor.InputPayload))

	source, err := store.GetExecutionRecord(t.Context(), root.ExecutionID)
	require.NoError(t, err)
	require.NotNil(t, source.RestartedAsExecutionID)
	require.Equal(t, successorID, *source.RestartedAsExecutionID)
	_, handedOffAgain := controller.handOffInterruptedRun(t.Context(), &staleRootSnapshot)
	require.False(t, handedOffAgain)

	router := gin.New()
	router.GET("/executions/:execution_id", GetExecutionStatusHandler(store))
	response := httptest.NewRecorder()
	router.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/executions/"+root.ExecutionID, nil))
	require.Equal(t, http.StatusOK, response.Code)
	var status ExecutionStatusResponse
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &status))
	require.Equal(t, &RestartedAsRef{ExecutionID: successorID, RunID: successor.RunID}, status.RestartedAs)

	newRun, err := store.GetWorkflowRun(t.Context(), successor.RunID)
	require.NoError(t, err)
	require.NotNil(t, newRun)
	var metadata map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(newRun.Metadata, &metadata))
	var lineage map[string]interface{}
	require.NoError(t, json.Unmarshal(metadata["lineage"], &lineage))
	require.Equal(t, "resume", lineage["kind"])
	require.Equal(t, "all-succeeded", lineage["reuse"])
	require.EqualValues(t, 1, lineage["resume_attempt"])

	select {
	case <-agentCalled:
	case <-time.After(time.Second):
		t.Fatal("successor was not dispatched")
	}
	require.Eventually(t, func() bool {
		completed, _ := store.GetExecutionRecord(t.Context(), successorID)
		return completed != nil && completed.Status == types.ExecutionStatusSucceeded
	}, time.Second, 10*time.Millisecond)
}

func TestHandOffInterruptedRunReplaysCompletedChild(t *testing.T) {
	withResumeInterruptedSettings(t, true, 1, 25, time.Hour, 0)
	withTestAsyncPool(t)

	type replayOutcome struct {
		status      int
		executionID string
		replayHit   string
		replayMode  string
		err         error
	}
	childOutcome := make(chan replayOutcome, 1)
	var childAgentCalls atomic.Int32
	var controlPlaneURL string
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if request.URL.Path == "/reasoners/reasoner-b" {
			childAgentCalls.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"unexpected":"child dispatch"}`))
			return
		}

		childRequest, err := http.NewRequest(
			http.MethodPost,
			controlPlaneURL+"/api/v1/execute/async/node-1.reasoner-b",
			strings.NewReader(`{"input":{"step":"done"}}`),
		)
		if err != nil {
			childOutcome <- replayOutcome{err: err}
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		childRequest.Header.Set("Content-Type", "application/json")
		childRequest.Header.Set("X-Run-ID", request.Header.Get("X-Run-ID"))
		childRequest.Header.Set("X-Parent-Execution-ID", request.Header.Get("X-Execution-ID"))
		for _, header := range []string{
			"X-AgentField-Replay-Source-Run-ID",
			"X-AgentField-Replay-Before-Execution-ID",
			"X-AgentField-Replay-Mode",
		} {
			childRequest.Header.Set(header, request.Header.Get(header))
		}

		response, err := http.DefaultClient.Do(childRequest)
		if err != nil {
			childOutcome <- replayOutcome{err: err}
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer response.Body.Close()
		var child AsyncExecuteResponse
		err = json.NewDecoder(response.Body).Decode(&child)
		childOutcome <- replayOutcome{
			status:      response.StatusCode,
			executionID: child.ExecutionID,
			replayHit:   response.Header.Get("X-AgentField-Replay-Hit"),
			replayMode:  request.Header.Get("X-AgentField-Replay-Mode"),
			err:         err,
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer agentServer.Close()

	store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	router := gin.New()
	router.POST("/api/v1/execute/async/:target", ExecuteAsyncHandler(store, services.NopPayloadStore{}, nil, time.Second, ""))
	controlPlane := httptest.NewServer(router)
	defer controlPlane.Close()
	controlPlaneURL = controlPlane.URL

	now := time.Now().UTC()
	reason := types.ExecutionReasonAgentShutdownCancelled
	root := &types.Execution{
		ExecutionID: "handoff-replay-root", RunID: "handoff-replay-source-run",
		AgentNodeID: "node-1", NodeID: "node-1", ReasonerID: "reasoner-a",
		Status: types.ExecutionStatusCancelled, StatusReason: &reason,
		InputPayload: json.RawMessage(`{"input":{}}`),
		StartedAt:    now.Add(-2 * time.Minute), CreatedAt: now.Add(-2 * time.Minute), UpdatedAt: now,
	}
	seedExecutionRecord(t, store, root)
	parentID := root.ExecutionID
	seedExecutionRecord(t, store, &types.Execution{
		ExecutionID: "handoff-replay-child", RunID: root.RunID, ParentExecutionID: &parentID,
		AgentNodeID: "node-1", NodeID: "node-1", ReasonerID: "reasoner-b",
		Status: types.ExecutionStatusSucceeded, InputPayload: json.RawMessage(`{"input":{"step":"done"}}`),
		ResultPayload: json.RawMessage(`{"from":"source-run"}`),
		StartedAt:     now.Add(-time.Minute), CreatedAt: now.Add(-time.Minute), UpdatedAt: now,
	})

	controller := newExecutionController(store, services.NopPayloadStore{}, nil, time.Second, "")
	successorID, handedOff := controller.handOffInterruptedRun(t.Context(), root)
	require.True(t, handedOff)

	var outcome replayOutcome
	select {
	case outcome = <-childOutcome:
	case <-time.After(2 * time.Second):
		t.Fatal("successor did not make its completed child call")
	}
	require.NoError(t, outcome.err)
	require.Equal(t, http.StatusAccepted, outcome.status)
	require.Equal(t, "all-succeeded", outcome.replayMode)
	require.Equal(t, "handoff-replay-child", outcome.replayHit)
	require.Zero(t, childAgentCalls.Load(), "the completed child must be replayed without agent dispatch")

	replayed, err := store.GetExecutionRecord(t.Context(), outcome.executionID)
	require.NoError(t, err)
	require.NotNil(t, replayed)
	require.Equal(t, successorID, *replayed.ParentExecutionID)
	require.Equal(t, types.ExecutionStatusSucceeded, replayed.Status)
	require.NotNil(t, replayed.StatusReason)
	require.Equal(t, "replayed_from_execution:handoff-replay-child", *replayed.StatusReason)
	require.JSONEq(t, `{"from":"source-run"}`, string(replayed.ResultPayload))
}

func TestGracefulShutdownStatusCallbackHandsRootOffWhenEnabled(t *testing.T) {
	withResumeInterruptedSettings(t, true, 1, 25, time.Hour, 0)
	withTestAsyncPool(t)
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer agentServer.Close()
	store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	now := time.Now().UTC()
	require.NoError(t, store.CreateExecutionRecord(t.Context(), &types.Execution{
		ExecutionID: "callback-handoff-root", RunID: "callback-handoff-run",
		AgentNodeID: "node-1", NodeID: "node-1", ReasonerID: "reasoner-a",
		Status: types.ExecutionStatusRunning, InputPayload: json.RawMessage(`{"input":{"from":"callback"}}`),
		StartedAt: now, CreatedAt: now, UpdatedAt: now,
	}))
	router := gin.New()
	router.PUT("/executions/:execution_id/status", UpdateExecutionStatusHandler(store, services.NopPayloadStore{}, nil, time.Second))
	request := httptest.NewRequest(http.MethodPut, "/executions/callback-handoff-root/status", strings.NewReader(`{"status":"cancelled","error":"cancelled during graceful shutdown"}`))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	require.Equal(t, http.StatusOK, response.Code)
	var source *types.Execution
	require.Eventually(t, func() bool {
		var err error
		source, err = store.GetExecutionRecord(t.Context(), "callback-handoff-root")
		return err == nil && source != nil && source.RestartedAsExecutionID != nil
	}, time.Second, 10*time.Millisecond)
	successor, err := store.GetExecutionRecord(t.Context(), *source.RestartedAsExecutionID)
	require.NoError(t, err)
	require.NotNil(t, successor)
	require.NotEqual(t, source.RunID, successor.RunID)
	require.Eventually(t, func() bool {
		completed, _ := store.GetExecutionRecord(t.Context(), successor.ExecutionID)
		return completed != nil && completed.Status == types.ExecutionStatusSucceeded
	}, time.Second, 10*time.Millisecond)
}

func TestStatusCallbackDelaysSuccessorDispatch(t *testing.T) {
	const handoffDelay = 100 * time.Millisecond
	withResumeInterruptedSettings(t, true, 1, 25, time.Hour, handoffDelay)
	withTestAsyncPool(t)
	dispatchedAt := make(chan time.Time, 1)
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		dispatchedAt <- time.Now()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer agentServer.Close()

	store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	now := time.Now().UTC()
	require.NoError(t, store.CreateExecutionRecord(t.Context(), &types.Execution{
		ExecutionID: "delayed-callback-root", RunID: "delayed-callback-run",
		AgentNodeID: "node-1", NodeID: "node-1", ReasonerID: "reasoner-a",
		Status: types.ExecutionStatusRunning, InputPayload: json.RawMessage(`{"input":{}}`),
		StartedAt: now, CreatedAt: now, UpdatedAt: now,
	}))
	router := gin.New()
	router.PUT("/executions/:execution_id/status", UpdateExecutionStatusHandler(store, services.NopPayloadStore{}, nil, time.Second))
	request := httptest.NewRequest(http.MethodPut, "/executions/delayed-callback-root/status", strings.NewReader(`{"status":"cancelled","error":"cancelled during graceful shutdown"}`))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	scheduledAt := time.Now()
	router.ServeHTTP(response, request)
	require.Equal(t, http.StatusOK, response.Code)

	select {
	case <-dispatchedAt:
		t.Fatal("successor dispatched before the configured handoff delay")
	default:
	}
	_, claimed := interruptedHandoffClaims.Load("delayed-callback-root")
	require.True(t, claimed, "the claim must cover the pending delay")

	select {
	case dispatchTime := <-dispatchedAt:
		require.False(t, dispatchTime.Before(scheduledAt.Add(handoffDelay)), "successor dispatched before the configured handoff delay")
	case <-time.After(time.Second):
		t.Fatal("successor was not dispatched after the configured handoff delay")
	}
	require.Eventually(t, func() bool {
		_, stillClaimed := interruptedHandoffClaims.Load("delayed-callback-root")
		return !stillClaimed
	}, time.Second, 10*time.Millisecond)
}

func TestGracefulShutdownLifecycleEventHandoffHonorsFeatureFlag(t *testing.T) {
	for _, test := range []struct {
		name    string
		enabled bool
	}{
		{name: "enabled", enabled: true},
		{name: "disabled", enabled: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			withResumeInterruptedSettings(t, test.enabled, 1, 25, time.Hour, 0)
			withTestAsyncPool(t)
			var agentCalls atomic.Int32
			agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				agentCalls.Add(1)
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"ok":true}`))
			}))
			defer agentServer.Close()

			store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
			now := time.Now().UTC()
			executionID := "event-handoff-root-" + test.name
			require.NoError(t, store.CreateExecutionRecord(t.Context(), &types.Execution{
				ExecutionID: executionID, RunID: "event-handoff-run-" + test.name,
				AgentNodeID: "node-1", NodeID: "node-1", ReasonerID: "reasoner-a",
				Status: types.ExecutionStatusRunning, InputPayload: json.RawMessage(`{"input":{"from":"event"}}`),
				StartedAt: now, CreatedAt: now, UpdatedAt: now,
			}))
			handler := WorkflowExecutionEventHandler(store, InterruptedRunResumeDependencies{
				Payloads: services.NopPayloadStore{}, Timeout: time.Second,
			})
			response := postWorkflowExecutionEvent(t, handler, WorkflowExecutionEventRequest{
				ExecutionID: executionID,
				RunID:       "event-handoff-run-" + test.name,
				AgentNodeID: "node-1",
				ReasonerID:  "reasoner-a",
				Status:      types.ExecutionStatusCancelled,
				Error:       types.AgentShutdownCancellationError,
			})
			require.Equal(t, http.StatusOK, response.Code)
			require.JSONEq(t, `{"success":true,"updated":true}`, response.Body.String())

			if !test.enabled {
				require.Never(t, func() bool {
					source, _ := store.GetExecutionRecord(t.Context(), executionID)
					return source != nil && source.RestartedAsExecutionID != nil
				}, 100*time.Millisecond, 10*time.Millisecond)
				require.Zero(t, agentCalls.Load())
				return
			}

			var successorID string
			require.Eventually(t, func() bool {
				source, _ := store.GetExecutionRecord(t.Context(), executionID)
				if source == nil || source.RestartedAsExecutionID == nil {
					return false
				}
				successorID = *source.RestartedAsExecutionID
				return successorID != ""
			}, time.Second, 10*time.Millisecond)
			require.Eventually(t, func() bool {
				successor, _ := store.GetExecutionRecord(t.Context(), successorID)
				return successor != nil && successor.Status == types.ExecutionStatusSucceeded
			}, time.Second, 10*time.Millisecond)

			router := gin.New()
			router.GET("/executions/:execution_id", GetExecutionStatusHandler(store))
			statusResponse := httptest.NewRecorder()
			router.ServeHTTP(statusResponse, httptest.NewRequest(http.MethodGet, "/executions/"+executionID, nil))
			require.Equal(t, http.StatusOK, statusResponse.Code)
			var status ExecutionStatusResponse
			require.NoError(t, json.Unmarshal(statusResponse.Body.Bytes(), &status))
			require.NotNil(t, status.RestartedAs)
			require.Equal(t, successorID, status.RestartedAs.ExecutionID)
			require.EqualValues(t, 1, agentCalls.Load())
		})
	}
}

func TestLifecycleEventAndStatusCallbackCreateExactlyOneSuccessor(t *testing.T) {
	withResumeInterruptedSettings(t, true, 1, 25, time.Hour, 100*time.Millisecond)
	withTestAsyncPool(t)
	var agentCalls atomic.Int32
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		agentCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer agentServer.Close()

	store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	now := time.Now().UTC()
	require.NoError(t, store.CreateExecutionRecord(t.Context(), &types.Execution{
		ExecutionID: "dual-signal-root", RunID: "dual-signal-run",
		AgentNodeID: "node-1", NodeID: "node-1", ReasonerID: "reasoner-a",
		Status: types.ExecutionStatusRunning, InputPayload: json.RawMessage(`{"input":{"from":"both"}}`),
		StartedAt: now, CreatedAt: now, UpdatedAt: now,
	}))

	router := gin.New()
	deps := InterruptedRunResumeDependencies{Payloads: services.NopPayloadStore{}, Timeout: time.Second}
	router.POST("/workflow/executions/events", WorkflowExecutionEventHandler(store, deps))
	router.POST("/executions/:execution_id/status", UpdateExecutionStatusHandler(store, services.NopPayloadStore{}, nil, time.Second))

	requests := []*http.Request{
		httptest.NewRequest(http.MethodPost, "/workflow/executions/events", strings.NewReader(`{"execution_id":"dual-signal-root","run_id":"dual-signal-run","agent_node_id":"node-1","reasoner_id":"reasoner-a","status":"cancelled","error":"cancelled during graceful shutdown"}`)),
		httptest.NewRequest(http.MethodPost, "/executions/dual-signal-root/status", strings.NewReader(`{"status":"cancelled","error":"cancelled during graceful shutdown"}`)),
	}
	for _, request := range requests {
		request.Header.Set("Content-Type", "application/json")
	}
	responses := []*httptest.ResponseRecorder{httptest.NewRecorder(), httptest.NewRecorder()}
	start := make(chan struct{})
	var requestsDone sync.WaitGroup
	for index := range requests {
		requestsDone.Add(1)
		go func(index int) {
			defer requestsDone.Done()
			<-start
			router.ServeHTTP(responses[index], requests[index])
		}(index)
	}
	close(start)
	requestsDone.Wait()
	for _, response := range responses {
		require.Equal(t, http.StatusOK, response.Code, response.Body.String())
	}
	_, claimed := interruptedHandoffClaims.Load("dual-signal-root")
	require.True(t, claimed, "one trigger must hold the claim across the delay")

	var successorID string
	require.Eventually(t, func() bool {
		source, _ := store.GetExecutionRecord(t.Context(), "dual-signal-root")
		if source == nil || source.RestartedAsExecutionID == nil {
			return false
		}
		successorID = *source.RestartedAsExecutionID
		return successorID != ""
	}, time.Second, 10*time.Millisecond)
	require.Eventually(t, func() bool {
		successor, _ := store.GetExecutionRecord(t.Context(), successorID)
		return successor != nil && successor.Status == types.ExecutionStatusSucceeded
	}, time.Second, 10*time.Millisecond)
	require.Never(t, func() bool { return agentCalls.Load() > 1 }, 100*time.Millisecond, 10*time.Millisecond)

	records, err := store.QueryExecutionRecords(t.Context(), types.ExecutionFilter{})
	require.NoError(t, err)
	require.Len(t, records, 2)
	require.EqualValues(t, 1, agentCalls.Load())
	require.Eventually(t, func() bool {
		_, stillClaimed := interruptedHandoffClaims.Load("dual-signal-root")
		return !stillClaimed
	}, time.Second, 10*time.Millisecond)
}

func TestScheduledHandoffRechecksFreshExecutionAfterDelay(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*types.Execution)
	}{
		{
			name: "completed by late callback",
			mutate: func(execution *types.Execution) {
				execution.Status = types.ExecutionStatusSucceeded
				execution.StatusReason = nil
			},
		},
		{
			name: "manually restarted",
			mutate: func(execution *types.Execution) {
				successorID := "manual-successor"
				execution.RestartedAsExecutionID = &successorID
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			withResumeInterruptedSettings(t, true, 1, 25, time.Hour, 100*time.Millisecond)
			store := newTestExecutionStorage(nil)
			reason := types.ExecutionReasonAgentShutdownCancelled
			executionID := "fresh-guard-" + strings.ReplaceAll(test.name, " ", "-")
			now := time.Now().UTC()
			seedExecutionRecord(t, store, &types.Execution{
				ExecutionID: executionID, RunID: executionID + "-run",
				AgentNodeID: "node-1", NodeID: "node-1", ReasonerID: "reasoner-a",
				Status: types.ExecutionStatusCancelled, StatusReason: &reason,
				InputPayload: json.RawMessage(`{"input":{}}`),
				StartedAt:    now, CreatedAt: now, UpdatedAt: now,
			})
			controller := newExecutionController(store, nil, nil, time.Second, "")
			candidate, err := store.GetExecutionRecord(t.Context(), executionID)
			require.NoError(t, err)
			controller.scheduleInterruptedRunHandoff(candidate)
			_, claimed := interruptedHandoffClaims.Load(executionID)
			require.True(t, claimed)

			_, err = store.UpdateExecutionRecord(t.Context(), executionID, func(current *types.Execution) (*types.Execution, error) {
				test.mutate(current)
				return current, nil
			})
			require.NoError(t, err)
			require.Eventually(t, func() bool {
				_, stillClaimed := interruptedHandoffClaims.Load(executionID)
				return !stillClaimed
			}, time.Second, 10*time.Millisecond)

			records, err := store.QueryExecutionRecords(t.Context(), types.ExecutionFilter{})
			require.NoError(t, err)
			require.Len(t, records, 1, "a changed execution must not gain an automatic successor")
		})
	}
}

func TestAgentRestartHandoffCreatesOneSuccessorForRootNotChildren(t *testing.T) {
	withResumeInterruptedSettings(t, true, 1, 25, time.Hour, 100*time.Millisecond)
	withTestAsyncPool(t)
	var calls atomic.Int32
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer agentServer.Close()
	store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	now := time.Now().UTC()
	reason := "agent_restart_orphaned: old instance gone"
	root := &types.Execution{ExecutionID: "reaped-root", RunID: "reaped-run", AgentNodeID: "node-1", InstanceID: "old-instance", NodeID: "node-1", ReasonerID: "reasoner-a", Status: types.ExecutionStatusFailed, StatusReason: &reason, InputPayload: json.RawMessage(`{"input":{}}`), StartedAt: now.Add(-time.Minute), CreatedAt: now, UpdatedAt: now}
	seedExecutionRecord(t, store, root)
	for i := 0; i < 5; i++ {
		parent := root.ExecutionID
		seedExecutionRecord(t, store, &types.Execution{ExecutionID: "reaped-child-" + string(rune('a'+i)), RunID: root.RunID, ParentExecutionID: &parent, AgentNodeID: "node-1", InstanceID: "old-instance", NodeID: "node-1", ReasonerID: "reasoner-b", Status: types.ExecutionStatusFailed, StatusReason: &reason, InputPayload: json.RawMessage(`{"input":{}}`), StartedAt: now, CreatedAt: now, UpdatedAt: now})
	}
	controller := newExecutionController(store, services.NopPayloadStore{}, nil, time.Second, "")
	controller.handOffInterruptedAgentExecutions(t.Context(), "node-1", "old-instance")
	updatedRoot, err := store.GetExecutionRecord(t.Context(), root.ExecutionID)
	require.NoError(t, err)
	require.Nil(t, updatedRoot.RestartedAsExecutionID, "post-reap handoff must wait for the replacement")
	_, claimed := interruptedHandoffClaims.Load(root.ExecutionID)
	require.True(t, claimed)
	require.Eventually(t, func() bool {
		updatedRoot, err = store.GetExecutionRecord(t.Context(), root.ExecutionID)
		return err == nil && updatedRoot != nil && updatedRoot.RestartedAsExecutionID != nil
	}, time.Second, 10*time.Millisecond)
	for i := 0; i < 5; i++ {
		child, err := store.GetExecutionRecord(t.Context(), "reaped-child-"+string(rune('a'+i)))
		require.NoError(t, err)
		require.Nil(t, child.RestartedAsExecutionID)
	}
	require.Eventually(t, func() bool { return calls.Load() == 1 }, time.Second, 10*time.Millisecond)
	require.Eventually(t, func() bool {
		completed, _ := store.GetExecutionRecord(t.Context(), *updatedRoot.RestartedAsExecutionID)
		return completed != nil && completed.Status == types.ExecutionStatusSucceeded
	}, time.Second, 10*time.Millisecond)
}

func TestManualWorkflowRestartStampsSourceAndRootForwardPointers(t *testing.T) {
	withResumeInterruptedSettings(t, false, 1, 25, time.Hour, 0)
	withTestAsyncPool(t)
	agentServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer agentServer.Close()
	store := newTestExecutionStorage(testRestartAgent(agentServer.URL))
	now := time.Now().UTC()
	root := &types.Execution{ExecutionID: "manual-root", RunID: "manual-run", AgentNodeID: "node-1", NodeID: "node-1", ReasonerID: "reasoner-a", Status: types.ExecutionStatusFailed, InputPayload: json.RawMessage(`{"input":{}}`), StartedAt: now.Add(-time.Minute), CreatedAt: now.Add(-time.Minute), UpdatedAt: now}
	parentID := root.ExecutionID
	child := &types.Execution{ExecutionID: "manual-child", RunID: root.RunID, ParentExecutionID: &parentID, AgentNodeID: "node-1", NodeID: "node-1", ReasonerID: "reasoner-b", Status: types.ExecutionStatusFailed, InputPayload: json.RawMessage(`{"input":{}}`), StartedAt: now, CreatedAt: now, UpdatedAt: now}
	seedExecutionRecord(t, store, root)
	seedExecutionRecord(t, store, child)
	require.NoError(t, store.UpdateWorkflowRunMetadata(t.Context(), root.RunID, func(metadata map[string]json.RawMessage) error {
		metadata["lineage"] = json.RawMessage(`{"kind":"original","source_run_id":"older-run"}`)
		return nil
	}))

	router := gin.New()
	router.POST("/executions/:execution_id/restart", RestartExecutionHandler(store, services.NopPayloadStore{}, nil, time.Second, ""))
	request := httptest.NewRequest(http.MethodPost, "/executions/manual-child/restart", strings.NewReader(`{"scope":"workflow"}`))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	require.Equal(t, http.StatusAccepted, response.Code)
	var restarted restartExecutionResponse
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &restarted))
	for _, sourceID := range []string{root.ExecutionID, child.ExecutionID} {
		record, err := store.GetExecutionRecord(t.Context(), sourceID)
		require.NoError(t, err)
		require.NotNil(t, record.RestartedAsExecutionID)
		require.Equal(t, restarted.ExecutionID, *record.RestartedAsExecutionID)
	}
	sourceRun, err := store.GetWorkflowRun(t.Context(), root.RunID)
	require.NoError(t, err)
	require.NotNil(t, sourceRun)
	var namespaces map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(sourceRun.Metadata, &namespaces))
	var lineage struct {
		Kind        string `json:"kind"`
		SourceRunID string `json:"source_run_id"`
		RestartedAs struct {
			RestartedAsRef
			At string `json:"at"`
		} `json:"restarted_as"`
	}
	require.NoError(t, json.Unmarshal(namespaces["lineage"], &lineage))
	require.Equal(t, "original", lineage.Kind)
	require.Equal(t, "older-run", lineage.SourceRunID)
	require.Equal(t, restarted.ExecutionID, lineage.RestartedAs.ExecutionID)
	require.Equal(t, restarted.RunID, lineage.RestartedAs.RunID)
	require.NotEmpty(t, lineage.RestartedAs.At)
	require.Eventually(t, func() bool {
		completed, _ := store.GetExecutionRecord(t.Context(), restarted.ExecutionID)
		return completed != nil && completed.Status == types.ExecutionStatusSucceeded
	}, time.Second, 10*time.Millisecond)
}

func TestHandOffInterruptedRunGuardsReasonRootPointerAndAttemptBound(t *testing.T) {
	withResumeInterruptedSettings(t, true, 1, 25, time.Hour, 0)
	store := newTestExecutionStorage(nil)
	controller := newExecutionController(store, nil, nil, time.Second, "")
	orphan := "agent_restart_orphaned: old pod gone"
	other := "agent_error"
	parent := "parent"
	successor := "already-restarted"

	tests := []struct {
		name string
		exec *types.Execution
	}{
		{name: "unrelated failure", exec: &types.Execution{ExecutionID: "guard-other", RunID: "guard-run-other", StatusReason: &other}},
		{name: "child", exec: &types.Execution{ExecutionID: "guard-child", RunID: "guard-run-child", ParentExecutionID: &parent, StatusReason: &orphan}},
		{name: "already restarted", exec: &types.Execution{ExecutionID: "guard-pointer", RunID: "guard-run-pointer", RestartedAsExecutionID: &successor, StatusReason: &orphan}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, ok := controller.handOffInterruptedRun(t.Context(), test.exec)
			require.False(t, ok)
		})
	}

	runID := "guard-max-attempt-run"
	maxAttempt := &types.Execution{
		ExecutionID: "guard-max-attempt", RunID: runID,
		Status: types.ExecutionStatusFailed, StatusReason: &orphan,
	}
	seedExecutionRecord(t, store, maxAttempt)
	require.NoError(t, store.UpdateWorkflowRunMetadata(t.Context(), runID, func(metadata map[string]json.RawMessage) error {
		metadata["lineage"] = json.RawMessage(`{"kind":"resume","resume_attempt":1}`)
		return nil
	}))
	_, ok := controller.handOffInterruptedRun(t.Context(), maxAttempt)
	require.False(t, ok)
}

type startupQueryCountingStore struct {
	*testExecutionStorage
	queries atomic.Int32
	last    types.ExecutionFilter
}

type executionReadCountingStore struct {
	*testExecutionStorage
	reads atomic.Int32
}

func (s *executionReadCountingStore) GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error) {
	s.reads.Add(1)
	return s.testExecutionStorage.GetExecutionRecord(ctx, executionID)
}

func TestExecutionStatusHealthyReadDoesNotResolveSuccessor(t *testing.T) {
	base := newTestExecutionStorage(nil)
	now := time.Now().UTC()
	require.NoError(t, base.CreateExecutionRecord(t.Context(), &types.Execution{
		ExecutionID: "healthy-poll", RunID: "healthy-run", Status: types.ExecutionStatusRunning,
		StartedAt: now, CreatedAt: now, UpdatedAt: now,
	}))
	store := &executionReadCountingStore{testExecutionStorage: base}
	router := gin.New()
	router.GET("/executions/:execution_id", GetExecutionStatusHandler(store))
	response := httptest.NewRecorder()
	router.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/executions/healthy-poll", nil))
	require.Equal(t, http.StatusOK, response.Code)
	require.EqualValues(t, 1, store.reads.Load(), "healthy polling must not perform a successor lookup")
}

func TestExecutionStatusOmitsUnresolvableRestartedAs(t *testing.T) {
	now := time.Now().UTC()
	missingSuccessor := "cleaned-up-successor"
	exec := &types.Execution{
		ExecutionID: "source-with-stale-pointer", RunID: "source-run",
		Status: types.ExecutionStatusFailed, RestartedAsExecutionID: &missingSuccessor,
		StartedAt: now, CreatedAt: now, UpdatedAt: now,
	}
	base := renderStatus(exec)
	require.Nil(t, base.RestartedAs, "the store-free renderer must never expose an unverified pointer")

	store := newTestExecutionStorage(nil)
	require.NoError(t, store.CreateExecutionRecord(t.Context(), exec))
	router := gin.New()
	router.GET("/executions/:execution_id", GetExecutionStatusHandler(store))
	response := httptest.NewRecorder()
	router.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/executions/"+exec.ExecutionID, nil))
	require.Equal(t, http.StatusOK, response.Code)
	var status ExecutionStatusResponse
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &status))
	require.Nil(t, status.RestartedAs, "a pointer to a deleted successor must be omitted")

	errStore := &executionRecordLookupErrorStore{testExecutionStorage: store, getErr: errors.New("lookup failed")}
	errorStatus := newExecutionController(errStore, nil, nil, 0, "").renderStatusWithApproval(t.Context(), exec)
	require.Nil(t, errorStatus.RestartedAs, "a pointer whose lookup fails must be omitted")
}

func (s *startupQueryCountingStore) QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
	s.queries.Add(1)
	s.last = filter
	return nil, nil
}

func TestResumeInterruptedRunsOnStartupIsDisabledWithoutQueryAndUsesBoundsWhenEnabled(t *testing.T) {
	store := &startupQueryCountingStore{testExecutionStorage: newTestExecutionStorage(nil)}
	withResumeInterruptedSettings(t, false, 1, 2, 30*time.Minute, 0)
	ResumeInterruptedRunsOnStartup(t.Context(), store, nil, nil, time.Second, "")
	require.Zero(t, store.queries.Load())

	SetResumeInterruptedRuns(true)
	ResumeInterruptedRunsOnStartup(t.Context(), store, nil, nil, time.Second, "")
	require.EqualValues(t, 1, store.queries.Load())
	require.Equal(t, 2, store.last.Limit)
	require.True(t, store.last.TerminalOnly)
	require.True(t, store.last.RootOnly)
	require.True(t, store.last.WithoutRestartedAs)
	require.Equal(t, "updated_at", store.last.SortBy)
	require.False(t, store.last.SortDescending)
	require.WithinDuration(t, time.Now().UTC().Add(-30*time.Minute), *store.last.UpdatedAfter, 2*time.Second)
}
