package handlers

import (
	"context"
	"encoding/json"
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

func TestExecuteAsyncHandler_QueueSaturation(t *testing.T) {
	gin.SetMode(gin.TestMode)
	oldPool, oldOnce := asyncPool, asyncPoolOnce
	asyncPool = newAsyncWorkerPool(1, 1)
	asyncPoolOnce = sync.Once{}
	asyncPoolOnce.Do(func() {})
	defer func() { asyncPool, asyncPoolOnce = oldPool, oldOnce }()

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

	oldPool := asyncPool
	oldOnce := asyncPoolOnce
	asyncPool = newAsyncWorkerPool(1, 2)
	asyncPoolOnce = sync.Once{}
	asyncPoolOnce.Do(func() {})
	defer func() { asyncPool, asyncPoolOnce = oldPool, oldOnce }()

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
	oldPool, oldOnce := asyncPool, asyncPoolOnce
	asyncPool = newAsyncWorkerPool(0, 1)
	asyncPoolOnce = sync.Once{}
	asyncPoolOnce.Do(func() {})
	defer func() { asyncPool, asyncPoolOnce = oldPool, oldOnce }()

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
	oldPool, oldOnce := asyncPool, asyncPoolOnce
	asyncPool = newAsyncWorkerPool(0, 1)
	require.True(t, asyncPool.reserve())
	asyncPoolOnce = sync.Once{}
	asyncPoolOnce.Do(func() {})
	defer func() { asyncPool, asyncPoolOnce = oldPool, oldOnce }()

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
	records, err := store.QueryExecutionRecords(context.Background(), types.ExecutionFilter{})
	require.NoError(t, err)
	require.Empty(t, records)
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
		exec := &types.Execution{ExecutionID: id, RunID: id, NodeID: "node-1", AgentNodeID: "node-1", ReasonerID: "reasoner-a", Status: types.ExecutionStatusRunning, CreatedAt: now, StartedAt: now, UpdatedAt: now}
		require.NoError(t, store.CreateExecutionRecord(context.Background(), exec))
		require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{ExecutionID: id, WorkflowID: id, RunID: &id, AgentNodeID: "node-1", ReasonerID: "reasoner-a", Status: types.ExecutionStatusRunning, StartedAt: now, CreatedAt: now, UpdatedAt: now}))
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
