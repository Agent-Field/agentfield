package ui

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

type countingFlusher struct {
	count int
}

func (f *countingFlusher) Flush() {
	f.count++
}

func TestStreamNDJSONWithIdleCoverage(t *testing.T) {
	t.Run("copies lines and flushes", func(t *testing.T) {
		var buf bytes.Buffer
		flusher := &countingFlusher{}

		err := streamNDJSONWithIdle(&buf, strings.NewReader("{\"a\":1}\n{\"b\":2}\n"), time.Second, flusher, context.Background())
		require.NoError(t, err)
		require.Equal(t, "{\"a\":1}\n{\"b\":2}\n", buf.String())
		require.Equal(t, 2, flusher.count)
	})

	t.Run("returns scanner error for oversized line", func(t *testing.T) {
		var buf bytes.Buffer
		tooLong := strings.Repeat("x", (2<<20)+1)
		err := streamNDJSONWithIdle(&buf, strings.NewReader(tooLong), time.Second, nil, context.Background())
		require.Error(t, err)
	})

	t.Run("returns idle timeout when reader is silent", func(t *testing.T) {
		reader, writer := io.Pipe()
		defer writer.Close()

		err := streamNDJSONWithIdle(io.Discard, reader, 15*time.Millisecond, nil, context.Background())
		require.ErrorIs(t, err, context.DeadlineExceeded)
	})

	t.Run("returns context cancellation", func(t *testing.T) {
		reader, writer := io.Pipe()
		ctx, cancel := context.WithCancel(context.Background())
		go func() {
			time.Sleep(10 * time.Millisecond)
			cancel()
			_ = writer.Close()
		}()

		err := streamNDJSONWithIdle(io.Discard, reader, time.Second, nil, ctx)
		require.ErrorIs(t, err, context.Canceled)
	})
}

func TestNodeLogsProxyHandlerCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	proxyCfg := config.NodeLogProxyConfig{
		MaxTailLines:      3,
		ConnectTimeout:    50 * time.Millisecond,
		StreamIdleTimeout: 20 * time.Millisecond,
		MaxStreamDuration: 200 * time.Millisecond,
	}

	t.Run("validates configuration and request", func(t *testing.T) {
		handler := &NodeLogsProxyHandler{}
		rec := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodGet, "/api/ui/v1/nodes/node-1/logs", nil)
		ctx.Params = gin.Params{{Key: "nodeId", Value: "node-1"}}
		handler.ProxyNodeLogsHandler(ctx)
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		store := &overrideStorage{StorageProvider: setupTestStorage(t)}
		handler = &NodeLogsProxyHandler{
			Storage: store,
			Snapshot: func() (config.NodeLogProxyConfig, string) {
				return proxyCfg, ""
			},
		}

		rec = httptest.NewRecorder()
		ctx, _ = gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodGet, "/api/ui/v1/nodes//logs", nil)
		handler.ProxyNodeLogsHandler(ctx)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		router := newTestUIRouter(func(r *gin.Engine) {
			r.GET("/api/ui/v1/nodes/:nodeId/logs", handler.ProxyNodeLogsHandler)
		})

		recorder := performJSONRequest(router, http.MethodGet, "/api/ui/v1/nodes/missing/logs", nil)
		require.Equal(t, http.StatusNotFound, recorder.Code)

		ctxStore := context.Background()
		require.NoError(t, store.RegisterAgent(ctxStore, &types.AgentNode{ID: "empty"}))
		recorder = performJSONRequest(router, http.MethodGet, "/api/ui/v1/nodes/empty/logs", nil)
		require.Equal(t, http.StatusBadGateway, recorder.Code)

		require.NoError(t, store.RegisterAgent(ctxStore, &types.AgentNode{ID: "invalid", BaseURL: "://bad"}))
		recorder = performJSONRequest(router, http.MethodGet, "/api/ui/v1/nodes/invalid/logs", nil)
		require.Equal(t, http.StatusBadRequest, recorder.Code)

		recorder = performJSONRequest(router, http.MethodGet, "/api/ui/v1/nodes/invalid/logs?tail_lines=nope", nil)
		require.Equal(t, http.StatusBadRequest, recorder.Code)

		require.NoError(t, store.RegisterAgent(ctxStore, &types.AgentNode{ID: "too-many", BaseURL: "http://127.0.0.1:1"}))
		recorder = performJSONRequest(router, http.MethodGet, "/api/ui/v1/nodes/too-many/logs?tail_lines=4", nil)
		require.Equal(t, http.StatusBadRequest, recorder.Code)
	})

	t.Run("proxies upstream statuses and bodies", func(t *testing.T) {
		for _, tc := range []struct {
			name         string
			status       int
			body         string
			expectStatus int
			expectBody   string
		}{
			{name: "unauthorized passthrough", status: http.StatusUnauthorized, body: "denied", expectStatus: http.StatusUnauthorized, expectBody: "denied"},
			{name: "bad gateway on unexpected status", status: http.StatusTeapot, body: "nope", expectStatus: http.StatusBadGateway},
			{name: "copies ndjson body", status: http.StatusOK, body: "{\"msg\":1}\n{\"msg\":2}\n", expectStatus: http.StatusOK, expectBody: "{\"msg\":1}\n{\"msg\":2}\n"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(tc.status)
					_, _ = w.Write([]byte(tc.body))
				}))
				defer upstream.Close()

				store := &overrideStorage{StorageProvider: setupTestStorage(t)}
				require.NoError(t, store.RegisterAgent(context.Background(), &types.AgentNode{
					ID:      "node-1",
					BaseURL: upstream.URL,
				}))

				handler := &NodeLogsProxyHandler{
					Storage: store,
					Snapshot: func() (config.NodeLogProxyConfig, string) {
						return proxyCfg, "token-123"
					},
				}
				router := newTestUIRouter(func(r *gin.Engine) {
					r.GET("/api/ui/v1/nodes/:nodeId/logs", handler.ProxyNodeLogsHandler)
				})

				rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/nodes/node-1/logs?tail_lines=2", nil)
				require.Equal(t, tc.expectStatus, rec.Code)
				if tc.expectBody != "" {
					require.Equal(t, tc.expectBody, rec.Body.String())
				}
			})
		}
	})

	t.Run("streams follow mode until idle timeout", func(t *testing.T) {
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			require.Equal(t, "Bearer token-123", r.Header.Get("Authorization"))
			w.Header().Set("Content-Type", "application/x-ndjson")
			_, _ = w.Write([]byte("{\"line\":1}\n"))
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
			time.Sleep(50 * time.Millisecond)
		}))
		defer upstream.Close()

		store := &overrideStorage{StorageProvider: setupTestStorage(t)}
		require.NoError(t, store.RegisterAgent(context.Background(), &types.AgentNode{
			ID:      "node-1",
			BaseURL: upstream.URL,
		}))

		handler := &NodeLogsProxyHandler{
			Storage: store,
			Snapshot: func() (config.NodeLogProxyConfig, string) {
				return proxyCfg, "token-123"
			},
		}
		router := newTestUIRouter(func(r *gin.Engine) {
			r.GET("/api/ui/v1/nodes/:nodeId/logs", handler.ProxyNodeLogsHandler)
		})

		rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/nodes/node-1/logs?follow=yes", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		require.Contains(t, rec.Body.String(), "{\"line\":1}\n")
		require.Equal(t, "application/x-ndjson", rec.Header().Get("Content-Type"))
	})
}

func TestExecutionAndNodeStreamCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("execution log stream validates request and terminal status", func(t *testing.T) {
		handler := NewExecutionLogsHandler(nil, nil, func() config.ExecutionLogsConfig {
			return config.ExecutionLogsConfig{MaxTailEntries: 2}
		})
		rec := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/exec-1/logs/stream", nil)
		ctx.Params = gin.Params{{Key: "execution_id", Value: "exec-1"}}
		handler.StreamExecutionLogsHandler(ctx)
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		store, storeCtx := setupExecutionLogsStorage(t)
		require.NoError(t, store.StoreExecutionLogEntry(storeCtx, &types.ExecutionLogEntry{
			ExecutionID: "exec-terminal",
			WorkflowID:  "wf-1",
			Sequence:    1,
			AgentNodeID: "node-1",
			Level:       "info",
			Source:      "sdk",
			Message:     "started",
			EmittedAt:   time.Now().UTC(),
		}))
		require.NoError(t, store.StoreWorkflowExecution(storeCtx, &types.WorkflowExecution{
			WorkflowID:  "wf-1",
			ExecutionID: "exec-terminal",
			Status:      string(types.ExecutionStatusSucceeded),
			CreatedAt:   time.Now().UTC(),
			UpdatedAt:   time.Now().UTC(),
		}))

		handler = NewExecutionLogsHandler(store, nil, func() config.ExecutionLogsConfig {
			return config.ExecutionLogsConfig{
				MaxTailEntries:     5,
				StreamIdleTimeout:  20 * time.Millisecond,
				MaxStreamDuration:  100 * time.Millisecond,
			}
		})
		router := newTestUIRouter(func(r *gin.Engine) {
			r.GET("/api/ui/v1/executions/:execution_id/logs/stream", handler.StreamExecutionLogsHandler)
		})

		recorder := performJSONRequest(router, http.MethodGet, "/api/ui/v1/executions/exec-terminal/logs/stream?tail=1", nil)
		require.Equal(t, http.StatusOK, recorder.Code)
		require.Contains(t, recorder.Body.String(), `"message":"started"`)
		require.Contains(t, recorder.Body.String(), `"type":"connected"`)
		require.Contains(t, recorder.Body.String(), `"reason":"terminal_status"`)

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/executions/exec-terminal/logs/stream?tail=nope", nil)
		require.Equal(t, http.StatusBadRequest, rec.Code)
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/executions/exec-terminal/logs/stream?since_seq=-1", nil)
		require.Equal(t, http.StatusBadRequest, rec.Code)
	})

	t.Run("execution log stream delivers matching live entries", func(t *testing.T) {
		store, storeCtx := setupExecutionLogsStorage(t)
		now := time.Now().UTC()
		require.NoError(t, store.StoreWorkflowExecution(storeCtx, &types.WorkflowExecution{
			WorkflowID:  "wf-live",
			ExecutionID: "exec-live",
			Status:      string(types.ExecutionStatusRunning),
			CreatedAt:   now,
			UpdatedAt:   now,
		}))

		handler := NewExecutionLogsHandler(store, nil, func() config.ExecutionLogsConfig {
			return config.ExecutionLogsConfig{
				MaxTailEntries:     5,
				StreamIdleTimeout:  35 * time.Millisecond,
				MaxStreamDuration:  250 * time.Millisecond,
			}
		})
		router := newTestUIRouter(func(r *gin.Engine) {
			r.GET("/api/ui/v1/executions/:execution_id/logs/stream", handler.StreamExecutionLogsHandler)
		})

		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/exec-live/logs/stream?since_seq=2", nil)
		resp := httptest.NewRecorder()
		done := make(chan struct{})
		go func() {
			router.ServeHTTP(resp, req)
			close(done)
		}()

		time.Sleep(20 * time.Millisecond)
		store.GetExecutionLogEventBus().Publish(&types.ExecutionLogEntry{
			ExecutionID: "exec-other",
			WorkflowID:  "wf-live",
			Sequence:    3,
			AgentNodeID: "node-1",
			Level:       "info",
			Source:      "sdk",
			Message:     "wrong execution",
			EmittedAt:   now,
		})
		store.GetExecutionLogEventBus().Publish(&types.ExecutionLogEntry{
			ExecutionID: "exec-live",
			WorkflowID:  "wf-live",
			Sequence:    2,
			AgentNodeID: "node-1",
			Level:       "info",
			Source:      "sdk",
			Message:     "old sequence",
			EmittedAt:   now,
		})
		store.GetExecutionLogEventBus().Publish(&types.ExecutionLogEntry{
			ExecutionID: "exec-live",
			WorkflowID:  "wf-live",
			Sequence:    3,
			AgentNodeID: "node-1",
			Level:       "info",
			Source:      "sdk",
			Message:     "delivered",
			EmittedAt:   now,
		})

		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatal("stream did not finish")
		}

		body := resp.Body.String()
		require.Contains(t, body, `"type":"connected"`)
		require.Contains(t, body, `"message":"delivered"`)
		require.NotContains(t, body, `"message":"wrong execution"`)
		require.NotContains(t, body, `"message":"old sequence"`)
	})

	t.Run("execution notes and node streams emit events", func(t *testing.T) {
		store := setupTestStorage(t)

		executionHandler := NewExecutionHandler(store, nil, nil)
		executionRouter := newTestUIRouter(func(r *gin.Engine) {
			r.GET("/api/ui/v1/workflows/:workflowId/notes/events", executionHandler.StreamWorkflowNodeNotesHandler)
		})

		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/workflows/wf-1/notes/events", nil)
		ctx, cancel := context.WithCancel(req.Context())
		req = req.WithContext(ctx)
		resp := httptest.NewRecorder()
		done := make(chan struct{})
		go func() {
			executionRouter.ServeHTTP(resp, req)
			close(done)
		}()

		time.Sleep(20 * time.Millisecond)
		store.GetExecutionEventBus().Publish(events.ExecutionEvent{
			Type:        events.ExecutionUpdated,
			ExecutionID: "exec-1",
			WorkflowID:  "wf-1",
			AgentNodeID: "node-1",
			Status:      "running",
			Timestamp:   time.Now().UTC(),
		})
		time.Sleep(20 * time.Millisecond)
		cancel()

		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatal("workflow notes stream did not finish")
		}
		require.Contains(t, resp.Body.String(), `"workflow_id":"wf-1"`)
		require.Contains(t, resp.Body.String(), `"execution_id":"exec-1"`)

		nodesHandler := NewNodesHandler(nil)
		nodeRouter := newTestUIRouter(func(r *gin.Engine) {
			r.GET("/api/ui/v1/nodes/events", nodesHandler.StreamNodeEventsHandler)
		})

		nodeReq := httptest.NewRequest(http.MethodGet, "/api/ui/v1/nodes/events", nil)
		nodeCtx, nodeCancel := context.WithCancel(nodeReq.Context())
		nodeReq = nodeReq.WithContext(nodeCtx)
		nodeResp := httptest.NewRecorder()
		nodeDone := make(chan struct{})
		go func() {
			nodeRouter.ServeHTTP(nodeResp, nodeReq)
			close(nodeDone)
		}()

		time.Sleep(20 * time.Millisecond)
		events.GlobalNodeEventBus.Publish(events.NodeEvent{
			Type:      events.NodeOnline,
			NodeID:    "node-1",
			Status:    "online",
			Timestamp: time.Now().UTC(),
		})
		time.Sleep(20 * time.Millisecond)
		nodeCancel()

		select {
		case <-nodeDone:
		case <-time.After(time.Second):
			t.Fatal("node stream did not finish")
		}
		require.Contains(t, nodeResp.Body.String(), `"message":"Node events stream connected"`)
		require.Contains(t, nodeResp.Body.String(), `"node_id":"node-1"`)
	})
}

func TestNodeRefreshAndTrendCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("refresh all node statuses covers error and success", func(t *testing.T) {
		nilService := services.NewUIService(setupTestStorage(t), nil, nil, nil)
		nilHandler := NewNodesHandler(nilService)
		rec := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodPost, "/api/ui/v1/nodes/status/refresh", nil)
		nilHandler.RefreshAllNodeStatusHandler(ctx)
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		ctxStore := context.Background()
		tempDir := t.TempDir()
		cfg := storage.StorageConfig{
			Mode: "local",
			Local: storage.LocalStorageConfig{
				DatabasePath: tempDir + "/test.db",
				KVStorePath:  tempDir + "/test.bolt",
			},
		}

		realStorage := storage.NewLocalStorage(storage.LocalStorageConfig{})
		err := realStorage.Initialize(ctxStore, cfg)
		if err != nil && strings.Contains(strings.ToLower(err.Error()), "fts5") {
			t.Skip("sqlite3 compiled without FTS5")
		}
		require.NoError(t, err)
		t.Cleanup(func() {
			_ = realStorage.Close(ctxStore)
		})

		require.NoError(t, realStorage.RegisterAgent(ctxStore, &types.AgentNode{ID: "node-1", BaseURL: "http://localhost:9999"}))

		mockClient := &MockAgentClientForUI{}
		mockAgentService := &MockAgentServiceForUI{}
		mockClient.On("GetAgentStatus", mock.Anything, "node-1").Return(nil, errors.New("offline"))

		statusManager := services.NewStatusManager(realStorage, services.StatusManagerConfig{}, nil, mockClient)
		handler := NewNodesHandler(services.NewUIService(realStorage, mockClient, mockAgentService, statusManager))
		router := newTestUIRouter(func(r *gin.Engine) {
			r.POST("/api/ui/v1/nodes/status/refresh", handler.RefreshAllNodeStatusHandler)
		})

		recorder := performJSONRequest(router, http.MethodPost, "/api/ui/v1/nodes/status/refresh", nil)
		require.Equal(t, http.StatusOK, recorder.Code)
		require.Contains(t, recorder.Body.String(), "node-1")
	})

	t.Run("builds execution trends directly", func(t *testing.T) {
		now := time.Date(2026, 4, 8, 12, 0, 0, 0, time.UTC)
		fast := int64(1000)
		slow := int64(2000)
		executions := []*types.Execution{
			{ExecutionID: "1", Status: string(types.ExecutionStatusSucceeded), StartedAt: now.Add(-time.Hour), DurationMS: &fast},
			{ExecutionID: "2", Status: string(types.ExecutionStatusFailed), StartedAt: now.Add(-2 * time.Hour), DurationMS: &slow},
			{ExecutionID: "3", Status: string(types.ExecutionStatusCancelled), StartedAt: now.AddDate(0, 0, -5)},
		}

		trends := buildExecutionTrends(now, executions)
		require.Len(t, trends.Last7Days, 7)
		require.Equal(t, 2, trends.Last24h.Total)
		require.Equal(t, 1, trends.Last24h.Succeeded)
		require.Equal(t, 1, trends.Last24h.Failed)
		require.Equal(t, 50.0, trends.Last24h.SuccessRate)
		require.Greater(t, trends.Last24h.AverageDurationMs, 0.0)
		require.Greater(t, trends.Last24h.ThroughputPerHour, 0.0)
	})

	t.Run("reasoner events and llm health enabled branch", func(t *testing.T) {
		reasonerHandler := NewReasonersHandler(nil)
		reasonerRouter := newTestUIRouter(func(r *gin.Engine) {
			r.GET("/api/ui/v1/reasoners/events", reasonerHandler.StreamReasonerEventsHandler)
		})

		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/reasoners/events", nil)
		ctx, cancel := context.WithCancel(req.Context())
		req = req.WithContext(ctx)
		resp := httptest.NewRecorder()
		done := make(chan struct{})
		go func() {
			reasonerRouter.ServeHTTP(resp, req)
			close(done)
		}()

		time.Sleep(20 * time.Millisecond)
		events.GlobalReasonerEventBus.Publish(events.ReasonerEvent{
			Type:       events.ReasonerOnline,
			ReasonerID: "node-1.plan",
			NodeID:     "node-1",
			Status:     "online",
			Timestamp:  time.Now().UTC(),
		})
		time.Sleep(20 * time.Millisecond)
		cancel()

		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatal("reasoner stream did not finish")
		}
		require.Contains(t, resp.Body.String(), `"message":"Reasoner events stream connected"`)
		require.Contains(t, resp.Body.String(), `"reasoner_id":"node-1.plan"`)

		monitor := services.NewLLMHealthMonitor(config.LLMHealthConfig{
			Enabled: true,
			Endpoints: []config.LLMEndpoint{
				{Name: "primary", URL: "http://llm.test/health"},
			},
		}, nil)
		llmHandler := NewExecutionLogsHandler(nil, monitor, func() config.ExecutionLogsConfig {
			return config.ExecutionLogsConfig{}
		})
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(http.MethodGet, "/api/ui/v1/llm/health", nil)
		llmHandler.GetLLMHealthHandler(c)
		require.Equal(t, http.StatusOK, rec.Code)
		require.Contains(t, rec.Body.String(), `"enabled":true`)
		require.Contains(t, rec.Body.String(), `"healthy":true`)
		require.Contains(t, rec.Body.String(), `"name":"primary"`)
	})
}
