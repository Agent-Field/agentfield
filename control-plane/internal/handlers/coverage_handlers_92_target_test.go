package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type payloadStoreStub struct {
	record *services.PayloadRecord
	err    error
	saved  [][]byte
}

func (s *payloadStoreStub) SaveFromReader(_ context.Context, r io.Reader) (*services.PayloadRecord, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return s.SaveBytes(context.Background(), data)
}

func (s *payloadStoreStub) SaveBytes(_ context.Context, data []byte) (*services.PayloadRecord, error) {
	s.saved = append(s.saved, append([]byte(nil), data...))
	if s.err != nil {
		return nil, s.err
	}
	return s.record, nil
}

func (s *payloadStoreStub) Open(context.Context, string) (io.ReadCloser, error) {
	return io.NopCloser(strings.NewReader("")), nil
}

func (s *payloadStoreStub) Remove(context.Context, string) error {
	return nil
}

type executionVCStoreStub struct {
	storage.StorageProvider
	vcs []*types.ExecutionVCInfo
	err error
}

func (s *executionVCStoreStub) ListExecutionVCs(_ context.Context, _ types.VCFilters) ([]*types.ExecutionVCInfo, error) {
	if s.err != nil {
		return nil, s.err
	}
	return s.vcs, nil
}

type cancelHandlerErrorStorage struct {
	storage.StorageProvider
	exec               *types.Execution
	wfExec             *types.WorkflowExecution
	getExecErr         error
	getWorkflowErr     error
	updateExecErr      error
	updateWorkflowErr  error
	storeEventErr      error
	updatedExecStatus  string
	updatedReason      *string
	storedWorkflowEvent *types.WorkflowExecutionEvent
}

func (s *cancelHandlerErrorStorage) GetExecutionRecord(_ context.Context, _ string) (*types.Execution, error) {
	if s.getExecErr != nil {
		return nil, s.getExecErr
	}
	if s.exec == nil {
		return nil, nil
	}
	copy := *s.exec
	return &copy, nil
}

func (s *cancelHandlerErrorStorage) GetWorkflowExecution(_ context.Context, _ string) (*types.WorkflowExecution, error) {
	if s.getWorkflowErr != nil {
		return nil, s.getWorkflowErr
	}
	if s.wfExec == nil {
		return nil, nil
	}
	copy := *s.wfExec
	return &copy, nil
}

func (s *cancelHandlerErrorStorage) UpdateExecutionRecord(_ context.Context, _ string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error) {
	if s.updateExecErr != nil {
		return nil, s.updateExecErr
	}
	if s.exec == nil {
		return nil, errors.New("missing execution")
	}
	copy := *s.exec
	updated, err := update(&copy)
	if err != nil {
		return nil, err
	}
	if updated != nil {
		copy = *updated
	}
	s.exec = &copy
	s.updatedExecStatus = copy.Status
	s.updatedReason = copy.StatusReason
	out := copy
	return &out, nil
}

func (s *cancelHandlerErrorStorage) UpdateWorkflowExecution(_ context.Context, _ string, update func(*types.WorkflowExecution) (*types.WorkflowExecution, error)) error {
	if s.updateWorkflowErr != nil {
		return s.updateWorkflowErr
	}
	if s.wfExec == nil {
		return nil
	}
	copy := *s.wfExec
	updated, err := update(&copy)
	if err != nil {
		return err
	}
	if updated != nil {
		copy = *updated
	}
	s.wfExec = &copy
	return nil
}

func (s *cancelHandlerErrorStorage) StoreWorkflowExecutionEvent(_ context.Context, event *types.WorkflowExecutionEvent) error {
	s.storedWorkflowEvent = event
	return s.storeEventErr
}

type workflowEventStoreStub struct {
	storage.StorageProvider
	exec              *types.Execution
	getErr            error
	createErr         error
	updateErr         error
	storeWorkflowErr  error
	createdExec       *types.Execution
	storedWorkflow    *types.WorkflowExecution
	updatedExecution  *types.Execution
}

func (s *workflowEventStoreStub) GetExecutionRecord(_ context.Context, _ string) (*types.Execution, error) {
	if s.getErr != nil {
		return nil, s.getErr
	}
	if s.exec == nil {
		return nil, nil
	}
	copy := *s.exec
	return &copy, nil
}

func (s *workflowEventStoreStub) CreateExecutionRecord(_ context.Context, execution *types.Execution) error {
	s.createdExec = execution
	return s.createErr
}

func (s *workflowEventStoreStub) StoreWorkflowExecution(_ context.Context, execution *types.WorkflowExecution) error {
	s.storedWorkflow = execution
	return s.storeWorkflowErr
}

func (s *workflowEventStoreStub) UpdateExecutionRecord(_ context.Context, _ string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error) {
	if s.updateErr != nil {
		return nil, s.updateErr
	}
	copy := *s.exec
	updated, err := update(&copy)
	if err != nil {
		return nil, err
	}
	if updated != nil {
		copy = *updated
	}
	s.exec = &copy
	s.updatedExecution = &copy
	return &copy, nil
}

func TestAgentConcurrencyLimiter_MaxPerAgentNilAndConfigured(t *testing.T) {
	t.Parallel()

	limiter := &AgentConcurrencyLimiter{maxPerAgent: 7}
	require.Equal(t, 7, limiter.MaxPerAgent())

	var nilLimiter *AgentConcurrencyLimiter
	require.Equal(t, 0, nilLimiter.MaxPerAgent())
}

func TestExecutionCleanupService_StartStopBranches(t *testing.T) {
	t.Parallel()

	t.Run("disabled start is no-op", func(t *testing.T) {
		service := NewExecutionCleanupService(&cleanupStoreMock{}, config.ExecutionCleanupConfig{Enabled: false})
		require.NoError(t, service.Start(context.Background()))
		require.False(t, service.isRunning)
		require.NoError(t, service.Stop())
	})

	t.Run("stop channel branch exits cleanly", func(t *testing.T) {
		logBuffer := setupExecutionCleanupTestLogger(t)
		service := NewExecutionCleanupService(&cleanupStoreMock{}, config.ExecutionCleanupConfig{
			Enabled:         true,
			RetentionPeriod: time.Hour,
			CleanupInterval: time.Hour,
			BatchSize:       1,
		})

		ctx := context.Background()
		require.NoError(t, service.Start(ctx))
		require.NoError(t, service.Start(ctx))
		require.NoError(t, service.Stop())
		require.NoError(t, service.Stop())
		require.False(t, service.isRunning)
		assert.Contains(t, logBuffer.String(), "Execution cleanup loop stopped")
	})
}

func TestExecutionControllerSavePayload(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		store       services.PayloadStore
		data        []byte
		wantURI     *string
		wantSaves   int
	}{
		{
			name:      "nil store returns nil",
			data:      []byte("payload"),
			wantSaves: 0,
		},
		{
			name:      "empty payload is ignored",
			store:     &payloadStoreStub{record: &services.PayloadRecord{URI: "payload://unused"}},
			data:      nil,
			wantSaves: 0,
		},
		{
			name:      "save error is swallowed",
			store:     &payloadStoreStub{err: errors.New("disk full")},
			data:      []byte("payload"),
			wantSaves: 1,
		},
		{
			name:      "successful save returns uri",
			store:     &payloadStoreStub{record: &services.PayloadRecord{URI: "payload://saved"}},
			data:      []byte("payload"),
			wantURI:   testStringPtr("payload://saved"),
			wantSaves: 1,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			controller := &executionController{payloads: tc.store}
			got := controller.savePayload(context.Background(), tc.data)
			if tc.wantURI == nil {
				require.Nil(t, got)
			} else {
				require.NotNil(t, got)
				require.Equal(t, *tc.wantURI, *got)
			}

			stub, _ := tc.store.(*payloadStoreStub)
			if stub != nil {
				require.Len(t, stub.saved, tc.wantSaves)
			}
		})
	}
}

func TestLookupWorkflowIssuerDID(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		vcs  []*types.ExecutionVCInfo
		err  error
		want *string
	}{
		{name: "storage error", err: errors.New("boom")},
		{name: "no vcs"},
		{name: "nil vc entry", vcs: []*types.ExecutionVCInfo{nil}},
		{name: "blank issuer", vcs: []*types.ExecutionVCInfo{{IssuerDID: "   "}}},
		{name: "trimmed issuer", vcs: []*types.ExecutionVCInfo{{IssuerDID: " did:example:issuer "}}, want: testStringPtr("did:example:issuer")},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			store := &executionVCStoreStub{vcs: tc.vcs, err: tc.err}
			require.Equal(t, tc.want, lookupWorkflowIssuerDID(context.Background(), store, "wf-1"))
		})
	}
}

func TestIsLightweightRequestQueryVariants(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	tests := []struct {
		target string
		want   bool
	}{
		{target: "/dag?mode=lightweight", want: true},
		{target: "/dag?mode=LIGHTWEIGHT", want: true},
		{target: "/dag?lightweight=true", want: true},
		{target: "/dag?lightweight=1", want: true},
		{target: "/dag?lightweight=false", want: false},
		{target: "/dag", want: false},
	}

	for _, tc := range tests {
		t.Run(tc.target, func(t *testing.T) {
			rec := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(rec)
			c.Request = httptest.NewRequest(http.MethodGet, tc.target, nil)
			require.Equal(t, tc.want, isLightweightRequest(c))
		})
	}
}

func TestDiscoveryLoggingIncludesOptionalRequestID(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	logBuffer := setupExecutionCleanupTestLogger(t)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodGet, "/api/v1/discovery/capabilities", nil)
	c.Set("request_id", "req-123")

	reasoner := "research"
	skill := "web"
	health := types.HealthStatusActive
	logDiscoverySuccess(c, DiscoveryFilters{
		Format:          "yaml",
		ReasonerPattern: &reasoner,
		SkillPattern:    &skill,
		HealthStatus:    &health,
		Limit:           5,
		Offset:          2,
	}, DiscoveryResponse{
		TotalAgents:    1,
		TotalReasoners: 2,
		TotalSkills:    3,
	}, true, 25*time.Millisecond)
	logDiscoveryError(c, "compact", time.Millisecond, errors.New("bad request"))

	logs := logBuffer.String()
	assert.Contains(t, logs, "discovery request completed")
	assert.Contains(t, logs, "discovery request failed")
	assert.Contains(t, logs, `"request_id":"req-123"`)
	assert.Contains(t, logs, `"format":"json"`)
	assert.Contains(t, logs, `"format":"compact"`)
}

func TestNodeRESTHandlersAdditionalBranches(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	t.Run("status lease respects pending approval", func(t *testing.T) {
		store := &nodeRESTStorageStub{
			agent: &types.AgentNode{ID: "node-1", Version: "v1", LifecycleStatus: types.AgentStatusPendingApproval},
		}
		presence := services.NewPresenceManager(nil, services.PresenceManagerConfig{})

		router := gin.New()
		router.PUT("/nodes/:node_id/status", NodeStatusLeaseHandler(store, nil, presence, 0))

		req := httptest.NewRequest(http.MethodPut, "/nodes/node-1/status", strings.NewReader(`{"phase":"ready"}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		assert.Equal(t, "node-1", store.lastHeartbeatID)
		assert.Equal(t, "v1", store.lastVersion)
		assert.True(t, presence.HasLease("node-1"))

		var body map[string]any
		require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
		assert.Equal(t, float64(int(DefaultLeaseTTL.Seconds())), body["lease_seconds"])
	})

	t.Run("status lease validates health score before update", func(t *testing.T) {
		store := &nodeRESTStorageStub{
			agent: &types.AgentNode{ID: "node-2", Version: "v2"},
		}
		router := gin.New()
		router.PUT("/nodes/:node_id/status", NodeStatusLeaseHandler(store, nil, nil, time.Minute))

		req := httptest.NewRequest(http.MethodPut, "/nodes/node-2/status", strings.NewReader(`{"health_score":101}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusBadRequest, resp.Code)
		assert.Empty(t, store.heartbeats)
	})

	t.Run("action ack and claim refresh heartbeats", func(t *testing.T) {
		store := &nodeRESTStorageStub{
			agent: &types.AgentNode{ID: "node-3", Version: "v3"},
		}
		presence := services.NewPresenceManager(nil, services.PresenceManagerConfig{})

		router := gin.New()
		router.POST("/nodes/:node_id/actions/ack", NodeActionAckHandler(store, presence, 2*time.Minute))
		router.POST("/actions/claim", ClaimActionsHandler(store, presence, 3*time.Minute))

		ackReq := httptest.NewRequest(http.MethodPost, "/nodes/node-3/actions/ack", strings.NewReader(`{"action_id":"act-1","status":"RUNNING"}`))
		ackReq.Header.Set("Content-Type", "application/json")
		ackResp := httptest.NewRecorder()
		router.ServeHTTP(ackResp, ackReq)
		require.Equal(t, http.StatusOK, ackResp.Code)

		claimReq := httptest.NewRequest(http.MethodPost, "/actions/claim", strings.NewReader(`{"node_id":"node-3","wait_seconds":0}`))
		claimReq.Header.Set("Content-Type", "application/json")
		claimResp := httptest.NewRecorder()
		router.ServeHTTP(claimResp, claimReq)
		require.Equal(t, http.StatusOK, claimResp.Code)

		require.Len(t, store.heartbeats, 2)
		assert.True(t, presence.HasLease("node-3"))

		var body map[string]any
		require.NoError(t, json.Unmarshal(claimResp.Body.Bytes(), &body))
		assert.Equal(t, float64(5), body["next_poll_after"])
	})

	t.Run("shutdown uses version lookup and clears presence", func(t *testing.T) {
		store := &nodeRESTStorageStub{
			versionedAgent: &types.AgentNode{ID: "node-4", Version: "v4"},
		}
		presence := services.NewPresenceManager(nil, services.PresenceManagerConfig{})
		presence.Touch("node-4", "v4", time.Now())

		router := gin.New()
		router.POST("/nodes/:node_id/shutdown", NodeShutdownHandler(store, nil, presence))

		req := httptest.NewRequest(http.MethodPost, "/nodes/node-4/shutdown", strings.NewReader(`{"version":"v4"}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusAccepted, resp.Code)
		assert.Equal(t, "node-4", store.lastHeartbeatID)
		assert.Equal(t, "v4", store.lastVersion)
		assert.False(t, presence.HasLease("node-4"))
	})
}

func TestCancelExecutionHandlerAdditionalErrorPaths(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	t.Run("invalid json returns bad request", func(t *testing.T) {
		router := gin.New()
		router.POST("/executions/:execution_id/cancel", CancelExecutionHandler(&cancelHandlerErrorStorage{}))

		req := httptest.NewRequest(http.MethodPost, "/executions/exec-1/cancel", bytes.NewBufferString("{"))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusBadRequest, resp.Code)
		assert.Contains(t, resp.Body.String(), "invalid request body")
	})

	t.Run("execution lookup failure returns internal error", func(t *testing.T) {
		router := gin.New()
		router.POST("/executions/:execution_id/cancel", CancelExecutionHandler(&cancelHandlerErrorStorage{
			getExecErr: errors.New("db down"),
		}))

		req := httptest.NewRequest(http.MethodPost, "/executions/exec-2/cancel", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusInternalServerError, resp.Code)
		assert.Contains(t, resp.Body.String(), "failed to look up execution")
	})

	t.Run("workflow lookup and event store failures are non fatal", func(t *testing.T) {
		exec := &types.Execution{
			ExecutionID: "exec-3",
			RunID:       "run-3",
			AgentNodeID: "agent-3",
			Status:      types.ExecutionStatusRunning,
			StartedAt:   time.Now().UTC(),
		}
		store := &cancelHandlerErrorStorage{
			exec:           exec,
			getWorkflowErr: errors.New("workflow table offline"),
			storeEventErr:  errors.New("event write failed"),
		}

		router := gin.New()
		router.POST("/executions/:execution_id/cancel", CancelExecutionHandler(store))

		req := httptest.NewRequest(http.MethodPost, "/executions/exec-3/cancel", strings.NewReader(`{"reason":"  operator stop  "}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		require.Equal(t, types.ExecutionStatusCancelled, store.updatedExecStatus)
		require.NotNil(t, store.updatedReason)
		require.Equal(t, "operator stop", *store.updatedReason)
		require.NotNil(t, store.storedWorkflowEvent)
		assert.Equal(t, "run-3", store.storedWorkflowEvent.WorkflowID)
		require.NotNil(t, store.storedWorkflowEvent.RunID)
		assert.Equal(t, "run-3", *store.storedWorkflowEvent.RunID)
	})
}

func TestCleanupWorkflowHandlerAdditionalBranches(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	t.Run("missing workflow id is rejected", func(t *testing.T) {
		store := &cleanupStorageStub{}
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(http.MethodDelete, "/cleanup?confirm=true", nil)

		CleanupWorkflowHandler(store)(c)

		require.Equal(t, http.StatusBadRequest, rec.Code)
		assert.Contains(t, rec.Body.String(), "workflow_id_required")
	})

	t.Run("unsuccessful result without error returns 500 response", func(t *testing.T) {
		store := &cleanupStorageStub{
			result: &types.WorkflowCleanupResult{
				WorkflowID: "wf-500",
				Success:    false,
				DryRun:     false,
			},
		}

		router := gin.New()
		router.DELETE("/workflows/:workflow_id/cleanup", CleanupWorkflowHandler(store))

		req := httptest.NewRequest(http.MethodDelete, "/workflows/wf-500/cleanup?confirm=true", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusInternalServerError, resp.Code)
		assert.Contains(t, resp.Body.String(), `"workflow_id":"wf-500"`)
	})
}

func TestWorkflowExecutionEventHandlerAdditionalBranches(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	t.Run("invalid payload returns 400", func(t *testing.T) {
		router := gin.New()
		router.POST("/events", WorkflowExecutionEventHandler(&workflowEventStoreStub{}))

		req := httptest.NewRequest(http.MethodPost, "/events", bytes.NewBufferString("{"))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusBadRequest, resp.Code)
	})

	t.Run("load failure returns 500", func(t *testing.T) {
		router := gin.New()
		router.POST("/events", WorkflowExecutionEventHandler(&workflowEventStoreStub{getErr: errors.New("load failed")}))

		req := httptest.NewRequest(http.MethodPost, "/events", strings.NewReader(`{"execution_id":"exec-1"}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusInternalServerError, resp.Code)
		assert.Contains(t, resp.Body.String(), "failed to load execution")
	})

	t.Run("create failure returns 500", func(t *testing.T) {
		router := gin.New()
		store := &workflowEventStoreStub{createErr: errors.New("create failed")}
		router.POST("/events", WorkflowExecutionEventHandler(store))

		req := httptest.NewRequest(http.MethodPost, "/events", strings.NewReader(`{"execution_id":"exec-2","status":"running"}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusInternalServerError, resp.Code)
		assert.Contains(t, resp.Body.String(), "failed to create execution")
	})

	t.Run("workflow execution store failure is non fatal on create", func(t *testing.T) {
		router := gin.New()
		store := &workflowEventStoreStub{storeWorkflowErr: errors.New("workflow write failed")}
		router.POST("/events", WorkflowExecutionEventHandler(store))

		req := httptest.NewRequest(http.MethodPost, "/events", strings.NewReader(`{"execution_id":"exec-3","status":"running","type":"local"}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		require.NotNil(t, store.createdExec)
		require.NotNil(t, store.storedWorkflow)
		assert.Contains(t, resp.Body.String(), `"created":true`)
	})

	t.Run("update failure returns 500", func(t *testing.T) {
		router := gin.New()
		store := &workflowEventStoreStub{
			exec:      &types.Execution{ExecutionID: "exec-4", Status: types.ExecutionStatusRunning, StartedAt: time.Now().UTC()},
			updateErr: errors.New("update failed"),
		}
		router.POST("/events", WorkflowExecutionEventHandler(store))

		req := httptest.NewRequest(http.MethodPost, "/events", strings.NewReader(`{"execution_id":"exec-4","status":"failed"}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusInternalServerError, resp.Code)
		assert.Contains(t, resp.Body.String(), "failed to update execution")
	})
}

func TestWorkflowDAGAdditionalBranches(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	t.Run("dag error and not found", func(t *testing.T) {
		errStore := &workflowDAGStorageStub{queryErr: errors.New("query failed")}
		router := gin.New()
		router.GET("/workflows/:workflowId/dag", GetWorkflowDAGHandler(errStore))

		req := httptest.NewRequest(http.MethodGet, "/workflows/run-err/dag", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusInternalServerError, resp.Code)

		emptyStore := &workflowDAGStorageStub{}
		router = gin.New()
		router.GET("/workflows/:workflowId/dag", GetWorkflowDAGHandler(emptyStore))
		req = httptest.NewRequest(http.MethodGet, "/workflows/run-missing/dag", nil)
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusNotFound, resp.Code)
	})

	t.Run("children and session error paths", func(t *testing.T) {
		store := &workflowDAGStorageStub{queryErr: errors.New("query failed")}
		router := gin.New()
		router.GET("/children/:execution_id", GetWorkflowChildrenHandler(store))
		router.GET("/sessions/:session_id/workflows", GetSessionWorkflowsHandler(store))

		req := httptest.NewRequest(http.MethodGet, "/children/root", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusInternalServerError, resp.Code)

		req = httptest.NewRequest(http.MethodGet, "/sessions/session-1/workflows", nil)
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusInternalServerError, resp.Code)
	})

	t.Run("session empty result succeeds", func(t *testing.T) {
		store := &workflowDAGStorageStub{}
		router := gin.New()
		router.GET("/sessions/:session_id/workflows", GetSessionWorkflowsHandler(store))

		req := httptest.NewRequest(http.MethodGet, "/sessions/session-empty/workflows", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		assert.Contains(t, resp.Body.String(), `"total_workflows":0`)
	})

	t.Run("helpers dedupe and cap webhook previews", func(t *testing.T) {
		execs := []*types.Execution{
			nil,
			{AgentNodeID: " agent-a "},
			{AgentNodeID: "agent-a"},
			{AgentNodeID: ""},
			{AgentNodeID: "agent-b"},
		}
		require.Equal(t, []string{"agent-a", "agent-b"}, collectUniqueAgentNodeIDs(execs))

		execByID := map[string]*types.Execution{}
		evMap := map[string][]*types.ExecutionWebhookEvent{}
		for i := 0; i < maxWebhookFailurePreviews+5; i++ {
			id := "exec-" + string(rune('a'+i))
			execByID[id] = &types.Execution{ExecutionID: id, AgentNodeID: "agent", ReasonerID: "reasoner"}
			evMap[id] = []*types.ExecutionWebhookEvent{
				{Status: "failed", EventType: "deliver", CreatedAt: time.Unix(int64(i), 0)},
				{Status: "succeeded", EventType: "deliver", CreatedAt: time.Unix(int64(i+100), 0)},
			}
		}

		previews := buildWebhookFailurePreviews(execByID, evMap)
		require.Len(t, previews, maxWebhookFailurePreviews)
		assert.GreaterOrEqual(t, previews[0].CreatedAt, previews[len(previews)-1].CreatedAt)
	})
}

func TestCheckLLMEndpointHealthAdditionalBranches(t *testing.T) {
	t.Parallel()

	t.Run("nil monitor is ignored", func(t *testing.T) {
		require.NoError(t, checkLLMEndpointHealth(nil, ""))
	})

	t.Run("single unavailable endpoint returns detailed error", func(t *testing.T) {
		failingServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
		}))
		defer failingServer.Close()

		monitor := services.NewLLMHealthMonitor(config.LLMHealthConfig{
			Enabled:          true,
			CheckInterval:    10 * time.Millisecond,
			CheckTimeout:     100 * time.Millisecond,
			FailureThreshold: 1,
			RecoveryTimeout:  time.Second,
			Endpoints: []config.LLMEndpoint{
				{Name: "solo", URL: failingServer.URL},
			},
		}, nil)
		go monitor.Start()
		defer monitor.Stop()

		// Poll until the monitor detects the unhealthy endpoint.
		require.Eventually(t, func() bool {
			err := checkLLMEndpointHealth(monitor, "unknown-endpoint")
			return err != nil && strings.Contains(err.Error(), `LLM backend "solo" unavailable`)
		}, 5*time.Second, 50*time.Millisecond)
	})
}

func TestNodeHandlerAdditionalErrorBranches(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	t.Run("list nodes surfaces storage error", func(t *testing.T) {
		store := &nodeRESTStorageStub{listAgentsErr: errors.New("list failed")}
		router := gin.New()
		router.GET("/nodes", ListNodesHandler(store))

		req := httptest.NewRequest(http.MethodGet, "/nodes", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusInternalServerError, resp.Code)
	})

	t.Run("get node returns not found", func(t *testing.T) {
		router := gin.New()
		router.GET("/nodes/:node_id", GetNodeHandler(&nodeRESTStorageStub{}))

		req := httptest.NewRequest(http.MethodGet, "/nodes/missing?version=v1", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusNotFound, resp.Code)
	})

	t.Run("bulk node status validates request payload", func(t *testing.T) {
		router := gin.New()
		router.POST("/nodes/bulk-status", BulkNodeStatusHandler(nil, &nodeRESTStorageStub{}))

		req := httptest.NewRequest(http.MethodPost, "/nodes/bulk-status", strings.NewReader(`{`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusBadRequest, resp.Code)

		var ids []string
		for i := 0; i < 51; i++ {
			ids = append(ids, "node-"+time.Unix(int64(i), 0).UTC().Format("150405"))
		}
		body, err := json.Marshal(gin.H{"node_ids": ids})
		require.NoError(t, err)

		req = httptest.NewRequest(http.MethodPost, "/nodes/bulk-status", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusBadRequest, resp.Code)
		assert.Contains(t, resp.Body.String(), "Too many node IDs requested")
	})

	t.Run("start and stop return node not found before manager checks", func(t *testing.T) {
		store := &nodeRESTStorageStub{getAgentErr: errors.New("missing")}
		router := gin.New()
		router.POST("/nodes/:node_id/start", StartNodeHandler(nil, store))
		router.POST("/nodes/:node_id/stop", StopNodeHandler(nil, store))

		req := httptest.NewRequest(http.MethodPost, "/nodes/node-x/start", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusNotFound, resp.Code)

		req = httptest.NewRequest(http.MethodPost, "/nodes/node-x/stop", nil)
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusNotFound, resp.Code)
	})

	t.Run("register node surfaces storage failure", func(t *testing.T) {
		store := &registerNodeStorageStub{
			nodeRESTStorageStub: &nodeRESTStorageStub{},
			registerErr:         errors.New("write failed"),
		}
		router := gin.New()
		router.POST("/nodes/register", RegisterNodeHandler(store, nil, nil, nil, nil, nil))

		req := httptest.NewRequest(http.MethodPost, "/nodes/register", strings.NewReader(`{"id":"node-fail","base_url":"https://example.com"}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusInternalServerError, resp.Code)
		assert.Contains(t, resp.Body.String(), "Failed to store node")
	})
}

func TestNormalizeServerlessDiscoveryURLAdditionalBranches(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		rawURL  string
		allowed []string
		want    string
		wantErr string
	}{
		{
			name:    "userinfo rejected",
			rawURL:  "https://user@example.com",
			allowed: []string{"example.com"},
			wantErr: "must not include user info",
		},
		{
			name:    "query rejected",
			rawURL:  "https://example.com?x=1",
			allowed: []string{"example.com"},
			wantErr: "must not include query parameters or fragments",
		},
		{
			name:    "fragment rejected",
			rawURL:  "https://example.com/#frag",
			allowed: []string{"example.com"},
			wantErr: "must not include query parameters or fragments",
		},
		{
			name:    "wildcard host allowlist accepted",
			rawURL:  "https://api.sub.example.com/",
			allowed: []string{"*.example.com"},
			want:    "https://api.sub.example.com",
		},
		{
			name:    "cidr allowlist accepted",
			rawURL:  "http://127.0.0.2:8080/",
			allowed: []string{"127.0.0.0/8"},
			want:    "http://127.0.0.2:8080",
		},
		{
			name:    "zero host normalized to localhost",
			rawURL:  "http://0.0.0.0:9090/",
			allowed: []string{"example.com"},
			want:    "http://localhost:9090",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := normalizeServerlessDiscoveryURL(tc.rawURL, tc.allowed)
			if tc.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.want, got)
		})
	}

	assert.True(t, isServerlessDiscoveryHostAllowed("service.example.com", []string{"*.example.com"}))
	assert.False(t, isServerlessDiscoveryHostAllowed("example.com", []string{"*.example.com"}))
}
