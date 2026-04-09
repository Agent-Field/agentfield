package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	afcli "github.com/Agent-Field/agentfield/control-plane/internal/cli"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type eventHistoryStub struct {
	storage.StorageProvider
	events     []*types.MemoryChangeEvent
	err        error
	lastFilter types.EventFilter
}

func (s *eventHistoryStub) GetEventHistory(ctx context.Context, filter types.EventFilter) ([]*types.MemoryChangeEvent, error) {
	s.lastFilter = filter
	return s.events, s.err
}

type workflowDAGStorageStub struct {
	storage.StorageProvider
	executions         []*types.Execution
	queryErr           error
	webhookRegistered  map[string]bool
	webhookEvents      map[string][]*types.ExecutionWebhookEvent
	webhookRegisterErr error
	webhookEventsErr   error
	executionVCs       []*types.ExecutionVCInfo
	vcErr              error
	lastFilter         types.ExecutionFilter
}

func (s *workflowDAGStorageStub) QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
	s.lastFilter = filter
	if s.queryErr != nil {
		return nil, s.queryErr
	}
	var out []*types.Execution
	for _, exec := range s.executions {
		if exec == nil {
			continue
		}
		if filter.RunID != nil && exec.RunID != *filter.RunID {
			continue
		}
		if filter.ParentExecutionID != nil {
			if exec.ParentExecutionID == nil || *exec.ParentExecutionID != *filter.ParentExecutionID {
				continue
			}
		}
		if filter.SessionID != nil {
			if exec.SessionID == nil || *exec.SessionID != *filter.SessionID {
				continue
			}
		}
		copyExec := *exec
		out = append(out, &copyExec)
	}
	return out, nil
}

func (s *workflowDAGStorageStub) ListExecutionWebhooksRegistered(ctx context.Context, executionIDs []string) (map[string]bool, error) {
	if s.webhookRegisterErr != nil {
		return nil, s.webhookRegisterErr
	}
	out := make(map[string]bool, len(executionIDs))
	for _, id := range executionIDs {
		out[id] = s.webhookRegistered[id]
	}
	return out, nil
}

func (s *workflowDAGStorageStub) ListExecutionWebhookEventsBatch(ctx context.Context, executionIDs []string) (map[string][]*types.ExecutionWebhookEvent, error) {
	if s.webhookEventsErr != nil {
		return nil, s.webhookEventsErr
	}
	out := make(map[string][]*types.ExecutionWebhookEvent, len(executionIDs))
	for _, id := range executionIDs {
		out[id] = s.webhookEvents[id]
	}
	return out, nil
}

func (s *workflowDAGStorageStub) ListExecutionVCs(ctx context.Context, filter types.VCFilters) ([]*types.ExecutionVCInfo, error) {
	if s.vcErr != nil {
		return nil, s.vcErr
	}
	if filter.WorkflowID == nil {
		return s.executionVCs, nil
	}
	var out []*types.ExecutionVCInfo
	for _, vc := range s.executionVCs {
		if vc != nil && vc.WorkflowID == *filter.WorkflowID {
			out = append(out, vc)
		}
	}
	return out, nil
}

type cleanupStorageStub struct {
	storage.StorageProvider
	result       *types.WorkflowCleanupResult
	err          error
	lastWorkflow string
	lastDryRun   bool
}

func (s *cleanupStorageStub) CleanupWorkflow(ctx context.Context, workflowID string, dryRun bool) (*types.WorkflowCleanupResult, error) {
	s.lastWorkflow = workflowID
	s.lastDryRun = dryRun
	return s.result, s.err
}

type nodeRESTStorageStub struct {
	storage.StorageProvider
	agent           *types.AgentNode
	versionedAgent  *types.AgentNode
	listAgents      []*types.AgentNode
	getAgentErr     error
	getVersionErr   error
	listAgentsErr   error
	heartbeats      []time.Time
	lastHeartbeatID string
	lastVersion     string
	updatedLifecycle *types.AgentLifecycleStatus
	registeredAgent  *types.AgentNode
}

func (s *nodeRESTStorageStub) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	if s.getAgentErr != nil {
		return nil, s.getAgentErr
	}
	if s.agent != nil && s.agent.ID == id {
		return s.agent, nil
	}
	return nil, nil
}

func (s *nodeRESTStorageStub) GetAgentVersion(ctx context.Context, id, version string) (*types.AgentNode, error) {
	if s.getVersionErr != nil {
		return nil, s.getVersionErr
	}
	if s.versionedAgent != nil && s.versionedAgent.ID == id && s.versionedAgent.Version == version {
		return s.versionedAgent, nil
	}
	return nil, nil
}

func (s *nodeRESTStorageStub) UpdateAgentHeartbeat(ctx context.Context, id, version string, ts time.Time) error {
	s.lastHeartbeatID = id
	s.lastVersion = version
	s.heartbeats = append(s.heartbeats, ts)
	return nil
}

func (s *nodeRESTStorageStub) ListAgents(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
	if s.listAgentsErr != nil {
		return nil, s.listAgentsErr
	}
	return s.listAgents, nil
}

func (s *nodeRESTStorageStub) UpdateAgentLifecycleStatus(ctx context.Context, nodeID string, status types.AgentLifecycleStatus) error {
	s.updatedLifecycle = &status
	return nil
}

func (s *nodeRESTStorageStub) RegisterAgent(ctx context.Context, agent *types.AgentNode) error {
	s.registeredAgent = agent
	s.agent = agent
	return nil
}

type didServiceStub struct{}

func (didServiceStub) RegisterAgent(req *types.DIDRegistrationRequest) (*types.DIDRegistrationResponse, error) {
	return &types.DIDRegistrationResponse{}, nil
}

func (didServiceStub) ResolveDID(did string) (*types.DIDIdentity, error) { return nil, nil }
func (didServiceStub) ListAllAgentDIDs() ([]string, error)                { return nil, nil }

type vcServiceStub struct{}

func (vcServiceStub) VerifyVC(vcDocument json.RawMessage) (*types.VCVerificationResponse, error) {
	return &types.VCVerificationResponse{}, nil
}

func (vcServiceStub) GetWorkflowVCChain(workflowID string) (*types.WorkflowVCChainResponse, error) {
	return nil, nil
}

func (vcServiceStub) CreateWorkflowVC(workflowID, sessionID string, executionVCIDs []string) (*types.WorkflowVC, error) {
	return nil, nil
}

func (vcServiceStub) GenerateExecutionVC(ctx *types.ExecutionContext, inputData, outputData []byte, status string, errorMessage *string, durationMS int) (*types.ExecutionVC, error) {
	return nil, nil
}

func (vcServiceStub) QueryExecutionVCs(filters *types.VCFilters) ([]types.ExecutionVC, error) {
	return nil, nil
}

func (vcServiceStub) ListWorkflowVCs() ([]*types.WorkflowVC, error) { return nil, nil }
func (vcServiceStub) GetExecutionVCByExecutionID(executionID string) (*types.ExecutionVC, error) {
	return nil, nil
}
func (vcServiceStub) ListAgentTagVCs() ([]*types.AgentTagVCRecord, error) { return nil, nil }

func newExecution(executionID, runID, agentID, reasonerID, status string, started time.Time) *types.Execution {
	return &types.Execution{
		ExecutionID: executionID,
		RunID:       runID,
		AgentNodeID: agentID,
		ReasonerID:  reasonerID,
		Status:      status,
		StartedAt:   started,
		CreatedAt:   started,
		UpdatedAt:   started,
	}
}

func TestRespondErrorHelpers(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name       string
		invoke     func(*gin.Context)
		wantStatus int
		wantError  string
	}{
		{
			name:       "generic",
			invoke:     func(c *gin.Context) { RespondError(c, http.StatusTeapot, "brew") },
			wantStatus: http.StatusTeapot,
			wantError:  "brew",
		},
		{
			name:       "bad_request",
			invoke:     func(c *gin.Context) { RespondBadRequest(c, "bad") },
			wantStatus: http.StatusBadRequest,
			wantError:  "bad",
		},
		{
			name:       "not_found",
			invoke:     func(c *gin.Context) { RespondNotFound(c, "missing") },
			wantStatus: http.StatusNotFound,
			wantError:  "missing",
		},
		{
			name:       "internal",
			invoke:     func(c *gin.Context) { RespondInternalError(c, "boom") },
			wantStatus: http.StatusInternalServerError,
			wantError:  "boom",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(recorder)
			tc.invoke(c)

			var response ErrorResponse
			require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &response))
			assert.Equal(t, tc.wantStatus, recorder.Code)
			assert.Equal(t, tc.wantError, response.Error)
		})
	}
}

func TestConcurrencyLimiterInitializationAndPreconditions(t *testing.T) {
	gin.SetMode(gin.TestMode)

	llmHealthMonitorMu.Lock()
	llmHealthMonitor = nil
	llmHealthMonitorMu.Unlock()
	prevLimiter := concurrencyLimiter
	concurrencyLimiterOnce = sync.Once{}
	concurrencyLimiter = nil
	t.Cleanup(func() {
		// Restore the guarded value and reset the Once so the next
		// InitConcurrencyLimiter call can re-initialise if needed.
		concurrencyLimiter = prevLimiter
		concurrencyLimiterOnce = sync.Once{}
	})

	InitConcurrencyLimiter(1)
	limiter := GetConcurrencyLimiter()
	require.NotNil(t, limiter)
	assert.Equal(t, 1, limiter.MaxPerAgent())

	require.NoError(t, CheckExecutionPreconditions("agent-1", ""))
	err := CheckExecutionPreconditions("agent-1", "")
	require.Error(t, err)

	preconditionErr, ok := err.(*executionPreconditionError)
	require.True(t, ok)
	assert.Equal(t, 429, preconditionErr.HTTPStatusCode())
	assert.Equal(t, ErrorCategoryConcurrencyLimit, preconditionErr.Category())
	assert.Contains(t, preconditionErr.Error(), "agent-1")

	ReleaseExecutionSlot("agent-1")
	require.NoError(t, CheckExecutionPreconditions("agent-1", ""))
}

func TestGetEventHistoryHandler_SuccessAndFailure(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	since := time.Now().UTC().Add(-time.Hour).Truncate(time.Second)
	successStub := &eventHistoryStub{
		events: []*types.MemoryChangeEvent{
			{ID: "evt-1", Key: "memory.key"},
		},
	}

	router := gin.New()
	router.GET("/history", GetEventHistoryHandler(successStub))

	req := httptest.NewRequest(http.MethodGet, "/history?scope=session&scope_id=s-1&patterns=foo.*,bar.*&since="+since.Format(time.RFC3339)+"&limit=7", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	require.NotNil(t, successStub.lastFilter.Scope)
	require.Equal(t, "session", *successStub.lastFilter.Scope)
	require.NotNil(t, successStub.lastFilter.ScopeID)
	require.Equal(t, "s-1", *successStub.lastFilter.ScopeID)
	require.Equal(t, []string{"foo.*", "bar.*"}, successStub.lastFilter.Patterns)
	require.NotNil(t, successStub.lastFilter.Since)
	require.Equal(t, since, successStub.lastFilter.Since.UTC())
	require.Equal(t, 7, successStub.lastFilter.Limit)

	failureStub := &eventHistoryStub{err: errors.New("history unavailable")}
	router = gin.New()
	router.GET("/history", GetEventHistoryHandler(failureStub))

	req = httptest.NewRequest(http.MethodGet, "/history?since=not-a-time&limit=bad", nil)
	resp = httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusInternalServerError, resp.Code)
	assert.Nil(t, failureStub.lastFilter.Since)
	assert.Equal(t, 0, failureStub.lastFilter.Limit)
}

func TestCleanupWorkflowHandler_MapsErrorsAndSupportsBodyOverride(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name         string
		method       string
		target       string
		body         string
		result       *types.WorkflowCleanupResult
		err          error
		wantCode     int
		wantContains string
		wantDryRun   bool
	}{
		{
			name:         "body overrides query and succeeds",
			method:       http.MethodPost,
			target:       "/workflows/run-1/cleanup?dry_run=false&confirm=false",
			body:         `{"confirm":true,"dry_run":true}`,
			result:       &types.WorkflowCleanupResult{WorkflowID: "run-1", DryRun: true, Success: true},
			wantCode:     http.StatusOK,
			wantContains: `"dry_run":true`,
			wantDryRun:   true,
		},
		{
			name:         "not found maps to 404",
			method:       http.MethodDelete,
			target:       "/workflows/run-2/cleanup?confirm=true",
			result:       &types.WorkflowCleanupResult{ErrorMessage: testStringPtr("workflow not found")},
			err:          errors.New("cleanup failed"),
			wantCode:     http.StatusNotFound,
			wantContains: "workflow_not_found",
			wantDryRun:   false,
		},
		{
			name:         "active maps to 409",
			method:       http.MethodDelete,
			target:       "/workflows/run-3/cleanup?confirm=true",
			result:       &types.WorkflowCleanupResult{ErrorMessage: testStringPtr("workflow is still active")},
			err:          errors.New("cleanup failed"),
			wantCode:     http.StatusConflict,
			wantContains: "workflow_active",
			wantDryRun:   false,
		},
		{
			name:         "empty maps to 400",
			method:       http.MethodDelete,
			target:       "/workflows/run-4/cleanup?confirm=true",
			result:       &types.WorkflowCleanupResult{ErrorMessage: testStringPtr("workflow id empty")},
			err:          errors.New("cleanup failed"),
			wantCode:     http.StatusBadRequest,
			wantContains: "invalid_workflow_id",
			wantDryRun:   false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			store := &cleanupStorageStub{result: tc.result, err: tc.err}
			router := gin.New()
			router.Any("/workflows/:workflow_id/cleanup", CleanupWorkflowHandler(store))

			req := httptest.NewRequest(tc.method, tc.target, strings.NewReader(tc.body))
			if tc.body != "" {
				req.Header.Set("Content-Type", "application/json")
			}
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			assert.Equal(t, tc.wantCode, resp.Code)
			assert.Contains(t, resp.Body.String(), tc.wantContains)
			assert.Equal(t, tc.wantDryRun, store.lastDryRun)
		})
	}
}

func TestWorkflowDAGHandlers(t *testing.T) {
	gin.SetMode(gin.TestMode)

	now := time.Now().UTC().Truncate(time.Second)
	sessionID := "session-1"
	actorID := "actor-1"
	root := newExecution("exec-root", "run-1", "agent-a", "root-reasoner", string(types.ExecutionStatusSucceeded), now)
	root.SessionID = &sessionID
	root.ActorID = &actorID

	childParent := root.ExecutionID
	child := newExecution("exec-child", "run-1", "agent-b", "child-reasoner", string(types.ExecutionStatusFailed), now.Add(time.Minute))
	child.ParentExecutionID = &childParent

	other := newExecution("exec-other", "run-2", "agent-c", "other-reasoner", string(types.ExecutionStatusRunning), now.Add(2*time.Minute))
	other.SessionID = &sessionID
	other.ActorID = &actorID

	status := 502
	store := &workflowDAGStorageStub{
		executions: []*types.Execution{root, child, other},
		webhookRegistered: map[string]bool{
			"exec-root":  true,
			"exec-child": true,
		},
		webhookEvents: map[string][]*types.ExecutionWebhookEvent{
			"exec-child": {
				{
					ExecutionID: "exec-child",
					EventType:   "completed",
					Status:      "failed",
					HTTPStatus:  &status,
					CreatedAt:   now.Add(3 * time.Minute),
				},
			},
		},
		executionVCs: []*types.ExecutionVCInfo{
			{WorkflowID: "run-1", IssuerDID: "did:example:issuer"},
		},
	}

	router := gin.New()
	router.GET("/workflows/:workflowId/dag", GetWorkflowDAGHandler(store))
	router.GET("/children/:execution_id", GetWorkflowChildrenHandler(store))
	router.GET("/sessions/:session_id/workflows", GetSessionWorkflowsHandler(store))

	t.Run("dag_requires_workflow_id", func(t *testing.T) {
		recorder := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(recorder)
		req := httptest.NewRequest(http.MethodGet, "/dag", nil)
		ctx.Request = req
		newExecutionGraphService(store).handleGetWorkflowDAG(ctx)
		assert.Equal(t, http.StatusBadRequest, recorder.Code)
	})

	t.Run("dag_full_success", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/workflows/run-1/dag", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		var payload WorkflowDAGResponse
		require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
		assert.Equal(t, "run-1", payload.RootWorkflowID)
		assert.Equal(t, 2, payload.TotalNodes)
		assert.Equal(t, "failed", payload.WorkflowStatus)
		assert.Equal(t, "root-reasoner", payload.WorkflowName)
		require.Len(t, payload.DAG.Children, 1)
		assert.Equal(t, "exec-child", payload.DAG.Children[0].ExecutionID)
	})

	t.Run("dag_lightweight_success", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/workflows/run-1/dag?lightweight=1", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		var payload WorkflowDAGLightweightResponse
		require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
		assert.Equal(t, "lightweight", payload.Mode)
		assert.Equal(t, []string{"agent-a", "agent-b"}, payload.UniqueAgentNodeIDs)
		require.NotNil(t, payload.WorkflowIssuerDID)
		assert.Equal(t, "did:example:issuer", *payload.WorkflowIssuerDID)
		require.NotNil(t, payload.WebhookSummary)
		assert.Equal(t, 2, payload.WebhookSummary.StepsWithWebhook)
		assert.Equal(t, 1, payload.WebhookSummary.TotalDeliveries)
		assert.Equal(t, 1, payload.WebhookSummary.FailedDeliveries)
		require.Len(t, payload.WebhookFailures, 1)
		assert.Equal(t, "exec-child", payload.WebhookFailures[0].ExecutionID)
	})

	t.Run("children_success", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/children/exec-root", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		var payload map[string]any
		require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
		assert.Equal(t, float64(1), payload["count"])
		assert.Equal(t, "exec-root", payload["execution_id"])
	})

	t.Run("session_workflows_success", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/sessions/session-1/workflows", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		var payload SessionWorkflowsResponse
		require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
		assert.Equal(t, "session-1", payload.SessionID)
		assert.Equal(t, 2, payload.TotalWorkflows)
		require.Len(t, payload.RootWorkflows, 2)
	})
}

func TestNodeRESTHandlers_SuccessPaths(t *testing.T) {
	gin.SetMode(gin.TestMode)

	presence := services.NewPresenceManager(nil, services.PresenceManagerConfig{HeartbeatTTL: time.Minute})
	agent := &types.AgentNode{ID: "node-1", Version: "v1", LifecycleStatus: types.AgentStatusReady}
	pendingAgent := &types.AgentNode{ID: "node-pending", Version: "v2", LifecycleStatus: types.AgentStatusPendingApproval}

	t.Run("status lease updates heartbeat and returns lease", func(t *testing.T) {
		store := &nodeRESTStorageStub{agent: agent}
		router := gin.New()
		router.PUT("/nodes/:node_id/status", NodeStatusLeaseHandler(store, nil, presence, 2*time.Minute))

		req := httptest.NewRequest(http.MethodPut, "/nodes/node-1/status", strings.NewReader(`{"phase":"ready","health_score":99}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		assert.Equal(t, "node-1", store.lastHeartbeatID)
		assert.Equal(t, "v1", store.lastVersion)
		assert.Contains(t, resp.Body.String(), `"lease_seconds":120`)
	})

	t.Run("pending approval only renews lease", func(t *testing.T) {
		store := &nodeRESTStorageStub{agent: pendingAgent}
		router := gin.New()
		router.PUT("/nodes/:node_id/status", NodeStatusLeaseHandler(store, nil, presence, 0))

		req := httptest.NewRequest(http.MethodPut, "/nodes/node-pending/status", strings.NewReader(`{"phase":"offline"}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		assert.Equal(t, "node-pending", store.lastHeartbeatID)
		assert.Contains(t, resp.Body.String(), `"lease_seconds":300`)
	})

	t.Run("action ack renews lease", func(t *testing.T) {
		store := &nodeRESTStorageStub{agent: agent}
		router := gin.New()
		router.POST("/nodes/:node_id/actions/ack", NodeActionAckHandler(store, presence, time.Minute))

		req := httptest.NewRequest(http.MethodPost, "/nodes/node-1/actions/ack", strings.NewReader(`{"action_id":"a1","status":"SUCCEEDED"}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		assert.Equal(t, "node-1", store.lastHeartbeatID)
		assert.Contains(t, resp.Body.String(), `"lease_seconds":60`)
	})

	t.Run("claim actions defaults wait seconds", func(t *testing.T) {
		store := &nodeRESTStorageStub{agent: agent}
		router := gin.New()
		router.POST("/actions/claim", ClaimActionsHandler(store, presence, time.Minute))

		req := httptest.NewRequest(http.MethodPost, "/actions/claim", strings.NewReader(`{"node_id":"node-1","max_items":0,"wait_seconds":0}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		assert.Equal(t, "node-1", store.lastHeartbeatID)
		assert.Contains(t, resp.Body.String(), `"next_poll_after":5`)
	})

	t.Run("shutdown uses version lookup and acknowledges", func(t *testing.T) {
		store := &nodeRESTStorageStub{versionedAgent: agent}
		router := gin.New()
		router.POST("/nodes/:node_id/shutdown", NodeShutdownHandler(store, nil, presence))

		req := httptest.NewRequest(http.MethodPost, "/nodes/node-1/shutdown", strings.NewReader(`{"version":"v1","reason":"restart"}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusAccepted, resp.Code)
		assert.Equal(t, "node-1", store.lastHeartbeatID)
		assert.Contains(t, resp.Body.String(), "shutdown acknowledged")
	})
}

func TestVerifyAuditBundleAndDIDRoutes(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	t.Run("empty and oversized audit bodies", func(t *testing.T) {
		router := gin.New()
		router.POST("/verify-audit", HandleVerifyAuditBundle)

		req := httptest.NewRequest(http.MethodPost, "/verify-audit", strings.NewReader(""))
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusBadRequest, resp.Code)
		assert.Contains(t, resp.Body.String(), "empty body")

		req = httptest.NewRequest(http.MethodPost, "/verify-audit", strings.NewReader(strings.Repeat("a", afcli.MaxVerifyAuditBodyBytes+1)))
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusRequestEntityTooLarge, resp.Code)
		assert.Contains(t, resp.Body.String(), "request body too large")
	})

	t.Run("did handler wrapper and route registration", func(t *testing.T) {
		router := gin.New()
		api := router.Group("/api/v1")
		h := NewDIDHandlers(didServiceStub{}, vcServiceStub{})
		h.RegisterRoutes(api)

		req := httptest.NewRequest(http.MethodPost, "/api/v1/did/verify-audit", strings.NewReader(""))
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusBadRequest, resp.Code)

		paths := make(map[string]struct{})
		for _, route := range router.Routes() {
			paths[route.Path] = struct{}{}
		}
		assert.Contains(t, paths, "/api/v1/did/verify-audit")
		assert.Contains(t, paths, "/api/v1/execution/vc")
	})
}

func TestNodeHandlers_BasicCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{ID: "node-1", Version: "v1", LifecycleStatus: types.AgentStatusReady}

	t.Run("list nodes default and show_all", func(t *testing.T) {
		store := &nodeRESTStorageStub{listAgents: []*types.AgentNode{agent}}
		router := gin.New()
		router.GET("/nodes", ListNodesHandler(store))

		req := httptest.NewRequest(http.MethodGet, "/nodes?team_id=t1&group_id=g1", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusOK, resp.Code)
		assert.Contains(t, resp.Body.String(), `"count":1`)

		req = httptest.NewRequest(http.MethodGet, "/nodes?show_all=true", nil)
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusOK, resp.Code)
	})

	t.Run("get node by id and version", func(t *testing.T) {
		store := &nodeRESTStorageStub{agent: agent, versionedAgent: agent}
		router := gin.New()
		router.GET("/nodes/:node_id", GetNodeHandler(store))

		req := httptest.NewRequest(http.MethodGet, "/nodes/node-1", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusOK, resp.Code)

		req = httptest.NewRequest(http.MethodGet, "/nodes/node-1?version=v1", nil)
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusOK, resp.Code)
	})

	t.Run("update lifecycle fallback success and pending approval rejection", func(t *testing.T) {
		store := &nodeRESTStorageStub{agent: agent}
		router := gin.New()
		router.POST("/nodes/:node_id/lifecycle", UpdateLifecycleStatusHandler(store, nil, nil))

		req := httptest.NewRequest(http.MethodPost, "/nodes/node-1/lifecycle", strings.NewReader(`{"lifecycle_status":"ready"}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusOK, resp.Code)
		require.NotNil(t, store.updatedLifecycle)
		assert.Equal(t, types.AgentStatusReady, *store.updatedLifecycle)

		store = &nodeRESTStorageStub{agent: &types.AgentNode{ID: "node-1", LifecycleStatus: types.AgentStatusPendingApproval}}
		router = gin.New()
		router.POST("/nodes/:node_id/lifecycle", UpdateLifecycleStatusHandler(store, nil, nil))
		req = httptest.NewRequest(http.MethodPost, "/nodes/node-1/lifecycle", strings.NewReader(`{"lifecycle_status":"offline"}`))
		req.Header.Set("Content-Type", "application/json")
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusConflict, resp.Code)
	})

	t.Run("status and refresh endpoints return service unavailable when manager missing", func(t *testing.T) {
		router := gin.New()
		router.GET("/nodes/:node_id/status", GetNodeStatusHandler(nil))
		router.POST("/nodes/:node_id/status/refresh", RefreshNodeStatusHandler(nil))

		req := httptest.NewRequest(http.MethodGet, "/nodes/node-1/status", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusServiceUnavailable, resp.Code)

		req = httptest.NewRequest(http.MethodPost, "/nodes/node-1/status/refresh", nil)
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusServiceUnavailable, resp.Code)
	})

	t.Run("bulk refresh and lifecycle handlers with nil manager", func(t *testing.T) {
		store := &nodeRESTStorageStub{agent: agent, listAgents: []*types.AgentNode{agent}}
		router := gin.New()
		router.POST("/nodes/bulk-status", BulkNodeStatusHandler(nil, store))
		router.POST("/nodes/status/refresh-all", RefreshAllNodeStatusHandler(nil, store))
		router.POST("/nodes/:node_id/start", StartNodeHandler(nil, store))
		router.POST("/nodes/:node_id/stop", StopNodeHandler(nil, store))

		req := httptest.NewRequest(http.MethodPost, "/nodes/bulk-status", strings.NewReader(`{"node_ids":["node-1"]}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusServiceUnavailable, resp.Code)

		req = httptest.NewRequest(http.MethodPost, "/nodes/status/refresh-all", nil)
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusServiceUnavailable, resp.Code)

		req = httptest.NewRequest(http.MethodPost, "/nodes/node-1/start", nil)
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusServiceUnavailable, resp.Code)

		req = httptest.NewRequest(http.MethodPost, "/nodes/node-1/stop", nil)
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusServiceUnavailable, resp.Code)
	})
}

func TestRegisterServerlessAgentHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("successfully registers discovered agent", func(t *testing.T) {
		discoveryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			require.Equal(t, "/discover", r.URL.Path)
			_, _ = w.Write([]byte(`{
				"node_id":"serverless-1",
				"version":"2026.04.08",
				"reasoners":[{"id":"r1","input_schema":{"type":"object"},"output_schema":{"type":"object"},"tags":["ops"]}],
				"skills":[{"id":"s1","input_schema":{"type":"object"}}]
			}`))
		}))
		defer discoveryServer.Close()

		store := &nodeRESTStorageStub{}
		router := gin.New()
		router.POST("/serverless/register", RegisterServerlessAgentHandler(store, nil, nil, nil, nil, []string{"127.0.0.1", "localhost"}))

		req := httptest.NewRequest(http.MethodPost, "/serverless/register", strings.NewReader(`{"invocation_url":"`+discoveryServer.URL+`"}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusCreated, resp.Code)
		require.NotNil(t, store.registeredAgent)
		assert.Equal(t, "serverless-1", store.registeredAgent.ID)
		assert.Equal(t, "serverless", store.registeredAgent.DeploymentType)
		assert.Contains(t, resp.Body.String(), "Serverless agent registered successfully")
	})

	t.Run("invalid discovery response is rejected", func(t *testing.T) {
		discoveryServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`{"version":"missing-node-id"}`))
		}))
		defer discoveryServer.Close()

		router := gin.New()
		router.POST("/serverless/register", RegisterServerlessAgentHandler(&nodeRESTStorageStub{}, nil, nil, nil, nil, []string{"127.0.0.1", "localhost"}))

		req := httptest.NewRequest(http.MethodPost, "/serverless/register", strings.NewReader(`{"invocation_url":"`+discoveryServer.URL+`"}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusBadRequest, resp.Code)
		assert.Contains(t, resp.Body.String(), "node_id is missing")
	})
}

func TestBuildWorkflowDAGWrapper(t *testing.T) {
	t.Parallel()

	root := newExecution("exec-root", "run-1", "agent-a", "reasoner", string(types.ExecutionStatusSucceeded), time.Now().UTC())
	parentID := root.ExecutionID
	child := newExecution("exec-child", "run-1", "agent-b", "child", string(types.ExecutionStatusSucceeded), time.Now().UTC().Add(time.Second))
	child.ParentExecutionID = &parentID

	dag, timeline, status, workflowName, _, _, maxDepth := BuildWorkflowDAG([]*types.Execution{root, child})
	assert.Equal(t, "exec-root", dag.ExecutionID)
	require.Len(t, timeline, 2)
	assert.Equal(t, string(types.ExecutionStatusSucceeded), status)
	assert.Equal(t, "reasoner", workflowName)
	assert.Equal(t, 1, maxDepth)
}

func testStringPtr(value string) *string {
	return &value
}
