package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type versionedExecutionStore struct {
	*testExecutionStorage
	versions    []*types.AgentNode
	getAgentErr error
}

func (s *versionedExecutionStore) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	if s.getAgentErr != nil {
		return nil, s.getAgentErr
	}
	return s.testExecutionStorage.GetAgent(ctx, id)
}

func (s *versionedExecutionStore) ListAgentVersions(ctx context.Context, id string) ([]*types.AgentNode, error) {
	return s.versions, nil
}

type errReadCloser struct{}

func (errReadCloser) Read([]byte) (int, error) { return 0, errors.New("read failed") }
func (errReadCloser) Close() error             { return nil }

type statusManagerErrorStore struct {
	*nodeRESTStorageStub
	getAgentErrByID map[string]error
	listErr         error
}

type registerCoverageStore struct {
	*nodeRESTStorageStub
	versioned       map[string]*types.AgentNode
	deleteCalls     []string
	updateLifeErr   error
	registerErr     error
}

func (s *statusManagerErrorStore) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	if err := s.getAgentErrByID[id]; err != nil {
		return nil, err
	}
	if s.nodeRESTStorageStub != nil {
		if s.nodeRESTStorageStub.agent != nil && s.nodeRESTStorageStub.agent.ID == id {
			return s.nodeRESTStorageStub.agent, nil
		}
		for _, agent := range s.nodeRESTStorageStub.listAgents {
			if agent != nil && agent.ID == id {
				return agent, nil
			}
		}
	}
	return nil, errors.New("missing")
}

func (s *statusManagerErrorStore) ListAgents(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
	if s.listErr != nil {
		return nil, s.listErr
	}
	return s.nodeRESTStorageStub.ListAgents(ctx, filters)
}

func (s *statusManagerErrorStore) UpdateAgentHealth(ctx context.Context, id string, status types.HealthStatus) error {
	return nil
}

func (s *registerCoverageStore) GetAgentVersion(ctx context.Context, id, version string) (*types.AgentNode, error) {
	if s.versioned != nil {
		return s.versioned[id+"@"+version], nil
	}
	return nil, nil
}

func (s *registerCoverageStore) DeleteAgentVersion(ctx context.Context, id, version string) error {
	s.deleteCalls = append(s.deleteCalls, id+"@"+version)
	return nil
}

func (s *registerCoverageStore) UpdateAgentLifecycleStatus(ctx context.Context, nodeID string, status types.AgentLifecycleStatus) error {
	if s.updateLifeErr != nil {
		return s.updateLifeErr
	}
	return s.nodeRESTStorageStub.UpdateAgentLifecycleStatus(ctx, nodeID, status)
}

func (s *registerCoverageStore) RegisterAgent(ctx context.Context, agent *types.AgentNode) error {
	if s.registerErr != nil {
		return s.registerErr
	}
	return s.nodeRESTStorageStub.RegisterAgent(ctx, agent)
}

func TestExecuteReasonerAndSkillHandlers_AdditionalCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("reasoner serverless propagates optional headers and context", func(t *testing.T) {
		var receivedHeaders http.Header
		var receivedBody map[string]interface{}

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header.Clone()
			defer r.Body.Close()

			require.Equal(t, "/execute", r.URL.Path)
			require.NoError(t, json.NewDecoder(r.Body).Decode(&receivedBody))

			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"ok":true}`))
		}))
		defer server.Close()

		store := newReasonerHandlerStorage(&types.AgentNode{
			ID:              "node-1",
			BaseURL:         server.URL,
			DeploymentType:  "serverless",
			Reasoners:       []types.ReasonerDefinition{{ID: "ping"}},
			HealthStatus:    types.HealthStatusActive,
			LifecycleStatus: types.AgentStatusReady,
		})

		router := gin.New()
		router.POST("/reasoners/:reasoner_id", ExecuteReasonerHandler(store))

		req := httptest.NewRequest(http.MethodPost, "/reasoners/node-1.ping", strings.NewReader(`{"input":{"topic":"status"}}`))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Workflow-ID", "wf-serverless")
		req.Header.Set("X-Session-ID", "session-1")
		req.Header.Set("X-Actor-ID", "actor-1")
		req.Header.Set("X-Parent-Workflow-ID", "wf-parent")
		req.Header.Set("X-Parent-Execution-ID", "exec-parent")
		req.Header.Set("X-Root-Workflow-ID", "wf-root")
		req.Header.Set("X-Workflow-Name", "Serverless Flow")
		req.Header.Set("X-Workflow-Tags", "ops, platform")
		req.Header.Set("X-Caller-DID", "did:web:caller.example")
		req.Header.Set("X-Target-DID", "did:web:target.example")
		req.Header.Set("X-Agent-Node-DID", "did:web:node.example")

		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		require.Equal(t, "did:web:caller.example", receivedHeaders.Get("X-Caller-DID"))
		require.Equal(t, "did:web:target.example", receivedHeaders.Get("X-Target-DID"))
		require.Equal(t, "did:web:node.example", receivedHeaders.Get("X-Agent-Node-DID"))
		require.Equal(t, "wf-parent", receivedHeaders.Get("X-Parent-Workflow-ID"))
		require.Equal(t, "exec-parent", receivedHeaders.Get("X-Parent-Execution-ID"))
		require.Equal(t, "wf-root", receivedHeaders.Get("X-Root-Workflow-ID"))
		require.Equal(t, "Serverless Flow", receivedHeaders.Get("X-Workflow-Name"))
		require.Equal(t, "ops, platform", receivedHeaders.Get("X-Workflow-Tags"))
		require.Equal(t, "session-1", receivedHeaders.Get("X-Session-ID"))
		require.Equal(t, "actor-1", receivedHeaders.Get("X-Actor-ID"))

		execCtx, ok := receivedBody["execution_context"].(map[string]interface{})
		require.True(t, ok)
		assert.Equal(t, "exec-parent", execCtx["parent_execution_id"])
		assert.Equal(t, "session-1", execCtx["session_id"])
		assert.Equal(t, "actor-1", execCtx["actor_id"])
		assert.Equal(t, "reasoner", receivedBody["type"])

		records, err := store.QueryWorkflowExecutions(context.Background(), types.WorkflowExecutionFilters{})
		require.NoError(t, err)
		require.Len(t, records, 1)
		assert.Equal(t, "actor-1", *records[0].ActorID)
		assert.Equal(t, "wf-parent", *records[0].ParentWorkflowID)
		assert.Equal(t, "exec-parent", *records[0].ParentExecutionID)
		assert.Equal(t, "wf-root", *records[0].RootWorkflowID)
		assert.Equal(t, "Serverless Flow", *records[0].WorkflowName)
		assert.Equal(t, []string{"ops", "platform"}, records[0].WorkflowTags)
	})

	t.Run("skill forwards optional headers and persists metadata", func(t *testing.T) {
		var receivedHeaders http.Header

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header.Clone()
			defer r.Body.Close()
			require.Equal(t, "/skills/summarize", r.URL.Path)
			_, _ = io.ReadAll(r.Body)

			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"summary":"ok"}`))
		}))
		defer server.Close()

		store := newReasonerHandlerStorage(newSkillAgent(server.URL))
		router := gin.New()
		router.POST("/skills/:skill_id", ExecuteSkillHandler(store))

		req := httptest.NewRequest(http.MethodPost, "/skills/node-1.summarize", strings.NewReader(`{"input":{"topic":"status"}}`))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Workflow-ID", "wf-skill")
		req.Header.Set("X-Session-ID", "session-1")
		req.Header.Set("X-Actor-ID", "actor-1")
		req.Header.Set("X-Parent-Workflow-ID", "wf-parent")
		req.Header.Set("X-Parent-Execution-ID", "exec-parent")
		req.Header.Set("X-Root-Workflow-ID", "wf-root")
		req.Header.Set("X-Workflow-Name", "Skill Flow")
		req.Header.Set("X-Workflow-Tags", "ops, platform")
		req.Header.Set("X-Caller-DID", "did:web:caller.example")
		req.Header.Set("X-Target-DID", "did:web:target.example")
		req.Header.Set("X-Agent-Node-DID", "did:web:node.example")

		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		require.Equal(t, "did:web:caller.example", receivedHeaders.Get("X-Caller-DID"))
		require.Equal(t, "did:web:target.example", receivedHeaders.Get("X-Target-DID"))
		require.Equal(t, "did:web:node.example", receivedHeaders.Get("X-Agent-Node-DID"))
		require.Equal(t, "wf-parent", receivedHeaders.Get("X-Parent-Workflow-ID"))
		require.Equal(t, "exec-parent", receivedHeaders.Get("X-Parent-Execution-ID"))
		require.Equal(t, "wf-root", receivedHeaders.Get("X-Root-Workflow-ID"))
		require.Equal(t, "Skill Flow", receivedHeaders.Get("X-Workflow-Name"))
		require.Equal(t, "ops, platform", receivedHeaders.Get("X-Workflow-Tags"))
		require.Equal(t, "actor-1", receivedHeaders.Get("X-Actor-ID"))

		records, err := store.QueryWorkflowExecutions(context.Background(), types.WorkflowExecutionFilters{})
		require.NoError(t, err)
		require.Len(t, records, 1)
		assert.Equal(t, "actor-1", *records[0].ActorID)
		assert.Equal(t, "wf-parent", *records[0].ParentWorkflowID)
		assert.Equal(t, "exec-parent", *records[0].ParentExecutionID)
		assert.Equal(t, "wf-root", *records[0].RootWorkflowID)
		assert.Equal(t, "Skill Flow", *records[0].WorkflowName)
		assert.Equal(t, []string{"ops", "platform"}, records[0].WorkflowTags)
	})
}

func TestExecuteReasonerAndSkillHandlers_TransportFailures(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name      string
		route     string
		target    string
		newStore  func() *reasonerHandlerStorage
		newHandle func(*reasonerHandlerStorage) gin.HandlerFunc
	}{
		{
			name:      "reasoner",
			route:     "/reasoners/:reasoner_id",
			target:    "/reasoners/node-1.ping",
			newStore:  func() *reasonerHandlerStorage { return newReasonerHandlerStorage(newReasonerAgent("http://127.0.0.1:1")) },
			newHandle: func(s *reasonerHandlerStorage) gin.HandlerFunc { return ExecuteReasonerHandler(s) },
		},
		{
			name:      "skill",
			route:     "/skills/:skill_id",
			target:    "/skills/node-1.summarize",
			newStore:  func() *reasonerHandlerStorage { return newReasonerHandlerStorage(newSkillAgent("http://127.0.0.1:1")) },
			newHandle: func(s *reasonerHandlerStorage) gin.HandlerFunc { return ExecuteSkillHandler(s) },
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			store := tt.newStore()
			router := gin.New()
			router.POST(tt.route, tt.newHandle(store))

			req := httptest.NewRequest(http.MethodPost, tt.target, strings.NewReader(`{"input":{"topic":"status"}}`))
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			require.Equal(t, http.StatusServiceUnavailable, resp.Code)

			records, err := store.QueryWorkflowExecutions(context.Background(), types.WorkflowExecutionFilters{})
			require.NoError(t, err)
			require.Len(t, records, 1)
			assert.Equal(t, string(types.ExecutionStatusFailed), records[0].Status)
			require.NotNil(t, records[0].ErrorMessage)
			assert.NotEmpty(t, *records[0].ErrorMessage)
		})
	}
}

func TestExecutionController_AdditionalCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("sync and async handlers surface precondition failures", func(t *testing.T) {
		previousLimiter := concurrencyLimiter
		concurrencyLimiter = &AgentConcurrencyLimiter{maxPerAgent: 1}
		require.NoError(t, concurrencyLimiter.Acquire("node-1"))
		defer func() {
			concurrencyLimiter.Release("node-1")
			concurrencyLimiter = previousLimiter
		}()

		agent := &types.AgentNode{
			ID:              "node-1",
			BaseURL:         "http://127.0.0.1:1",
			Reasoners:       []types.ReasonerDefinition{{ID: "reasoner-a"}},
			HealthStatus:    types.HealthStatusActive,
			LifecycleStatus: types.AgentStatusReady,
		}

		for _, asyncPath := range []string{"/api/v1/execute/node-1.reasoner-a", "/api/v1/execute/async/node-1.reasoner-a"} {
			store := newTestExecutionStorage(agent)
			router := gin.New()
			router.POST("/api/v1/execute/:target", ExecuteHandler(store, nil, nil, 90*time.Second, ""))
			router.POST("/api/v1/execute/async/:target", ExecuteAsyncHandler(store, nil, nil, 90*time.Second, ""))

			req := httptest.NewRequest(http.MethodPost, asyncPath, strings.NewReader(`{"input":{"foo":"bar"}}`))
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			require.Equal(t, http.StatusTooManyRequests, resp.Code)
			assert.Contains(t, resp.Body.String(), string(ErrorCategoryConcurrencyLimit))
		}
	})

	t.Run("version routing adds routed version header", func(t *testing.T) {
		var requestPath string
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			requestPath = r.URL.Path
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"ok":true}`))
		}))
		defer server.Close()

		versionedAgent := &types.AgentNode{
			ID:              "node-1",
			Version:         "v2",
			BaseURL:         server.URL,
			Reasoners:       []types.ReasonerDefinition{{ID: "reasoner-a"}},
			HealthStatus:    types.HealthStatusActive,
			LifecycleStatus: types.AgentStatusReady,
		}
		store := &versionedExecutionStore{
			testExecutionStorage: newTestExecutionStorage(nil),
			versions:             []*types.AgentNode{versionedAgent},
			getAgentErr:          errors.New("missing"),
		}

		router := gin.New()
		router.POST("/api/v1/execute/:target", ExecuteHandler(store, nil, nil, 90*time.Second, ""))

		req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		assert.Equal(t, "/reasoners/reasoner-a", requestPath)
		assert.Equal(t, "v2", resp.Header().Get("X-Routed-Version"))
	})

	t.Run("sync async-acknowledged failure returns bad gateway", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusAccepted)
		}))
		defer server.Close()

		agent := &types.AgentNode{
			ID:              "node-1",
			BaseURL:         server.URL,
			Reasoners:       []types.ReasonerDefinition{{ID: "reasoner-a"}},
			HealthStatus:    types.HealthStatusActive,
			LifecycleStatus: types.AgentStatusReady,
		}
		store := newTestExecutionStorage(agent)

		router := gin.New()
		router.POST("/api/v1/execute/:target", ExecuteHandler(store, nil, nil, 250*time.Millisecond, ""))

		req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()

		done := make(chan struct{})
		go func() {
			router.ServeHTTP(resp, req)
			close(done)
		}()

		var executionID string
		require.Eventually(t, func() bool {
			records, err := store.QueryExecutionRecords(context.Background(), types.ExecutionFilter{})
			if err != nil || len(records) == 0 {
				return false
			}
			executionID = records[0].ExecutionID
			return executionID != ""
		}, 5*time.Second, 50*time.Millisecond)

		_, err := store.UpdateExecutionRecord(context.Background(), executionID, func(current *types.Execution) (*types.Execution, error) {
			current.Status = types.ExecutionStatusFailed
			current.ResultPayload = json.RawMessage(`{"error":"bad"}`)
			current.ErrorMessage = pointerString("callback failed")
			duration := int64(25)
			current.DurationMS = &duration
			current.CompletedAt = nil
			return current, nil
		})
		require.NoError(t, err)

		store.GetExecutionEventBus().Publish(events.ExecutionEvent{
			Type:        events.ExecutionFailed,
			ExecutionID: executionID,
			WorkflowID:  "run-1",
			Status:      string(types.ExecutionStatusFailed),
			Timestamp:   time.Now(),
		})

		select {
		case <-done:
		case <-time.After(2 * time.Second):
			t.Fatal("sync execution did not complete")
		}

		require.Equal(t, http.StatusBadGateway, resp.Code)
		assert.Contains(t, resp.Body.String(), "callback failed")
	})

	t.Run("async queue saturation and sync completion saturation return errors", func(t *testing.T) {
		agent := &types.AgentNode{
			ID:              "node-1",
			BaseURL:         "http://127.0.0.1:1",
			Reasoners:       []types.ReasonerDefinition{{ID: "reasoner-a"}},
			HealthStatus:    types.HealthStatusActive,
			LifecycleStatus: types.AgentStatusReady,
		}

		prevAsyncPool := asyncPool
		asyncPool = &asyncWorkerPool{queue: make(chan asyncExecutionJob)}
		asyncPoolOnce = sync.Once{}
		asyncPoolOnce.Do(func() {})
		t.Cleanup(func() {
			asyncPool = prevAsyncPool
			asyncPoolOnce = sync.Once{}
		})

		asyncStore := newTestExecutionStorage(agent)
		asyncRouter := gin.New()
		asyncRouter.POST("/api/v1/execute/async/:target", ExecuteAsyncHandler(asyncStore, nil, nil, 90*time.Second, ""))

		asyncReq := httptest.NewRequest(http.MethodPost, "/api/v1/execute/async/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
		asyncReq.Header.Set("Content-Type", "application/json")
		asyncResp := httptest.NewRecorder()
		asyncRouter.ServeHTTP(asyncResp, asyncReq)
		require.Equal(t, http.StatusServiceUnavailable, asyncResp.Code)

		prevCompletionQueue := completionQueue
		completionQueue = make(chan completionJob, 1)
		completionQueue <- completionJob{}
		completionOnce = sync.Once{}
		completionOnce.Do(func() {})
		t.Cleanup(func() {
			completionQueue = prevCompletionQueue
			completionOnce = sync.Once{}
		})

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"ok":true}`))
		}))
		defer server.Close()

		syncStore := newTestExecutionStorage(&types.AgentNode{
			ID:              "node-1",
			BaseURL:         server.URL,
			Reasoners:       []types.ReasonerDefinition{{ID: "reasoner-a"}},
			HealthStatus:    types.HealthStatusActive,
			LifecycleStatus: types.AgentStatusReady,
		})
		syncRouter := gin.New()
		syncRouter.POST("/api/v1/execute/:target", ExecuteHandler(syncStore, nil, nil, 90*time.Second, ""))

		syncReq := httptest.NewRequest(http.MethodPost, "/api/v1/execute/node-1.reasoner-a", strings.NewReader(`{"input":{"foo":"bar"}}`))
		syncReq.Header.Set("Content-Type", "application/json")
		syncResp := httptest.NewRecorder()
		syncRouter.ServeHTTP(syncResp, syncReq)
		require.Equal(t, http.StatusBadRequest, syncResp.Code)
		assert.Contains(t, syncResp.Body.String(), "completion queue is full")
	})

	t.Run("workflow status updates normalize reasons and clear terminal fields", func(t *testing.T) {
		store := newTestExecutionStorage(nil)
		now := time.Now().UTC()
		duration := int64(10)
		require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
			ExecutionID:  "exec-1",
			WorkflowID:   "run-1",
			Status:       string(types.ExecutionStatusSucceeded),
			CompletedAt:  &now,
			DurationMS:   &duration,
			StatusReason: pointerString("old"),
			StartedAt:    now,
		}))

		controller := newExecutionController(store, nil, nil, 90*time.Second, "")
		reason := "  waiting_for_input  "
		controller.updateWorkflowExecutionStatus(context.Background(), "exec-1", string(types.ExecutionStatusWaiting), &reason)

		updated, err := store.GetWorkflowExecution(context.Background(), "exec-1")
		require.NoError(t, err)
		require.NotNil(t, updated)
		assert.Equal(t, string(types.ExecutionStatusWaiting), updated.Status)
		require.NotNil(t, updated.StatusReason)
		assert.Equal(t, "waiting_for_input", *updated.StatusReason)
		assert.Nil(t, updated.CompletedAt)
		assert.Nil(t, updated.DurationMS)
	})

	t.Run("callAgent forwards internal auth and did headers", func(t *testing.T) {
		var receivedHeaders http.Header
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header.Clone()
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"ok":true}`))
		}))
		defer server.Close()

		agent := &types.AgentNode{
			ID:              "node-1",
			BaseURL:         server.URL,
			DeploymentType:  "serverless",
			Reasoners:       []types.ReasonerDefinition{{ID: "reasoner-a"}},
			HealthStatus:    types.HealthStatusActive,
			LifecycleStatus: types.AgentStatusReady,
		}
		store := newTestExecutionStorage(agent)
		controller := newExecutionController(store, nil, nil, 90*time.Second, "internal-token")
		plan := &preparedExecution{
			exec: &types.Execution{
				ExecutionID: "exec-1",
				RunID:       "run-1",
			},
			requestBody: []byte(`{"foo":"bar"}`),
			agent:       agent,
			target:      &parsedTarget{NodeID: "node-1", TargetName: "reasoner-a"},
			callerDID:   "did:web:caller.example",
			targetDID:   "did:web:target.example",
		}

		_, _, _, err := controller.callAgent(context.Background(), plan)
		require.NoError(t, err)
		assert.Equal(t, "Bearer internal-token", receivedHeaders.Get("Authorization"))
		assert.Equal(t, "did:web:caller.example", receivedHeaders.Get("X-Caller-DID"))
		assert.Equal(t, "did:web:target.example", receivedHeaders.Get("X-Target-DID"))
	})
}

func TestHandleVerifyAuditBundle_AdditionalCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("generic read errors return bad request", func(t *testing.T) {
		router := gin.New()
		router.POST("/verify", func(c *gin.Context) {
			c.Request.Body = errReadCloser{}
			HandleVerifyAuditBundle(c)
		})

		req := httptest.NewRequest(http.MethodPost, "/verify", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusBadRequest, resp.Code)
		assert.Contains(t, resp.Body.String(), "failed to read request body")
	})

	t.Run("malformed provenance returns unprocessable entity", func(t *testing.T) {
		router := gin.New()
		router.POST("/verify", HandleVerifyAuditBundle)

		req := httptest.NewRequest(http.MethodPost, "/verify?verbose=true", strings.NewReader(`not-json`))
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusUnprocessableEntity, resp.Code)
	})
}

func TestDiscoveryHelpers_AdditionalCoverage(t *testing.T) {
	t.Run("parameter and parsing helpers", func(t *testing.T) {
		assert.Equal(t, "because", (&parameterError{Reason: "because"}).Error())
		assert.Equal(t, "invalid limit parameter", (&parameterError{Parameter: "limit"}).Error())

		_, err := parseInt("oops", 0, 10)
		require.Error(t, err)
		_, err = parseInt("11", 0, 10)
		require.Error(t, err)

		assert.True(t, matchesPattern("anything", "*"))
		assert.True(t, matchesTags([]string{"ml"}, nil))
		assert.False(t, matchesTags([]string{"ops"}, []string{"ml*"}))
		assert.Equal(t, []string{"a", "b"}, dedupeStrings([]string{"", "a", "a", "b"}))
	})

	t.Run("metadata extraction handles nil and typed variants", func(t *testing.T) {
		assert.Nil(t, extractDescription(types.AgentMetadata{}, "id"))
		assert.Nil(t, extractDescription(types.AgentMetadata{Custom: map[string]interface{}{"descriptions": map[string]interface{}{"id": " "}}}, "id"))

		meta := types.AgentMetadata{Custom: map[string]interface{}{
			"descriptions": map[string]interface{}{"id": "desc"},
			"examples": map[string]interface{}{
				"typed": []map[string]interface{}{{"name": "typed"}},
				"list":  []interface{}{map[string]interface{}{"name": "basic"}, "skip"},
			},
		}}
		require.NotNil(t, extractDescription(meta, "id"))
		assert.Len(t, extractExamples(meta, "typed"), 1)
		assert.Len(t, extractExamples(meta, "list"), 1)
		assert.Nil(t, extractExamples(types.AgentMetadata{Custom: map[string]interface{}{"examples": "bad"}}, "id"))
	})

	t.Run("build discovery response filters by health and offset", func(t *testing.T) {
		health := types.HealthStatusActive
		filters := DiscoveryFilters{
			HealthStatus:        &health,
			ReasonerPattern:     optionalString("sum*"),
			Tags:                []string{"summ*"},
			IncludeInputSchema:  true,
			IncludeDescriptions: true,
			IncludeExamples:     true,
			Limit:               1,
			Offset:              5,
		}

		resp := buildDiscoveryResponse(buildDiscoveryAgents(), filters)
		assert.Equal(t, 1, resp.TotalAgents)
		assert.Empty(t, resp.Capabilities)
		assert.False(t, resp.Pagination.HasMore)
	})
}

func TestNodeHelpers_AdditionalCoverage(t *testing.T) {
	t.Run("normalize candidate and resolution edge cases", func(t *testing.T) {
		_, err := normalizeCandidate("", "8080")
		require.Error(t, err)

		normalized, err := normalizeCandidate("[2001:db8::1]", "")
		require.NoError(t, err)
		assert.Equal(t, "http://[2001:db8::1]", normalized)

		resolved, normalizedList, probeResults := resolveCallbackCandidates([]string{"", "://bad"}, "8080")
		assert.Empty(t, resolved)
		assert.Nil(t, normalizedList)
		assert.Nil(t, probeResults)
	})

	t.Run("serverless discovery validation and host allowlist branches", func(t *testing.T) {
		tests := []struct {
			rawURL  string
			allowed []string
			wantErr string
		}{
			{rawURL: "ftp://example.com", wantErr: "must use http or https"},
			{rawURL: "https://user@example.com", wantErr: "must not include user info"},
			{rawURL: "https://example.com/path?q=1", wantErr: "must not include query parameters or fragments"},
		}

		for _, tt := range tests {
			_, err := normalizeServerlessDiscoveryURL(tt.rawURL, tt.allowed)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)
		}

		assert.True(t, isServerlessDiscoveryHostAllowed("10.1.2.3", []string{"10.0.0.0/8"}))
		assert.True(t, isServerlessDiscoveryHostAllowed("api.trusted.example", []string{"*.trusted.example"}))
		assert.False(t, isServerlessDiscoveryHostAllowed("trusted.example", []string{"*.trusted.example"}))
	})

	t.Run("async heartbeat exits when version lookup fails", func(t *testing.T) {
		store := &heartbeatAsyncStorageStub{
			nodeRESTStorageStub: &nodeRESTStorageStub{getVersionErr: errors.New("missing")},
		}
		processHeartbeatAsync(store, nil, "node-1", "v1", &CachedNodeData{LastDBUpdate: time.Now().UTC()})

		// Assert that the async goroutine does not record a heartbeat when version lookup fails.
		require.Never(t, func() bool { return len(store.heartbeats) > 0 }, 200*time.Millisecond, 10*time.Millisecond)
	})

	t.Run("status handlers exercise error branches", func(t *testing.T) {
		makeManager := func(store *statusManagerErrorStore) *services.StatusManager {
			return services.NewStatusManager(store, services.StatusManagerConfig{}, nil, nil)
		}

		t.Run("get and refresh return storage-backed errors", func(t *testing.T) {
			store := &statusManagerErrorStore{
				nodeRESTStorageStub: &nodeRESTStorageStub{},
				getAgentErrByID:     map[string]error{"node-err": errors.New("boom")},
			}
			router := gin.New()
			router.GET("/nodes/:node_id/status", GetNodeStatusHandler(makeManager(store)))
			router.POST("/nodes/:node_id/status/refresh", RefreshNodeStatusHandler(makeManager(store)))

			req := httptest.NewRequest(http.MethodGet, "/nodes/node-err/status", nil)
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)
			require.Equal(t, http.StatusNotFound, resp.Code)

			req = httptest.NewRequest(http.MethodPost, "/nodes/node-err/status/refresh", nil)
			resp = httptest.NewRecorder()
			router.ServeHTTP(resp, req)
			require.Equal(t, http.StatusInternalServerError, resp.Code)
		})

		t.Run("bulk status returns multi-status when only some nodes fail", func(t *testing.T) {
			now := time.Now().UTC()
			store := &statusManagerErrorStore{
				nodeRESTStorageStub: &nodeRESTStorageStub{
					listAgents: []*types.AgentNode{
						{ID: "node-1", HealthStatus: types.HealthStatusActive, LifecycleStatus: types.AgentStatusReady, LastHeartbeat: now},
					},
				},
				getAgentErrByID: map[string]error{"node-2": errors.New("missing")},
			}
			router := gin.New()
			router.POST("/nodes/bulk-status", BulkNodeStatusHandler(makeManager(store), store))

			req := httptest.NewRequest(http.MethodPost, "/nodes/bulk-status", strings.NewReader(`{"node_ids":["node-1","node-2"]}`))
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			require.Equal(t, 207, resp.Code)
			assert.Contains(t, resp.Body.String(), `"failed":1`)
		})

		t.Run("refresh all handles list failures and all-node failures", func(t *testing.T) {
			router := gin.New()

			listErrStore := &statusManagerErrorStore{
				nodeRESTStorageStub: &nodeRESTStorageStub{},
				listErr:             errors.New("list failed"),
			}
			router.POST("/nodes/status/refresh-all/list", RefreshAllNodeStatusHandler(makeManager(listErrStore), listErrStore))

			req := httptest.NewRequest(http.MethodPost, "/nodes/status/refresh-all/list", nil)
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)
			require.Equal(t, http.StatusInternalServerError, resp.Code)

			now := time.Now().UTC()
			allFailStore := &statusManagerErrorStore{
				nodeRESTStorageStub: &nodeRESTStorageStub{
					listAgents: []*types.AgentNode{
						{ID: "node-1", HealthStatus: types.HealthStatusUnknown, LifecycleStatus: types.AgentStatusStarting, LastHeartbeat: now},
						{ID: "node-2", HealthStatus: types.HealthStatusUnknown, LifecycleStatus: types.AgentStatusStarting, LastHeartbeat: now},
					},
				},
				getAgentErrByID: map[string]error{
					"node-1": errors.New("missing"),
					"node-2": errors.New("missing"),
				},
			}
			router.POST("/nodes/status/refresh-all/fail", RefreshAllNodeStatusHandler(makeManager(allFailStore), allFailStore))

			req = httptest.NewRequest(http.MethodPost, "/nodes/status/refresh-all/fail", nil)
			resp = httptest.NewRecorder()
			router.ServeHTTP(resp, req)
			require.Equal(t, http.StatusInternalServerError, resp.Code)
			assert.Contains(t, resp.Body.String(), `"failed":2`)
		})
	})

	t.Run("shutdown covers default lookup and not found", func(t *testing.T) {
		now := time.Now().UTC()
		statusStore := &statusManagerStorageStub{
			nodeRESTStorageStub: &nodeRESTStorageStub{
				agent: &types.AgentNode{
					ID:              "node-1",
					Version:         "v1",
					HealthStatus:    types.HealthStatusActive,
					LifecycleStatus: types.AgentStatusReady,
					LastHeartbeat:   now,
				},
			},
		}
		statusManager := services.NewStatusManager(statusStore, services.StatusManagerConfig{}, nil, nil)
		router := gin.New()
		router.POST("/nodes/:node_id/shutdown", NodeShutdownHandler(statusStore, statusManager, nil))

		req := httptest.NewRequest(http.MethodPost, "/nodes/node-1/shutdown", strings.NewReader(`{"reason":"maintenance"}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusAccepted, resp.Code)

		missingStore := &nodeRESTStorageStub{getAgentErr: errors.New("missing")}
		router = gin.New()
		router.POST("/nodes/:node_id/shutdown", NodeShutdownHandler(missingStore, nil, nil))

		req = httptest.NewRequest(http.MethodPost, "/nodes/missing/shutdown", strings.NewReader(`{"reason":"maintenance"}`))
		req.Header.Set("Content-Type", "application/json")
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusNotFound, resp.Code)
	})

	t.Run("heartbeat covers parse fallback, pending approval, and legacy update failure", func(t *testing.T) {
		heartbeatCache = &HeartbeatCache{nodes: make(map[string]*CachedNodeData)}

		presence := services.NewPresenceManager(nil, services.PresenceManagerConfig{})
		presence.Touch("node-1", "v1", time.Now().UTC())

		pendingStore := &statusManagerStorageStub{
			nodeRESTStorageStub: &nodeRESTStorageStub{
				agent: &types.AgentNode{
					ID:              "node-1",
					Version:         "v1",
					BaseURL:         "https://example.com",
					LifecycleStatus: types.AgentStatusPendingApproval,
				},
			},
		}
		manager := services.NewStatusManager(pendingStore, services.StatusManagerConfig{}, nil, nil)
		router := gin.New()
		router.POST("/nodes/:node_id/heartbeat", HeartbeatHandler(pendingStore, nil, nil, manager, presence))

		req := httptest.NewRequest(http.MethodPost, "/nodes/node-1/heartbeat", strings.NewReader(`{"status":"ready","health_score":91}`))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusOK, resp.Code)

		heartbeatCache = &HeartbeatCache{nodes: make(map[string]*CachedNodeData)}
		parseFallbackStore := &nodeRESTStorageStub{
			agent: &types.AgentNode{ID: "node-2", Version: "v1", LifecycleStatus: types.AgentStatusReady},
		}
		router = gin.New()
		router.POST("/nodes/:node_id/heartbeat", HeartbeatHandler(parseFallbackStore, nil, nil, nil, nil))

		req = httptest.NewRequest(http.MethodPost, "/nodes/node-2/heartbeat", strings.NewReader(`{"status":`))
		req.Header.Set("Content-Type", "application/json")
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusOK, resp.Code)

		heartbeatCache = &HeartbeatCache{nodes: make(map[string]*CachedNodeData)}
		legacyErrStore := &registerCoverageStore{
			nodeRESTStorageStub: &nodeRESTStorageStub{
				agent: &types.AgentNode{ID: "node-3", Version: "v1", LifecycleStatus: types.AgentStatusOffline},
			},
			updateLifeErr: errors.New("lifecycle failed"),
		}
		router = gin.New()
		router.POST("/nodes/:node_id/heartbeat", HeartbeatHandler(legacyErrStore, nil, nil, nil, nil))

		req = httptest.NewRequest(http.MethodPost, "/nodes/node-3/heartbeat", strings.NewReader(`{"status":"ready"}`))
		req.Header.Set("Content-Type", "application/json")
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusOK, resp.Code)
	})

	t.Run("register handler covers invalid bodies, version cleanup, and registration errors", func(t *testing.T) {
		router := gin.New()
		store := &registerCoverageStore{nodeRESTStorageStub: &nodeRESTStorageStub{}}
		router.POST("/nodes/register", RegisterNodeHandler(store, nil, nil, nil, nil, nil))

		req := httptest.NewRequest(http.MethodPost, "/nodes/register", strings.NewReader("{"))
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusBadRequest, resp.Code)

		presence := services.NewPresenceManager(nil, services.PresenceManagerConfig{})
		store = &registerCoverageStore{
			nodeRESTStorageStub: &nodeRESTStorageStub{},
			versioned: map[string]*types.AgentNode{
				"node-v@v2": {ID: "node-v", Version: "v2", LifecycleStatus: types.AgentStatusReady},
				"node-v@":   {ID: "node-v", Version: "", LifecycleStatus: types.AgentStatusOffline},
			},
		}
		router = gin.New()
		router.POST("/nodes/register", RegisterNodeHandler(store, nil, nil, presence, nil, nil))

		req = httptest.NewRequest(http.MethodPost, "/nodes/register", strings.NewReader(`{"id":"node-v","version":"v2","base_url":"https://example.com","callback_discovery":{"mode":"manual"}}`))
		req.Header.Set("Content-Type", "application/json")
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusCreated, resp.Code)
		assert.Contains(t, store.deleteCalls, "node-v@")
		assert.True(t, presence.HasLease("node-v"))

		store = &registerCoverageStore{
			nodeRESTStorageStub: &nodeRESTStorageStub{},
			registerErr:         errors.New("store failed"),
		}
		router = gin.New()
		router.POST("/nodes/register", RegisterNodeHandler(store, nil, nil, nil, nil, nil))

		req = httptest.NewRequest(http.MethodPost, "/nodes/register", strings.NewReader(`{"id":"node-fail","base_url":"https://example.com"}`))
		req.Header.Set("Content-Type", "application/json")
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusInternalServerError, resp.Code)
	})
}

func TestAsyncExecutionJob_ProcessAdditionalCoverage(t *testing.T) {
	t.Run("returns early for cancelled execution and releases slot", func(t *testing.T) {
		previousLimiter := concurrencyLimiter
		concurrencyLimiter = &AgentConcurrencyLimiter{maxPerAgent: 2}
		require.NoError(t, concurrencyLimiter.Acquire("node-1"))
		defer func() { concurrencyLimiter = previousLimiter }()

		store := newTestExecutionStorage(nil)
		require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
			ExecutionID: "exec-1",
			Status:      types.ExecutionStatusCancelled,
			StartedAt:   time.Now().UTC(),
			CreatedAt:   time.Now().UTC(),
			UpdatedAt:   time.Now().UTC(),
		}))

		job := asyncExecutionJob{
			controller: newExecutionController(store, nil, nil, time.Second, ""),
			plan: preparedExecution{
				exec:   &types.Execution{ExecutionID: "exec-1"},
				target: &parsedTarget{NodeID: "node-1"},
			},
		}

		job.process()
		assert.Equal(t, int64(0), concurrencyLimiter.GetRunningCount("node-1"))
	})

	t.Run("async acceptance does not enqueue completion", func(t *testing.T) {
		var calls int32
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			atomic.AddInt32(&calls, 1)
			w.WriteHeader(http.StatusAccepted)
		}))
		defer server.Close()

		agent := &types.AgentNode{
			ID:              "node-1",
			BaseURL:         server.URL,
			Reasoners:       []types.ReasonerDefinition{{ID: "reasoner-a"}},
			HealthStatus:    types.HealthStatusActive,
			LifecycleStatus: types.AgentStatusReady,
		}

		store := newTestExecutionStorage(agent)
		require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
			ExecutionID: "exec-2",
			RunID:       "run-2",
			Status:      types.ExecutionStatusRunning,
			StartedAt:   time.Now().UTC(),
			CreatedAt:   time.Now().UTC(),
			UpdatedAt:   time.Now().UTC(),
		}))

		job := asyncExecutionJob{
			controller: newExecutionController(store, nil, nil, time.Second, ""),
			plan: preparedExecution{
				exec:        &types.Execution{ExecutionID: "exec-2", RunID: "run-2"},
				requestBody: []byte(`{"foo":"bar"}`),
				agent:       agent,
				target:      &parsedTarget{NodeID: "node-1", TargetName: "reasoner-a"},
			},
		}

		job.process()
		assert.Equal(t, int32(1), atomic.LoadInt32(&calls))
	})
}
