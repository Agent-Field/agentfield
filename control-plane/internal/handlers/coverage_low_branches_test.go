package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type hierarchyErrorStore struct {
	*testExecutionStorage
	err error
}

type statusManagerStorageStub struct {
	*nodeRESTStorageStub
	healthUpdates []types.HealthStatus
}

type registerNodeStorageStub struct {
	*nodeRESTStorageStub
	registerErr error
}

type heartbeatAsyncStorageStub struct {
	*nodeRESTStorageStub
	versions []*types.AgentNode
}

func (s *hierarchyErrorStore) GetWorkflowExecution(ctx context.Context, executionID string) (*types.WorkflowExecution, error) {
	if s.err != nil {
		return nil, s.err
	}
	return s.testExecutionStorage.GetWorkflowExecution(ctx, executionID)
}

func (s *statusManagerStorageStub) UpdateAgentHealth(ctx context.Context, id string, status types.HealthStatus) error {
	s.healthUpdates = append(s.healthUpdates, status)
	if s.agent != nil && s.agent.ID == id {
		s.agent.HealthStatus = status
	}
	for _, agent := range s.listAgents {
		if agent != nil && agent.ID == id {
			agent.HealthStatus = status
		}
	}
	return nil
}

func (s *statusManagerStorageStub) UpdateAgentHeartbeat(ctx context.Context, id, version string, ts time.Time) error {
	if s.agent != nil && s.agent.ID == id {
		s.agent.LastHeartbeat = ts
	}
	for _, agent := range s.listAgents {
		if agent != nil && agent.ID == id {
			agent.LastHeartbeat = ts
		}
	}
	return s.nodeRESTStorageStub.UpdateAgentHeartbeat(ctx, id, version, ts)
}

func (s *statusManagerStorageStub) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	if s.nodeRESTStorageStub.agent != nil && s.nodeRESTStorageStub.agent.ID == id {
		return s.nodeRESTStorageStub.agent, nil
	}
	for _, agent := range s.nodeRESTStorageStub.listAgents {
		if agent != nil && agent.ID == id {
			return agent, nil
		}
	}
	return nil, nil
}

func (s *registerNodeStorageStub) RegisterAgent(ctx context.Context, agent *types.AgentNode) error {
	if s.registerErr != nil {
		return s.registerErr
	}
	return s.nodeRESTStorageStub.RegisterAgent(ctx, agent)
}

func (s *heartbeatAsyncStorageStub) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	if s.nodeRESTStorageStub.getAgentErr != nil {
		return nil, s.nodeRESTStorageStub.getAgentErr
	}
	return s.nodeRESTStorageStub.GetAgent(ctx, id)
}

func (s *heartbeatAsyncStorageStub) ListAgentVersions(ctx context.Context, id string) ([]*types.AgentNode, error) {
	return s.versions, nil
}

func TestRequestApprovalHandler_WrapperHandlers(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("request approval wrapper", func(t *testing.T) {
		store := newTestExecutionStorage(&types.AgentNode{ID: "agent-1"})
		now := time.Now().UTC()
		require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
			ExecutionID: "exec-1",
			RunID:       "run-1",
			AgentNodeID: "agent-1",
			Status:      types.ExecutionStatusRunning,
			StartedAt:   now,
			CreatedAt:   now,
		}))
		require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
			ExecutionID: "exec-1",
			WorkflowID:  "wf-1",
			RunID:       lowBranchStringPtr("run-1"),
			AgentNodeID: "agent-1",
			Status:      types.ExecutionStatusRunning,
			StartedAt:   now,
		}))

		router := gin.New()
		router.POST("/executions/:execution_id/request-approval", RequestApprovalHandler(store))

		req := httptest.NewRequest(http.MethodPost, "/executions/exec-1/request-approval", strings.NewReader(`{"approval_request_id":"req-1"}`))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		wfExec, err := store.GetWorkflowExecution(context.Background(), "exec-1")
		require.NoError(t, err)
		require.NotNil(t, wfExec)
		assert.Equal(t, types.ExecutionStatusWaiting, wfExec.Status)
	})

	t.Run("get approval status wrapper", func(t *testing.T) {
		store := newTestExecutionStorage(&types.AgentNode{ID: "agent-1"})
		now := time.Now().UTC()
		status := "approved"
		requestID := "req-1"
		requestURL := "https://example.com/approvals/req-1"
		require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
			ExecutionID:         "exec-2",
			WorkflowID:          "wf-1",
			AgentNodeID:         "agent-1",
			ApprovalRequestID:   &requestID,
			ApprovalRequestURL:  &requestURL,
			ApprovalStatus:      &status,
			ApprovalRequestedAt: &now,
		}))

		router := gin.New()
		router.GET("/executions/:execution_id/approval-status", GetApprovalStatusHandler(store))

		req := httptest.NewRequest(http.MethodGet, "/executions/exec-2/approval-status", nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		assert.Contains(t, rec.Body.String(), `"status":"approved"`)
	})
}

func TestSelectVersionedAgent(t *testing.T) {
	t.Run("returns nil when no node is eligible", func(t *testing.T) {
		atomic.StoreUint64(&versionRoundRobinCounter, 0)
		selected, version := selectVersionedAgent([]*types.AgentNode{
			{ID: "offline-1", Version: "v1", LifecycleStatus: types.AgentStatusOffline},
		})
		assert.Nil(t, selected)
		assert.Empty(t, version)
	})

	t.Run("falls back to non-offline nodes when none are healthy", func(t *testing.T) {
		atomic.StoreUint64(&versionRoundRobinCounter, 0)
		selected, version := selectVersionedAgent([]*types.AgentNode{
			{ID: "offline-1", Version: "v1", LifecycleStatus: types.AgentStatusOffline},
			{ID: "starting-1", Version: "v2", LifecycleStatus: types.AgentStatusStarting},
		})
		require.NotNil(t, selected)
		assert.Equal(t, "starting-1", selected.ID)
		assert.Equal(t, "v2", version)
	})

	t.Run("uses simple round robin when healthy weights are equal", func(t *testing.T) {
		atomic.StoreUint64(&versionRoundRobinCounter, 0)
		versions := []*types.AgentNode{
			{ID: "node-a", Version: "v1", HealthStatus: types.HealthStatusActive, LifecycleStatus: types.AgentStatusReady, TrafficWeight: 100},
			{ID: "node-b", Version: "v2", HealthStatus: types.HealthStatusActive, LifecycleStatus: types.AgentStatusReady, TrafficWeight: 100},
		}

		first, firstVersion := selectVersionedAgent(versions)
		second, secondVersion := selectVersionedAgent(versions)

		require.NotNil(t, first)
		require.NotNil(t, second)
		assert.Equal(t, "node-a", first.ID)
		assert.Equal(t, "v1", firstVersion)
		assert.Equal(t, "node-b", second.ID)
		assert.Equal(t, "v2", secondVersion)
	})

	t.Run("uses weighted selection when healthy weights differ", func(t *testing.T) {
		atomic.StoreUint64(&versionRoundRobinCounter, 0)
		versions := []*types.AgentNode{
			{ID: "node-a", Version: "v1", HealthStatus: types.HealthStatusActive, LifecycleStatus: types.AgentStatusReady, TrafficWeight: 1},
			{ID: "node-b", Version: "v2", HealthStatus: types.HealthStatusActive, LifecycleStatus: types.AgentStatusReady, TrafficWeight: 3},
		}

		first, firstVersion := selectVersionedAgent(versions)
		second, secondVersion := selectVersionedAgent(versions)

		require.NotNil(t, first)
		require.NotNil(t, second)
		assert.Equal(t, "node-a", first.ID)
		assert.Equal(t, "v1", firstVersion)
		assert.Equal(t, "node-b", second.ID)
		assert.Equal(t, "v2", secondVersion)
	})
}

func TestNormalizeWebhookRequest(t *testing.T) {
	longValue := strings.Repeat("a", maxWebhookSecretLength+1)
	longHeader := strings.Repeat("h", maxWebhookHeaderLength+1)

	tests := []struct {
		name    string
		req     *WebhookRequest
		wantErr string
		assert  func(t *testing.T, got *normalizedWebhookConfig)
	}{
		{
			name:   "nil request",
			req:    nil,
			assert: func(t *testing.T, got *normalizedWebhookConfig) { assert.Nil(t, got) },
		},
		{
			name:    "blank url",
			req:     &WebhookRequest{URL: "   "},
			wantErr: "webhook.url is required",
		},
		{
			name:    "missing scheme and host",
			req:     &WebhookRequest{URL: "example.com/path"},
			wantErr: "webhook url must include scheme and host",
		},
		{
			name:    "unsupported scheme",
			req:     &WebhookRequest{URL: "ftp://example.com/path"},
			wantErr: "webhook url must use http or https",
		},
		{
			name:    "embedded credentials",
			req:     &WebhookRequest{URL: "https://user:pass@example.com/path"},
			wantErr: "webhook url must not contain embedded credentials",
		},
		{
			name:    "ssrf loopback ipv4",
			req:     &WebhookRequest{URL: "http://127.0.0.1:9999/cb"},
			wantErr: "webhook url rejected",
		},
		{
			name:    "ssrf cloud metadata",
			req:     &WebhookRequest{URL: "http://169.254.169.254/latest/meta-data/"},
			wantErr: "webhook url rejected",
		},
		{
			name:    "ssrf rfc1918 10.x",
			req:     &WebhookRequest{URL: "http://10.0.0.1:8080/cb"},
			wantErr: "webhook url rejected",
		},
		{
			name:    "ssrf rfc1918 192.168.x",
			req:     &WebhookRequest{URL: "http://192.168.1.1:8080/cb"},
			wantErr: "webhook url rejected",
		},
		{
			name:    "ssrf rfc1918 172.16.x",
			req:     &WebhookRequest{URL: "http://172.16.0.1:8080/cb"},
			wantErr: "webhook url rejected",
		},
		{
			name:    "ssrf localhost",
			req:     &WebhookRequest{URL: "http://localhost:9999/cb"},
			wantErr: "webhook url rejected",
		},
		{
			name:    "ssrf ipv6 loopback",
			req:     &WebhookRequest{URL: "http://[::1]:9999/cb"},
			wantErr: "webhook url rejected",
		},
		{
			name:    "ssrf unspecified",
			req:     &WebhookRequest{URL: "http://0.0.0.0:9999/cb"},
			wantErr: "webhook url rejected",
		},
		{
			name:    "too many headers",
			req:     &WebhookRequest{URL: "https://example.com/path", Headers: func() map[string]string { m := map[string]string{}; for i := 0; i < maxWebhookHeaders+1; i++ { m[time.Unix(int64(i), 0).UTC().Format(time.RFC3339)] = "v" }; return m }()},
			wantErr: "webhook.headers supports at most",
		},
		{
			name:    "header name too long",
			req:     &WebhookRequest{URL: "https://example.com/path", Headers: map[string]string{longHeader: "value"}},
			wantErr: "webhook header name",
		},
		{
			name:    "secret too long",
			req:     &WebhookRequest{URL: "https://example.com/path", Secret: longValue},
			wantErr: "webhook secret exceeds",
		},
		{
			name: "valid request",
			req: &WebhookRequest{
				URL:    " https://example.com/path#fragment ",
				Secret: "  top-secret  ",
				Headers: map[string]string{
					" X-Test ": " value ",
					"   ":      "ignored",
				},
			},
			assert: func(t *testing.T, got *normalizedWebhookConfig) {
				require.NotNil(t, got)
				assert.Equal(t, "https://example.com/path", got.URL)
				require.NotNil(t, got.Secret)
				assert.Equal(t, "top-secret", *got.Secret)
				assert.Equal(t, map[string]string{"X-Test": "value"}, got.Headers)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := normalizeWebhookRequest(tt.req)
			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErr)
				return
			}
			require.NoError(t, err)
			if tt.assert != nil {
				tt.assert(t, got)
			}
		})
	}
}

func TestClassifyCallError(t *testing.T) {
	tests := []struct {
		name string
		err  *callError
		want ErrorCategory
	}{
		{
			name: "server error",
			err:  &callError{statusCode: http.StatusInternalServerError},
			want: ErrorCategoryAgentError,
		},
		{
			name: "timeout status",
			err:  &callError{statusCode: http.StatusRequestTimeout},
			want: ErrorCategoryAgentTimeout,
		},
		{
			name: "invalid json body",
			err:  &callError{statusCode: http.StatusBadRequest, body: []byte("not-json")},
			want: ErrorCategoryBadResponse,
		},
		{
			name: "valid json body",
			err:  &callError{statusCode: http.StatusBadGateway, body: []byte(`{"error":"upstream"}`)},
			want: ErrorCategoryAgentError,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, classifyCallError(tt.err, tt.err))
		})
	}
}

func TestDeriveWorkflowHierarchy(t *testing.T) {
	t.Run("uses run id when execution has no parent", func(t *testing.T) {
		store := newTestExecutionStorage(nil)
		controller := newExecutionController(store, nil, nil, 0, "")
		root, parent, depth := controller.deriveWorkflowHierarchy(context.Background(), &types.Execution{
			ExecutionID: "exec-1",
			RunID:       "run-1",
		})
		require.NotNil(t, root)
		assert.Equal(t, "run-1", *root)
		assert.Nil(t, parent)
		assert.Equal(t, 0, depth)
	})

	t.Run("uses parent workflow when parent exists", func(t *testing.T) {
		store := newTestExecutionStorage(nil)
		parentID := "parent-exec"
		rootWorkflowID := "root-wf"
		require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
			ExecutionID:     parentID,
			WorkflowID:      "parent-wf",
			RootWorkflowID:  &rootWorkflowID,
			WorkflowDepth:   2,
			AgentNodeID:     "agent-1",
			Status:          types.ExecutionStatusRunning,
			StartedAt:       time.Now().UTC(),
		}))

		controller := newExecutionController(store, nil, nil, 0, "")
		root, parent, depth := controller.deriveWorkflowHierarchy(context.Background(), &types.Execution{
			ExecutionID:       "exec-2",
			RunID:             "run-2",
			ParentExecutionID: &parentID,
		})
		require.NotNil(t, root)
		require.NotNil(t, parent)
		assert.Equal(t, "root-wf", *root)
		assert.Equal(t, "parent-wf", *parent)
		assert.Equal(t, 3, depth)
	})

	t.Run("falls back when parent lookup fails", func(t *testing.T) {
		store := &hierarchyErrorStore{
			testExecutionStorage: newTestExecutionStorage(nil),
			err:                  errors.New("boom"),
		}
		controller := newExecutionController(store, nil, nil, 0, "")
		parentID := "missing-parent"
		root, parent, depth := controller.deriveWorkflowHierarchy(context.Background(), &types.Execution{
			ExecutionID:       "exec-3",
			RunID:             "run-3",
			ParentExecutionID: &parentID,
		})
		require.NotNil(t, root)
		assert.Equal(t, "run-3", *root)
		assert.Nil(t, parent)
		assert.Equal(t, 1, depth)
	})
}

func TestExecutionCleanupService_ForceCleanup(t *testing.T) {
	t.Run("cleans until final partial batch and updates metrics", func(t *testing.T) {
		store := &cleanupStoreMock{
			cleanupResponses: []cleanupResponse{
				{count: 2},
				{count: 1},
			},
		}
		svc := NewExecutionCleanupService(store, config.ExecutionCleanupConfig{
			BatchSize:       2,
			RetentionPeriod: 24 * time.Hour,
		})

		cleaned, err := svc.ForceCleanup(context.Background())
		require.NoError(t, err)
		assert.Equal(t, 3, cleaned)

		calls := store.getCleanupCalls()
		require.Len(t, calls, 2)
		assert.Equal(t, 2, calls[0].batchSize)

		total, lastCleanupTime, lastErr := svc.GetMetrics()
		assert.Equal(t, int64(3), total)
		assert.NoError(t, lastErr)
		assert.False(t, lastCleanupTime.IsZero())
	})

	t.Run("returns storage error", func(t *testing.T) {
		store := &cleanupStoreMock{
			cleanupResponses: []cleanupResponse{
				{err: errors.New("cleanup failed")},
			},
		}
		svc := NewExecutionCleanupService(store, config.ExecutionCleanupConfig{
			BatchSize:       5,
			RetentionPeriod: time.Hour,
		})

		cleaned, err := svc.ForceCleanup(context.Background())
		require.Error(t, err)
		assert.Equal(t, 0, cleaned)
	})

	t.Run("honors cancelled context between batches", func(t *testing.T) {
		store := &cleanupStoreMock{
			cleanupResponses: []cleanupResponse{
				{count: 1},
			},
		}
		svc := NewExecutionCleanupService(store, config.ExecutionCleanupConfig{
			BatchSize:       1,
			RetentionPeriod: time.Hour,
		})

		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		cleaned, err := svc.ForceCleanup(ctx)
		require.Error(t, err)
		assert.Equal(t, 1, cleaned)
		assert.ErrorIs(t, err, context.Canceled)
	})
}

func TestSchemaHelpersAndHeartbeatCache(t *testing.T) {
	t.Run("schema helpers handle empty invalid and valid values", func(t *testing.T) {
		assert.Nil(t, decodeSchema(nil))
		assert.Nil(t, decodeSchema([]byte("{invalid")))
		assert.Equal(t, "", encodeSchema(nil))
		assert.Equal(t, "", encodeSchema(map[string]interface{}{}))
		assert.JSONEq(t, `{"enabled":true}`, encodeSchema(map[string]interface{}{"enabled": true}))

		decoded := decodeSchema([]byte(`{"enabled":true}`))
		require.NotNil(t, decoded)
		assert.Equal(t, true, decoded["enabled"])
	})

	t.Run("heartbeat cache updates only after threshold", func(t *testing.T) {
		cache := &HeartbeatCache{nodes: make(map[string]*CachedNodeData)}
		start := time.Now().UTC()

		shouldWrite, cached := cache.shouldUpdateDatabase("node-1", start, "healthy")
		require.True(t, shouldWrite)
		require.NotNil(t, cached)
		assert.Equal(t, "healthy", cached.Status)

		shouldWrite, cached = cache.shouldUpdateDatabase("node-1", start.Add(time.Second), "degraded")
		require.False(t, shouldWrite)
		assert.Equal(t, "degraded", cached.Status)

		shouldWrite, cached = cache.shouldUpdateDatabase("node-1", start.Add(dbUpdateThreshold+time.Millisecond), "ready")
		require.True(t, shouldWrite)
		assert.Equal(t, "ready", cached.Status)
		assert.Equal(t, start.Add(dbUpdateThreshold+time.Millisecond), cached.LastDBUpdate)
	})
}

func TestRegisterNodeHandler_AdditionalCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("rejects invalid callback url", func(t *testing.T) {
		store := &nodeRESTStorageStub{}
		router := gin.New()
		router.POST("/nodes/register", RegisterNodeHandler(store, nil, nil, nil, nil, nil))

		req := httptest.NewRequest(http.MethodPost, "/nodes/register", strings.NewReader(`{"id":"node-1","base_url":"example.com"}`))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusBadRequest, rec.Code)
		assert.Contains(t, rec.Body.String(), "Invalid callback URL")
	})

	t.Run("registers new node and normalizes fields", func(t *testing.T) {
		store := &nodeRESTStorageStub{}
		router := gin.New()
		router.POST("/nodes/register", RegisterNodeHandler(store, nil, nil, nil, nil, nil))

		body := `{
			"id":"node-1",
			"base_url":"https://example.com",
			"reasoners":[{"id":"reasoner-1","proposed_tags":["alpha","beta"]}],
			"skills":[{"id":"skill-1","tags":["ops"]}]
		}`
		req := httptest.NewRequest(http.MethodPost, "/nodes/register", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusCreated, rec.Code)
		require.NotNil(t, store.registeredAgent)
		assert.Equal(t, "node-1", store.registeredAgent.GroupID)
		assert.Equal(t, types.HealthStatusUnknown, store.registeredAgent.HealthStatus)
		assert.Equal(t, types.AgentStatusStarting, store.registeredAgent.LifecycleStatus)
		require.NotNil(t, store.registeredAgent.CallbackDiscovery)
		assert.Equal(t, "auto", store.registeredAgent.CallbackDiscovery.Mode)
		assert.Equal(t, "https://example.com", store.registeredAgent.CallbackDiscovery.Preferred)
		assert.Equal(t, "https://example.com", store.registeredAgent.CallbackDiscovery.Resolved)
		assert.Equal(t, []string{"alpha", "beta"}, store.registeredAgent.Reasoners[0].Tags)
		assert.Equal(t, []string{"ops"}, store.registeredAgent.Skills[0].ProposedTags)
	})

	t.Run("re-registration preserves approved tags and lifts offline status to starting", func(t *testing.T) {
		store := &nodeRESTStorageStub{
			agent: &types.AgentNode{
				ID:              "node-1",
				LifecycleStatus: types.AgentStatusOffline,
				ApprovedTags:    []string{"alpha"},
			},
		}
		router := gin.New()
		router.POST("/nodes/register", RegisterNodeHandler(store, nil, nil, nil, nil, nil))

		req := httptest.NewRequest(http.MethodPost, "/nodes/register", strings.NewReader(`{
			"id":"node-1",
			"base_url":"https://example.com",
			"reasoners":[{"id":"reasoner-1","tags":["alpha","beta"]}],
			"skills":[{"id":"skill-1","proposed_tags":["alpha","gamma"]}]
		}`))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusCreated, rec.Code)
		require.NotNil(t, store.registeredAgent)
		assert.Equal(t, []string{"alpha"}, store.registeredAgent.ApprovedTags)
		assert.Equal(t, types.AgentStatusStarting, store.registeredAgent.LifecycleStatus)
		assert.Equal(t, []string{"alpha"}, store.registeredAgent.Reasoners[0].ApprovedTags)
		assert.Equal(t, []string{"alpha"}, store.registeredAgent.Skills[0].ApprovedTags)
	})

	t.Run("admin revoked re-registration remains pending approval", func(t *testing.T) {
		store := &nodeRESTStorageStub{
			agent: &types.AgentNode{
				ID:              "node-2",
				LifecycleStatus: types.AgentStatusPendingApproval,
				ApprovedTags:    []string{},
			},
		}
		router := gin.New()
		router.POST("/nodes/register", RegisterNodeHandler(store, nil, nil, nil, nil, nil))

		req := httptest.NewRequest(http.MethodPost, "/nodes/register", strings.NewReader(`{"id":"node-2","base_url":"https://example.com"}`))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusCreated, rec.Code)
		require.NotNil(t, store.registeredAgent)
		assert.Equal(t, types.AgentStatusPendingApproval, store.registeredAgent.LifecycleStatus)
	})

}

func TestHeartbeatHandler_AdditionalCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Run("legacy heartbeat updates lifecycle and async heartbeat timestamp", func(t *testing.T) {
		heartbeatCache = &HeartbeatCache{nodes: make(map[string]*CachedNodeData)}
		store := &nodeRESTStorageStub{
			agent: &types.AgentNode{
				ID:              "node-1",
				Version:         "v1",
				BaseURL:         "https://example.com",
				LifecycleStatus: types.AgentStatusStarting,
			},
		}
		router := gin.New()
		router.POST("/nodes/:node_id/heartbeat", HeartbeatHandler(store, nil, nil, nil, nil))

		req := httptest.NewRequest(http.MethodPost, "/nodes/node-1/heartbeat", strings.NewReader(`{"status":"ready"}`))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		require.Eventually(t, func() bool { return len(store.heartbeats) == 1 }, 5*time.Second, 50*time.Millisecond)
		assert.Equal(t, "node-1", store.lastHeartbeatID)
		assert.Equal(t, "v1", store.lastVersion)
		require.NotNil(t, store.updatedLifecycle)
		assert.Equal(t, types.AgentStatusReady, *store.updatedLifecycle)
	})

	t.Run("status manager path processes lifecycle and health score", func(t *testing.T) {
		heartbeatCache = &HeartbeatCache{nodes: make(map[string]*CachedNodeData)}
		store := &statusManagerStorageStub{
			nodeRESTStorageStub: &nodeRESTStorageStub{
				agent: &types.AgentNode{
					ID:              "node-2",
					Version:         "v2",
					HealthStatus:    types.HealthStatusUnknown,
					LifecycleStatus: types.AgentStatusStarting,
					LastHeartbeat:   time.Now().UTC(),
				},
				versionedAgent: &types.AgentNode{
					ID:              "node-2",
					Version:         "v2",
					HealthStatus:    types.HealthStatusUnknown,
					LifecycleStatus: types.AgentStatusStarting,
					LastHeartbeat:   time.Now().UTC(),
				},
			},
		}
		statusManager := services.NewStatusManager(store, services.StatusManagerConfig{}, nil, nil)
		router := gin.New()
		router.POST("/nodes/:node_id/heartbeat", HeartbeatHandler(store, nil, nil, statusManager, nil))

		req := httptest.NewRequest(http.MethodPost, "/nodes/node-2/heartbeat", strings.NewReader(`{"version":"v2","status":"ready","health_score":88}`))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		assert.NotEmpty(t, store.healthUpdates)
	})

	t.Run("returns not found when node cannot be loaded for db update", func(t *testing.T) {
		heartbeatCache = &HeartbeatCache{nodes: make(map[string]*CachedNodeData)}
		store := &nodeRESTStorageStub{}
		router := gin.New()
		router.POST("/nodes/:node_id/heartbeat", HeartbeatHandler(store, nil, nil, nil, nil))

		req := httptest.NewRequest(http.MethodPost, "/nodes/missing/heartbeat", strings.NewReader(`{}`))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusNotFound, rec.Code)
	})
}

func TestProcessHeartbeatAsync_UsesVersionFallback(t *testing.T) {
	store := &heartbeatAsyncStorageStub{
		nodeRESTStorageStub: &nodeRESTStorageStub{
			getAgentErr: errors.New("missing"),
		},
		versions: []*types.AgentNode{{ID: "node-async", Version: "v1"}},
	}

	processHeartbeatAsync(store, nil, "node-async", "", &CachedNodeData{LastDBUpdate: time.Now().UTC()})

	require.Eventually(t, func() bool { return len(store.heartbeats) == 1 }, 5*time.Second, 50*time.Millisecond)
	assert.Equal(t, "node-async", store.lastHeartbeatID)
}

func TestNodeStatusHandlers_WithStatusManager(t *testing.T) {
	gin.SetMode(gin.TestMode)

	newManager := func(store *statusManagerStorageStub) *services.StatusManager {
		return services.NewStatusManager(store, services.StatusManagerConfig{}, nil, nil)
	}

	t.Run("get node status succeeds", func(t *testing.T) {
		store := &statusManagerStorageStub{
			nodeRESTStorageStub: &nodeRESTStorageStub{
				agent: &types.AgentNode{
					ID:              "node-1",
					HealthStatus:    types.HealthStatusActive,
					LifecycleStatus: types.AgentStatusReady,
					LastHeartbeat:   time.Now().UTC(),
				},
			},
		}
		router := gin.New()
		router.GET("/nodes/:node_id/status", GetNodeStatusHandler(newManager(store)))

		req := httptest.NewRequest(http.MethodGet, "/nodes/node-1/status", nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		assert.Contains(t, rec.Body.String(), `"success":true`)
	})

	t.Run("refresh node status succeeds", func(t *testing.T) {
		store := &statusManagerStorageStub{
			nodeRESTStorageStub: &nodeRESTStorageStub{
				agent: &types.AgentNode{
					ID:              "node-1",
					HealthStatus:    types.HealthStatusUnknown,
					LifecycleStatus: types.AgentStatusReady,
					LastHeartbeat:   time.Now().UTC(),
				},
			},
		}
		router := gin.New()
		router.POST("/nodes/:node_id/status/refresh", RefreshNodeStatusHandler(newManager(store)))

		req := httptest.NewRequest(http.MethodPost, "/nodes/node-1/status/refresh", nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		assert.NotEmpty(t, store.healthUpdates)
	})

	t.Run("bulk node status succeeds", func(t *testing.T) {
		store := &statusManagerStorageStub{
			nodeRESTStorageStub: &nodeRESTStorageStub{
				agent: &types.AgentNode{
					ID:              "node-1",
					HealthStatus:    types.HealthStatusActive,
					LifecycleStatus: types.AgentStatusReady,
					LastHeartbeat:   time.Now().UTC(),
				},
			},
		}
		router := gin.New()
		router.POST("/nodes/bulk-status", BulkNodeStatusHandler(newManager(store), store))

		req := httptest.NewRequest(http.MethodPost, "/nodes/bulk-status", strings.NewReader(`{"node_ids":["node-1"]}`))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		assert.Contains(t, rec.Body.String(), `"successful":1`)
	})

	t.Run("bulk node status validates request size", func(t *testing.T) {
		store := &statusManagerStorageStub{nodeRESTStorageStub: &nodeRESTStorageStub{}}
		router := gin.New()
		router.POST("/nodes/bulk-status", BulkNodeStatusHandler(newManager(store), store))

		var body strings.Builder
		body.WriteString(`{"node_ids":[`)
		for i := 0; i < 51; i++ {
			if i > 0 {
				body.WriteString(",")
			}
			body.WriteString(`"node"`)
		}
		body.WriteString(`]}`)

		req := httptest.NewRequest(http.MethodPost, "/nodes/bulk-status", strings.NewReader(body.String()))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusBadRequest, rec.Code)
		assert.Contains(t, rec.Body.String(), "Too many node IDs")
	})

	t.Run("refresh all node statuses succeeds", func(t *testing.T) {
		now := time.Now().UTC()
		store := &statusManagerStorageStub{
			nodeRESTStorageStub: &nodeRESTStorageStub{
				listAgents: []*types.AgentNode{
					{ID: "node-1", HealthStatus: types.HealthStatusActive, LifecycleStatus: types.AgentStatusReady, LastHeartbeat: now},
					{ID: "node-2", HealthStatus: types.HealthStatusUnknown, LifecycleStatus: types.AgentStatusStarting, LastHeartbeat: now},
				},
			},
		}
		router := gin.New()
		router.POST("/nodes/status/refresh-all", RefreshAllNodeStatusHandler(newManager(store), store))

		req := httptest.NewRequest(http.MethodPost, "/nodes/status/refresh-all", nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		assert.Contains(t, rec.Body.String(), `"successful":2`)
	})

	t.Run("start and stop node succeed", func(t *testing.T) {
		router := gin.New()

		startStore := &statusManagerStorageStub{
			nodeRESTStorageStub: &nodeRESTStorageStub{
				agent: &types.AgentNode{
					ID:              "node-1",
					HealthStatus:    types.HealthStatusInactive,
					LifecycleStatus: types.AgentStatusOffline,
					LastHeartbeat:   time.Now().UTC(),
				},
			},
		}
		stopStore := &statusManagerStorageStub{
			nodeRESTStorageStub: &nodeRESTStorageStub{
				agent: &types.AgentNode{
					ID:              "node-2",
					HealthStatus:    types.HealthStatusActive,
					LifecycleStatus: types.AgentStatusReady,
					LastHeartbeat:   time.Now().UTC(),
				},
			},
		}

		router.POST("/nodes/:node_id/start", StartNodeHandler(newManager(startStore), startStore))
		router.POST("/nodes/:node_id/stop", StopNodeHandler(newManager(stopStore), stopStore))

		startReq := httptest.NewRequest(http.MethodPost, "/nodes/node-1/start", nil)
		startRec := httptest.NewRecorder()
		router.ServeHTTP(startRec, startReq)
		require.Equal(t, http.StatusOK, startRec.Code)
		assert.Equal(t, types.AgentStatusStarting, *startStore.updatedLifecycle)

		stopReq := httptest.NewRequest(http.MethodPost, "/nodes/node-2/stop", nil)
		stopRec := httptest.NewRecorder()
		router.ServeHTTP(stopRec, stopReq)
		require.Equal(t, http.StatusOK, stopRec.Code)
		assert.Equal(t, types.AgentStatusOffline, *stopStore.updatedLifecycle)
	})

	t.Run("start and stop return not found when node lookup fails", func(t *testing.T) {
		store := &nodeRESTStorageStub{getAgentErr: errors.New("missing")}
		router := gin.New()
		router.POST("/nodes/:node_id/start", StartNodeHandler(nil, store))
		router.POST("/nodes/:node_id/stop", StopNodeHandler(nil, store))

		startReq := httptest.NewRequest(http.MethodPost, "/nodes/missing/start", nil)
		startRec := httptest.NewRecorder()
		router.ServeHTTP(startRec, startReq)
		require.Equal(t, http.StatusNotFound, startRec.Code)

		stopReq := httptest.NewRequest(http.MethodPost, "/nodes/missing/stop", nil)
		stopRec := httptest.NewRecorder()
		router.ServeHTTP(stopRec, stopReq)
		require.Equal(t, http.StatusNotFound, stopRec.Code)
	})
}

func TestLogMemoryAccess(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(rec)
	ctx.Request = httptest.NewRequest(http.MethodGet, "/api/v1/memory/get", nil)
	ctx.Request.RemoteAddr = "127.0.0.1:12345"

	logMemoryAccess(ctx, &types.Memory{
		Key:     "key-1",
		Scope:   "workflow",
		ScopeID: "wf-1",
	}, "agent-1")
}

func TestDIDHandlers_AdditionalCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("register agent handles bad request missing id and service error", func(t *testing.T) {
		router := gin.New()
		handler := NewDIDHandlers(&fakeDIDService{
			registerFn: func(*types.DIDRegistrationRequest) (*types.DIDRegistrationResponse, error) {
				return nil, errors.New("register failed")
			},
		}, &fakeVCService{})
		router.POST("/did/register", handler.RegisterAgent)

		req := httptest.NewRequest(http.MethodPost, "/did/register", strings.NewReader(`not-json`))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		req = httptest.NewRequest(http.MethodPost, "/did/register", strings.NewReader(`{"agent_node_id":""}`))
		req.Header.Set("Content-Type", "application/json")
		rec = httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		req = httptest.NewRequest(http.MethodPost, "/did/register", strings.NewReader(`{"agent_node_id":"agent-1"}`))
		req.Header.Set("Content-Type", "application/json")
		rec = httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusInternalServerError, rec.Code)
	})

	t.Run("verify vc and workflow vc handlers exercise error paths", func(t *testing.T) {
		handler := NewDIDHandlers(&fakeDIDService{}, &fakeVCService{
			verifyFn: func(json.RawMessage) (*types.VCVerificationResponse, error) {
				return nil, errors.New("verify failed")
			},
			workflowChainFn: func(string) (*types.WorkflowVCChainResponse, error) {
				return nil, errors.New("chain failed")
			},
			createWorkflowFn: func(string, string, []string) (*types.WorkflowVC, error) {
				return nil, errors.New("create failed")
			},
			generateExecFn: func(*types.ExecutionContext, []byte, []byte, string, *string, int) (*types.ExecutionVC, error) {
				return nil, nil
			},
		})
		router := gin.New()
		router.POST("/did/verify", handler.VerifyVC)
		router.GET("/did/workflow/:workflow_id/vc-chain", handler.GetWorkflowVCChain)
		router.POST("/did/workflow/:workflow_id/vc", handler.CreateWorkflowVC)
		router.POST("/execution/vc", handler.CreateExecutionVC)

		req := httptest.NewRequest(http.MethodPost, "/did/verify", strings.NewReader(`not-json`))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		req = httptest.NewRequest(http.MethodPost, "/did/verify", strings.NewReader(`{"vc_document":{"ok":true}}`))
		req.Header.Set("Content-Type", "application/json")
		rec = httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		req = httptest.NewRequest(http.MethodGet, "/did/workflow/wf-1/vc-chain", nil)
		rec = httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		req = httptest.NewRequest(http.MethodPost, "/did/workflow/wf-1/vc", strings.NewReader(`not-json`))
		req.Header.Set("Content-Type", "application/json")
		rec = httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		req = httptest.NewRequest(http.MethodPost, "/did/workflow/wf-1/vc", strings.NewReader(`{"session_id":"s1"}`))
		req.Header.Set("Content-Type", "application/json")
		rec = httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		req = httptest.NewRequest(http.MethodPost, "/execution/vc", strings.NewReader(`{"execution_context":{"execution_id":"e1","workflow_id":"wf-1","timestamp":"bad-time"}}`))
		req.Header.Set("Content-Type", "application/json")
		rec = httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		req = httptest.NewRequest(http.MethodPost, "/execution/vc", strings.NewReader(`{"execution_context":{"execution_id":"e1","workflow_id":"wf-1","timestamp":"2026-04-08T00:00:00Z"}}`))
		req.Header.Set("Content-Type", "application/json")
		rec = httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusOK, rec.Code)
		assert.Contains(t, rec.Body.String(), "VC generation is disabled")
	})

	t.Run("get did document handles resolve and jwk failures", func(t *testing.T) {
		router := gin.New()
		handler := NewDIDHandlers(&fakeDIDService{
			resolveFn: func(did string) (*types.DIDIdentity, error) {
				switch did {
				case "did:example:missing":
					return nil, errors.New("missing")
				case "did:example:bad-jwk":
					return &types.DIDIdentity{DID: did, PublicKeyJWK: `not-json`}, nil
				default:
					return &types.DIDIdentity{DID: did, PublicKeyJWK: `{"kty":"OKP"}`}, nil
				}
			},
		}, &fakeVCService{})
		router.GET("/did/document/:did", handler.GetDIDDocument)

		req := httptest.NewRequest(http.MethodGet, "/did/document/did:example:missing", nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusNotFound, rec.Code)

		req = httptest.NewRequest(http.MethodGet, "/did/document/did:example:bad-jwk", nil)
		rec = httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusInternalServerError, rec.Code)
	})
}

func lowBranchStringPtr(v string) *string {
	return &v
}
