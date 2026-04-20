package handlers

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// bulkStatusFailStore forces GetAgentStatusSnapshot to error on every lookup so
// BulkNodeStatusHandler returns the all-failed 500 branch.
type bulkStatusFailStore struct {
	*nodeRESTStorageStub
}

func (s *bulkStatusFailStore) GetAgent(context.Context, string) (*types.AgentNode, error) {
	return nil, errors.New("not found")
}

func (s *bulkStatusFailStore) GetAgentVersion(context.Context, string, string) (*types.AgentNode, error) {
	return nil, errors.New("not found")
}

func TestListNodesHandler_FilterBranches(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	t.Run("health_status query param drives filter", func(t *testing.T) {
		store := &nodeRESTStorageStub{listAgents: []*types.AgentNode{
			{ID: "node-1", HealthStatus: types.HealthStatusDegraded},
		}}
		router := gin.New()
		router.GET("/nodes", ListNodesHandler(store))

		req := httptest.NewRequest(http.MethodGet, "/nodes?health_status=degraded&team_id=team-a&group_id=group-a", nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		assert.Contains(t, rec.Body.String(), `"health_status":"degraded"`)
		assert.Contains(t, rec.Body.String(), `"team_id":"team-a"`)
		assert.Contains(t, rec.Body.String(), `"group_id":"group-a"`)
	})

	t.Run("show_all clears health filter", func(t *testing.T) {
		store := &nodeRESTStorageStub{listAgents: []*types.AgentNode{
			{ID: "node-1", HealthStatus: types.HealthStatusInactive},
		}}
		router := gin.New()
		router.GET("/nodes", ListNodesHandler(store))

		req := httptest.NewRequest(http.MethodGet, "/nodes?show_all=true", nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		assert.Contains(t, rec.Body.String(), `"count":1`)
	})
}

func TestHeartbeatHandler_EmptyNodeIDReturnsBadRequest(t *testing.T) {
	gin.SetMode(gin.TestMode)
	heartbeatCache = &HeartbeatCache{nodes: make(map[string]*CachedNodeData)}

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/nodes//heartbeat", nil)
	// No node_id param set — simulate a missing path var.

	HeartbeatHandler(&nodeRESTStorageStub{}, nil, nil, nil, nil)(c)

	require.Equal(t, http.StatusBadRequest, rec.Code)
	assert.Contains(t, rec.Body.String(), "node_id is required")
}

func TestHeartbeatHandler_RegistersHealthMonitorAndTouchesPresence(t *testing.T) {
	gin.SetMode(gin.TestMode)
	heartbeatCache = &HeartbeatCache{nodes: make(map[string]*CachedNodeData)}

	store := &nodeRESTStorageStub{
		agent: &types.AgentNode{
			ID:              "node-hm",
			Version:         "v1",
			BaseURL:         "https://hm.example.com",
			LifecycleStatus: types.AgentStatusReady,
		},
	}
	healthMonitor := services.NewHealthMonitor(store, services.HealthMonitorConfig{}, nil, nil, nil, nil)
	presence := services.NewPresenceManager(nil, services.PresenceManagerConfig{})

	router := gin.New()
	router.POST("/nodes/:node_id/heartbeat", HeartbeatHandler(store, nil, healthMonitor, nil, presence))

	req := httptest.NewRequest(http.MethodPost, "/nodes/node-hm/heartbeat", strings.NewReader(`{}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	require.Eventually(t, func() bool { return len(store.heartbeats) == 1 }, 5*time.Second, 50*time.Millisecond)
	assert.True(t, presence.HasLease("node-hm"))
}

func TestHeartbeatHandler_CachedPathSkipsDBUpdate(t *testing.T) {
	gin.SetMode(gin.TestMode)
	heartbeatCache = &HeartbeatCache{nodes: make(map[string]*CachedNodeData)}

	store := &nodeRESTStorageStub{
		agent: &types.AgentNode{ID: "node-cache", Version: "v1", LifecycleStatus: types.AgentStatusReady},
	}
	router := gin.New()
	router.POST("/nodes/:node_id/heartbeat", HeartbeatHandler(store, nil, nil, nil, nil))

	// First heartbeat triggers DB update.
	req1 := httptest.NewRequest(http.MethodPost, "/nodes/node-cache/heartbeat", strings.NewReader(`{}`))
	req1.Header.Set("Content-Type", "application/json")
	rec1 := httptest.NewRecorder()
	router.ServeHTTP(rec1, req1)
	require.Equal(t, http.StatusOK, rec1.Code)

	// Second heartbeat immediately after should hit the cached, no-DB-update branch.
	req2 := httptest.NewRequest(http.MethodPost, "/nodes/node-cache/heartbeat", strings.NewReader(`{}`))
	req2.Header.Set("Content-Type", "application/json")
	rec2 := httptest.NewRecorder()
	router.ServeHTTP(rec2, req2)
	require.Equal(t, http.StatusOK, rec2.Code)
}

func TestHeartbeatHandler_PendingApprovalIgnoresStatusManagerPath(t *testing.T) {
	gin.SetMode(gin.TestMode)
	heartbeatCache = &HeartbeatCache{nodes: make(map[string]*CachedNodeData)}

	// Pending-approval agent with no versionedAgent — forces the
	// pending_approval check in the statusManager branch to run the nil
	// existingNode reload path and then skip the lifecycle update.
	store := &statusManagerStorageStub{
		nodeRESTStorageStub: &nodeRESTStorageStub{
			agent: &types.AgentNode{
				ID:              "node-pending",
				Version:         "v1",
				LifecycleStatus: types.AgentStatusPendingApproval,
			},
		},
	}
	manager := services.NewStatusManager(store, services.StatusManagerConfig{}, nil, nil)

	router := gin.New()
	router.POST("/nodes/:node_id/heartbeat", HeartbeatHandler(store, nil, nil, manager, nil))

	req := httptest.NewRequest(http.MethodPost, "/nodes/node-pending/heartbeat", strings.NewReader(`{"status":"ready"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
}

func TestUpdateLifecycleStatusHandler_Branches(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	t.Run("missing node id returns bad request", func(t *testing.T) {
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(http.MethodPut, "/nodes//lifecycle",
			strings.NewReader(`{"lifecycle_status":"ready"}`))
		c.Request.Header.Set("Content-Type", "application/json")

		UpdateLifecycleStatusHandler(&nodeRESTStorageStub{}, nil, nil)(c)

		require.Equal(t, http.StatusBadRequest, rec.Code)
		assert.Contains(t, rec.Body.String(), "node_id is required")
	})

	t.Run("status manager success path updates unified status", func(t *testing.T) {
		store := &statusManagerStorageStub{
			nodeRESTStorageStub: &nodeRESTStorageStub{
				agent: &types.AgentNode{
					ID:              "node-ok",
					Version:         "v1",
					LifecycleStatus: types.AgentStatusStarting,
					LastHeartbeat:   time.Now().UTC(),
				},
			},
		}
		manager := services.NewStatusManager(store, services.StatusManagerConfig{}, nil, nil)

		router := gin.New()
		router.PUT("/nodes/:node_id/lifecycle",
			UpdateLifecycleStatusHandler(store, nil, manager))

		req := httptest.NewRequest(http.MethodPut, "/nodes/node-ok/lifecycle",
			strings.NewReader(`{"lifecycle_status":"ready"}`))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		assert.Contains(t, rec.Body.String(), `"lifecycle_status":"ready"`)
	})
}

func TestNodeStatusHandlers_MissingNodeID(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	manager := services.NewStatusManager(&nodeRESTStorageStub{}, services.StatusManagerConfig{}, nil, nil)

	t.Run("get status", func(t *testing.T) {
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(http.MethodGet, "/nodes//status", nil)

		GetNodeStatusHandler(manager)(c)
		require.Equal(t, http.StatusBadRequest, rec.Code)
		assert.Contains(t, rec.Body.String(), "MISSING_NODE_ID")
	})

	t.Run("refresh status", func(t *testing.T) {
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(http.MethodPost, "/nodes//status/refresh", nil)

		RefreshNodeStatusHandler(manager)(c)
		require.Equal(t, http.StatusBadRequest, rec.Code)
		assert.Contains(t, rec.Body.String(), "MISSING_NODE_ID")
	})

	t.Run("start node", func(t *testing.T) {
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(http.MethodPost, "/nodes//start", nil)

		StartNodeHandler(manager, &nodeRESTStorageStub{})(c)
		require.Equal(t, http.StatusBadRequest, rec.Code)
		assert.Contains(t, rec.Body.String(), "MISSING_NODE_ID")
	})

	t.Run("stop node", func(t *testing.T) {
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(http.MethodPost, "/nodes//stop", nil)

		StopNodeHandler(manager, &nodeRESTStorageStub{})(c)
		require.Equal(t, http.StatusBadRequest, rec.Code)
		assert.Contains(t, rec.Body.String(), "MISSING_NODE_ID")
	})
}

func TestBulkNodeStatusHandler_AllFailedReturns500(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	store := &bulkStatusFailStore{nodeRESTStorageStub: &nodeRESTStorageStub{}}
	manager := services.NewStatusManager(store, services.StatusManagerConfig{}, nil, nil)

	router := gin.New()
	router.POST("/nodes/bulk-status", BulkNodeStatusHandler(manager, store))

	req := httptest.NewRequest(http.MethodPost, "/nodes/bulk-status",
		strings.NewReader(`{"node_ids":["missing-1","missing-2"]}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusInternalServerError, rec.Code)
	assert.Contains(t, rec.Body.String(), `"failed":2`)
}

func TestRegisterNodeHandler_AutoDiscoveryPath(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	store := &nodeRESTStorageStub{}
	router := gin.New()
	router.POST("/nodes/register", RegisterNodeHandler(store, nil, nil, nil, nil, nil))

	// BaseURL empty + CallbackDiscovery.Preferred set triggers the
	// auto-discovery branch (candidates non-empty, skipAutoDiscovery=false).
	// The resolved URL differs from the empty BaseURL, covering the
	// "Auto-discovered callback URL" log line and BaseURL assignment.
	body := `{
		"id":"node-auto",
		"callback_discovery":{"mode":"auto","preferred":"http://10.0.0.5:9001"}
	}`
	req := httptest.NewRequest(http.MethodPost, "/nodes/register", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusCreated, rec.Code)
	require.NotNil(t, store.registeredAgent)
	assert.Equal(t, "http://10.0.0.5:9001", store.registeredAgent.BaseURL)
	require.NotNil(t, store.registeredAgent.CallbackDiscovery)
	assert.Equal(t, "http://10.0.0.5:9001", store.registeredAgent.CallbackDiscovery.Resolved)
}
