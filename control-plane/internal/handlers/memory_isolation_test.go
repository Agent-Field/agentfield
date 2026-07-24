package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// setupIsolationRouter creates a test router with memory permission middleware enabled.
func setupIsolationRouter(storage MemoryStorage, config middleware.MemoryPermissionConfig) *gin.Engine {
	gin.SetMode(gin.TestMode)
	router := gin.New()

	resolver := &isolationTestAgentResolver{agents: map[string]*types.AgentNode{
		"agent-alpha": {ID: "agent-alpha", ApprovedTags: []string{"team-alpha"}},
		"agent-beta":  {ID: "agent-beta", ApprovedTags: []string{"team-beta"}},
	}}

	router.Use(middleware.MemoryPermissionMiddleware(
		nil, // no policy service
		resolver,
		&isolationTestDIDResolver{},
		config,
	))

	router.POST("/api/v1/memory/set", SetMemoryHandler(storage))
	router.POST("/api/v1/memory/get", GetMemoryHandler(storage))
	router.POST("/api/v1/memory/delete", DeleteMemoryHandler(storage))
	router.GET("/api/v1/memory/list", ListMemoryHandler(storage))

	return router
}

// --- Mock types for isolation tests ---

type isolationTestAgentResolver struct {
	agents map[string]*types.AgentNode
}

func (r *isolationTestAgentResolver) GetAgent(_ context.Context, agentID string) (*types.AgentNode, error) {
	if a, ok := r.agents[agentID]; ok {
		return a, nil
	}
	return nil, nil
}

type isolationTestDIDResolver struct{}

func (r *isolationTestDIDResolver) GenerateDIDWeb(agentID string) string {
	return "did:web:localhost:agents:" + agentID
}

func (r *isolationTestDIDResolver) ResolveAgentIDByDID(_ context.Context, _ string) string {
	return ""
}

// --- Cross-Agent Isolation Tests ---

func TestCrossAgentIsolation_AgentCannotReadOtherAgentActorScope(t *testing.T) {
	storage := newMemoryStorageStub()
	config := middleware.MemoryPermissionConfig{
		Enabled:               true,
		EnforceScopeOwnership: true,
	}
	router := setupIsolationRouter(storage, config)

	// Agent Alpha writes to its own actor scope
	setBody := `{"key":"secret","data":"alpha-secret-data","scope":"actor"}`
	setReq := httptest.NewRequest(http.MethodPost, "/api/v1/memory/set", strings.NewReader(setBody))
	setReq.Header.Set("Content-Type", "application/json")
	setReq.Header.Set("X-Agent-Node-ID", "agent-alpha")
	setReq.Header.Set("X-Actor-ID", "agent-alpha")

	setResp := httptest.NewRecorder()
	router.ServeHTTP(setResp, setReq)
	require.Equal(t, http.StatusOK, setResp.Code)

	// Agent Beta tries to read Agent Alpha's actor scope
	getBody := `{"key":"secret","scope":"actor"}`
	getReq := httptest.NewRequest(http.MethodPost, "/api/v1/memory/get", strings.NewReader(getBody))
	getReq.Header.Set("Content-Type", "application/json")
	getReq.Header.Set("X-Agent-Node-ID", "agent-beta")
	getReq.Header.Set("X-Actor-ID", "agent-alpha") // trying to access alpha's scope

	getResp := httptest.NewRecorder()
	router.ServeHTTP(getResp, getReq)

	// Should be denied by scope ownership validation
	assert.Equal(t, http.StatusForbidden, getResp.Code)
	assert.Contains(t, getResp.Body.String(), "scope_ownership_denied")
}

func TestCrossAgentIsolation_AgentCanReadOwnActorScope(t *testing.T) {
	storage := newMemoryStorageStub()
	config := middleware.MemoryPermissionConfig{
		Enabled:               true,
		EnforceScopeOwnership: true,
	}
	router := setupIsolationRouter(storage, config)

	// Agent Alpha writes to its own actor scope
	setBody := `{"key":"my-data","data":"alpha-data","scope":"actor"}`
	setReq := httptest.NewRequest(http.MethodPost, "/api/v1/memory/set", strings.NewReader(setBody))
	setReq.Header.Set("Content-Type", "application/json")
	setReq.Header.Set("X-Agent-Node-ID", "agent-alpha")
	setReq.Header.Set("X-Actor-ID", "agent-alpha")

	setResp := httptest.NewRecorder()
	router.ServeHTTP(setResp, setReq)
	require.Equal(t, http.StatusOK, setResp.Code)

	// Agent Alpha reads its own actor scope
	getBody := `{"key":"my-data","scope":"actor"}`
	getReq := httptest.NewRequest(http.MethodPost, "/api/v1/memory/get", strings.NewReader(getBody))
	getReq.Header.Set("Content-Type", "application/json")
	getReq.Header.Set("X-Agent-Node-ID", "agent-alpha")
	getReq.Header.Set("X-Actor-ID", "agent-alpha")

	getResp := httptest.NewRecorder()
	router.ServeHTTP(getResp, getReq)

	assert.Equal(t, http.StatusOK, getResp.Code)

	var memory types.Memory
	require.NoError(t, json.Unmarshal(getResp.Body.Bytes(), &memory))
	assert.Equal(t, "my-data", memory.Key)
}

func TestCrossAgentIsolation_AgentCannotDeleteOtherAgentMemory(t *testing.T) {
	storage := newMemoryStorageStub()
	config := middleware.MemoryPermissionConfig{
		Enabled:               true,
		EnforceScopeOwnership: true,
	}
	router := setupIsolationRouter(storage, config)

	// Agent Alpha writes data
	setBody := `{"key":"protected","data":"important","scope":"actor"}`
	setReq := httptest.NewRequest(http.MethodPost, "/api/v1/memory/set", strings.NewReader(setBody))
	setReq.Header.Set("Content-Type", "application/json")
	setReq.Header.Set("X-Agent-Node-ID", "agent-alpha")
	setReq.Header.Set("X-Actor-ID", "agent-alpha")

	setResp := httptest.NewRecorder()
	router.ServeHTTP(setResp, setReq)
	require.Equal(t, http.StatusOK, setResp.Code)

	// Agent Beta tries to delete Agent Alpha's data
	delBody := `{"key":"protected","scope":"actor"}`
	delReq := httptest.NewRequest(http.MethodPost, "/api/v1/memory/delete", strings.NewReader(delBody))
	delReq.Header.Set("Content-Type", "application/json")
	delReq.Header.Set("X-Agent-Node-ID", "agent-beta")
	delReq.Header.Set("X-Actor-ID", "agent-alpha")

	delResp := httptest.NewRecorder()
	router.ServeHTTP(delResp, delReq)

	assert.Equal(t, http.StatusForbidden, delResp.Code)

	// Verify data still exists
	verifyReq := httptest.NewRequest(http.MethodPost, "/api/v1/memory/get", strings.NewReader(`{"key":"protected","scope":"actor"}`))
	verifyReq.Header.Set("Content-Type", "application/json")
	verifyReq.Header.Set("X-Agent-Node-ID", "agent-alpha")
	verifyReq.Header.Set("X-Actor-ID", "agent-alpha")

	verifyResp := httptest.NewRecorder()
	router.ServeHTTP(verifyResp, verifyReq)
	assert.Equal(t, http.StatusOK, verifyResp.Code)
}

func TestCrossAgentIsolation_GlobalScopeSharedAcrossAgents(t *testing.T) {
	storage := newMemoryStorageStub()
	config := middleware.MemoryPermissionConfig{
		Enabled:               true,
		EnforceScopeOwnership: true,
	}
	router := setupIsolationRouter(storage, config)

	// Agent Alpha writes to global scope
	setBody := `{"key":"shared-config","data":"global-value","scope":"global"}`
	setReq := httptest.NewRequest(http.MethodPost, "/api/v1/memory/set", strings.NewReader(setBody))
	setReq.Header.Set("Content-Type", "application/json")
	setReq.Header.Set("X-Agent-Node-ID", "agent-alpha")

	setResp := httptest.NewRecorder()
	router.ServeHTTP(setResp, setReq)
	require.Equal(t, http.StatusOK, setResp.Code)

	// Agent Beta can read from global scope
	getBody := `{"key":"shared-config","scope":"global"}`
	getReq := httptest.NewRequest(http.MethodPost, "/api/v1/memory/get", strings.NewReader(getBody))
	getReq.Header.Set("Content-Type", "application/json")
	getReq.Header.Set("X-Agent-Node-ID", "agent-beta")

	getResp := httptest.NewRecorder()
	router.ServeHTTP(getResp, getReq)

	assert.Equal(t, http.StatusOK, getResp.Code)
}

func TestCrossAgentIsolation_SessionScopeRequiresSessionID(t *testing.T) {
	storage := newMemoryStorageStub()
	config := middleware.MemoryPermissionConfig{
		Enabled:               true,
		EnforceScopeOwnership: true,
	}
	router := setupIsolationRouter(storage, config)

	// Agent tries session scope without providing session ID
	setBody := `{"key":"test","data":"value"}`
	setReq := httptest.NewRequest(http.MethodPost, "/api/v1/memory/set", strings.NewReader(setBody))
	setReq.Header.Set("Content-Type", "application/json")
	setReq.Header.Set("X-Agent-Node-ID", "agent-alpha")
	setReq.Header.Set("X-Memory-Scope", "session")

	setResp := httptest.NewRecorder()
	router.ServeHTTP(setResp, setReq)

	// Should be denied - no session ID provided
	assert.Equal(t, http.StatusForbidden, setResp.Code)
}

func TestCrossAgentIsolation_WorkflowScopeSharedAmongParticipants(t *testing.T) {
	storage := newMemoryStorageStub()
	config := middleware.MemoryPermissionConfig{
		Enabled:               true,
		EnforceScopeOwnership: true,
	}
	router := setupIsolationRouter(storage, config)

	workflowID := "wf-shared-123"

	// Agent Alpha writes to workflow scope
	setBody := `{"key":"step-result","data":"computed-value","scope":"workflow"}`
	setReq := httptest.NewRequest(http.MethodPost, "/api/v1/memory/set", strings.NewReader(setBody))
	setReq.Header.Set("Content-Type", "application/json")
	setReq.Header.Set("X-Agent-Node-ID", "agent-alpha")
	setReq.Header.Set("X-Workflow-ID", workflowID)

	setResp := httptest.NewRecorder()
	router.ServeHTTP(setResp, setReq)
	require.Equal(t, http.StatusOK, setResp.Code)

	// Agent Beta can read from the same workflow scope
	getBody := `{"key":"step-result","scope":"workflow"}`
	getReq := httptest.NewRequest(http.MethodPost, "/api/v1/memory/get", strings.NewReader(getBody))
	getReq.Header.Set("Content-Type", "application/json")
	getReq.Header.Set("X-Agent-Node-ID", "agent-beta")
	getReq.Header.Set("X-Workflow-ID", workflowID)

	getResp := httptest.NewRecorder()
	router.ServeHTTP(getResp, getReq)

	assert.Equal(t, http.StatusOK, getResp.Code)
}

func TestCrossAgentIsolation_ListMemoryRespectsScopeIsolation(t *testing.T) {
	storage := newMemoryStorageStub()
	config := middleware.MemoryPermissionConfig{
		Enabled:               true,
		EnforceScopeOwnership: true,
	}
	router := setupIsolationRouter(storage, config)

	// Agent Alpha writes to its actor scope
	_ = storage.SetMemory(context.Background(), &types.Memory{
		Scope:   "actor",
		ScopeID: "agent-alpha",
		Key:     "alpha-key-1",
		Data:    json.RawMessage(`"alpha-data"`),
	})
	_ = storage.SetMemory(context.Background(), &types.Memory{
		Scope:   "actor",
		ScopeID: "agent-beta",
		Key:     "beta-key-1",
		Data:    json.RawMessage(`"beta-data"`),
	})

	// Agent Alpha lists its own actor scope - should only see its own keys
	listReq := httptest.NewRequest(http.MethodGet, "/api/v1/memory/list?scope=actor", nil)
	listReq.Header.Set("X-Agent-Node-ID", "agent-alpha")
	listReq.Header.Set("X-Actor-ID", "agent-alpha")

	listResp := httptest.NewRecorder()
	router.ServeHTTP(listResp, listReq)

	require.Equal(t, http.StatusOK, listResp.Code)

	var memories []types.Memory
	require.NoError(t, json.Unmarshal(listResp.Body.Bytes(), &memories))
	require.Len(t, memories, 1)
	assert.Equal(t, "alpha-key-1", memories[0].Key)
}

// TestAccessControl_RequiredRoles tests that required roles are enforced.
func TestAccessControl_RequiredRoles(t *testing.T) {
	config := AccessControlConfig{
		Enabled:         true,
		AuditLogEnabled: false,
	}

	memory := &types.Memory{
		Key:   "restricted-data",
		Scope: "global",
		Metadata: types.MemoryMetadata{
			AccessControl: &types.AccessControlMetadata{
				RequiredRoles: []string{"admin", "super-admin"},
			},
		},
	}

	gin.SetMode(gin.TestMode)

	// Test with correct role
	t.Run("allowed with matching role", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodGet, "/test", nil)
		c.Request.Header.Set("X-Agent-Roles", "admin, viewer")

		allowed := checkAccessControl(c, memory, "agent-a", config)
		assert.True(t, allowed)
	})

	// Test without required role
	t.Run("denied without matching role", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodGet, "/test", nil)
		c.Request.Header.Set("X-Agent-Roles", "viewer, editor")

		allowed := checkAccessControl(c, memory, "agent-b", config)
		assert.False(t, allowed)
	})

	// Test with no roles header
	t.Run("denied with no roles", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodGet, "/test", nil)

		allowed := checkAccessControl(c, memory, "agent-c", config)
		assert.False(t, allowed)
	})
}

// TestAccessControl_TeamRestriction tests team-based access restriction.
func TestAccessControl_TeamRestriction(t *testing.T) {
	config := AccessControlConfig{
		Enabled: true,
	}

	memory := &types.Memory{
		Key:   "team-data",
		Scope: "global",
		Metadata: types.MemoryMetadata{
			AccessControl: &types.AccessControlMetadata{
				TeamRestricted: true,
			},
			Custom: map[string]interface{}{
				"team": "engineering",
			},
		},
	}

	gin.SetMode(gin.TestMode)

	t.Run("allowed for same team", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodGet, "/test", nil)
		c.Request.Header.Set("X-Team-ID", "engineering")

		allowed := checkAccessControl(c, memory, "agent-a", config)
		assert.True(t, allowed)
	})

	t.Run("denied for different team", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodGet, "/test", nil)
		c.Request.Header.Set("X-Team-ID", "marketing")

		allowed := checkAccessControl(c, memory, "agent-b", config)
		assert.False(t, allowed)
	})
}

// TestAccessControl_Disabled tests that access control checks are skipped when disabled.
func TestAccessControl_Disabled(t *testing.T) {
	config := AccessControlConfig{
		Enabled: false,
	}

	memory := &types.Memory{
		Key: "restricted",
		Metadata: types.MemoryMetadata{
			AccessControl: &types.AccessControlMetadata{
				RequiredRoles: []string{"admin"},
			},
		},
	}

	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/test", nil)

	// Should be allowed even without admin role when disabled
	allowed := checkAccessControl(c, memory, "agent-a", config)
	assert.True(t, allowed)
}

// TestAccessControl_NilMetadata tests that nil access control metadata is handled gracefully.
func TestAccessControl_NilMetadata(t *testing.T) {
	config := AccessControlConfig{
		Enabled: true,
	}

	memory := &types.Memory{
		Key: "normal-data",
	}

	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/test", nil)

	allowed := checkAccessControl(c, memory, "agent-a", config)
	assert.True(t, allowed)
}
