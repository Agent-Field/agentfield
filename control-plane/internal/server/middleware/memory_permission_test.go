package middleware

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

// --- Memory Permission Test Helpers ---

func setupMemoryTestRoute(policyService AccessPolicyServiceInterface, config MemoryPermissionConfig) *gin.Engine {
	gin.SetMode(gin.TestMode)
	router := gin.New()

	resolver := &testAgentResolver{agents: map[string]*types.AgentNode{
		"agent-a": {ID: "agent-a", ApprovedTags: []string{"data-reader", "team-alpha"}},
		"agent-b": {ID: "agent-b", ApprovedTags: []string{"data-writer", "team-beta"}},
	}}

	router.Use(MemoryPermissionMiddleware(
		policyService,
		resolver,
		&testDIDResolver{},
		config,
	))

	router.POST("/api/v1/memory/set", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"ok": true})
	})
	router.POST("/api/v1/memory/get", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"ok": true})
	})
	router.POST("/api/v1/memory/delete", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"ok": true})
	})
	router.GET("/api/v1/memory/list", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"ok": true})
	})

	return router
}

// --- Memory Permission Tests ---

func TestMemoryPermission_DisabledAllowsAll(t *testing.T) {
	config := MemoryPermissionConfig{Enabled: false}
	router := setupMemoryTestRoute(nil, config)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/memory/set", strings.NewReader(`{"key":"test","data":"value"}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestMemoryPermission_GlobalScopeAlwaysAllowed(t *testing.T) {
	config := MemoryPermissionConfig{
		Enabled:               true,
		EnforceScopeOwnership: true,
	}
	router := setupMemoryTestRoute(nil, config)

	// No scope headers = global scope
	req := httptest.NewRequest(http.MethodPost, "/api/v1/memory/set", strings.NewReader(`{"key":"test","data":"value"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Agent-Node-ID", "agent-a")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestMemoryPermission_ScopeOwnership_ActorScope_OwnAgent(t *testing.T) {
	config := MemoryPermissionConfig{
		Enabled:               true,
		EnforceScopeOwnership: true,
	}
	router := setupMemoryTestRoute(nil, config)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/memory/set", strings.NewReader(`{"key":"test","data":"value"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Agent-Node-ID", "agent-a")
	req.Header.Set("X-Actor-ID", "agent-a")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestMemoryPermission_ScopeOwnership_ActorScope_DifferentAgent(t *testing.T) {
	config := MemoryPermissionConfig{
		Enabled:               true,
		EnforceScopeOwnership: true,
	}
	router := setupMemoryTestRoute(nil, config)

	// agent-b trying to access agent-a's actor scope
	req := httptest.NewRequest(http.MethodPost, "/api/v1/memory/set", strings.NewReader(`{"key":"test","data":"value"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Agent-Node-ID", "agent-b")
	req.Header.Set("X-Actor-ID", "agent-a")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusForbidden, w.Code)
	assert.Contains(t, w.Body.String(), "scope_ownership_denied")
}

func TestMemoryPermission_ScopeOwnership_WorkflowScope_Allowed(t *testing.T) {
	config := MemoryPermissionConfig{
		Enabled:               true,
		EnforceScopeOwnership: true,
	}
	router := setupMemoryTestRoute(nil, config)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/memory/set", strings.NewReader(`{"key":"test","data":"value"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Agent-Node-ID", "agent-a")
	req.Header.Set("X-Workflow-ID", "wf-123")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestMemoryPermission_PolicyDeniesAccess(t *testing.T) {
	config := MemoryPermissionConfig{
		Enabled:               true,
		EnforceScopeOwnership: false,
	}
	policy := &testPolicyService{result: &types.PolicyEvaluationResult{
		Matched:    true,
		Allowed:    false,
		PolicyName: "deny-memory-write",
		Reason:     "agent-b cannot write memory",
	}}
	router := setupMemoryTestRoute(policy, config)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/memory/set", strings.NewReader(`{"key":"test","data":"value"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Agent-Node-ID", "agent-b")
	req.Header.Set("X-Workflow-ID", "wf-123")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusForbidden, w.Code)
	assert.Contains(t, w.Body.String(), "memory_access_denied")
}

func TestMemoryPermission_PolicyAllowsAccess(t *testing.T) {
	config := MemoryPermissionConfig{
		Enabled:               true,
		EnforceScopeOwnership: false,
	}
	policy := &testPolicyService{result: &types.PolicyEvaluationResult{
		Matched:    true,
		Allowed:    true,
		PolicyName: "allow-memory-read",
	}}
	router := setupMemoryTestRoute(policy, config)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/memory/get", strings.NewReader(`{"key":"test"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Agent-Node-ID", "agent-a")
	req.Header.Set("X-Workflow-ID", "wf-123")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestMemoryPermission_NoPolicyMatchAllows(t *testing.T) {
	config := MemoryPermissionConfig{
		Enabled:               true,
		EnforceScopeOwnership: false,
	}
	policy := &testPolicyService{result: &types.PolicyEvaluationResult{Matched: false}}
	router := setupMemoryTestRoute(policy, config)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/memory/get", strings.NewReader(`{"key":"test"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Agent-Node-ID", "agent-a")
	req.Header.Set("X-Workflow-ID", "wf-123")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestMemoryPermission_AnonymousAccess_GlobalScope(t *testing.T) {
	config := MemoryPermissionConfig{
		Enabled:               true,
		EnforceScopeOwnership: true,
	}
	router := setupMemoryTestRoute(nil, config)

	// No caller identity, no scope headers = global scope = allowed
	req := httptest.NewRequest(http.MethodPost, "/api/v1/memory/get", strings.NewReader(`{"key":"test"}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestResolveMemoryOperation(t *testing.T) {
	tests := []struct {
		name     string
		method   string
		path     string
		expected string
	}{
		{"set memory", "POST", "/api/v1/memory/set", "memory.write"},
		{"get memory", "POST", "/api/v1/memory/get", "memory.read"},
		{"delete memory", "POST", "/api/v1/memory/delete", "memory.delete"},
		{"list memory", "GET", "/api/v1/memory/list", "memory.read"},
		{"set vector", "POST", "/api/v1/memory/vector", "vector.write"},
		{"get vector", "GET", "/api/v1/memory/vector/my-key", "vector.read"},
		{"search vector", "POST", "/api/v1/memory/vector/search", "vector.search"},
		{"delete vector", "DELETE", "/api/v1/memory/vector/my-key", "vector.delete"},
		{"events ws", "GET", "/api/v1/memory/events/ws", "memory.subscribe"},
		{"events sse", "GET", "/api/v1/memory/events/sse", "memory.subscribe"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gin.SetMode(gin.TestMode)
			c, _ := gin.CreateTestContext(httptest.NewRecorder())
			c.Request = httptest.NewRequest(tt.method, tt.path, nil)
			result := resolveMemoryOperation(c)
			assert.Equal(t, tt.expected, result)
		})
	}
}
