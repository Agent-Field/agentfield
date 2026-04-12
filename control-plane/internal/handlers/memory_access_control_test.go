package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// newTestContext creates a minimal gin.Context for unit tests.
func newTestContext(headers map[string]string) *gin.Context {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	req, _ := http.NewRequest("GET", "/memory/test", nil)
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	c.Request = req
	return c
}

// --- checkAccessControl unit tests ---

func TestCheckAccessControl_DisabledConfig_AlwaysAllows(t *testing.T) {
	config := AccessControlConfig{Enabled: false}
	mem := &types.Memory{
		Key:     "secret",
		Scope:   "agent",
		ScopeID: "agent-1",
		Metadata: types.MemoryMetadata{
			AccessControl: &types.AccessControlMetadata{
				RequiredRoles:  []string{"admin"},
				TeamRestricted: true,
			},
		},
	}

	c := newTestContext(nil)
	result := checkAccessControl(c, mem, "any-agent", config)
	assert.True(t, result, "access control disabled should always allow")
}

func TestCheckAccessControl_NilMemory_Allows(t *testing.T) {
	config := AccessControlConfig{Enabled: true}
	c := newTestContext(nil)
	result := checkAccessControl(c, nil, "agent-1", config)
	assert.True(t, result, "nil memory should allow access")
}

func TestCheckAccessControl_NilACL_Allows(t *testing.T) {
	config := AccessControlConfig{Enabled: true}
	mem := &types.Memory{
		Key:      "key",
		Metadata: types.MemoryMetadata{AccessControl: nil},
	}
	c := newTestContext(nil)
	result := checkAccessControl(c, mem, "agent-1", config)
	assert.True(t, result, "nil ACL should allow access")
}

func TestCheckAccessControl_RequiredRoles_CallerHasRole(t *testing.T) {
	config := AccessControlConfig{Enabled: true}
	mem := &types.Memory{
		Key: "protected",
		Metadata: types.MemoryMetadata{
			AccessControl: &types.AccessControlMetadata{
				RequiredRoles: []string{"admin", "operator"},
			},
		},
	}

	c := newTestContext(map[string]string{"X-Agent-Roles": "admin,reader"})
	result := checkAccessControl(c, mem, "agent-1", config)
	assert.True(t, result, "caller with required role should be allowed")
}

func TestCheckAccessControl_RequiredRoles_CallerLacksRole(t *testing.T) {
	config := AccessControlConfig{Enabled: true}
	mem := &types.Memory{
		Key: "protected",
		Metadata: types.MemoryMetadata{
			AccessControl: &types.AccessControlMetadata{
				RequiredRoles: []string{"admin"},
			},
		},
	}

	c := newTestContext(map[string]string{"X-Agent-Roles": "reader"})
	result := checkAccessControl(c, mem, "agent-1", config)
	assert.False(t, result, "caller without required role should be denied")
}

func TestCheckAccessControl_RequiredRoles_NoRolesHeader(t *testing.T) {
	config := AccessControlConfig{Enabled: true}
	mem := &types.Memory{
		Key: "protected",
		Metadata: types.MemoryMetadata{
			AccessControl: &types.AccessControlMetadata{
				RequiredRoles: []string{"admin"},
			},
		},
	}

	c := newTestContext(nil)
	result := checkAccessControl(c, mem, "agent-1", config)
	assert.False(t, result, "caller with no roles should be denied when roles required")
}

func TestCheckAccessControl_TeamRestricted_SameTeam(t *testing.T) {
	config := AccessControlConfig{Enabled: true}
	mem := &types.Memory{
		Key: "team-data",
		Metadata: types.MemoryMetadata{
			AccessControl: &types.AccessControlMetadata{
				TeamRestricted: true,
			},
			Custom: map[string]interface{}{"team": "team-alpha"},
		},
	}

	c := newTestContext(map[string]string{"X-Team-ID": "team-alpha"})
	result := checkAccessControl(c, mem, "agent-alpha", config)
	assert.True(t, result, "same team should be allowed")
}

func TestCheckAccessControl_TeamRestricted_DifferentTeam(t *testing.T) {
	config := AccessControlConfig{Enabled: true}
	mem := &types.Memory{
		Key: "team-data",
		Metadata: types.MemoryMetadata{
			AccessControl: &types.AccessControlMetadata{
				TeamRestricted: true,
			},
			Custom: map[string]interface{}{"team": "team-alpha"},
		},
	}

	c := newTestContext(map[string]string{"X-Team-ID": "team-beta"})
	result := checkAccessControl(c, mem, "agent-beta", config)
	assert.False(t, result, "different team should be denied")
}

func TestCheckAccessControl_TeamRestricted_NoTeamHeader(t *testing.T) {
	config := AccessControlConfig{Enabled: true}
	mem := &types.Memory{
		Key: "team-data",
		Metadata: types.MemoryMetadata{
			AccessControl: &types.AccessControlMetadata{
				TeamRestricted: true,
			},
			Custom: map[string]interface{}{"team": "team-alpha"},
		},
	}

	c := newTestContext(nil)
	result := checkAccessControl(c, mem, "agent-unknown", config)
	assert.False(t, result, "missing team header should be denied for team-restricted memory")
}

func TestCheckAccessControl_NoRequirements_Allows(t *testing.T) {
	config := AccessControlConfig{Enabled: true}
	mem := &types.Memory{
		Key: "open",
		Metadata: types.MemoryMetadata{
			AccessControl: &types.AccessControlMetadata{
				TeamRestricted: false,
				RequiredRoles:  nil,
			},
		},
	}

	c := newTestContext(nil)
	result := checkAccessControl(c, mem, "any-agent", config)
	assert.True(t, result, "no restrictions should allow all callers")
}

// --- hasRequiredRole unit tests ---

func TestHasRequiredRole_Match(t *testing.T) {
	assert.True(t, hasRequiredRole([]string{"reader", "admin"}, []string{"admin"}))
}

func TestHasRequiredRole_NoMatch(t *testing.T) {
	assert.False(t, hasRequiredRole([]string{"reader"}, []string{"admin", "operator"}))
}

func TestHasRequiredRole_EmptyCallerRoles(t *testing.T) {
	assert.False(t, hasRequiredRole(nil, []string{"admin"}))
}

func TestHasRequiredRole_EmptyRequiredRoles(t *testing.T) {
	// If required is empty, loop doesn't run — returns false.
	assert.False(t, hasRequiredRole([]string{"admin"}, nil))
}

// --- splitAndTrim unit tests ---

func TestSplitAndTrim_Normal(t *testing.T) {
	result := splitAndTrim("admin, reader , operator", ",")
	assert.Equal(t, []string{"admin", "reader", "operator"}, result)
}

func TestSplitAndTrim_Empty(t *testing.T) {
	result := splitAndTrim("", ",")
	assert.Nil(t, result)
}

func TestSplitAndTrim_SingleValue(t *testing.T) {
	result := splitAndTrim("admin", ",")
	assert.Equal(t, []string{"admin"}, result)
}

func TestSplitAndTrim_WhitespaceOnly(t *testing.T) {
	result := splitAndTrim("  ,  ,  ", ",")
	// splitAndTrim filters empty parts but still returns a non-nil slice
	assert.Empty(t, result)
}

// --- getMemoryTeam unit tests ---

func TestGetMemoryTeam_HasTeam(t *testing.T) {
	mem := &types.Memory{
		Metadata: types.MemoryMetadata{
			Custom: map[string]interface{}{"team": "team-alpha"},
		},
	}
	assert.Equal(t, "team-alpha", getMemoryTeam(mem))
}

func TestGetMemoryTeam_NoCustom(t *testing.T) {
	mem := &types.Memory{Metadata: types.MemoryMetadata{}}
	assert.Equal(t, "", getMemoryTeam(mem))
}

func TestGetMemoryTeam_TeamNotString(t *testing.T) {
	mem := &types.Memory{
		Metadata: types.MemoryMetadata{
			Custom: map[string]interface{}{"team": 123},
		},
	}
	assert.Equal(t, "", getMemoryTeam(mem))
}

// --- EnforceAccessControlHandler middleware tests ---

func TestEnforceAccessControlHandler_Disabled_PassesThrough(t *testing.T) {
	config := AccessControlConfig{Enabled: false}
	handlerCalled := false

	router := gin.New()
	router.Use(EnforceAccessControlHandler(nil, config))
	router.GET("/test", func(c *gin.Context) {
		handlerCalled = true
		c.JSON(http.StatusOK, gin.H{"ok": true})
	})

	req := httptest.NewRequest("GET", "/test", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.True(t, handlerCalled)
}

func TestEnforceAccessControlHandler_Enabled_SetsContextValues(t *testing.T) {
	config := AccessControlConfig{Enabled: true}

	router := gin.New()
	router.Use(EnforceAccessControlHandler(nil, config))
	router.GET("/test", func(c *gin.Context) {
		storedConfig, exists := c.Get("access_control_config")
		assert.True(t, exists, "access_control_config should be set")
		cfg, ok := storedConfig.(AccessControlConfig)
		assert.True(t, ok)
		assert.True(t, cfg.Enabled)

		callerID, _ := c.Get("access_control_caller_id")
		assert.Equal(t, "agent-xyz", callerID)

		c.JSON(http.StatusOK, gin.H{"ok": true})
	})

	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("X-Caller-Agent-ID", "agent-xyz")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
}

func TestEnforceAccessControlHandler_Enabled_FallsBackToNodeID(t *testing.T) {
	config := AccessControlConfig{Enabled: true}

	router := gin.New()
	router.Use(EnforceAccessControlHandler(nil, config))
	router.GET("/test", func(c *gin.Context) {
		callerID, _ := c.Get("access_control_caller_id")
		assert.Equal(t, "node-123", callerID)
		c.JSON(http.StatusOK, gin.H{"ok": true})
	})

	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("X-Agent-Node-ID", "node-123")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
}

// --- SetMemoryWithAccessControl tests ---

func TestSetMemoryWithAccessControl_ValidRequest(t *testing.T) {
	mockStorage := new(MockMemoryStorage)
	mockStorage.On("SetMemory", mock.Anything, mock.AnythingOfType("*types.Memory")).Return(nil)

	config := AccessControlConfig{Enabled: true}

	router := gin.New()
	router.POST("/memory/acl", SetMemoryWithAccessControl(mockStorage, config))

	body := `{"key":"test-key","data":"test-value"}`
	req := httptest.NewRequest("POST", "/memory/acl", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	mockStorage.AssertExpectations(t)
}

func TestSetMemoryWithAccessControl_MissingRequiredFields(t *testing.T) {
	config := AccessControlConfig{Enabled: true}

	router := gin.New()
	router.POST("/memory/acl", SetMemoryWithAccessControl(nil, config))

	// Missing required "key" field — binding should fail
	body := `{"data":"test-value"}`
	req := httptest.NewRequest("POST", "/memory/acl", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusBadRequest, rec.Code)
}

func TestSetMemoryWithAccessControl_WithACLMetadata(t *testing.T) {
	mockStorage := new(MockMemoryStorage)
	mockStorage.On("SetMemory", mock.Anything, mock.AnythingOfType("*types.Memory")).Return(nil)

	config := AccessControlConfig{Enabled: true}

	router := gin.New()
	router.POST("/memory/acl", SetMemoryWithAccessControl(mockStorage, config))

	body := `{
		"key":"secure-key",
		"data":"secure-value",
		"access_control": {
			"required_roles": ["admin"],
			"team_restricted": false
		}
	}`
	req := httptest.NewRequest("POST", "/memory/acl", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var respMem types.Memory
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &respMem))
	assert.Equal(t, "secure-key", respMem.Key)
	require.NotNil(t, respMem.Metadata.AccessControl)
	assert.Equal(t, []string{"admin"}, respMem.Metadata.AccessControl.RequiredRoles)

	mockStorage.AssertExpectations(t)
}

func TestSetMemoryWithAccessControl_MalformedJSON(t *testing.T) {
	config := AccessControlConfig{Enabled: true}

	router := gin.New()
	router.POST("/memory/acl", SetMemoryWithAccessControl(nil, config))

	req := httptest.NewRequest("POST", "/memory/acl", strings.NewReader("{invalid json"))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusBadRequest, rec.Code)
}

func TestDefaultAccessControlConfig(t *testing.T) {
	cfg := DefaultAccessControlConfig()
	assert.False(t, cfg.Enabled)
	assert.False(t, cfg.AuditLogEnabled)
}
