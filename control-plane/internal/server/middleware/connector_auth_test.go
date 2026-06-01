package middleware

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// APIKeyAuth — connector route skip
// ---------------------------------------------------------------------------

func TestAPIKeyAuth_SkipConnectorRoutes(t *testing.T) {
	// Connector routes should bypass global API key auth entirely.
	// They are protected by ConnectorTokenAuth instead.
	router := gin.New()
	router.Use(APIKeyAuth(AuthConfig{APIKey: "secret-key"}))
	router.GET("/api/v1/connector/manifest", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"message": "manifest"})
	})
	router.GET("/api/v1/connector/reasoners", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"message": "reasoners"})
	})
	paths := []string{
		"/api/v1/connector/manifest",
		"/api/v1/connector/reasoners",
	}
	for _, path := range paths {
		t.Run(path, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, path, nil)
			// No API key — should still be allowed through global middleware
			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)
			assert.Equal(t, http.StatusOK, w.Code, "connector route %s should bypass API key auth", path)
		})
	}
}

func TestAPIKeyAuth_DoesNotSkipNonConnectorRoutes(t *testing.T) {
	// Ensure the connector skip doesn't accidentally bypass other /api/v1/ routes
	router := gin.New()
	router.Use(APIKeyAuth(AuthConfig{APIKey: "secret-key"}))
	router.GET("/api/v1/agents", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"message": "agents"})
	})
	router.GET("/api/v1/connectors", func(c *gin.Context) {
		// Note: /api/v1/connectors (plural) is NOT /api/v1/connector/
		c.JSON(http.StatusOK, gin.H{"message": "connectors"})
	})

	tests := []struct {
		name string
		path string
	}{
		{"agents endpoint", "/api/v1/agents"},
		{"connectors (plural, not connector/)", "/api/v1/connectors"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, tt.path, nil)
			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)
			assert.Equal(t, http.StatusUnauthorized, w.Code,
				"non-connector route %s should still require API key", tt.path)
		})
	}
}

// ---------------------------------------------------------------------------
// ConnectorTokenAuth
// ---------------------------------------------------------------------------

func TestConnectorTokenAuth_ValidToken(t *testing.T) {
	router := gin.New()
	router.Use(ConnectorTokenAuth("connector-secret"))
	router.GET("/test", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"message": "ok"})
	})

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	req.Header.Set("X-Connector-Token", "connector-secret")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestConnectorTokenAuth_InvalidToken(t *testing.T) {
	router := gin.New()
	router.Use(ConnectorTokenAuth("connector-secret"))
	router.GET("/test", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"message": "ok"})
	})

	tests := []struct {
		name  string
		token string
	}{
		{"wrong token", "wrong-token"},
		{"empty token", ""},
		{"partial match", "connector-secre"},
		{"extra chars", "connector-secret-extra"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/test", nil)
			if tt.token != "" {
				req.Header.Set("X-Connector-Token", tt.token)
			}
			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)

			assert.Equal(t, http.StatusForbidden, w.Code)
			var resp map[string]string
			err := json.Unmarshal(w.Body.Bytes(), &resp)
			require.NoError(t, err)
			assert.Equal(t, "forbidden", resp["error"])
		})
	}
}

func TestConnectorTokenAuth_NoTokenConfigured(t *testing.T) {
	// When connector token is not configured on CP, all requests should be rejected
	router := gin.New()
	router.Use(ConnectorTokenAuth(""))
	router.GET("/test", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"message": "ok"})
	})

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	req.Header.Set("X-Connector-Token", "some-token")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusForbidden, w.Code)
	var resp map[string]string
	err := json.Unmarshal(w.Body.Bytes(), &resp)
	require.NoError(t, err)
	assert.Contains(t, resp["message"], "not configured")
}

func TestConnectorTokenAuth_DoesNotAcceptAPIKey(t *testing.T) {
	// Connector token auth only accepts X-Connector-Token, NOT X-API-Key or Bearer
	router := gin.New()
	router.Use(ConnectorTokenAuth("the-token"))
	router.GET("/test", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"message": "ok"})
	})

	tests := []struct {
		name   string
		header string
		value  string
	}{
		{"X-API-Key header", "X-API-Key", "the-token"},
		{"Bearer token", "Authorization", "Bearer the-token"},
		{"query param", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			url := "/test"
			if tt.name == "query param" {
				url = "/test?api_key=the-token"
			}
			req := httptest.NewRequest(http.MethodGet, url, nil)
			if tt.header != "" {
				req.Header.Set(tt.header, tt.value)
			}
			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)
			assert.Equal(t, http.StatusForbidden, w.Code,
				"connector auth should only accept X-Connector-Token, not %s", tt.name)
		})
	}
}

func TestConnectorTokenAuth_InjectsAuditMetadata(t *testing.T) {
	var capturedCmdID, capturedCmdSource string

	router := gin.New()
	router.Use(ConnectorTokenAuth("connector-secret"))
	router.GET("/test", func(c *gin.Context) {
		if v, ok := c.Get("connector_command_id"); ok {
			capturedCmdID, _ = v.(string)
		}
		if v, ok := c.Get("connector_command_source"); ok {
			capturedCmdSource, _ = v.(string)
		}
		c.JSON(http.StatusOK, gin.H{"message": "ok"})
	})

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	req.Header.Set("X-Connector-Token", "connector-secret")
	req.Header.Set("X-Command-ID", "cmd_123")
	req.Header.Set("X-Command-Source", "hax-sdk")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "cmd_123", capturedCmdID)
	assert.Equal(t, "hax-sdk", capturedCmdSource)
}

// ---------------------------------------------------------------------------
// ConnectorCapabilityCheck
// ---------------------------------------------------------------------------

func TestConnectorCapabilityCheck_EnabledCapability(t *testing.T) {
	caps := map[string]config.ConnectorCapability{
		"reasoner_management": {Enabled: true, ReadOnly: false},
	}

	router := gin.New()
	router.Use(ConnectorCapabilityCheck("reasoner_management", caps))
	router.GET("/test", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"ok": true}) })
	router.POST("/test", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"ok": true}) })

	// GET should work
	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)

	// POST should work (not read-only)
	req = httptest.NewRequest(http.MethodPost, "/test", nil)
	w = httptest.NewRecorder()
	router.ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)
}

func TestConnectorCapabilityCheck_DisabledCapability(t *testing.T) {
	caps := map[string]config.ConnectorCapability{
		"reasoner_management": {Enabled: false},
	}

	router := gin.New()
	router.Use(ConnectorCapabilityCheck("reasoner_management", caps))
	router.GET("/test", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"ok": true}) })

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusForbidden, w.Code)
	var resp map[string]string
	err := json.Unmarshal(w.Body.Bytes(), &resp)
	require.NoError(t, err)
	assert.Equal(t, "capability_disabled", resp["error"])
}

func TestConnectorCapabilityCheck_MissingCapability(t *testing.T) {
	caps := map[string]config.ConnectorCapability{} // empty

	router := gin.New()
	router.Use(ConnectorCapabilityCheck("nonexistent", caps))
	router.GET("/test", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"ok": true}) })

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusForbidden, w.Code)
}

func TestConnectorCapabilityCheck_ReadOnly(t *testing.T) {
	caps := map[string]config.ConnectorCapability{
		"policy_management": {Enabled: true, ReadOnly: true},
	}

	router := gin.New()
	router.Use(ConnectorCapabilityCheck("policy_management", caps))
	router.GET("/test", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"ok": true}) })
	router.POST("/test", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"ok": true}) })
	router.PUT("/test", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"ok": true}) })
	router.DELETE("/test", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"ok": true}) })
	router.PATCH("/test", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"ok": true}) })

	// GET should work in read-only mode
	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)

	// Write methods should be rejected
	writeMethods := []string{http.MethodPost, http.MethodPut, http.MethodDelete, http.MethodPatch}
	for _, method := range writeMethods {
		t.Run(method, func(t *testing.T) {
			req := httptest.NewRequest(method, "/test", nil)
			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)
			assert.Equal(t, http.StatusForbidden, w.Code)
			var resp map[string]string
			err := json.Unmarshal(w.Body.Bytes(), &resp)
			require.NoError(t, err)
			assert.Equal(t, "read_only", resp["error"])
		})
	}
}

// ---------------------------------------------------------------------------
// Integration: Full middleware chain (APIKeyAuth skip → ConnectorTokenAuth)
// ---------------------------------------------------------------------------

func TestConnectorRoutes_FullMiddlewareChain(t *testing.T) {
	// Simulates the real server middleware stack:
	// 1. Global APIKeyAuth (should skip connector routes)
	// 2. ConnectorTokenAuth (should enforce connector token)
	// This proves connector routes are NOT accessible without a valid connector token,
	// even though they bypass the global API key check.

	connectorToken := "connector-secret-token"
	apiKey := "global-api-key"

	router := gin.New()
	router.Use(APIKeyAuth(AuthConfig{APIKey: apiKey}))

	// Connector route group with its own auth
	connectorGroup := router.Group("/api/v1/connector")
	connectorGroup.Use(ConnectorTokenAuth(connectorToken))
	connectorGroup.GET("/manifest", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"capabilities": []string{}})
	})
	connectorGroup.GET("/reasoners", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"reasoners": []string{}})
	})

	// Regular API route for comparison
	router.GET("/api/v1/agents", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"agents": []string{}})
	})

	t.Run("connector route with valid connector token — allowed", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/v1/connector/manifest", nil)
		req.Header.Set("X-Connector-Token", connectorToken)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code)
	})

	t.Run("connector route with no auth — rejected by ConnectorTokenAuth", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/v1/connector/manifest", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
		assert.Equal(t, http.StatusForbidden, w.Code)
		var resp map[string]string
		err := json.Unmarshal(w.Body.Bytes(), &resp)
		require.NoError(t, err)
		assert.Equal(t, "forbidden", resp["error"])
	})

	t.Run("connector route with only API key — rejected by ConnectorTokenAuth", func(t *testing.T) {
		// Even with a valid global API key, connector routes require their own token
		req := httptest.NewRequest(http.MethodGet, "/api/v1/connector/reasoners", nil)
		req.Header.Set("X-API-Key", apiKey)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
		assert.Equal(t, http.StatusForbidden, w.Code)
	})

	t.Run("connector route with wrong connector token — rejected", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/v1/connector/manifest", nil)
		req.Header.Set("X-Connector-Token", "wrong-token")
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
		assert.Equal(t, http.StatusForbidden, w.Code)
	})

	t.Run("regular API route still requires API key", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/v1/agents", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
		assert.Equal(t, http.StatusUnauthorized, w.Code)
	})

	t.Run("regular API route with valid API key — allowed", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/v1/agents", nil)
		req.Header.Set("X-API-Key", apiKey)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
		assert.Equal(t, http.StatusOK, w.Code)
	})
}

func TestConnectorRoutes_FullChainWithCapabilities(t *testing.T) {
	// Full chain: APIKeyAuth skip → ConnectorTokenAuth → ConnectorCapabilityCheck
	connectorToken := "ct-secret"
	caps := map[string]config.ConnectorCapability{
		"reasoner_management": {Enabled: true, ReadOnly: false},
		"policy_management":   {Enabled: true, ReadOnly: true},
		"tag_management":      {Enabled: false},
	}

	router := gin.New()
	router.Use(APIKeyAuth(AuthConfig{APIKey: "api-key"}))

	connectorGroup := router.Group("/api/v1/connector")
	connectorGroup.Use(ConnectorTokenAuth(connectorToken))

	// Manifest — no capability check
	connectorGroup.GET("/manifest", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"ok": true})
	})

	// Reasoner routes — full access
	reasonerGroup := connectorGroup.Group("")
	reasonerGroup.Use(ConnectorCapabilityCheck("reasoner_management", caps))
	reasonerGroup.GET("/reasoners", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"ok": true}) })
	reasonerGroup.POST("/reasoners/r1/restart", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"ok": true}) })

	// Policy routes — read-only
	policyGroup := connectorGroup.Group("")
	policyGroup.Use(ConnectorCapabilityCheck("policy_management", caps))
	policyGroup.GET("/policies", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"ok": true}) })
	policyGroup.POST("/policies", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"ok": true}) })

	// Tag routes — disabled
	tagGroup := connectorGroup.Group("")
	tagGroup.Use(ConnectorCapabilityCheck("tag_management", caps))
	tagGroup.GET("/tags", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"ok": true}) })

	makeReq := func(method, path string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(method, path, nil)
		req.Header.Set("X-Connector-Token", connectorToken)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
		return w
	}

	t.Run("manifest always accessible with valid token", func(t *testing.T) {
		assert.Equal(t, http.StatusOK, makeReq(http.MethodGet, "/api/v1/connector/manifest").Code)
	})

	t.Run("enabled capability GET", func(t *testing.T) {
		assert.Equal(t, http.StatusOK, makeReq(http.MethodGet, "/api/v1/connector/reasoners").Code)
	})

	t.Run("enabled capability POST", func(t *testing.T) {
		assert.Equal(t, http.StatusOK, makeReq(http.MethodPost, "/api/v1/connector/reasoners/r1/restart").Code)
	})

	t.Run("read-only capability GET allowed", func(t *testing.T) {
		assert.Equal(t, http.StatusOK, makeReq(http.MethodGet, "/api/v1/connector/policies").Code)
	})

	t.Run("read-only capability POST rejected", func(t *testing.T) {
		w := makeReq(http.MethodPost, "/api/v1/connector/policies")
		assert.Equal(t, http.StatusForbidden, w.Code)
		var resp map[string]string
		err := json.Unmarshal(w.Body.Bytes(), &resp)
		require.NoError(t, err)
		assert.Equal(t, "read_only", resp["error"])
	})

	t.Run("disabled capability rejected", func(t *testing.T) {
		w := makeReq(http.MethodGet, "/api/v1/connector/tags")
		assert.Equal(t, http.StatusForbidden, w.Code)
		var resp map[string]string
		err := json.Unmarshal(w.Body.Bytes(), &resp)
		require.NoError(t, err)
		assert.Equal(t, "capability_disabled", resp["error"])
	})
}
