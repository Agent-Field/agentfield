package agentic

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/server/apicatalog"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDiscoverHandler_WithMethodFilter(t *testing.T) {
	catalog := setupCatalog()
	router := gin.New()
	router.Use(func(c *gin.Context) { c.Set("auth_level", "api_key"); c.Next() })
	router.GET("/api/v1/agentic/discover", DiscoverHandler(catalog))

	req := httptest.NewRequest("GET", "/api/v1/agentic/discover?method=POST", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.True(t, resp.OK)

	data := resp.Data.(map[string]interface{})
	endpoints := data["endpoints"].([]interface{})
	for _, ep := range endpoints {
		entry := ep.(map[string]interface{})
		assert.Equal(t, "POST", entry["method"])
	}
}

func TestDiscoverHandler_LimitClampAbove100(t *testing.T) {
	catalog := setupCatalog()
	router := gin.New()
	router.Use(func(c *gin.Context) { c.Set("auth_level", "api_key"); c.Next() })
	router.GET("/api/v1/agentic/discover", DiscoverHandler(catalog))

	req := httptest.NewRequest("GET", "/api/v1/agentic/discover?limit=500", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.True(t, resp.OK)

	data := resp.Data.(map[string]interface{})
	endpoints := data["endpoints"].([]interface{})
	assert.LessOrEqual(t, len(endpoints), 100)
}

func TestDiscoverHandler_AllFiltersCombined(t *testing.T) {
	catalog := setupCatalog()
	router := gin.New()
	router.Use(func(c *gin.Context) { c.Set("auth_level", "api_key"); c.Next() })
	router.GET("/api/v1/agentic/discover", DiscoverHandler(catalog))

	req := httptest.NewRequest("GET", "/api/v1/agentic/discover?q=agent&group=discovery&method=GET&limit=5", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.True(t, resp.OK)

	data := resp.Data.(map[string]interface{})
	endpoints := data["endpoints"].([]interface{})
	assert.LessOrEqual(t, len(endpoints), 5)

	filters := data["filters"].(map[string]interface{})
	assert.Equal(t, "agent", filters["q"])
	assert.Equal(t, "discovery", filters["group"])
	assert.Equal(t, "GET", filters["method"])
}

func TestDiscoverHandler_SeeAlsoContainsExpectedKeys(t *testing.T) {
	catalog := setupCatalog()
	router := gin.New()
	router.GET("/api/v1/agentic/discover", DiscoverHandler(catalog))

	req := httptest.NewRequest("GET", "/api/v1/agentic/discover", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	data := resp.Data.(map[string]interface{})

	seeAlso := data["see_also"].(map[string]interface{})
	assert.Contains(t, seeAlso, "live_agents")
	assert.Contains(t, seeAlso, "kb")
}

func TestSmart404Handler_SuggestionsAreAuthFiltered(t *testing.T) {
	catalog := setupCatalog()
	router := gin.New()
	router.Use(func(c *gin.Context) { c.Set("auth_level", "api_key"); c.Next() })
	router.NoRoute(Smart404Handler(catalog, nil))

	req := httptest.NewRequest("POST", "/api/v1/nonexistent-endpoint", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusNotFound, rec.Code)

	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
	assert.Equal(t, "endpoint_not_found", body["error"])
	suggestions, ok := body["suggestions"].([]interface{})
	assert.True(t, ok)
	for _, s := range suggestions {
		entry := s.(map[string]interface{})
		assert.Contains(t, entry, "method")
		assert.Contains(t, entry, "path")
		assert.Contains(t, entry, "summary")
		assert.Contains(t, entry, "score")
	}
}

func TestDiscoverAndSmart404_UseAuthenticatedCatalogVisibility(t *testing.T) {
	catalog := apicatalog.New()
	catalog.RegisterBatch([]apicatalog.EndpointEntry{
		{Method: http.MethodGet, Path: "/api/v1/visible", Group: "test", Summary: "Visible endpoint", AuthLevel: "api_key"},
		{Method: http.MethodGet, Path: "/api/v1/admin-only", Group: "test", Summary: "Admin endpoint", AuthLevel: "admin"},
	})
	router := gin.New()
	router.Use(middleware.APIKeyAuth(middleware.AuthConfig{APIKey: "test-api-key"}))
	router.GET("/api/v1/agentic/discover", DiscoverHandler(catalog))
	router.NoRoute(Smart404Handler(catalog, nil))

	t.Run("discover exposes only API-key-accessible endpoints", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/v1/agentic/discover?group=test", nil)
		req.Header.Set("X-API-Key", "test-api-key")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		var resp AgenticResponse
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
		endpoints := resp.Data.(map[string]interface{})["endpoints"].([]interface{})
		require.Len(t, endpoints, 1)
		assert.Equal(t, "/api/v1/visible", endpoints[0].(map[string]interface{})["path"])
	})

	t.Run("smart 404 excludes admin-only suggestions", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/v1/admin-onli", nil)
		req.Header.Set("X-API-Key", "test-api-key")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusNotFound, rec.Code)
		var body map[string]interface{}
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
		for _, suggestion := range body["suggestions"].([]interface{}) {
			assert.NotEqual(t, "/api/v1/admin-only", suggestion.(map[string]interface{})["path"])
		}
	})
}

func TestSmart404Handler_HelpLinksContainExpectedKeys(t *testing.T) {
	catalog := setupCatalog()
	router := gin.New()
	router.NoRoute(Smart404Handler(catalog, nil))

	req := httptest.NewRequest("GET", "/api/v1/not-real", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusNotFound, rec.Code)

	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))

	help := body["help"].(map[string]interface{})
	assert.Contains(t, help, "api_search")
	assert.Contains(t, help, "live_agents")
	assert.Contains(t, help, "kb")
	assert.Contains(t, help, "guide")
}

func TestDiscoverHandler_NegativeLimitDefaults(t *testing.T) {
	catalog := setupCatalog()
	router := gin.New()
	router.Use(func(c *gin.Context) { c.Set("auth_level", "api_key"); c.Next() })
	router.GET("/api/v1/agentic/discover", DiscoverHandler(catalog))

	req := httptest.NewRequest("GET", "/api/v1/agentic/discover?limit=-1", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	data := resp.Data.(map[string]interface{})
	endpoints := data["endpoints"].([]interface{})
	assert.LessOrEqual(t, len(endpoints), 20)
}

func TestDiscoverHandler_SeeAlsoReferences(t *testing.T) {
	entries := apicatalog.DefaultEntries()
	c := apicatalog.New()
	c.RegisterBatch(entries)

	router := gin.New()
	router.GET("/api/v1/agentic/discover", DiscoverHandler(c))

	req := httptest.NewRequest("GET", "/api/v1/agentic/discover", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.True(t, resp.OK)

	data := resp.Data.(map[string]interface{})
	seeAlso := data["see_also"].(map[string]interface{})
	assert.NotEmpty(t, seeAlso["live_agents"])
	assert.NotEmpty(t, seeAlso["kb"])
}

func TestSmart404Handler_MessageContainsMethodPath(t *testing.T) {
	catalog := setupCatalog()
	router := gin.New()
	router.NoRoute(Smart404Handler(catalog, nil))

	req := httptest.NewRequest("DELETE", "/api/v2/old-endpoint", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusNotFound, rec.Code)

	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
	msg, ok := body["message"].(string)
	assert.True(t, ok)
	assert.Contains(t, msg, "DELETE")
	assert.Contains(t, msg, "/api/v2/old-endpoint")
}
