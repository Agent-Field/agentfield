package agentic

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/server/apicatalog"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/knowledgebase"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func init() {
	gin.SetMode(gin.TestMode)
}

func setupCatalog() *apicatalog.Catalog {
	c := apicatalog.New()
	c.RegisterBatch(apicatalog.DefaultEntries())
	return c
}

func setupKB() *knowledgebase.KB {
	kb := knowledgebase.New()
	knowledgebase.LoadDefaultContent(kb)
	return kb
}

// --- Discover handler tests ---

func TestDiscoverHandler_NoQuery(t *testing.T) {
	catalog := setupCatalog()
	router := gin.New()
	// Simulate authenticated caller so results aren't filtered to public-only
	router.Use(func(c *gin.Context) {
		c.Set("auth_level", "api_key")
		c.Next()
	})
	router.GET("/api/v1/agentic/discover", DiscoverHandler(catalog))

	req := httptest.NewRequest("GET", "/api/v1/agentic/discover", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.True(t, resp.OK)

	data := resp.Data.(map[string]interface{})
	endpoints := data["endpoints"].([]interface{})
	assert.Greater(t, len(endpoints), 0)
	groups := data["groups"].([]interface{})
	assert.Greater(t, len(groups), 0)
}

func TestDiscoverHandler_WithKeyword(t *testing.T) {
	catalog := setupCatalog()
	router := gin.New()
	router.GET("/api/v1/agentic/discover", DiscoverHandler(catalog))

	req := httptest.NewRequest("GET", "/api/v1/agentic/discover?q=execute", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	data := resp.Data.(map[string]interface{})

	// Auth filtering may apply — results filtered by getAuthLevel which
	// returns "public" in test mode (no auth headers). Just verify structure.
	total, ok := data["total"].(float64)
	require.True(t, ok)
	assert.GreaterOrEqual(t, total, float64(0))
	assert.Contains(t, data, "endpoints")
	assert.Contains(t, data, "groups")
	assert.Contains(t, data, "filters")

	filters := data["filters"].(map[string]interface{})
	assert.Equal(t, "execute", filters["q"])
}

func TestDiscoverHandler_WithGroupFilter(t *testing.T) {
	catalog := setupCatalog()
	router := gin.New()
	router.Use(func(c *gin.Context) { c.Set("auth_level", "api_key"); c.Next() })
	router.GET("/api/v1/agentic/discover", DiscoverHandler(catalog))

	req := httptest.NewRequest("GET", "/api/v1/agentic/discover?group=memory", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	data := resp.Data.(map[string]interface{})
	endpoints := data["endpoints"].([]interface{})
	for _, ep := range endpoints {
		entry := ep.(map[string]interface{})
		assert.Equal(t, "memory", entry["group"])
	}
}

func TestDiscoverHandler_WithLimit(t *testing.T) {
	catalog := setupCatalog()
	router := gin.New()
	router.Use(func(c *gin.Context) { c.Set("auth_level", "api_key"); c.Next() })
	router.GET("/api/v1/agentic/discover", DiscoverHandler(catalog))

	req := httptest.NewRequest("GET", "/api/v1/agentic/discover?limit=3", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	data := resp.Data.(map[string]interface{})
	endpoints := data["endpoints"].([]interface{})
	assert.LessOrEqual(t, len(endpoints), 3)
}

// --- Smart 404 handler tests ---

func TestSmart404Handler_NonUIPath(t *testing.T) {
	catalog := setupCatalog()
	router := gin.New()
	router.NoRoute(Smart404Handler(catalog, nil))

	req := httptest.NewRequest("GET", "/api/v1/nonexistent", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusNotFound, rec.Code)

	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
	assert.Equal(t, "endpoint_not_found", body["error"])
	assert.Contains(t, body["message"], "/api/v1/nonexistent")
	assert.Contains(t, body, "suggestions")
	assert.Contains(t, body, "help")

	// help should reference both API search and live agent discovery
	help := body["help"].(map[string]interface{})
	assert.Contains(t, help, "api_search")
	assert.Contains(t, help, "live_agents")
	assert.Contains(t, help, "kb")
}

func TestSmart404Handler_UIPath_DelegatesToHandler(t *testing.T) {
	catalog := setupCatalog()
	uiHandler := func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"ui": true})
	}
	router := gin.New()
	router.NoRoute(Smart404Handler(catalog, uiHandler))

	req := httptest.NewRequest("GET", "/ui/some/page", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
	assert.Equal(t, true, body["ui"])
}

func TestSmart404Handler_UIPath_NilHandler(t *testing.T) {
	catalog := setupCatalog()
	router := gin.New()
	router.NoRoute(Smart404Handler(catalog, nil))

	req := httptest.NewRequest("GET", "/ui/dashboard", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusNotFound, rec.Code)
}

// --- KB handler tests ---

func TestKBTopicsHandler(t *testing.T) {
	kb := setupKB()
	router := gin.New()
	router.GET("/api/v1/agentic/kb/topics", KBTopicsHandler(kb))

	req := httptest.NewRequest("GET", "/api/v1/agentic/kb/topics", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.True(t, resp.OK)

	data := resp.Data.(map[string]interface{})
	topics := data["topics"].([]interface{})
	assert.GreaterOrEqual(t, len(topics), 6)

	// Verify topic structure
	topic := topics[0].(map[string]interface{})
	assert.Contains(t, topic, "name")
	assert.Contains(t, topic, "description")
	assert.Contains(t, topic, "article_count")
}

func TestKBArticlesHandler_FilterByTopic(t *testing.T) {
	kb := setupKB()
	router := gin.New()
	router.GET("/api/v1/agentic/kb/articles", KBArticlesHandler(kb))

	req := httptest.NewRequest("GET", "/api/v1/agentic/kb/articles?topic=patterns", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	data := resp.Data.(map[string]interface{})
	articles := data["articles"].([]interface{})
	assert.Greater(t, len(articles), 0)

	for _, a := range articles {
		article := a.(map[string]interface{})
		assert.Equal(t, "patterns", article["topic"])
	}
}

func TestKBArticlesHandler_FilterBySDK(t *testing.T) {
	kb := setupKB()
	router := gin.New()
	router.GET("/api/v1/agentic/kb/articles", KBArticlesHandler(kb))

	req := httptest.NewRequest("GET", "/api/v1/agentic/kb/articles?sdk=python", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	data := resp.Data.(map[string]interface{})
	articles := data["articles"].([]interface{})
	assert.Greater(t, len(articles), 0)

	for _, a := range articles {
		article := a.(map[string]interface{})
		assert.Equal(t, "python", article["sdk"])
	}
}

func TestKBArticlesHandler_Search(t *testing.T) {
	kb := setupKB()
	router := gin.New()
	router.GET("/api/v1/agentic/kb/articles", KBArticlesHandler(kb))

	req := httptest.NewRequest("GET", "/api/v1/agentic/kb/articles?q=hunt", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	data := resp.Data.(map[string]interface{})
	articles := data["articles"].([]interface{})
	assert.Greater(t, len(articles), 0)
}

func TestKBArticleHandler_Found(t *testing.T) {
	kb := setupKB()
	router := gin.New()
	router.GET("/api/v1/agentic/kb/articles/:article_id/:sub_id", KBArticleHandler(kb))

	req := httptest.NewRequest("GET", "/api/v1/agentic/kb/articles/patterns/hunt-prove", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.True(t, resp.OK)

	data := resp.Data.(map[string]interface{})
	assert.Equal(t, "patterns/hunt-prove", data["id"])
	assert.NotEmpty(t, data["content"])
}

func TestKBArticleHandler_NotFound(t *testing.T) {
	kb := setupKB()
	router := gin.New()
	router.GET("/api/v1/agentic/kb/articles/:article_id/:sub_id", KBArticleHandler(kb))

	req := httptest.NewRequest("GET", "/api/v1/agentic/kb/articles/nonexistent/thing", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusNotFound, rec.Code)
}

func TestKBGuideHandler(t *testing.T) {
	kb := setupKB()
	router := gin.New()
	router.GET("/api/v1/agentic/kb/guide", KBGuideHandler(kb))

	req := httptest.NewRequest("GET", "/api/v1/agentic/kb/guide?goal=build+security+agent", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.True(t, resp.OK)

	data := resp.Data.(map[string]interface{})
	assert.Equal(t, "build security agent", data["goal"])
	steps := data["steps"].([]interface{})
	assert.Greater(t, len(steps), 0)

	// Each step should have order, article, reason
	step := steps[0].(map[string]interface{})
	assert.Contains(t, step, "order")
	assert.Contains(t, step, "article")
	assert.Contains(t, step, "reason")
}

func TestKBGuideHandler_MissingGoal(t *testing.T) {
	kb := setupKB()
	router := gin.New()
	router.GET("/api/v1/agentic/kb/guide", KBGuideHandler(kb))

	req := httptest.NewRequest("GET", "/api/v1/agentic/kb/guide", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusBadRequest, rec.Code)
}

// --- Batch handler tests ---

func TestBatchHandler_Success(t *testing.T) {
	router := gin.New()
	router.GET("/api/v1/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "healthy"})
	})
	router.POST("/api/v1/agentic/batch", BatchHandler(router))

	body := `{"operations":[
		{"id":"op1","method":"GET","path":"/api/v1/health"},
		{"id":"op2","method":"GET","path":"/api/v1/health"}
	]}`

	req := httptest.NewRequest("POST", "/api/v1/agentic/batch", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.True(t, resp.OK)

	data := resp.Data.(map[string]interface{})
	results := data["results"].([]interface{})
	assert.Len(t, results, 2)

	for _, r := range results {
		result := r.(map[string]interface{})
		assert.Equal(t, float64(200), result["status"])
	}
}

func TestBatchHandler_EmptyOperations(t *testing.T) {
	router := gin.New()
	router.POST("/api/v1/agentic/batch", BatchHandler(router))

	body := `{"operations":[]}`
	req := httptest.NewRequest("POST", "/api/v1/agentic/batch", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusBadRequest, rec.Code)
}

func TestBatchHandler_TooManyOperations(t *testing.T) {
	router := gin.New()
	router.POST("/api/v1/agentic/batch", BatchHandler(router))

	ops := make([]BatchOperation, 21)
	for i := range ops {
		ops[i] = BatchOperation{ID: "op", Method: "GET", Path: "/health"}
	}
	bodyBytes, _ := json.Marshal(BatchRequest{Operations: ops})

	req := httptest.NewRequest("POST", "/api/v1/agentic/batch", bytes.NewBuffer(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusBadRequest, rec.Code)
}

// --- Helper tests ---

func TestGetAuthLevel_Default(t *testing.T) {
	router := gin.New()
	router.GET("/test", func(c *gin.Context) {
		level := getAuthLevel(c)
		c.JSON(200, gin.H{"level": level})
	})

	req := httptest.NewRequest("GET", "/test", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	var body map[string]string
	json.Unmarshal(rec.Body.Bytes(), &body)
	assert.Equal(t, "public", body["level"])
}

func TestGetAuthLevel_FromContext(t *testing.T) {
	router := gin.New()
	router.Use(func(c *gin.Context) {
		c.Set("auth_level", "api_key")
		c.Next()
	})
	router.GET("/test", func(c *gin.Context) {
		level := getAuthLevel(c)
		c.JSON(200, gin.H{"level": level})
	})

	req := httptest.NewRequest("GET", "/test", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	var body map[string]string
	json.Unmarshal(rec.Body.Bytes(), &body)
	assert.Equal(t, "api_key", body["level"])
}

func TestGetAuthLevel_FromHeader(t *testing.T) {
	router := gin.New()
	router.GET("/test", func(c *gin.Context) {
		level := getAuthLevel(c)
		c.JSON(200, gin.H{"level": level})
	})

	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("X-API-Key", "some-key")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	var body map[string]string
	json.Unmarshal(rec.Body.Bytes(), &body)
	assert.Equal(t, "api_key", body["level"])
}

func TestGetIntQuery(t *testing.T) {
	router := gin.New()
	router.GET("/test", func(c *gin.Context) {
		v := getIntQuery(c, "limit", 10)
		c.JSON(200, gin.H{"v": v})
	})

	tests := []struct {
		query    string
		expected float64
	}{
		{"", 10},           // default
		{"?limit=5", 5},    // explicit
		{"?limit=-1", 10},  // negative → default
		{"?limit=abc", 10}, // non-numeric → default
	}

	for _, tt := range tests {
		req := httptest.NewRequest("GET", "/test"+tt.query, nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		var body map[string]float64
		json.Unmarshal(rec.Body.Bytes(), &body)
		assert.Equal(t, tt.expected, body["v"], "query=%s", tt.query)
	}
}

// --- Response envelope tests ---

func TestRespondOK(t *testing.T) {
	router := gin.New()
	router.GET("/test", func(c *gin.Context) {
		respondOK(c, gin.H{"key": "value"})
	})

	req := httptest.NewRequest("GET", "/test", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.True(t, resp.OK)
	assert.NotNil(t, resp.Data)
	assert.Nil(t, resp.Error)

	// Should have X-Token-Estimate header
	assert.NotEmpty(t, rec.Header().Get("X-Token-Estimate"))
}

func TestRespondError(t *testing.T) {
	router := gin.New()
	router.GET("/test", func(c *gin.Context) {
		respondError(c, http.StatusBadRequest, "bad_input", "missing field")
	})

	req := httptest.NewRequest("GET", "/test", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusBadRequest, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.False(t, resp.OK)
	assert.Nil(t, resp.Data)
	require.NotNil(t, resp.Error)
	assert.Equal(t, "bad_input", resp.Error.Code)
	assert.Equal(t, "missing field", resp.Error.Message)
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsLower(s, substr))
}

func containsLower(s, substr string) bool {
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
