package agentic

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBatchHandler_InvalidJSON(t *testing.T) {
	router := gin.New()
	router.POST("/api/v1/agentic/batch", BatchHandler(router))

	req := httptest.NewRequest("POST", "/api/v1/agentic/batch", bytes.NewBufferString(`{invalid`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusBadRequest, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.False(t, resp.OK)
	assert.Equal(t, "invalid_request", resp.Error.Code)
}

func TestBatchHandler_SingleOperation(t *testing.T) {
	router := gin.New()
	router.GET("/api/v1/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "healthy"})
	})
	router.POST("/api/v1/agentic/batch", BatchHandler(router))

	body := `{"operations":[{"id":"op1","method":"GET","path":"/api/v1/health"}]}`
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
	assert.Len(t, results, 1)
	assert.Equal(t, float64(1), data["total"])

	result := results[0].(map[string]interface{})
	assert.Equal(t, "op1", result["id"])
	assert.Equal(t, float64(200), result["status"])
}

func TestBatchHandler_WithPOSTBody(t *testing.T) {
	router := gin.New()
	router.POST("/api/v1/echo", func(c *gin.Context) {
		var body map[string]interface{}
		if err := c.ShouldBindJSON(&body); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		body["echoed"] = true
		c.JSON(http.StatusOK, body)
	})
	router.POST("/api/v1/agentic/batch", BatchHandler(router))

	body := `{"operations":[
		{"id":"op1","method":"POST","path":"/api/v1/echo","body":{"value":42}},
		{"id":"op2","method":"POST","path":"/api/v1/echo","body":{"msg":"hello"}}
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

func TestBatchHandler_SubrequestError(t *testing.T) {
	router := gin.New()
	router.GET("/api/v1/valid", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"ok": true})
	})
	router.NoRoute(func(c *gin.Context) {
		c.JSON(http.StatusNotFound, gin.H{"error": "not_found"})
	})
	router.POST("/api/v1/agentic/batch", BatchHandler(router))

	body := `{"operations":[
		{"id":"good","method":"GET","path":"/api/v1/valid"},
		{"id":"bad","method":"GET","path":"/api/v1/nonexistent"}
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

	good := results[0].(map[string]interface{})
	assert.Equal(t, float64(200), good["status"])

	bad := results[1].(map[string]interface{})
	assert.Equal(t, float64(404), bad["status"])
}

func TestBatchHandler_AuthHeaderPropagation(t *testing.T) {
	router := gin.New()
	router.GET("/api/v1/check-auth", func(c *gin.Context) {
		key := c.GetHeader("X-API-Key")
		auth := c.GetHeader("Authorization")
		c.JSON(http.StatusOK, gin.H{
			"has_key":  key != "",
			"has_auth": auth != "",
			"key":      key,
			"auth":     auth,
		})
	})
	router.POST("/api/v1/agentic/batch", BatchHandler(router))

	body := `{"operations":[{"id":"op1","method":"GET","path":"/api/v1/check-auth"}]}`
	req := httptest.NewRequest("POST", "/api/v1/agentic/batch", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-API-Key", "test-key-123")
	req.Header.Set("Authorization", "Bearer token-456")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.True(t, resp.OK)

	data := resp.Data.(map[string]interface{})
	results := data["results"].([]interface{})
	require.Len(t, results, 1)

	result := results[0].(map[string]interface{})
	assert.Equal(t, float64(200), result["status"])

	innerBody := result["body"].(map[string]interface{})
	assert.Equal(t, true, innerBody["has_key"])
	assert.Equal(t, true, innerBody["has_auth"])
	assert.Equal(t, "test-key-123", innerBody["key"])
	assert.Equal(t, "Bearer token-456", innerBody["auth"])
}

func TestBatchHandler_AtMaxBatchSize(t *testing.T) {
	router := gin.New()
	router.GET("/api/v1/ok", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})
	router.POST("/api/v1/agentic/batch", BatchHandler(router))

	ops := make([]BatchOperation, maxBatchSize)
	for i := range ops {
		ops[i] = BatchOperation{
			ID:     "op",
			Method: "GET",
			Path:   "/api/v1/ok",
		}
	}
	bodyBytes, _ := json.Marshal(BatchRequest{Operations: ops})

	req := httptest.NewRequest("POST", "/api/v1/agentic/batch", bytes.NewBuffer(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.True(t, resp.OK)

	data := resp.Data.(map[string]interface{})
	results := data["results"].([]interface{})
	assert.Len(t, results, maxBatchSize)
	assert.Equal(t, float64(maxBatchSize), data["total"])
}

func TestBatchHandler_ConcurrentResultIntegrity(t *testing.T) {
	router := gin.New()
	router.GET("/api/v1/item", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"item": true})
	})
	router.POST("/api/v1/agentic/batch", BatchHandler(router))

	var ops []BatchOperation
	for range 5 {
		ops = append(ops, BatchOperation{
			ID:     "op",
			Method: "GET",
			Path:   "/api/v1/item",
		})
	}
	bodyBytes, _ := json.Marshal(BatchRequest{Operations: ops})

	req := httptest.NewRequest("POST", "/api/v1/agentic/batch", bytes.NewBuffer(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.True(t, resp.OK)

	data := resp.Data.(map[string]interface{})
	results := data["results"].([]interface{})
	assert.Len(t, results, 5)

	for _, r := range results {
		result := r.(map[string]interface{})
		assert.Equal(t, float64(200), result["status"])
	}
}

func TestBatchHandler_PreservesCallerIdentityThroughAuthenticatedRoute(t *testing.T) {
	router := gin.New()
	router.Use(middleware.APIKeyAuth(middleware.AuthConfig{APIKey: "test-api-key"}))
	router.GET("/api/v1/protected", func(c *gin.Context) {
		callerAgentID, ok := c.Get(string(middleware.CallerAgentIDKey))
		if !ok {
			c.JSON(http.StatusForbidden, gin.H{"error": "caller identity missing"})
			return
		}
		c.JSON(http.StatusOK, gin.H{"caller_agent_id": callerAgentID})
	})
	router.POST("/api/v1/agentic/batch", BatchHandler(router))

	body := `{"operations":[{"id":"protected","method":"GET","path":"/api/v1/protected"}]}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/agentic/batch", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-API-Key", "test-api-key")
	req.Header.Set("X-Caller-Agent-ID", "calling-agent")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	results := resp.Data.(map[string]interface{})["results"].([]interface{})
	result := results[0].(map[string]interface{})
	assert.Equal(t, float64(http.StatusOK), result["status"])
	assert.Equal(t, "calling-agent", result["body"].(map[string]interface{})["caller_agent_id"])
}

func TestBatchHandler_ForwardsAgentNodeIDToSubRequests(t *testing.T) {
	router := gin.New()
	router.GET("/api/v1/echo-node", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"agent_node_id": c.GetHeader("X-Agent-Node-ID")})
	})
	router.POST("/api/v1/agentic/batch", BatchHandler(router))

	body := `{"operations":[{"id":"echo","method":"GET","path":"/api/v1/echo-node"}]}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/agentic/batch", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Agent-Node-ID", "node-7")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	results := resp.Data.(map[string]interface{})["results"].([]interface{})
	result := results[0].(map[string]interface{})
	assert.Equal(t, float64(http.StatusOK), result["status"])
	assert.Equal(t, "node-7", result["body"].(map[string]interface{})["agent_node_id"])
}
