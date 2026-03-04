package handlers

import (
	"net/http"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
)

// SetVectorRequest captures inputs for storing a vector embedding.
type SetVectorRequest struct {
	Key       string                 `json:"key" binding:"required"`
	Embedding []float32              `json:"embedding" binding:"required"`
	Metadata  map[string]interface{} `json:"metadata"`
	Scope     *string                `json:"scope,omitempty"`
}

// DeleteVectorRequest removes a vector by key.
type DeleteVectorRequest struct {
	Key   string  `json:"key" binding:"required"`
	Scope *string `json:"scope,omitempty"`
}

// DeleteNamespaceRequest removes all vectors by namespace prefix.
type DeleteNamespaceRequest struct {
	Namespace string  `json:"namespace" binding:"required"`
	Scope     *string `json:"scope,omitempty"`
}

// VectorSearchRequest describes a similarity search query.
type VectorSearchRequest struct {
	QueryEmbedding []float32              `json:"query_embedding" binding:"required"`
	TopK           int                    `json:"top_k"`
	Filters        map[string]interface{} `json:"filters"`
	Scope          *string                `json:"scope,omitempty"`
}

// SetVectorHandler stores or updates a vector embedding.
func SetVectorHandler(storage MemoryStorage) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req SetVectorRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			RespondBadRequest(c, "invalid request")
			return
		}
		if len(req.Embedding) == 0 {
			RespondBadRequest(c, "embedding cannot be empty")
			return
		}

		scope, scopeID := resolveScope(c, req.Scope)
		record := &types.VectorRecord{
			Scope:     scope,
			ScopeID:   scopeID,
			Key:       req.Key,
			Embedding: req.Embedding,
			Metadata:  req.Metadata,
		}

		if err := storage.SetVector(c.Request.Context(), record); err != nil {
			logger.Logger.Error().Err(err).Msg("failed to set vector")
			RespondInternalError(c, "failed to store vector")
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"key":      record.Key,
			"scope":    record.Scope,
			"scope_id": record.ScopeID,
			"metadata": record.Metadata,
		})
	}
}

// GetVectorHandler retrieves a vector by key.
func GetVectorHandler(storage MemoryStorage) gin.HandlerFunc {
	return func(c *gin.Context) {
		key := c.Param("key")
		if key == "" {
			RespondBadRequest(c, "key is required")
			return
		}

		scopeParam := c.Query("scope")
		var scopePtr *string
		if scopeParam != "" {
			scopePtr = &scopeParam
		}

		scope, scopeID := resolveScope(c, scopePtr)
		record, err := storage.GetVector(c.Request.Context(), scope, scopeID, key)
		if err != nil {
			logger.Logger.Error().Err(err).Msg("failed to get vector")
			RespondInternalError(c, "failed to get vector")
			return
		}

		if record == nil {
			RespondNotFound(c, "vector not found")
			return
		}

		c.JSON(http.StatusOK, record)
	}
}

// DeleteVectorHandler removes a vector by key.
func DeleteVectorHandler(storage MemoryStorage) gin.HandlerFunc {
	return func(c *gin.Context) {
		key := c.Param("key")
		if key == "" {
			// Fallback to body for backward compatibility if needed, but the plan says RESTful.
			var req DeleteVectorRequest
			if err := c.ShouldBindJSON(&req); err == nil {
				key = req.Key
			} else {
				RespondBadRequest(c, "key is required")
				return
			}
		}

		scopeParam := c.Query("scope")
		var scopePtr *string
		if scopeParam != "" {
			scopePtr = &scopeParam
		}

		scope, scopeID := resolveScope(c, scopePtr)
		if err := storage.DeleteVector(c.Request.Context(), scope, scopeID, key); err != nil {
			logger.Logger.Error().Err(err).Msg("failed to delete vector")
			RespondInternalError(c, "failed to delete vector")
			return
		}

		c.Status(http.StatusNoContent)
	}
}

// DeleteNamespaceVectorsHandler removes all vectors whose keys start with the namespace prefix.
func DeleteNamespaceVectorsHandler(storage MemoryStorage) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req DeleteNamespaceRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			RespondBadRequest(c, "invalid request")
			return
		}
		if req.Namespace == "" {
			RespondBadRequest(c, "namespace is required")
			return
		}

		scope, scopeID := resolveScope(c, req.Scope)
		deleted, err := storage.DeleteVectorsByPrefix(c.Request.Context(), scope, scopeID, req.Namespace)
		if err != nil {
			logger.Logger.Error().Err(err).Msg("failed to delete namespace vectors")
			RespondInternalError(c, "failed to delete namespace vectors")
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"namespace": req.Namespace,
			"deleted":   deleted,
			"scope":     scope,
			"scope_id":  scopeID,
			"status":    "deleted",
		})
	}
}

// SimilaritySearchHandler performs a similarity search.
func SimilaritySearchHandler(storage MemoryStorage) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req VectorSearchRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			RespondBadRequest(c, "invalid request")
			return
		}

		if len(req.QueryEmbedding) == 0 {
			RespondBadRequest(c, "query_embedding cannot be empty")
			return
		}

		if req.TopK <= 0 {
			req.TopK = 10
		}

		scope, scopeID := resolveScope(c, req.Scope)
		results, err := storage.SimilaritySearch(
			c.Request.Context(),
			scope,
			scopeID,
			req.QueryEmbedding,
			req.TopK,
			req.Filters,
		)
		if err != nil {
			logger.Logger.Error().Err(err).Msg("vector search failed")
			RespondInternalError(c, "vector search failed")
			return
		}

		c.JSON(http.StatusOK, results)
	}
}
