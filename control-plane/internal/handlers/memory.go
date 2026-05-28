package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// MemoryStorage captures the storage operations required by memory handlers.
type MemoryStorage interface {
	SetMemory(ctx context.Context, memory *types.Memory) error
	GetMemory(ctx context.Context, scope, scopeID, key string) (*types.Memory, error)
	DeleteMemory(ctx context.Context, scope, scopeID, key string) error
	ListMemory(ctx context.Context, scope, scopeID string) ([]*types.Memory, error)
	StoreEvent(ctx context.Context, event *types.MemoryChangeEvent) error
	PublishMemoryChange(ctx context.Context, event types.MemoryChangeEvent) error
	SetVector(ctx context.Context, record *types.VectorRecord) error
	GetVector(ctx context.Context, scope, scopeID, key string) (*types.VectorRecord, error)
	DeleteVector(ctx context.Context, scope, scopeID, key string) error
	DeleteVectorsByPrefix(ctx context.Context, scope, scopeID, prefix string) (int, error)
	SimilaritySearch(ctx context.Context, scope, scopeID string, queryEmbedding []float32, topK int, filters map[string]interface{}) ([]*types.VectorSearchResult, error)
}

// SetMemoryRequest defines the structure for setting a memory value.
type SetMemoryRequest struct {
	Key   string      `json:"key" binding:"required"`
	Data  interface{} `json:"data" binding:"required"`
	Scope *string     `json:"scope,omitempty"`
}

// GetMemoryRequest defines the structure for getting a memory value.
type GetMemoryRequest struct {
	Key   string  `json:"key" binding:"required"`
	Scope *string `json:"scope,omitempty"`
}

// MemoryResponse defines the structure for a memory API response.
type MemoryResponse struct {
	Key       string      `json:"key"`
	Data      interface{} `json:"data,omitempty"`
	Scope     string      `json:"scope"`
	ScopeID   string      `json:"scope_id"`
	CreatedAt time.Time   `json:"created_at"`
	UpdatedAt time.Time   `json:"updated_at"`
}

// SetMemoryHandler handles the request to set a memory value.
func SetMemoryHandler(storageProvider MemoryStorage) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		logger.Logger.Debug().
			Str("operation", "set_memory").
			Msg("handler called")

		var req SetMemoryRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			logger.Logger.Debug().
				Err(err).
				Str("operation", "bind_json").
				Msg("request binding failed")
			c.JSON(http.StatusBadRequest, ErrorResponse{
				Error:   "invalid_request",
				Details: err.Error(),
				Code:    http.StatusBadRequest,
			})
			return
		}
		logger.Logger.Debug().
			Str("operation", "set_memory").
			Str("key", req.Key).
			Msg("request parsed")

		scope, scopeID := resolveScope(c, req.Scope)
		logger.Logger.Debug().
			Str("operation", "resolve_scope").
			Str("scope", scope).
			Str("scope_id", scopeID).
			Msg("scope resolved")

		// Get existing memory value for event publishing
		logger.Logger.Debug().
			Str("operation", "get_memory").
			Str("scope", scope).
			Str("scope_id", scopeID).
			Str("key", req.Key).
			Msg("checking existing memory")
		var previousData json.RawMessage
		if existingMemory, err := storageProvider.GetMemory(ctx, scope, scopeID, req.Key); err == nil {
			previousData = existingMemory.Data
		}
		logger.Logger.Debug().
			Str("operation", "get_memory").
			Str("scope", scope).
			Str("scope_id", scopeID).
			Str("key", req.Key).
			Msg("existing memory check completed")

		logger.Logger.Debug().
			Str("operation", "marshal").
			Str("field", "memory_data").
			Msg("marshaling request data")
		dataJSON, err := marshalDataWithLogging(req.Data, "memory_data")
		if err != nil {
			logger.Logger.Error().
				Err(err).
				Str("operation", "marshal").
				Str("field", "memory_data").
				Msg("failed to marshal request data")
			c.JSON(http.StatusBadRequest, ErrorResponse{
				Error:   "marshal_error",
				Details: err.Error(),
				Code:    http.StatusBadRequest,
			})
			return
		}
		logger.Logger.Debug().
			Str("operation", "marshal").
			Str("field", "memory_data").
			Int("bytes", len(dataJSON)).
			Msg("request data marshaled")

		now := time.Now()
		memory := &types.Memory{
			Scope:     scope,
			ScopeID:   scopeID,
			Key:       req.Key,
			Data:      dataJSON,
			CreatedAt: now,
			UpdatedAt: now,
		}
		logger.Logger.Debug().
			Str("operation", "set_memory").
			Str("scope", scope).
			Str("scope_id", scopeID).
			Str("key", req.Key).
			Msg("memory object created")

		logger.Logger.Debug().
			Str("operation", "set_memory").
			Str("scope", scope).
			Str("scope_id", scopeID).
			Str("key", req.Key).
			Msg("persisting memory")
		if err := storageProvider.SetMemory(ctx, memory); err != nil {
			logger.Logger.Error().
				Err(err).
				Str("operation", "set_memory").
				Str("scope", scope).
				Str("scope_id", scopeID).
				Str("key", req.Key).
				Msg("failed to persist memory")
			c.JSON(http.StatusInternalServerError, ErrorResponse{
				Error:   "storage_error",
				Details: err.Error(),
				Code:    http.StatusInternalServerError,
			})
			return
		}
		logger.Logger.Debug().
			Str("operation", "set_memory").
			Str("scope", scope).
			Str("scope_id", scopeID).
			Str("key", req.Key).
			Msg("memory persisted")

		// Publish memory change event
		logger.Logger.Debug().
			Str("operation", "create_memory_change_event").
			Str("scope", scope).
			Str("scope_id", scopeID).
			Str("key", req.Key).
			Msg("creating memory change event")
		event := &types.MemoryChangeEvent{
			Type:         "memory_change",
			Scope:        scope,
			ScopeID:      scopeID,
			Key:          req.Key,
			Action:       "set",
			Data:         dataJSON,
			PreviousData: previousData,
			Metadata: types.EventMetadata{
				AgentID:    c.GetHeader("X-Agent-Node-ID"),
				ActorID:    c.GetHeader("X-Actor-ID"),
				WorkflowID: c.GetHeader("X-Workflow-ID"),
			},
		}

		// Store event (don't fail the request if event storage fails)
		logger.Logger.Debug().
			Str("operation", "store_memory_change_event").
			Str("scope", scope).
			Str("scope_id", scopeID).
			Str("key", req.Key).
			Msg("storing event")
		if err := storageProvider.StoreEvent(ctx, event); err != nil {
			// Log error but continue
			logger.Logger.Warn().
				Err(err).
				Str("operation", "store_memory_change_event").
				Str("scope", scope).
				Str("scope_id", scopeID).
				Str("key", req.Key).
				Msg("failed to store memory change event")
		} else if err := storageProvider.PublishMemoryChange(ctx, *event); err != nil {
			logger.Logger.Warn().
				Err(err).
				Str("operation", "publish_memory_change").
				Str("scope", scope).
				Str("scope_id", scopeID).
				Str("key", req.Key).
				Msg("failed to publish memory change event")
		}
		logger.Logger.Debug().
			Str("operation", "store_memory_change_event").
			Str("scope", scope).
			Str("scope_id", scopeID).
			Str("key", req.Key).
			Msg("event handling completed")

		logger.Logger.Debug().
			Str("operation", "set_memory").
			Str("scope", scope).
			Str("scope_id", scopeID).
			Str("key", req.Key).
			Msg("sending response")
		c.JSON(http.StatusOK, memory)
		logger.Logger.Debug().
			Str("operation", "set_memory").
			Str("scope", scope).
			Str("scope_id", scopeID).
			Str("key", req.Key).
			Msg("response sent")
	}
}

// GetMemoryHandler handles the request to get a memory value.
func GetMemoryHandler(storageProvider MemoryStorage) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		var req GetMemoryRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, ErrorResponse{
				Error:   "invalid_request",
				Details: err.Error(),
				Code:    http.StatusBadRequest,
			})
			return
		}

		if req.Scope != nil {
			// Explicit scope provided
			scope, scopeID := resolveScope(c, req.Scope)
			memory, err := storageProvider.GetMemory(ctx, scope, scopeID, req.Key)
			if err != nil {
				c.JSON(http.StatusNotFound, ErrorResponse{
					Error:   "not_found",
					Details: err.Error(),
					Code:    http.StatusNotFound,
				})
				return
			}
			c.JSON(http.StatusOK, memory)
			return
		}

		// Hierarchical search
		scopes := []string{"workflow", "session", "actor", "global"}
		for _, scope := range scopes {
			scopeID := getScopeID(c, scope)
			if scopeID != "" || scope == "global" {
				if scope == "global" {
					scopeID = "global" // Use a consistent ID for global scope
				}
				memory, err := storageProvider.GetMemory(ctx, scope, scopeID, req.Key)
				if err == nil {
					c.JSON(http.StatusOK, memory)
					return
				}
			}
		}

		c.JSON(http.StatusNotFound, ErrorResponse{
			Error:   "not_found",
			Details: "Memory key not found in any scope",
			Code:    http.StatusNotFound,
		})
	}
}

// DeleteMemoryHandler handles the request to delete a memory value.
func DeleteMemoryHandler(storageProvider MemoryStorage) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		var req GetMemoryRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, ErrorResponse{
				Error:   "invalid_request",
				Details: err.Error(),
				Code:    http.StatusBadRequest,
			})
			return
		}

		scope, scopeID := resolveScope(c, req.Scope)

		// Get existing memory value for event publishing
		var previousData json.RawMessage
		if existingMemory, err := storageProvider.GetMemory(ctx, scope, scopeID, req.Key); err == nil {
			previousData = existingMemory.Data
		}

		if err := storageProvider.DeleteMemory(ctx, scope, scopeID, req.Key); err != nil {
			c.JSON(http.StatusNotFound, ErrorResponse{
				Error:   "not_found",
				Details: err.Error(),
				Code:    http.StatusNotFound,
			})
			return
		}

		// Publish memory change event
		event := &types.MemoryChangeEvent{
			Type:         "memory_change",
			Scope:        scope,
			ScopeID:      scopeID,
			Key:          req.Key,
			Action:       "delete",
			Data:         nil, // No new data for delete
			PreviousData: previousData,
			Metadata: types.EventMetadata{
				AgentID:    c.GetHeader("X-Agent-Node-ID"),
				ActorID:    c.GetHeader("X-Actor-ID"),
				WorkflowID: c.GetHeader("X-Workflow-ID"),
			},
		}

		// Store event (don't fail the request if event storage fails)
		if err := storageProvider.StoreEvent(ctx, event); err != nil {
			// Log error but continue
			logger.Logger.Warn().
				Err(err).
				Str("operation", "store_memory_change_event").
				Str("scope", scope).
				Str("scope_id", scopeID).
				Str("key", req.Key).
				Msg("failed to store memory change event")
		} else if err := storageProvider.PublishMemoryChange(ctx, *event); err != nil {
			logger.Logger.Warn().
				Err(err).
				Str("operation", "publish_memory_change").
				Str("scope", scope).
				Str("scope_id", scopeID).
				Str("key", req.Key).
				Msg("failed to publish memory change event")
		}

		c.Status(http.StatusNoContent)
	}
}

// ListMemoryHandler handles the request to list memory values in a scope.
func ListMemoryHandler(storageProvider MemoryStorage) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		scopeParam := c.Query("scope")
		if scopeParam == "" {
			c.JSON(http.StatusBadRequest, ErrorResponse{
				Error:   "missing_scope",
				Details: "Scope parameter is required for listing memory",
				Code:    http.StatusBadRequest,
			})
			return
		}

		scope, scopeID := resolveScope(c, &scopeParam)

		memories, err := storageProvider.ListMemory(ctx, scope, scopeID)
		if err != nil {
			c.JSON(http.StatusInternalServerError, ErrorResponse{
				Error:   "storage_error",
				Details: err.Error(),
				Code:    http.StatusInternalServerError,
			})
			return
		}

		if memories == nil {
			memories = []*types.Memory{}
		}

		c.JSON(http.StatusOK, memories)
	}
}

// resolveScope determines the memory scope and scope ID to use.
func resolveScope(c *gin.Context, explicitScope *string) (string, string) {
	if explicitScope != nil {
		return *explicitScope, getScopeID(c, *explicitScope)
	}

	if id := c.GetHeader("X-Workflow-ID"); id != "" {
		return "workflow", id
	}
	if id := c.GetHeader("X-Session-ID"); id != "" {
		return "session", id
	}
	if id := c.GetHeader("X-Actor-ID"); id != "" {
		return "actor", id
	}

	return "global", "global"
}

// getScopeID retrieves the appropriate ID for a given scope from headers.
func getScopeID(c *gin.Context, scope string) string {
	switch scope {
	case "workflow":
		return c.GetHeader("X-Workflow-ID")
	case "session":
		return c.GetHeader("X-Session-ID")
	case "actor":
		return c.GetHeader("X-Actor-ID")
	case "global":
		return "global"
	default:
		return ""
	}
}
