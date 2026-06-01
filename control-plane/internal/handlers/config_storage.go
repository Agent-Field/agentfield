package handlers

import (
	"io"
	"net/http"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/gin-gonic/gin"
)

// maxConfigBodySize is the maximum allowed size for a config body (1 MB).
// Prevents DoS via unbounded request body reads.
const maxConfigBodySize = 1 << 20 // 1 MB

// ConfigReloadFunc is called to reload configuration from the database.
type ConfigReloadFunc func() error

// ConfigStorageHandlers provides HTTP handlers for database-backed configuration.
type ConfigStorageHandlers struct {
	storage  storage.StorageProvider
	reloadFn ConfigReloadFunc
}

// NewConfigStorageHandlers creates a new ConfigStorageHandlers instance.
func NewConfigStorageHandlers(store storage.StorageProvider, reloadFn ConfigReloadFunc) *ConfigStorageHandlers {
	return &ConfigStorageHandlers{storage: store, reloadFn: reloadFn}
}

// RegisterRoutes registers config storage routes on the given router group.
func (h *ConfigStorageHandlers) RegisterRoutes(group *gin.RouterGroup) {
	group.GET("/configs", h.ListConfigs)
	group.GET("/configs/:key", h.GetConfig)
	group.PUT("/configs/:key", h.SetConfig)
	group.DELETE("/configs/:key", h.DeleteConfig)
	group.POST("/configs/reload", h.ReloadConfig)
}

// ListConfigs returns all stored configuration entries.
func (h *ConfigStorageHandlers) ListConfigs(c *gin.Context) {
	entries, err := h.storage.ListConfigs(c.Request.Context())
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if entries == nil {
		entries = []*storage.ConfigEntry{}
	}
	c.JSON(http.StatusOK, gin.H{
		"configs": entries,
		"total":   len(entries),
	})
}

// GetConfig returns a specific configuration entry by key.
func (h *ConfigStorageHandlers) GetConfig(c *gin.Context) {
	key := c.Param("key")
	entry, err := h.storage.GetConfig(c.Request.Context(), key)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if entry == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "config not found", "key": key})
		return
	}
	c.JSON(http.StatusOK, entry)
}

// SetConfig creates or updates a configuration entry.
// Accepts raw YAML/text body as the config value.
func (h *ConfigStorageHandlers) SetConfig(c *gin.Context) {
	key := c.Param("key")

	body, err := io.ReadAll(io.LimitReader(c.Request.Body, maxConfigBodySize+1))
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "failed to read request body"})
		return
	}
	if len(body) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "request body is empty"})
		return
	}
	if len(body) > maxConfigBodySize {
		c.JSON(http.StatusRequestEntityTooLarge, gin.H{
			"error": "config body exceeds maximum size",
			"max":   maxConfigBodySize,
		})
		return
	}

	updatedBy := c.GetHeader("X-Updated-By")
	if updatedBy == "" {
		updatedBy = "api"
	}

	if err := h.storage.SetConfig(c.Request.Context(), key, string(body), updatedBy); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Return the saved entry
	entry, err := h.storage.GetConfig(c.Request.Context(), key)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message": "config saved",
		"config":  entry,
	})
}

// DeleteConfig removes a configuration entry by key.
func (h *ConfigStorageHandlers) DeleteConfig(c *gin.Context) {
	key := c.Param("key")
	if err := h.storage.DeleteConfig(c.Request.Context(), key); err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"message": "config deleted", "key": key})
}

// ReloadConfig triggers a hot-reload of configuration from the database.
func (h *ConfigStorageHandlers) ReloadConfig(c *gin.Context) {
	if h.reloadFn == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error": "config reload not available (AGENTFIELD_CONFIG_SOURCE != db)",
		})
		return
	}
	if err := h.reloadFn(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "config reload failed",
			"details": err.Error(),
		})
		return
	}
	c.JSON(http.StatusOK, gin.H{"message": "config reloaded from database"})
}
