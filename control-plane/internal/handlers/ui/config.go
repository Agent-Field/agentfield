package ui

import (
	"encoding/json"
	"net/http"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// ConfigHandler provides handlers for configuration management operations.
type ConfigHandler struct {
	storage storage.StorageProvider
}

// NewConfigHandler creates a new ConfigHandler.
func NewConfigHandler(storage storage.StorageProvider) *ConfigHandler {
	return &ConfigHandler{storage: storage}
}

// ErrorResponse represents an error response structure
type ErrorResponse struct {
	Error string `json:"error"`
}

// GetConfigSchemaHandler handles requests for getting configuration schema for an agent
// GET /api/ui/v1/agents/:agentId/config/schema
func (h *ConfigHandler) GetConfigSchemaHandler(c *gin.Context) {
	ctx := c.Request.Context()
	agentID := c.Param("agentId")
	if agentID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "agentId is required"})
		return
	}

	// Get packageId from query parameter
	packageID := c.Query("packageId")
	if packageID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "packageId query parameter is required"})
		return
	}

	// Get the agent package to retrieve the configuration schema
	agentPackage, err := h.storage.GetAgentPackage(ctx, packageID)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{Error: "package not found"})
		return
	}

	// Parse the configuration schema
	var schema map[string]interface{}
	if err := json.Unmarshal(agentPackage.ConfigurationSchema, &schema); err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to parse configuration schema"})
		return
	}

	response := map[string]interface{}{
		"agent_id":   agentID,
		"package_id": packageID,
		"schema":     schema,
		"metadata": map[string]interface{}{
			"package_name":    agentPackage.Name,
			"package_version": agentPackage.Version,
			"description":     agentPackage.Description,
		},
	}

	c.JSON(http.StatusOK, response)
}

// GetConfigHandler handles requests for getting current configuration for an agent
// GET /api/ui/v1/agents/:agentId/config
func (h *ConfigHandler) GetConfigHandler(c *gin.Context) {
	ctx := c.Request.Context()
	agentID := c.Param("agentId")
	if agentID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "agentId is required"})
		return
	}

	// Get packageId from query parameter
	packageID := c.Query("packageId")
	if packageID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "packageId query parameter is required"})
		return
	}

	// Get the current configuration
	config, err := h.storage.GetAgentConfiguration(ctx, agentID, packageID)
	if err != nil {
		// If configuration doesn't exist, return empty configuration
		response := map[string]interface{}{
			"agent_id":      agentID,
			"package_id":    packageID,
			"configuration": map[string]interface{}{},
			"status":        "draft",
			"version":       0,
		}
		c.JSON(http.StatusOK, response)
		return
	}

	response := map[string]interface{}{
		"agent_id":         agentID,
		"package_id":       packageID,
		"configuration":    config.Configuration,
		"encrypted_fields": config.EncryptedFields,
		"status":           config.Status,
		"version":          config.Version,
		"created_at":       config.CreatedAt,
		"updated_at":       config.UpdatedAt,
		"created_by":       config.CreatedBy,
		"updated_by":       config.UpdatedBy,
	}

	c.JSON(http.StatusOK, response)
}

// SetConfigRequest represents the request body for setting configuration
type SetConfigRequest struct {
	Configuration map[string]interface{} `json:"configuration" binding:"required"`
	Status        *string                `json:"status,omitempty"`
}

// SetConfigHandler handles requests for setting configuration for an agent
// POST /api/ui/v1/agents/:agentId/config
func (h *ConfigHandler) SetConfigHandler(c *gin.Context) {
	ctx := c.Request.Context()
	agentID := c.Param("agentId")
	if agentID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "agentId is required"})
		return
	}

	// Get packageId from query parameter
	packageID := c.Query("packageId")
	if packageID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "packageId query parameter is required"})
		return
	}

	// Parse request body
	var req SetConfigRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "invalid request body: " + err.Error()})
		return
	}

	// Validate configuration against package schema
	validationResult, err := h.storage.ValidateAgentConfiguration(ctx, agentID, packageID, req.Configuration)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to validate configuration"})
		return
	}

	if !validationResult.Valid {
		c.JSON(http.StatusBadRequest, map[string]interface{}{
			"error":             "configuration validation failed",
			"validation_errors": validationResult.Errors,
		})
		return
	}

	// Set default status if not provided
	status := types.ConfigurationStatusDraft
	if req.Status != nil {
		status = types.ConfigurationStatus(*req.Status)
	}

	// Check if configuration already exists
	existingConfig, err := h.storage.GetAgentConfiguration(ctx, agentID, packageID)

	if err != nil {
		// Configuration doesn't exist, create new one
		newConfig := &types.AgentConfiguration{
			AgentID:       agentID,
			PackageID:     packageID,
			Configuration: req.Configuration,
			Status:        status,
			Version:       1,
		}

		if err := h.storage.StoreAgentConfiguration(ctx, newConfig); err != nil {
			c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to store configuration"})
			return
		}

		response := map[string]interface{}{
			"agent_id":      agentID,
			"package_id":    packageID,
			"configuration": newConfig.Configuration,
			"status":        newConfig.Status,
			"version":       newConfig.Version,
			"message":       "configuration created successfully",
		}

		c.JSON(http.StatusCreated, response)
	} else {
		// Configuration exists, update it
		existingConfig.Configuration = req.Configuration
		existingConfig.Status = status
		existingConfig.Version++

		if err := h.storage.UpdateAgentConfiguration(ctx, existingConfig); err != nil {
			c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to update configuration"})
			return
		}

		response := map[string]interface{}{
			"agent_id":      agentID,
			"package_id":    packageID,
			"configuration": existingConfig.Configuration,
			"status":        existingConfig.Status,
			"version":       existingConfig.Version,
			"message":       "configuration updated successfully",
		}

		c.JSON(http.StatusOK, response)
	}
}
