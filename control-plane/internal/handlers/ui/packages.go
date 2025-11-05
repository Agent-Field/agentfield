package ui

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"

	"github.com/your-org/agentfield/control-plane/internal/storage"
	"github.com/your-org/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// PackageHandler provides handlers for agent package management operations.
type PackageHandler struct {
	storage storage.StorageProvider
}

// NewPackageHandler creates a new PackageHandler.
func NewPackageHandler(storage storage.StorageProvider) *PackageHandler {
	return &PackageHandler{storage: storage}
}

// PackageListResponse represents the response for listing packages
type PackageListResponse struct {
	Packages []PackageInfo `json:"packages"`
	Total    int           `json:"total"`
}

// PackageInfo represents package information in the list
type PackageInfo struct {
	ID                    string `json:"id"`
	Name                  string `json:"name"`
	Version               string `json:"version"`
	Status                string `json:"status"`
	InstallPath           string `json:"install_path"`
	ConfigurationRequired bool   `json:"configuration_required"`
	ConfigurationComplete bool   `json:"configuration_complete"`
	RunningNodeID         string `json:"running_node_id,omitempty"`
	LastStarted           string `json:"last_started,omitempty"`
	ProcessID             int    `json:"process_id,omitempty"`
	Port                  int    `json:"port,omitempty"`
	Description           string `json:"description"`
	Author                string `json:"author"`
}

// PackageDetailsResponse represents detailed package information
type PackageDetailsResponse struct {
	ID            string               `json:"id"`
	Name          string               `json:"name"`
	Version       string               `json:"version"`
	Description   string               `json:"description"`
	Author        string               `json:"author"`
	InstallPath   string               `json:"install_path"`
	Status        string               `json:"status"`
	Configuration PackageConfiguration `json:"configuration"`
	Capabilities  *PackageCapabilities `json:"capabilities,omitempty"`
	Runtime       *PackageRuntime      `json:"runtime,omitempty"`
}

// PackageConfiguration represents configuration information
type PackageConfiguration struct {
	Required bool                   `json:"required"`
	Complete bool                   `json:"complete"`
	Schema   map[string]interface{} `json:"schema"`
	Current  map[string]interface{} `json:"current"`
}

// PackageCapabilities represents package capabilities
type PackageCapabilities struct {
	Reasoners  []ReasonerDefinition  `json:"reasoners"`
	Skills     []SkillDefinition     `json:"skills"`
	MCPServers []MCPServerDefinition `json:"mcp_servers"`
}

// ReasonerDefinition represents a reasoner definition
type ReasonerDefinition struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Description  string                 `json:"description"`
	InputSchema  map[string]interface{} `json:"input_schema"`
	OutputSchema map[string]interface{} `json:"output_schema,omitempty"`
}

// SkillDefinition represents a skill definition
type SkillDefinition struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"input_schema"`
	Tags        []string               `json:"tags"`
}

// MCPServerDefinition represents an MCP server definition
type MCPServerDefinition struct {
	Alias       string   `json:"alias"`
	Command     []string `json:"command"`
	Description string   `json:"description"`
	Tools       []string `json:"tools"`
}

// PackageRuntime represents runtime information
type PackageRuntime struct {
	ProcessID int      `json:"process_id,omitempty"`
	Port      int      `json:"port,omitempty"`
	NodeID    string   `json:"node_id,omitempty"`
	StartedAt string   `json:"started_at,omitempty"`
	Logs      []string `json:"logs,omitempty"`
}

// ListPackagesHandler handles requests for listing installed agent packages
// GET /api/ui/v1/agents/packages
func (h *PackageHandler) ListPackagesHandler(c *gin.Context) {
	// Get query parameters for filtering
	status := c.Query("status")
	search := c.Query("search")

	// Get all agent packages from storage
	ctx := c.Request.Context()
	packages, err := h.storage.QueryAgentPackages(ctx, types.PackageFilters{})
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to list packages"})
		return
	}

	var packageInfos []PackageInfo
	for _, pkg := range packages {
		// Determine package status
		packageStatus := h.determinePackageStatus(ctx, pkg)

		// Apply filters
		if status != "" && packageStatus != status {
			continue
		}

		if search != "" && !h.matchesSearch(pkg, search) {
			continue
		}

		// Check configuration status
		configRequired := len(pkg.ConfigurationSchema) > 0
		configComplete := false

		if configRequired {
			// Check if configuration exists and is complete
			config, err := h.storage.GetAgentConfiguration(ctx, pkg.ID, pkg.ID)
			if err == nil && config.Status == types.ConfigurationStatusActive {
				configComplete = true
			}
		} else {
			// No configuration required means it's complete
			configComplete = true
		}

		packageInfo := PackageInfo{
			ID:                    pkg.ID,
			Name:                  pkg.Name,
			Version:               pkg.Version,
			Status:                packageStatus,
			InstallPath:           pkg.InstallPath,
			ConfigurationRequired: configRequired,
			ConfigurationComplete: configComplete,
			Description:           h.safeStringValue(pkg.Description),
			Author:                h.safeStringValue(pkg.Author),
		}

		// TODO: Add runtime information when agent lifecycle management is implemented
		// This would include RunningNodeID, LastStarted, ProcessID, Port

		packageInfos = append(packageInfos, packageInfo)
	}

	response := PackageListResponse{
		Packages: packageInfos,
		Total:    len(packageInfos),
	}

	c.JSON(http.StatusOK, response)
}

// GetPackageDetailsHandler handles requests for getting detailed package information
// GET /api/ui/v1/agents/packages/:packageId/details
func (h *PackageHandler) GetPackageDetailsHandler(c *gin.Context) {
	packageID := c.Param("packageId")
	if packageID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "packageId is required"})
		return
	}

	// Get the agent package
	ctx := c.Request.Context()
	pkg, err := h.storage.GetAgentPackage(ctx, packageID)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{Error: "package not found"})
		return
	}

	// Determine package status
	packageStatus := h.determinePackageStatus(ctx, pkg)

	// Parse configuration schema
	var schema map[string]interface{}
	if len(pkg.ConfigurationSchema) > 0 {
		if err := json.Unmarshal(pkg.ConfigurationSchema, &schema); err != nil {
			c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to parse configuration schema"})
			return
		}
	}

	// Get current configuration
	var currentConfig map[string]interface{}
	configRequired := len(pkg.ConfigurationSchema) > 0
	configComplete := false

	if configRequired {
		config, err := h.storage.GetAgentConfiguration(ctx, packageID, packageID)
		if err == nil {
			currentConfig = config.Configuration
			configComplete = config.Status == types.ConfigurationStatusActive
		} else {
			currentConfig = make(map[string]interface{})
		}
	} else {
		configComplete = true
		currentConfig = make(map[string]interface{})
	}

	// Build response
	response := PackageDetailsResponse{
		ID:          pkg.ID,
		Name:        pkg.Name,
		Version:     pkg.Version,
		Description: h.safeStringValue(pkg.Description),
		Author:      h.safeStringValue(pkg.Author),
		InstallPath: pkg.InstallPath,
		Status:      packageStatus,
		Configuration: PackageConfiguration{
			Required: configRequired,
			Complete: configComplete,
			Schema:   schema,
			Current:  currentConfig,
		},
	}

	// TODO: Add capabilities parsing when agent introspection is implemented
	// This would parse reasoners, skills, and MCP servers from the package

	// TODO: Add runtime information when agent lifecycle management is implemented
	// This would include process information, logs, etc.

	c.JSON(http.StatusOK, response)
}

// determinePackageStatus determines the current status of a package
func (h *PackageHandler) determinePackageStatus(ctx context.Context, pkg *types.AgentPackage) string {
	// Check if configuration is required
	configRequired := len(pkg.ConfigurationSchema) > 0

	if configRequired {
		// Check configuration status
		config, err := h.storage.GetAgentConfiguration(ctx, pkg.ID, pkg.ID)
		if err != nil {
			return "not_configured"
		}

		switch config.Status {
		case types.ConfigurationStatusDraft:
			return "configured"
		case types.ConfigurationStatusActive:
			// TODO: Check if agent is actually running
			// For now, return "configured" until lifecycle management is implemented
			return "configured"
		default:
			return "not_configured"
		}
	}

	// No configuration required
	// TODO: Check if agent is running
	// For now, return "configured" until lifecycle management is implemented
	return "configured"
}

// matchesSearch checks if a package matches the search query
func (h *PackageHandler) matchesSearch(pkg *types.AgentPackage, search string) bool {
	search = strings.ToLower(search)

	return strings.Contains(strings.ToLower(pkg.Name), search) ||
		strings.Contains(strings.ToLower(h.safeStringValue(pkg.Description)), search) ||
		strings.Contains(strings.ToLower(h.safeStringValue(pkg.Author)), search) ||
		strings.Contains(strings.ToLower(pkg.ID), search)
}

// safeStringValue safely converts a *string to string, returning empty string if nil
func (h *PackageHandler) safeStringValue(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}
