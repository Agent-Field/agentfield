package ui

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages/updatecheck"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// PackageHandler provides handlers for agent package management operations.
type PackageHandler struct {
	storage storage.StorageProvider
	updates packageUpdateReader
}

type packageUpdateReader interface {
	RegistryEntry(id string) (packages.InstalledPackage, bool, error)
	RegistryEntries() (map[string]packages.InstalledPackage, error)
	Checker() *updatecheck.Checker
}

// NewPackageHandler creates a new PackageHandler.
func NewPackageHandler(storage storage.StorageProvider) *PackageHandler {
	return &PackageHandler{storage: storage}
}

func (h *PackageHandler) ConfigurePackageUpdates(updates packageUpdateReader) {
	h.updates = updates
}

// PackageListResponse represents the response for listing packages
type PackageListResponse struct {
	Packages []PackageInfo `json:"packages"`
	Total    int           `json:"total"`
}

// PackageInfo represents package information in the list
type PackageInfo struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	Version string `json:"version"`
	Status  string `json:"status"`
	// InstallStatus is the raw registry-backed state ("installed", "running",
	// "stopped", "uninstalled", ...), unlike Status which is a derived
	// configuration summary. Clients listing actually-installed packages
	// (e.g. the desktop app) must filter on this.
	InstallStatus         string        `json:"install_status"`
	InstalledAt           string        `json:"installed_at,omitempty"`
	InstallPath           string        `json:"install_path"`
	Source                string        `json:"source"`
	ConfigurationRequired bool          `json:"configuration_required"`
	ConfigurationComplete bool          `json:"configuration_complete"`
	RunningNodeID         string        `json:"running_node_id,omitempty"`
	LastStarted           string        `json:"last_started,omitempty"`
	ProcessID             int           `json:"process_id,omitempty"`
	Port                  int           `json:"port,omitempty"`
	Description           string        `json:"description"`
	Author                string        `json:"author"`
	InstalledCommit       string        `json:"installed_commit"`
	SourceRef             string        `json:"source_ref"`
	AutoUpdate            bool          `json:"auto_update"`
	Update                PackageUpdate `json:"update"`
}

type PackageUpdate struct {
	Status       updatecheck.Status `json:"status"`
	LatestCommit string             `json:"latest_commit"`
	CheckedAt    string             `json:"checked_at"`
	Message      string             `json:"message"`
}

// PackageDetailsResponse represents detailed package information
type PackageDetailsResponse struct {
	ID              string               `json:"id"`
	Name            string               `json:"name"`
	Version         string               `json:"version"`
	Description     string               `json:"description"`
	Author          string               `json:"author"`
	InstallPath     string               `json:"install_path"`
	Source          string               `json:"source"`
	Status          string               `json:"status"`
	Configuration   PackageConfiguration `json:"configuration"`
	Capabilities    *PackageCapabilities `json:"capabilities,omitempty"`
	Runtime         *PackageRuntime      `json:"runtime,omitempty"`
	InstalledCommit string               `json:"installed_commit"`
	SourceRef       string               `json:"source_ref"`
	AutoUpdate      bool                 `json:"auto_update"`
	Update          PackageUpdate        `json:"update"`
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
	Reasoners []ReasonerDefinition `json:"reasoners"`
	Skills    []SkillDefinition    `json:"skills"`
}

// ReasonerDefinition represents a reasoner definition
type ReasonerDefinition struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Description  string                 `json:"description"`
	InputSchema  map[string]interface{} `json:"input_schema"`
	OutputSchema map[string]interface{} `json:"output_schema,omitempty"`
	Tags         []string               `json:"tags,omitempty"`
}

// SkillDefinition represents a skill definition
type SkillDefinition struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"input_schema"`
	Tags        []string               `json:"tags"`
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
	dbPackages, err := h.storage.QueryAgentPackages(ctx, types.PackageFilters{})
	if err != nil {
		RespondInternalError(c, "failed to list packages")
		return
	}
	registry := map[string]packages.InstalledPackage{}
	if h.updates != nil {
		if snapshot, snapshotErr := h.updates.RegistryEntries(); snapshotErr == nil {
			registry = snapshot
		}
	}

	var packageInfos []PackageInfo
	for _, pkg := range dbPackages {
		if search != "" && !h.matchesSearch(pkg, search) {
			continue
		}
		packageStatus := ""
		if status != "" {
			packageStatus = h.determinePackageStatus(ctx, pkg)
		}
		if status != "" && packageStatus != status {
			continue
		}
		packageInfo := h.buildPackageInfoWithStatus(ctx, pkg, packageStatus)
		if entry, ok := registry[pkg.ID]; ok {
			applyRegistryEntry(&packageInfo, pkg.ID, entry, h.updates.Checker().Cached(pkg.ID))
		}

		packageInfos = append(packageInfos, packageInfo)
	}

	response := PackageListResponse{
		Packages: packageInfos,
		Total:    len(packageInfos),
	}

	c.JSON(http.StatusOK, response)
}

func (h *PackageHandler) buildPackageInfo(ctx context.Context, pkg *types.AgentPackage) PackageInfo {
	return h.buildPackageInfoWithStatus(ctx, pkg, "")
}

func (h *PackageHandler) buildPackageInfoWithStatus(ctx context.Context, pkg *types.AgentPackage, packageStatus string) PackageInfo {
	configRequired := len(pkg.ConfigurationSchema) > 0
	configComplete := !configRequired
	if configRequired {
		config, err := h.storage.GetAgentConfiguration(ctx, pkg.ID, pkg.ID)
		configComplete = err == nil && config.Status == types.ConfigurationStatusActive
	}
	if packageStatus == "" {
		packageStatus = h.determinePackageStatus(ctx, pkg)
	}
	info := PackageInfo{
		ID: pkg.ID, Name: pkg.Name, Version: pkg.Version,
		Status: packageStatus, InstallStatus: string(pkg.Status),
		InstallPath: pkg.InstallPath, Source: h.safeStringValue(pkg.Repository),
		ConfigurationRequired: configRequired, ConfigurationComplete: configComplete,
		Description: h.safeStringValue(pkg.Description), Author: h.safeStringValue(pkg.Author),
		AutoUpdate: true, Update: PackageUpdate{Status: updatecheck.StatusUnknown},
	}
	if !pkg.InstalledAt.IsZero() {
		info.InstalledAt = pkg.InstalledAt.UTC().Format(time.RFC3339)
	}
	return info
}

// GetPackageDetailsHandler handles requests for getting detailed package information
// GET /api/ui/v1/agents/packages/:packageId/details
func (h *PackageHandler) GetPackageDetailsHandler(c *gin.Context) {
	packageID := c.Param("packageId")
	if packageID == "" {
		RespondBadRequest(c, "packageId is required")
		return
	}

	// Get the agent package
	ctx := c.Request.Context()
	pkg, err := h.storage.GetAgentPackage(ctx, packageID)
	if err != nil {
		RespondNotFound(c, "package not found")
		return
	}

	// Determine package status
	packageStatus := h.determinePackageStatus(ctx, pkg)

	// Parse configuration schema
	var schema map[string]interface{}
	if len(pkg.ConfigurationSchema) > 0 {
		if err := json.Unmarshal(pkg.ConfigurationSchema, &schema); err != nil {
			RespondInternalError(c, "failed to parse configuration schema")
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
		Source:      h.safeStringValue(pkg.Repository),
		Status:      packageStatus,
		AutoUpdate:  true,
		Update:      PackageUpdate{Status: updatecheck.StatusUnknown},
		Configuration: PackageConfiguration{
			Required: configRequired,
			Complete: configComplete,
			Schema:   schema,
			Current:  currentConfig,
		},
	}
	if h.updates != nil {
		if entry, ok, err := h.updates.RegistryEntry(packageID); err == nil && ok {
			if entry.SourcePath != "" {
				response.Source = entry.SourcePath
			}
			response.InstalledCommit = entry.Commit
			response.SourceRef = entry.Ref
			response.AutoUpdate = entry.AutoUpdateEnabled()
			response.Update = packageUpdate(h.updates.Checker().Cached(packageID))
		}
	}

	// TODO: Add capabilities parsing when agent introspection is implemented
	// This would parse reasoners and skills from the package

	// TODO: Add runtime information when agent lifecycle management is implemented
	// This would include process information, logs, etc.

	c.JSON(http.StatusOK, response)
}

func applyRegistryEntry(info *PackageInfo, id string, entry packages.InstalledPackage, update updatecheck.Update) {
	if info == nil {
		return
	}
	overlay := packageInfoFromRegistry(id, entry, update)
	if overlay.InstallStatus != "" {
		info.InstallStatus = overlay.InstallStatus
	}
	if overlay.InstalledAt != "" {
		info.InstalledAt = overlay.InstalledAt
	}
	if overlay.InstallPath != "" {
		info.InstallPath = overlay.InstallPath
	}
	if overlay.Source != "" {
		info.Source = overlay.Source
	}
	if overlay.LastStarted != "" {
		info.LastStarted = overlay.LastStarted
	}
	if entry.Status == "running" && entry.Runtime.PID != nil {
		info.ProcessID = overlay.ProcessID
	}
	if entry.Status == "running" && entry.Runtime.Port != nil {
		info.Port = overlay.Port
	}
	info.InstalledCommit = overlay.InstalledCommit
	info.SourceRef = overlay.SourceRef
	info.AutoUpdate = overlay.AutoUpdate
	info.Update = overlay.Update
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
