package ui

import (
	"context"
	"errors"
	"net/http"
	"os"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages/updatecheck"
	"github.com/Agent-Field/agentfield/control-plane/internal/services/packagemaint"
	"github.com/gin-gonic/gin"
)

type packageUpdateMaintenance interface {
	Entries() ([]updatecheck.Entry, error)
	Checker() *updatecheck.Checker
	SetAutoUpdate(id string, enabled bool) (packages.InstalledPackage, error)
	Status() packagemaint.Status
	StartPass() error
}

type PackageUpdateHandler struct {
	maintenance packageUpdateMaintenance
	packages    *PackageHandler
}

func NewPackageUpdateHandler(maintenance packageUpdateMaintenance, packageHandlers ...*PackageHandler) *PackageUpdateHandler {
	handler := &PackageUpdateHandler{maintenance: maintenance}
	if len(packageHandlers) > 0 {
		handler.packages = packageHandlers[0]
	}
	return handler
}

func (h *PackageUpdateHandler) CheckUpdatesHandler(c *gin.Context) {
	entries, err := h.maintenance.Entries()
	if err != nil {
		RespondInternalError(c, err.Error())
		return
	}
	budget := packageUpdateCheckBudget(len(entries))
	results := h.maintenance.Checker().CheckWithTimeout(context.Background(), entries, budget)
	checkedAt := time.Now().UTC()
	if len(results) > 0 {
		checkedAt = results[0].Update.CheckedAt
	}
	c.JSON(http.StatusOK, gin.H{"checked_at": checkedAt, "packages": results})
}

func packageUpdateCheckBudget(entryCount int) time.Duration {
	budget := time.Duration(entryCount) * 10 * time.Second
	if budget < 10*time.Second {
		budget = 10 * time.Second
	}
	if budget > 120*time.Second {
		budget = 120 * time.Second
	}
	return budget
}

type autoUpdateRequest struct {
	Enabled *bool `json:"enabled" binding:"required"`
}

func (h *PackageUpdateHandler) SetAutoUpdateHandler(c *gin.Context) {
	var request autoUpdateRequest
	if err := c.ShouldBindJSON(&request); err != nil || request.Enabled == nil {
		RespondBadRequest(c, "enabled is required")
		return
	}
	id := c.Param("packageId")
	entry, err := h.maintenance.SetAutoUpdate(id, *request.Enabled)
	if errors.Is(err, os.ErrNotExist) {
		RespondNotFound(c, "package not found")
		return
	}
	if err != nil {
		RespondInternalError(c, err.Error())
		return
	}
	info := packageInfoFromRegistry(id, entry, h.maintenance.Checker().Cached(id))
	if h.packages != nil {
		if pkg, getErr := h.packages.storage.GetAgentPackage(c.Request.Context(), id); getErr == nil {
			info = h.packages.buildPackageInfo(c.Request.Context(), pkg)
			applyRegistryEntry(&info, id, entry, h.maintenance.Checker().Cached(id))
		}
	}
	c.JSON(http.StatusOK, info)
}

func (h *PackageUpdateHandler) MaintenanceStatusHandler(c *gin.Context) {
	c.JSON(http.StatusOK, h.maintenance.Status())
}

func (h *PackageUpdateHandler) RunMaintenanceHandler(c *gin.Context) {
	if err := h.maintenance.StartPass(); errors.Is(err, packagemaint.ErrShuttingDown) {
		RespondError(c, http.StatusServiceUnavailable, "package maintenance is shutting down")
		return
	} else if err != nil {
		RespondError(c, http.StatusConflict, err.Error())
		return
	}
	c.JSON(http.StatusAccepted, gin.H{"started": true})
}

func packageInfoFromRegistry(id string, entry packages.InstalledPackage, update updatecheck.Update) PackageInfo {
	name := entry.Name
	if name == "" {
		name = id
	}
	info := PackageInfo{
		ID: id, Name: name, Version: entry.Version, InstallStatus: entry.Status,
		InstalledAt: entry.InstalledAt, InstallPath: entry.Path, Source: entry.SourcePath,
		Description: entry.Description, InstalledCommit: entry.Commit, SourceRef: entry.Ref,
		AutoUpdate: entry.AutoUpdateEnabled(), Update: packageUpdate(update),
	}
	if parsed, err := time.Parse(time.RFC3339, info.InstalledAt); err == nil {
		info.InstalledAt = parsed.UTC().Format(time.RFC3339)
	}
	if entry.Status == "running" && entry.Runtime.Port != nil {
		info.Port = *entry.Runtime.Port
	}
	if entry.Status == "running" && entry.Runtime.PID != nil {
		info.ProcessID = *entry.Runtime.PID
	}
	if entry.Runtime.StartedAt != nil {
		info.LastStarted = *entry.Runtime.StartedAt
	}
	return info
}

func packageUpdate(update updatecheck.Update) PackageUpdate {
	checkedAt := ""
	if !update.CheckedAt.IsZero() {
		checkedAt = update.CheckedAt.UTC().Format(time.RFC3339)
	}
	return PackageUpdate{Status: update.Status, LatestCommit: update.LatestCommit, CheckedAt: checkedAt, Message: update.Message}
}
