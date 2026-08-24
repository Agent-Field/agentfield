package ui

import (
	"errors"
	"io"
	"net/http"

	"github.com/Agent-Field/agentfield/control-plane/internal/services/packagejobs"
	"github.com/gin-gonic/gin"
)

type PackageInstallHandler struct {
	manager packageJobManager
}

type packageJobManager interface {
	StartInstall(source string, force bool) (*packagejobs.Job, error)
	StartUpdate(packageName, source string) (*packagejobs.Job, error)
	Uninstall(packageName string) error
	GetJob(id string) (*packagejobs.Job, bool)
	ListJobs() []*packagejobs.Job
}

func NewPackageInstallHandler(manager packageJobManager) *PackageInstallHandler {
	return &PackageInstallHandler{manager: manager}
}

type installPackageRequest struct {
	Source string `json:"source" binding:"required"`
	Force  bool   `json:"force"`
}

type updatePackageRequest struct {
	Source string `json:"source"`
}

func (h *PackageInstallHandler) InstallPackageHandler(c *gin.Context) {
	var req installPackageRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		RespondBadRequest(c, "source is required")
		return
	}
	job, err := h.manager.StartInstall(req.Source, req.Force)
	if err != nil {
		h.respondOperationError(c, err)
		return
	}
	c.JSON(http.StatusAccepted, gin.H{"job_id": job.ID})
}

func (h *PackageInstallHandler) GetInstallJobHandler(c *gin.Context) {
	job, ok := h.manager.GetJob(c.Param("jobId"))
	if !ok {
		RespondNotFound(c, "install job not found")
		return
	}
	c.JSON(http.StatusOK, job)
}

func (h *PackageInstallHandler) ListInstallJobsHandler(c *gin.Context) {
	c.JSON(http.StatusOK, h.manager.ListJobs())
}

func (h *PackageInstallHandler) UninstallPackageHandler(c *gin.Context) {
	packageID := c.Param("packageId")
	if packageID == "" {
		RespondBadRequest(c, "packageId is required")
		return
	}
	if err := h.manager.Uninstall(packageID); err != nil {
		h.respondOperationError(c, err)
		return
	}
	c.JSON(http.StatusOK, gin.H{"package_id": packageID, "status": "uninstalled"})
}

func (h *PackageInstallHandler) UpdatePackageHandler(c *gin.Context) {
	packageID := c.Param("packageId")
	if packageID == "" {
		RespondBadRequest(c, "packageId is required")
		return
	}
	var req updatePackageRequest
	// Older clients send no body; EOF preserves their recorded-source update.
	if err := c.ShouldBindJSON(&req); err != nil && !errors.Is(err, io.EOF) {
		RespondBadRequest(c, "invalid request body")
		return
	}
	job, err := h.manager.StartUpdate(packageID, req.Source)
	if err != nil {
		h.respondOperationError(c, err)
		return
	}
	c.JSON(http.StatusAccepted, gin.H{"job_id": job.ID})
}

func (h *PackageInstallHandler) respondOperationError(c *gin.Context, err error) {
	switch {
	case errors.Is(err, packagejobs.ErrInvalidSource):
		RespondBadRequest(c, err.Error())
	case errors.Is(err, packagejobs.ErrBusy):
		RespondError(c, http.StatusConflict, err.Error())
	case errors.Is(err, packagejobs.ErrNotFound):
		RespondNotFound(c, err.Error())
	default:
		RespondInternalError(c, err.Error())
	}
}
