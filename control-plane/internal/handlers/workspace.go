package handlers

import (
	"io"
	"net/http"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/workspace"

	"github.com/gin-gonic/gin"
)

// The control plane exposes the same workspace transport endpoints the spec
// defines for nodes, backed by its own content store. This lets the `af` CLI
// seal a folder and push its blobs over HTTP so it works identically whether the
// control plane is local or remote — CLI-vs-control-plane location stays a pure
// configuration detail (the control plane URL).

// maxWorkspaceBlobBytes caps a single uploaded blob to guard against unbounded
// memory use. Whole-file blobs above this size are out of scope for the POC.
const maxWorkspaceBlobBytes = 256 << 20 // 256 MiB

type workspacePrepareRequest struct {
	Manifest *workspace.Manifest `json:"manifest"`
}

type workspacePrepareResponse struct {
	Missing []string `json:"missing"`
}

// workspaceStagedResponse is the diff-fetch view returned to the CLI. It
// intentionally omits internal node routing details.
type workspaceStagedResponse struct {
	RunID       string          `json:"run_id"`
	ExecutionID string          `json:"execution_id,omitempty"`
	ManifestID  string          `json:"manifest_id,omitempty"`
	Diff        *workspace.Diff `json:"workspace_diff"`
}

func controlPlaneCAS() *workspace.CAS {
	return workspace.NewCAS(workspace.DefaultCASDir())
}

// WorkspacePrepareHandler reports which of a manifest's blobs the content store
// is missing so the caller can upload just those.
func WorkspacePrepareHandler() gin.HandlerFunc {
	return func(ctx *gin.Context) {
		var req workspacePrepareRequest
		if err := ctx.ShouldBindJSON(&req); err != nil {
			ctx.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body: " + err.Error()})
			return
		}
		if req.Manifest == nil {
			ctx.JSON(http.StatusBadRequest, gin.H{"error": "manifest is required"})
			return
		}
		cas := controlPlaneCAS()
		seen := make(map[string]struct{})
		missing := make([]string, 0)
		for _, f := range req.Manifest.Files {
			if f.SHA256 == "" {
				continue
			}
			if _, dup := seen[f.SHA256]; dup {
				continue
			}
			seen[f.SHA256] = struct{}{}
			if !cas.Has(f.SHA256) {
				missing = append(missing, f.SHA256)
			}
		}
		ctx.JSON(http.StatusOK, workspacePrepareResponse{Missing: missing})
	}
}

// WorkspaceBlobPutHandler stores an uploaded blob keyed by its sha256. The body
// is raw bytes; the content is verified against the path sha before storage.
func WorkspaceBlobPutHandler() gin.HandlerFunc {
	return func(ctx *gin.Context) {
		sha := ctx.Param("sha")
		if sha == "" {
			ctx.JSON(http.StatusBadRequest, gin.H{"error": "sha256 is required"})
			return
		}
		data, err := io.ReadAll(io.LimitReader(ctx.Request.Body, maxWorkspaceBlobBytes+1))
		if err != nil {
			ctx.JSON(http.StatusBadRequest, gin.H{"error": "read blob body: " + err.Error()})
			return
		}
		if len(data) > maxWorkspaceBlobBytes {
			ctx.JSON(http.StatusRequestEntityTooLarge, gin.H{"error": "blob exceeds maximum size"})
			return
		}
		if err := controlPlaneCAS().PutVerified(sha, data); err != nil {
			ctx.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		ctx.Status(http.StatusNoContent)
	}
}

// WorkspaceBlobGetHandler streams a stored blob, or 404 when absent.
func WorkspaceBlobGetHandler() gin.HandlerFunc {
	return func(ctx *gin.Context) {
		sha := ctx.Param("sha")
		if sha == "" {
			ctx.JSON(http.StatusBadRequest, gin.H{"error": "sha256 is required"})
			return
		}
		cas := controlPlaneCAS()
		if !cas.Has(sha) {
			ctx.JSON(http.StatusNotFound, gin.H{"error": "blob not found"})
			return
		}
		data, err := cas.Get(sha)
		if err != nil {
			ctx.JSON(http.StatusInternalServerError, gin.H{"error": "read blob: " + err.Error()})
			return
		}
		ctx.Data(http.StatusOK, "application/octet-stream", data)
	}
}

// WorkspaceStagedHandler returns the staged workspace diff for a run so the CLI
// can render `af diff` and drive `af apply` against a remote control plane.
func WorkspaceStagedHandler() gin.HandlerFunc {
	return func(ctx *gin.Context) {
		runID := ctx.Param("run_id")
		if runID == "" {
			ctx.JSON(http.StatusBadRequest, gin.H{"error": "run_id is required"})
			return
		}
		rec, err := workspace.LoadStaged(runID)
		if err != nil {
			logger.Logger.Error().Err(err).Str("run_id", runID).Msg("failed to load staged workspace record")
			ctx.JSON(http.StatusInternalServerError, gin.H{"error": "load staged record: " + err.Error()})
			return
		}
		if rec == nil || rec.Diff == nil {
			ctx.JSON(http.StatusNotFound, gin.H{"error": "no workspace diff staged for run " + runID})
			return
		}
		ctx.JSON(http.StatusOK, workspaceStagedResponse{
			RunID:       rec.RunID,
			ExecutionID: rec.ExecutionID,
			ManifestID:  rec.ManifestID,
			Diff:        rec.Diff,
		})
	}
}
