package handlers

import (
	"errors"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"

	afcli "github.com/Agent-Field/agentfield/control-plane/internal/cli"
)

// HandleVerifyAuditBundle reads, validates, and verifies an exported provenance JSON body.
// Shared by both the public API and UI handlers.
func HandleVerifyAuditBundle(c *gin.Context) {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, afcli.MaxVerifyAuditBodyBytes)
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		var maxBytesErr *http.MaxBytesError
		if errors.As(err, &maxBytesErr) {
			c.JSON(http.StatusRequestEntityTooLarge, gin.H{"error": "request body too large", "max_bytes": afcli.MaxVerifyAuditBodyBytes})
		} else {
			c.JSON(http.StatusBadRequest, gin.H{"error": "failed to read request body"})
		}
		return
	}
	if len(body) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "empty body"})
		return
	}
	if c.Query("resolve_web") == "true" || c.Query("did_resolver") != "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "remote DID resolution is only supported in the CLI; upload a bundle with embedded DID data for HTTP verification",
		})
		return
	}
	opts := afcli.VerifyOptions{
		OutputFormat: "json",
		Verbose:      c.Query("verbose") == "true",
	}
	result := afcli.VerifyProvenanceJSON(body, opts)
	if !result.FormatValid {
		c.JSON(http.StatusUnprocessableEntity, result)
		return
	}
	c.JSON(http.StatusOK, result)
}
