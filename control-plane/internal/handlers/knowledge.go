package handlers

import (
	"net/http"

	"github.com/Agent-Field/agentfield/control-plane/internal/knowledge"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/gin-gonic/gin"
)

// knowledgeScopeBody is the JSON scope shape shared by knowledge requests.
type knowledgeScopeBody struct {
	Tier        string `json:"tier" binding:"required"`
	WorkspaceID string `json:"workspace_id" binding:"required"`
	ProjectID   string `json:"project_id,omitempty"`
	SenderID    string `json:"sender_id,omitempty"`
}

func (b knowledgeScopeBody) toScope() knowledge.Scope {
	return knowledge.Scope{
		Tier:        knowledge.Tier(b.Tier),
		WorkspaceID: b.WorkspaceID,
		ProjectID:   b.ProjectID,
		SenderID:    b.SenderID,
	}
}

// KnowledgeUpsertRequest is the body for POST /knowledge/upsert.
type KnowledgeUpsertRequest struct {
	Scope    knowledgeScopeBody `json:"scope" binding:"required"`
	SourceID string             `json:"source_id" binding:"required"`
	Chunks   []struct {
		Text     string                 `json:"text"`
		Page     *int                   `json:"page,omitempty"`
		Ordinal  *int                   `json:"ordinal,omitempty"`
		Metadata map[string]interface{} `json:"metadata,omitempty"`
	} `json:"chunks" binding:"required"`
}

// KnowledgeSearchRequest is the body for POST /knowledge/search.
type KnowledgeSearchRequest struct {
	Scope knowledgeScopeBody `json:"scope" binding:"required"`
	Query string             `json:"query" binding:"required"`
	TopK  int                `json:"top_k"`
}

// KnowledgeDeleteRequest is the optional body for DELETE /knowledge/source/:id,
// carrying the scope (the source ID comes from the path).
type KnowledgeDeleteRequest struct {
	Scope knowledgeScopeBody `json:"scope" binding:"required"`
}

// KnowledgeUpsertHandler embeds and stores a source's chunks under a scope.
func KnowledgeUpsertHandler(svc *knowledge.Service) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req KnowledgeUpsertRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, ErrorResponse{Error: "invalid_request", Details: err.Error(), Code: http.StatusBadRequest})
			return
		}

		chunks := make([]knowledge.Chunk, len(req.Chunks))
		for i, ch := range req.Chunks {
			chunks[i] = knowledge.Chunk{Text: ch.Text, Page: ch.Page, Ordinal: ch.Ordinal, Metadata: ch.Metadata}
		}

		count, err := svc.Upsert(c.Request.Context(), req.Scope.toScope(), req.SourceID, chunks)
		if err != nil {
			writeKnowledgeError(c, err)
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"source_id": req.SourceID,
			"upserted":  count,
			"status":    "ok",
		})
	}
}

// KnowledgeSearchHandler runs a scoped semantic search.
func KnowledgeSearchHandler(svc *knowledge.Service) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req KnowledgeSearchRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, ErrorResponse{Error: "invalid_request", Details: err.Error(), Code: http.StatusBadRequest})
			return
		}

		hits, err := svc.Search(c.Request.Context(), req.Scope.toScope(), req.Query, req.TopK)
		if err != nil {
			writeKnowledgeError(c, err)
			return
		}

		c.JSON(http.StatusOK, gin.H{"results": hits})
	}
}

// KnowledgeDeleteSourceHandler removes a source's chunks within a scope.
func KnowledgeDeleteSourceHandler(svc *knowledge.Service) gin.HandlerFunc {
	return func(c *gin.Context) {
		sourceID := c.Param("id")
		if sourceID == "" {
			c.JSON(http.StatusBadRequest, ErrorResponse{Error: "invalid_request", Details: "source id is required", Code: http.StatusBadRequest})
			return
		}

		var req KnowledgeDeleteRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, ErrorResponse{Error: "invalid_request", Details: err.Error(), Code: http.StatusBadRequest})
			return
		}

		deleted, err := svc.DeleteSource(c.Request.Context(), req.Scope.toScope(), sourceID)
		if err != nil {
			writeKnowledgeError(c, err)
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"source_id": sourceID,
			"deleted":   deleted,
			"status":    "deleted",
		})
	}
}

// writeKnowledgeError maps service errors to HTTP responses. Validation/scope
// errors are 400; everything else is 500.
func writeKnowledgeError(c *gin.Context, err error) {
	if isKnowledgeValidationError(err) {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "invalid_request", Details: err.Error(), Code: http.StatusBadRequest})
		return
	}
	logger.Logger.Error().Err(err).Msg("knowledge operation failed")
	c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "knowledge_error", Details: err.Error(), Code: http.StatusInternalServerError})
}

// isKnowledgeValidationError reports whether err is a caller-input error (vs an
// internal/storage failure). Scope/argument validation messages are surfaced as
// 400s; embed/store failures fall through to 500.
func isKnowledgeValidationError(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	for _, frag := range []string{
		"workspace_id is required",
		"project_id is required",
		"sender_id is required",
		"invalid tier",
		"source_id is required",
		"query is required",
		"at least one chunk is required",
		"has empty text",
	} {
		if containsSubstr(msg, frag) {
			return true
		}
	}
	return false
}

func containsSubstr(s, sub string) bool {
	if len(sub) == 0 {
		return true
	}
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
