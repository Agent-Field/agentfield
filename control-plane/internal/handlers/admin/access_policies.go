package admin

import (
	"net/http"
	"strconv"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
)

// AccessPolicyHandlers handles admin access policy HTTP requests.
type AccessPolicyHandlers struct {
	policyService *services.AccessPolicyService
}

// NewAccessPolicyHandlers creates a new access policy admin handlers instance.
func NewAccessPolicyHandlers(policyService *services.AccessPolicyService) *AccessPolicyHandlers {
	return &AccessPolicyHandlers{
		policyService: policyService,
	}
}

// ListPolicies returns all access policies.
// GET /api/v1/admin/policies
func (h *AccessPolicyHandlers) ListPolicies(c *gin.Context) {
	policies, err := h.policyService.ListPolicies(c.Request.Context())
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to list access policies")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "list_failed",
			"message": "Failed to list access policies",
		})
		return
	}

	c.JSON(http.StatusOK, types.AccessPolicyListResponse{
		Policies: policies,
		Total:    len(policies),
	})
}

// CreatePolicy creates a new access policy.
// POST /api/v1/admin/policies
func (h *AccessPolicyHandlers) CreatePolicy(c *gin.Context) {
	var req types.AccessPolicyRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "invalid_request",
			"message": "Invalid JSON: " + err.Error(),
		})
		return
	}

	if req.Action != "allow" && req.Action != "deny" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "invalid_action",
			"message": "Action must be 'allow' or 'deny'",
		})
		return
	}

	policy, err := h.policyService.AddPolicy(c.Request.Context(), &req)
	if err != nil {
		logger.Logger.Error().Err(err).Str("name", req.Name).Msg("Failed to create access policy")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "create_failed",
			"message": "Failed to create access policy: " + err.Error(),
		})
		return
	}

	c.JSON(http.StatusCreated, policy)
}

// GetPolicy returns a single access policy by ID.
// GET /api/v1/admin/policies/:id
func (h *AccessPolicyHandlers) GetPolicy(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "invalid_id",
			"message": "Policy ID must be a number",
		})
		return
	}

	policy, err := h.policyService.GetPolicyByID(c.Request.Context(), id)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{
			"error":   "not_found",
			"message": "Access policy not found",
		})
		return
	}

	c.JSON(http.StatusOK, policy)
}

// UpdatePolicy updates an existing access policy.
// PUT /api/v1/admin/policies/:id
func (h *AccessPolicyHandlers) UpdatePolicy(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "invalid_id",
			"message": "Policy ID must be a number",
		})
		return
	}

	var req types.AccessPolicyRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "invalid_request",
			"message": "Invalid JSON: " + err.Error(),
		})
		return
	}

	if req.Action != "allow" && req.Action != "deny" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "invalid_action",
			"message": "Action must be 'allow' or 'deny'",
		})
		return
	}

	policy, err := h.policyService.UpdatePolicy(c.Request.Context(), id, &req)
	if err != nil {
		logger.Logger.Error().Err(err).Int64("id", id).Msg("Failed to update access policy")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "update_failed",
			"message": "Failed to update access policy: " + err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, policy)
}

// DeletePolicy deletes an access policy.
// DELETE /api/v1/admin/policies/:id
func (h *AccessPolicyHandlers) DeletePolicy(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "invalid_id",
			"message": "Policy ID must be a number",
		})
		return
	}

	if err := h.policyService.RemovePolicy(c.Request.Context(), id); err != nil {
		logger.Logger.Error().Err(err).Int64("id", id).Msg("Failed to delete access policy")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "delete_failed",
			"message": "Failed to delete access policy: " + err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"message": "Access policy deleted",
	})
}

// RegisterRoutes registers the access policy admin routes.
func (h *AccessPolicyHandlers) RegisterRoutes(router *gin.RouterGroup) {
	adminGroup := router.Group("/admin")
	{
		policiesGroup := adminGroup.Group("/policies")
		{
			policiesGroup.GET("", h.ListPolicies)
			policiesGroup.POST("", h.CreatePolicy)
			policiesGroup.GET("/:id", h.GetPolicy)
			policiesGroup.PUT("/:id", h.UpdatePolicy)
			policiesGroup.DELETE("/:id", h.DeletePolicy)
		}
	}
}
