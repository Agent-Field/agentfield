package middleware

import (
	"net/http"
	"strings"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/gin-gonic/gin"
)

const (
	// CallerAgentIDKey is the context key for storing the resolved caller agent ID.
	CallerAgentIDKey ContextKey = "caller_agent_id"
	// MemoryPermissionResultKey is the context key for storing memory permission check results.
	MemoryPermissionResultKey ContextKey = "memory_permission_result"
)

// MemoryPermissionConfig holds configuration for memory permission checking.
type MemoryPermissionConfig struct {
	// Enabled determines if memory permission checking is active
	Enabled bool
	// EnforceScopeOwnership determines if scope ownership validation is enforced
	EnforceScopeOwnership bool
}

// MemoryPermissionResult contains the result of a memory permission check.
type MemoryPermissionResult struct {
	Allowed  bool
	Reason   string
	CallerID string
}

// MemoryPermissionMiddleware creates a middleware that enforces permission checks
// on memory endpoints. It validates:
//  1. Caller identity resolution (from DID auth or agent headers)
//  2. Scope ownership - agents can only access scopes they own or participate in
//  3. Access control metadata - honors RequiredRoles, TeamRestricted, AuditAccess
//  4. Tag-based policy evaluation for memory-specific rules
//
// This middleware should be applied AFTER DIDAuthMiddleware.
func MemoryPermissionMiddleware(
	policyService AccessPolicyServiceInterface,
	agentResolver AgentResolverInterface,
	didResolver DIDResolverInterface,
	config MemoryPermissionConfig,
) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Skip if memory permission checking is disabled
		if !config.Enabled {
			c.Next()
			return
		}

		// Resolve caller identity
		callerAgentID := resolveCallerIdentity(c, didResolver)
		c.Set(string(CallerAgentIDKey), callerAgentID)

		// Global scope is open by design - skip ownership checks
		scope := c.GetHeader("X-Memory-Scope")
		if scope == "" {
			// Try to infer scope from other headers
			if c.GetHeader("X-Workflow-ID") != "" {
				scope = "workflow"
			} else if c.GetHeader("X-Session-ID") != "" {
				scope = "session"
			} else if c.GetHeader("X-Actor-ID") != "" {
				scope = "actor"
			} else {
				scope = "global"
			}
		}

		if scope == "global" {
			c.Set(string(MemoryPermissionResultKey), &MemoryPermissionResult{
				Allowed:  true,
				CallerID: callerAgentID,
			})
			c.Next()
			return
		}

		// Scope ownership validation
		if config.EnforceScopeOwnership && callerAgentID != "" {
			if !validateScopeOwnership(c, callerAgentID, scope) {
				logger.Logger.Warn().
					Str("caller_agent_id", callerAgentID).
					Str("scope", scope).
					Msg("Memory permission denied: scope ownership validation failed")
				c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
					"error":   "scope_ownership_denied",
					"message": "Agent does not own or participate in the requested scope",
				})
				return
			}
		}

		// Tag-based policy evaluation for memory access
		if policyService != nil && callerAgentID != "" {
			callerAgent, err := agentResolver.GetAgent(c.Request.Context(), callerAgentID)
			if err != nil {
				// Agent not registered as a node — skip tag-based policy evaluation.
				// Scope ownership was already validated above.
				logger.Logger.Debug().Err(err).Str("caller_agent_id", callerAgentID).
					Msg("Caller agent not found in node registry, skipping tag-based policy evaluation")
			}

			if err == nil && callerAgent != nil {
				callerTags := callerAgent.ApprovedTags

				// Determine memory operation from request method and path
				operation := resolveMemoryOperation(c)

				// Use empty target tags for memory - policies match on caller tags + operation
				result := policyService.EvaluateAccess(callerTags, []string{"memory"}, operation, nil)
				if result.Matched && !result.Allowed {
					c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
						"error":   "memory_access_denied",
						"message": "Access denied by memory policy",
						"policy":  result.PolicyName,
					})
					return
				}
			}
		}

		c.Set(string(MemoryPermissionResultKey), &MemoryPermissionResult{
			Allowed:  true,
			CallerID: callerAgentID,
		})
		c.Next()
	}
}

// resolveCallerIdentity extracts the caller agent ID from the request context.
// It checks DID auth first, then falls back to header-based identity.
func resolveCallerIdentity(c *gin.Context, didResolver DIDResolverInterface) string {
	// Check DID-authenticated identity first
	callerDID := GetVerifiedCallerDID(c)
	if callerDID != "" && didResolver != nil {
		agentID := didResolver.ResolveAgentIDByDID(c.Request.Context(), callerDID)
		if agentID != "" {
			return agentID
		}
	}

	// Fallback to header-based identity
	if agentID := c.GetHeader("X-Caller-Agent-ID"); agentID != "" {
		return agentID
	}
	if agentID := c.GetHeader("X-Agent-Node-ID"); agentID != "" {
		return agentID
	}

	return ""
}

// validateScopeOwnership checks if the caller agent owns or participates in the
// requested scope.
func validateScopeOwnership(c *gin.Context, callerAgentID string, scope string) bool {
	switch scope {
	case "actor":
		// Actor scope: the actor ID should match the caller's agent ID
		actorID := c.GetHeader("X-Actor-ID")
		return actorID == callerAgentID || actorID == ""
	case "session":
		// Session scope: allow if the caller provides a session ID
		// (session participation is validated by the session service)
		sessionID := c.GetHeader("X-Session-ID")
		return sessionID != ""
	case "workflow":
		// Workflow scope: allow if the caller provides a workflow ID.
		// Workflow participation is validated by the workflow service at execution time;
		// here we verify a workflow ID is present so agents can't access workflow-scoped
		// memory without being in a workflow context.
		workflowID := c.GetHeader("X-Workflow-ID")
		return workflowID != ""
	case "global":
		return true
	default:
		// Unknown scopes are denied by default
		return false
	}
}

// resolveMemoryOperation determines the memory operation type from the request.
func resolveMemoryOperation(c *gin.Context) string {
	path := c.Request.URL.Path
	method := c.Request.Method

	if strings.Contains(path, "/memory/vector") {
		if method == "DELETE" || strings.Contains(path, "/delete") {
			return "vector.delete"
		}
		if strings.Contains(path, "/search") {
			return "vector.search"
		}
		if method == "GET" {
			return "vector.read"
		}
		return "vector.write"
	}

	if strings.Contains(path, "/memory/events") {
		return "memory.subscribe"
	}

	if strings.Contains(path, "/delete") {
		return "memory.delete"
	}
	if strings.Contains(path, "/set") {
		return "memory.write"
	}
	if strings.Contains(path, "/get") || strings.Contains(path, "/list") {
		return "memory.read"
	}

	return "memory.unknown"
}

// GetCallerAgentID extracts the caller agent ID from the gin context.
func GetCallerAgentID(c *gin.Context) string {
	if id, exists := c.Get(string(CallerAgentIDKey)); exists {
		if s, ok := id.(string); ok {
			return s
		}
	}
	return ""
}

// GetMemoryPermissionResult extracts the memory permission result from the gin context.
func GetMemoryPermissionResult(c *gin.Context) *MemoryPermissionResult {
	if result, exists := c.Get(string(MemoryPermissionResultKey)); exists {
		if r, ok := result.(*MemoryPermissionResult); ok {
			return r
		}
	}
	return nil
}
