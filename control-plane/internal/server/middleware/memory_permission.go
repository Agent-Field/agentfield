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
	// ExecutionReader supplies participation checks for wildcard memory events.
	ExecutionReader MemoryExecutionReader
}

// MemoryPermissionResult contains the result of a memory permission check.
type MemoryPermissionResult struct {
	Allowed  bool
	Reason   string
	CallerID string
}

// MemoryPermissionMiddleware resolves identity and evaluates memory tag policies.
// Handlers authorize the actual resolved scope before accessing storage.
// This middleware should be applied AFTER DIDAuthMiddleware.
func MemoryPermissionMiddleware(
	policyService AccessPolicyServiceInterface,
	agentResolver AgentResolverInterface,
	didResolver DIDResolverInterface,
	config MemoryPermissionConfig,
) gin.HandlerFunc {
	registryReader, _ := agentResolver.(MemoryAgentLister)
	registry := newMemoryAgentSnapshot(registryReader)
	return func(c *gin.Context) {
		if !config.Enabled {
			c.Next()
			return
		}

		callerAgentID := resolveCallerIdentity(c, didResolver)
		// Overwrite any identity inferred by APIKeyAuth, including anonymous callers.
		c.Set(string(CallerAgentIDKey), callerAgentID)

		if policyService != nil {
			callerTags := []string{}
			if callerAgentID != "" && agentResolver != nil {
				agent, err := agentResolver.GetAgent(c.Request.Context(), callerAgentID)
				if err != nil && !isMemoryAgentNotFound(err, callerAgentID) {
					c.AbortWithStatusJSON(http.StatusServiceUnavailable, gin.H{
						"error":   "memory_authorization_unavailable",
						"message": "Unable to resolve caller tags for memory authorization",
					})
					return
				}
				if err == nil && agent != nil {
					callerTags = agent.ApprovedTags
				}
			}
			result := policyService.EvaluateAccess(callerTags, []string{"memory"}, resolveMemoryOperation(c), nil)
			if result.Matched && !result.Allowed {
				c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
					"error":   "memory_access_denied",
					"message": "Access denied by memory policy",
					"policy":  result.PolicyName,
				})
				return
			}
		}

		c.Set(memoryAuthorizerKey, &memoryAuthorizer{
			callerID: callerAgentID, registry: registry, config: config,
		})
		c.Set(string(MemoryPermissionResultKey), &MemoryPermissionResult{
			Allowed: true, CallerID: callerAgentID,
		})
		c.Next()
	}
}

// resolveCallerIdentity never falls back to asserted headers when a verified DID
// is present, even if that DID cannot be resolved to an agent.
func resolveCallerIdentity(c *gin.Context, didResolver DIDResolverInterface) string {
	if callerDID := GetVerifiedCallerDID(c); callerDID != "" {
		agentID := ""
		if didResolver != nil {
			agentID = didResolver.ResolveAgentIDByDID(c.Request.Context(), callerDID)
		}
		for _, header := range []string{"X-Caller-Agent-ID", "X-Agent-Node-ID"} {
			if asserted := strings.TrimSpace(c.GetHeader(header)); asserted != "" && asserted != agentID {
				logger.Logger.Debug().Str("header", header).Str("caller_agent_id", agentID).
					Msg("Ignoring memory identity header that disagrees with verified DID")
			}
		}
		return agentID
	}
	if agentID := strings.TrimSpace(c.GetHeader("X-Caller-Agent-ID")); agentID != "" {
		return agentID
	}
	return strings.TrimSpace(c.GetHeader("X-Agent-Node-ID"))
}

// resolveMemoryOperation determines the memory operation type from the request.
func resolveMemoryOperation(c *gin.Context) string {
	path := c.Request.URL.Path
	method := c.Request.Method

	if strings.Contains(path, "/memory/vector") {
		if method == "DELETE" || (method == "POST" && path == "/api/v1/memory/vector/delete") {
			return "vector.delete"
		}
		if method == "POST" && path == "/api/v1/memory/vector/search" {
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
