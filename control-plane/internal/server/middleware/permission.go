package middleware

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
)

// AgentResolverInterface provides methods for resolving agent information.
type AgentResolverInterface interface {
	GetAgent(ctx context.Context, agentID string) (*types.AgentNode, error)
}

// DIDResolverInterface provides methods for resolving agent DIDs.
type DIDResolverInterface interface {
	GenerateDIDWeb(agentID string) string
	// ResolveAgentIDByDID looks up the agent ID associated with a DID.
	// Returns empty string if the DID cannot be resolved.
	ResolveAgentIDByDID(ctx context.Context, did string) string
}

// AccessPolicyServiceInterface defines the methods required for tag-based policy evaluation.
type AccessPolicyServiceInterface interface {
	EvaluateAccess(callerTags, targetTags []string, functionName string, inputParams map[string]any) *types.PolicyEvaluationResult
}

// TagVCVerifierInterface defines the methods required for verifying Agent Tag VCs.
type TagVCVerifierInterface interface {
	VerifyAgentTagVC(ctx context.Context, agentID string) (*types.AgentTagVCDocument, error)
}

// PermissionConfig holds configuration for permission checking.
type PermissionConfig struct {
	// Enabled determines if permission checking is active
	Enabled bool
	// DefaultDeny returns 403 instead of allowing fall-through when no
	// access policy matches. Default false preserves backward compat.
	DefaultDeny bool
}

// PermissionCheckResult contains the result of a permission check.
type PermissionCheckResult struct {
	Allowed            bool
	RequiresPermission bool
	Error              error
}

const (
	// PermissionCheckResultKey is the context key for storing permission check results.
	PermissionCheckResultKey ContextKey = "permission_check_result"
	// TargetAgentKey is the context key for storing the resolved target agent.
	TargetAgentKey ContextKey = "target_agent"
	// TargetDIDKey is the context key for storing the target agent's DID.
	TargetDIDKey ContextKey = "target_did"
)

// PermissionCheckMiddleware creates a middleware that checks permissions before allowing
// requests to protected agents.
//
// This middleware should be applied AFTER DIDAuthMiddleware so that the verified
// caller DID is available in the context.
//
// The middleware:
//  1. Extracts the verified caller DID from context (set by DIDAuthMiddleware)
//  2. Resolves the target agent from the request path
//  3. Evaluates access policies based on caller/target tags
//  4. If a policy denies access, returns 403 Forbidden
//  5. If no policy matches and DefaultDeny is false, allows the request
//     (backward compat for untagged agents). When DefaultDeny is true,
//     returns HTTP 403 with an opaque error. The unmatched
//     (caller_tags, target_tags, function) tuple is logged at DEBUG in
//     both modes; it is not included in the response body to avoid
//     leaking tag taxonomy to denied callers.
func PermissionCheckMiddleware(
	policyService AccessPolicyServiceInterface,
	tagVCVerifier TagVCVerifierInterface,
	agentResolver AgentResolverInterface,
	didResolver DIDResolverInterface,
	config PermissionConfig,
) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Skip if permission checking is disabled
		if !config.Enabled {
			c.Next()
			return
		}

		// Extract target from path parameter
		target := c.Param("target")
		if target == "" {
			// No target specified - let the handler deal with it
			c.Next()
			return
		}

		// Parse target (format: "agent_id.reasoner_name")
		agentID, _, err := parseTargetParam(target)
		if err != nil {
			c.Next()
			return
		}

		// Resolve the target agent
		ctx := c.Request.Context()
		agent, err := agentResolver.GetAgent(ctx, agentID)
		if err != nil {
			// Fail closed if target resolution fails to avoid bypass on transient backend errors.
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
				"error":           "target_resolution_failed",
				"message":         "Unable to resolve target agent for permission enforcement",
				"target_agent_id": agentID,
			})
			return
		}
		if agent == nil {
			// Agent not found - let the handler deal with the error
			c.Next()
			return
		}

		// Store the resolved agent in context for downstream use
		c.Set(string(TargetAgentKey), agent)

		// Block calls to agents in pending_approval status
		if agent.LifecycleStatus == types.AgentStatusPendingApproval {
			c.AbortWithStatusJSON(http.StatusServiceUnavailable, gin.H{
				"error":           "agent_pending_approval",
				"message":         "Target agent is awaiting tag approval and cannot receive calls",
				"target_agent_id": agentID,
			})
			return
		}

		// Generate target DID
		targetDID := didResolver.GenerateDIDWeb(agentID)
		c.Set(string(TargetDIDKey), targetDID)

		// Get canonical plain tags for permission matching.
		tags := services.CanonicalAgentTags(agent)

		// Extract caller DID (needed for policy evaluation).
		callerDID := GetVerifiedCallerDID(c)

		// Parse function name from target param for policy evaluation.
		_, functionName, _ := parseTargetParam(target)

		// Resolve caller agent identity from the verified DID only. Caller identity
		// headers are client-controlled and must not influence authorization.
		var callerAgentID string
		if callerDID != "" {
			callerAgentID = didResolver.ResolveAgentIDByDID(ctx, callerDID)
			if callerAgentID == "" {
				logger.Logger.Warn().
					Str("caller_did", callerDID).
					Msg("Permission middleware: verified caller DID did not resolve to an agent ID, treating caller as anonymous")
			}
		} else if c.GetHeader("X-Caller-Agent-ID") != "" || c.GetHeader("X-Agent-Node-ID") != "" {
			logger.Logger.Warn().
				Bool("x_caller_agent_id_present", c.GetHeader("X-Caller-Agent-ID") != "").
				Bool("x_agent_node_id_present", c.GetHeader("X-Agent-Node-ID") != "").
				Msg("Permission middleware: caller identity headers ignored without a verified DID, treating caller as anonymous")
		}

		// --- Tag-based policy evaluation ---
		if policyService != nil {
			var callerTags []string

			if callerAgentID != "" {
				// Try to get VC-verified tags first (cryptographic proof of approved tags)
				vcChecked := false
				if tagVCVerifier != nil {
					tagVC, vcErr := tagVCVerifier.VerifyAgentTagVC(ctx, callerAgentID)
					if vcErr == nil && tagVC != nil {
						callerTags = tagVC.CredentialSubject.Permissions.Tags
						vcChecked = true
					} else if vcErr != nil {
						// VC exists but verification failed (revoked, expired, invalid signature).
						// Fail closed: use empty tags so policies requiring caller tags won't match.
						logger.Logger.Warn().Err(vcErr).Str("caller_agent_id", callerAgentID).Msg("Caller tag VC verification failed, using empty tags (fail-closed)")
						vcChecked = true
					}
				}
				// Fall back to registration tags only when no VC was found at all.
				// This covers auto-approved agents that haven't received a Tag VC yet.
				if !vcChecked && len(callerTags) == 0 {
					if callerAgent, agentErr := agentResolver.GetAgent(ctx, callerAgentID); agentErr == nil && callerAgent != nil {
						callerTags = services.CanonicalAgentTags(callerAgent)
					}
				}
			}

			// Read input params from request body (peek without consuming).
			// Always restore the body regardless of read success.
			var inputParams map[string]any
			if c.Request.Body != nil {
				body, readErr := io.ReadAll(c.Request.Body)
				if readErr == nil && len(body) > 0 {
					c.Request.Body = io.NopCloser(bytes.NewBuffer(body))
					json.Unmarshal(body, &inputParams) //nolint:errcheck
				} else if readErr == nil {
					c.Request.Body = io.NopCloser(bytes.NewBuffer(body))
				}
			}

			// Unwrap the "input" envelope so constraint evaluation sees flat params.
			// Execute requests send {"input": {"limit": 500, ...}} but constraints
			// reference parameter names directly (e.g. "limit").
			if nested, ok := inputParams["input"].(map[string]any); ok {
				inputParams = nested
			}

			logger.Logger.Debug().
				Str("target", target).
				Str("function", functionName).
				Str("caller_did", callerDID).
				Str("caller_agent_id", callerAgentID).
				Strs("caller_tags", callerTags).
				Strs("target_tags", tags).
				Msg("Permission middleware: evaluating policy")

			policyResult := policyService.EvaluateAccess(callerTags, tags, functionName, inputParams)
			if policyResult.Matched {
				result := &PermissionCheckResult{
					Allowed:            policyResult.Allowed,
					RequiresPermission: true,
				}
				c.Set(string(PermissionCheckResultKey), result)

				if !policyResult.Allowed {
					c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
						"error":   "access_denied",
						"message": "Access denied by policy",
					})
					return
				}

				// Policy allows — proceed
				c.Next()
				return
			}

			// No policy matched. Log the tuple at DEBUG so operators can
			// diagnose coverage gaps even when DefaultDeny is off.
			logger.Logger.Debug().
				Str("target", target).
				Str("function", functionName).
				Strs("caller_tags", callerTags).
				Strs("target_tags", tags).
				Msg("Permission middleware: no policy matched")

			if config.DefaultDeny {
				c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
					"error":   "no_policy_matched",
					"message": "no access policy matches this request; see server logs for the unmatched tuple",
				})
				return
			}
		}

		// No policy service configured, or no policy matched with DefaultDeny off.
		c.Set(string(PermissionCheckResultKey), &PermissionCheckResult{Allowed: true})
		c.Next()
	}
}

// GetPermissionCheckResult extracts the permission check result from the gin context.
func GetPermissionCheckResult(c *gin.Context) *PermissionCheckResult {
	if result, exists := c.Get(string(PermissionCheckResultKey)); exists {
		if r, ok := result.(*PermissionCheckResult); ok {
			return r
		}
	}
	return nil
}

// GetTargetAgent extracts the resolved target agent from the gin context.
func GetTargetAgent(c *gin.Context) *types.AgentNode {
	if agent, exists := c.Get(string(TargetAgentKey)); exists {
		if a, ok := agent.(*types.AgentNode); ok {
			return a
		}
	}
	return nil
}

// GetTargetDID extracts the target DID from the gin context.
func GetTargetDID(c *gin.Context) string {
	if did, exists := c.Get(string(TargetDIDKey)); exists {
		if d, ok := did.(string); ok {
			return d
		}
	}
	return ""
}

// parseTargetParam parses a target parameter in the format "agent_id.reasoner_name".
func parseTargetParam(target string) (agentID, reasonerName string, err error) {
	for i := 0; i < len(target); i++ {
		if target[i] == '.' {
			return target[:i], target[i+1:], nil
		}
	}
	return target, "", nil
}
