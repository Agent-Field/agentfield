package services

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// AccessPolicyStorage defines the storage interface subset for access policies.
type AccessPolicyStorage interface {
	GetAccessPolicies(ctx context.Context) ([]*types.AccessPolicy, error)
	GetAccessPolicyByID(ctx context.Context, id int64) (*types.AccessPolicy, error)
	CreateAccessPolicy(ctx context.Context, policy *types.AccessPolicy) error
	UpdateAccessPolicy(ctx context.Context, policy *types.AccessPolicy) error
	DeleteAccessPolicy(ctx context.Context, id int64) error
}

// AccessPolicyService handles tag-based access policy evaluation and management.
type AccessPolicyService struct {
	storage  AccessPolicyStorage
	mu       sync.RWMutex
	policies []*types.AccessPolicy // in-memory cache, sorted by priority desc
}

// NewAccessPolicyService creates a new access policy service instance.
func NewAccessPolicyService(storage AccessPolicyStorage) *AccessPolicyService {
	return &AccessPolicyService{
		storage:  storage,
		policies: make([]*types.AccessPolicy, 0),
	}
}

// Initialize loads access policies from storage into memory.
func (s *AccessPolicyService) Initialize(ctx context.Context) error {
	policies, err := s.storage.GetAccessPolicies(ctx)
	if err != nil {
		return fmt.Errorf("failed to load access policies: %w", err)
	}

	// Sort by priority descending (highest first), stable to ensure deterministic ordering
	sort.SliceStable(policies, func(i, j int) bool {
		if policies[i].Priority != policies[j].Priority {
			return policies[i].Priority > policies[j].Priority
		}
		return policies[i].ID < policies[j].ID // tie-break by ID for determinism
	})

	s.mu.Lock()
	s.policies = policies
	s.mu.Unlock()

	logger.Logger.Info().
		Int("policies_count", len(policies)).
		Msg("Loaded access policies")

	return nil
}

// EvaluateAccess evaluates access policies for a cross-agent call.
// Returns a PolicyEvaluationResult indicating whether access is allowed, denied, or no policy matched.
func (s *AccessPolicyService) EvaluateAccess(
	callerTags, targetTags []string,
	functionName string,
	inputParams map[string]any,
) *types.PolicyEvaluationResult {
	s.mu.RLock()
	policies := s.policies
	s.mu.RUnlock()

	// Normalize tags for comparison
	normalizedCallerTags := normalizeTags(callerTags)
	normalizedTargetTags := normalizeTags(targetTags)

	for _, policy := range policies {
		if !policy.Enabled {
			continue
		}

		// 1. Check caller tag intersection
		if !tagsIntersect(policy.CallerTags, normalizedCallerTags) {
			continue
		}

		// 2. Check target tag intersection
		if !tagsIntersect(policy.TargetTags, normalizedTargetTags) {
			continue
		}

		// 3. Check function against deny list first (deny takes precedence)
		if len(policy.DenyFunctions) > 0 {
			if functionName != "" && functionMatchesAny(functionName, policy.DenyFunctions) {
				return &types.PolicyEvaluationResult{
					Allowed:    false,
					Matched:    true,
					PolicyName: policy.Name,
					PolicyID:   policy.ID,
					Reason:     fmt.Sprintf("Function %q is denied by policy %q", functionName, policy.Name),
				}
			}
		}

		// 4. Check function against allow list — if allow list is set but function name
		// is empty or not in the list, this policy doesn't match (fail closed)
		if len(policy.AllowFunctions) > 0 {
			if functionName == "" || !functionMatchesAny(functionName, policy.AllowFunctions) {
				continue // Function not in allow list, try next policy
			}
		}

		// 5. Evaluate constraints — fail closed when constraints exist but inputParams is nil
		if len(policy.Constraints) > 0 {
			if inputParams == nil {
				return &types.PolicyEvaluationResult{
					Allowed:    false,
					Matched:    true,
					PolicyName: policy.Name,
					PolicyID:   policy.ID,
					Reason:     fmt.Sprintf("Policy %q requires parameter constraints but no input parameters provided", policy.Name),
				}
			}
			constraintViolation := evaluateConstraints(policy.Constraints, inputParams)
			if constraintViolation != "" {
				return &types.PolicyEvaluationResult{
					Allowed:    false,
					Matched:    true,
					PolicyName: policy.Name,
					PolicyID:   policy.ID,
					Reason:     constraintViolation,
				}
			}
		}

		// All checks passed — policy matches
		allowed := strings.ToLower(policy.Action) == "allow"
		reason := fmt.Sprintf("Policy %q matched: action=%s", policy.Name, policy.Action)
		if !allowed {
			reason = fmt.Sprintf("Policy %q explicitly denies access", policy.Name)
		}

		return &types.PolicyEvaluationResult{
			Allowed:    allowed,
			Matched:    true,
			PolicyName: policy.Name,
			PolicyID:   policy.ID,
			Reason:     reason,
		}
	}

	// No policy matched
	return &types.PolicyEvaluationResult{
		Matched: false,
		Reason:  "No access policy matched",
	}
}

// validConstraintOperators defines the allowed constraint operator set.
var validConstraintOperators = map[string]bool{
	"<=": true, ">=": true, "==": true, "!=": true, "<": true, ">": true,
}

// validatePolicyRequest validates policy fields before creation/update.
func validatePolicyRequest(req *types.AccessPolicyRequest) error {
	action := strings.ToLower(strings.TrimSpace(req.Action))
	if action != "allow" && action != "deny" {
		return fmt.Errorf("invalid policy action %q: must be 'allow' or 'deny'", req.Action)
	}
	req.Action = action // normalize

	for paramName, constraint := range req.Constraints {
		if !validConstraintOperators[constraint.Operator] {
			return fmt.Errorf("invalid constraint operator %q for parameter %q: must be one of <=, >=, ==, !=, <, >", constraint.Operator, paramName)
		}
	}
	return nil
}

// AddPolicy creates a new access policy and refreshes the cache.
func (s *AccessPolicyService) AddPolicy(ctx context.Context, req *types.AccessPolicyRequest) (*types.AccessPolicy, error) {
	if err := validatePolicyRequest(req); err != nil {
		return nil, err
	}

	now := time.Now()
	policy := &types.AccessPolicy{
		Name:           req.Name,
		CallerTags:     req.CallerTags,
		TargetTags:     req.TargetTags,
		AllowFunctions: req.AllowFunctions,
		DenyFunctions:  req.DenyFunctions,
		Constraints:    req.Constraints,
		Action:         req.Action,
		Priority:       req.Priority,
		Enabled:        true,
		CreatedAt:      now,
		UpdatedAt:      now,
	}
	if req.Description != "" {
		policy.Description = &req.Description
	}

	if err := s.storage.CreateAccessPolicy(ctx, policy); err != nil {
		return nil, fmt.Errorf("failed to create access policy: %w", err)
	}

	// Reload cache — propagate failure so caller knows enforcement may be stale
	if err := s.Initialize(ctx); err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to reload policies after adding new policy — cache may be stale")
		return policy, fmt.Errorf("policy created but cache reload failed: %w", err)
	}

	return policy, nil
}

// UpdatePolicy updates an existing access policy and refreshes the cache.
func (s *AccessPolicyService) UpdatePolicy(ctx context.Context, id int64, req *types.AccessPolicyRequest) (*types.AccessPolicy, error) {
	if err := validatePolicyRequest(req); err != nil {
		return nil, err
	}

	policy, err := s.storage.GetAccessPolicyByID(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("access policy not found: %w", err)
	}

	policy.Name = req.Name
	policy.CallerTags = req.CallerTags
	policy.TargetTags = req.TargetTags
	policy.AllowFunctions = req.AllowFunctions
	policy.DenyFunctions = req.DenyFunctions
	policy.Constraints = req.Constraints
	policy.Action = req.Action
	policy.Priority = req.Priority
	policy.UpdatedAt = time.Now()
	if req.Description != "" {
		policy.Description = &req.Description
	}

	if err := s.storage.UpdateAccessPolicy(ctx, policy); err != nil {
		return nil, fmt.Errorf("failed to update access policy: %w", err)
	}

	// Reload cache — propagate failure so caller knows enforcement may be stale
	if err := s.Initialize(ctx); err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to reload policies after updating policy — cache may be stale")
		return policy, fmt.Errorf("policy updated but cache reload failed: %w", err)
	}

	return policy, nil
}

// RemovePolicy deletes an access policy and refreshes the cache.
func (s *AccessPolicyService) RemovePolicy(ctx context.Context, id int64) error {
	if err := s.storage.DeleteAccessPolicy(ctx, id); err != nil {
		return fmt.Errorf("failed to delete access policy: %w", err)
	}

	// Reload cache — propagate failure so caller knows enforcement may be stale
	if err := s.Initialize(ctx); err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to reload policies after removing policy — cache may be stale")
		return fmt.Errorf("policy deleted but cache reload failed: %w", err)
	}

	return nil
}

// ListPolicies returns all access policies from storage.
func (s *AccessPolicyService) ListPolicies(ctx context.Context) ([]*types.AccessPolicy, error) {
	return s.storage.GetAccessPolicies(ctx)
}

// GetPolicyByID returns a single access policy by ID.
func (s *AccessPolicyService) GetPolicyByID(ctx context.Context, id int64) (*types.AccessPolicy, error) {
	return s.storage.GetAccessPolicyByID(ctx, id)
}

// ============================================================================
// Internal helpers
// ============================================================================

// normalizeTags lowercases and trims all tags.
func normalizeTags(tags []string) []string {
	normalized := make([]string, 0, len(tags))
	for _, tag := range tags {
		if t := normalizeTag(tag); t != "" {
			normalized = append(normalized, t)
		}
	}
	return normalized
}

// tagsIntersect returns true if at least one policy tag matches at least one agent tag.
// Empty policy tags are treated as wildcard (match any agent tags).
// Policy tags support wildcards via matchesPattern.
func tagsIntersect(policyTags, agentTags []string) bool {
	if len(policyTags) == 0 {
		return true // empty policy tags = wildcard, matches any agent
	}
	for _, pt := range policyTags {
		normalizedPT := normalizeTag(pt)
		for _, at := range agentTags {
			if matchesPattern(normalizedPT, at) {
				return true
			}
		}
	}
	return false
}

// functionMatchesAny returns true if the function name matches any of the patterns.
func functionMatchesAny(functionName string, patterns []string) bool {
	normalized := strings.ToLower(strings.TrimSpace(functionName))
	for _, pattern := range patterns {
		if matchesPattern(strings.ToLower(strings.TrimSpace(pattern)), normalized) {
			return true
		}
	}
	return false
}

// evaluateConstraints checks all constraints against input parameters.
// Returns empty string if all pass, or a violation description if any fail.
func evaluateConstraints(constraints map[string]types.AccessConstraint, inputParams map[string]any) string {
	for paramName, constraint := range constraints {
		paramValue, exists := inputParams[paramName]
		if !exists {
			// Fail closed: constraint references a parameter not in input
			return fmt.Sprintf("Constraint violation: parameter %q not found in input", paramName)
		}

		if !evaluateConstraint(paramValue, constraint) {
			return fmt.Sprintf("Constraint violation: %s %s %v (actual: %v)",
				paramName, constraint.Operator, constraint.Value, paramValue)
		}
	}
	return ""
}

// evaluateConstraint checks a single parameter value against a constraint.
func evaluateConstraint(paramValue any, constraint types.AccessConstraint) bool {
	// Try numeric comparison
	paramNum, paramOK := toFloat64(paramValue)
	constraintNum, constraintOK := toFloat64(constraint.Value)

	if paramOK && constraintOK {
		switch constraint.Operator {
		case "<=":
			return paramNum <= constraintNum
		case ">=":
			return paramNum >= constraintNum
		case "<":
			return paramNum < constraintNum
		case ">":
			return paramNum > constraintNum
		case "==":
			return paramNum == constraintNum
		case "!=":
			return paramNum != constraintNum
		}
	}

	// Fall back to string comparison for == and !=
	paramStr := fmt.Sprintf("%v", paramValue)
	constraintStr := fmt.Sprintf("%v", constraint.Value)

	switch constraint.Operator {
	case "==":
		return paramStr == constraintStr
	case "!=":
		return paramStr != constraintStr
	}

	// Unsupported operator for non-numeric types — fail closed
	return false
}

// toFloat64 attempts to convert a value to float64.
func toFloat64(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case int32:
		return float64(n), true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	case string:
		// Don't parse strings as numbers
		return 0, false
	default:
		return 0, false
	}
}

// matchesPattern checks if a value matches a pattern (supports wildcards).
func matchesPattern(pattern, value string) bool {
	if pattern == value {
		return true
	}
	if pattern == "*" {
		return true
	}
	if strings.HasSuffix(pattern, "*") {
		prefix := strings.TrimSuffix(pattern, "*")
		return strings.HasPrefix(value, prefix)
	}
	if strings.HasPrefix(pattern, "*") {
		suffix := strings.TrimPrefix(pattern, "*")
		return strings.HasSuffix(value, suffix)
	}
	return false
}
