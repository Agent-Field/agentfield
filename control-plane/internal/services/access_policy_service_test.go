package services

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockAccessPolicyStorage is an in-memory mock for AccessPolicyStorage.
type mockAccessPolicyStorage struct {
	policies  []*types.AccessPolicy
	nextID    int64
	createErr error
	getErr    error
}

func (m *mockAccessPolicyStorage) GetAccessPolicies(_ context.Context) ([]*types.AccessPolicy, error) {
	if m.getErr != nil {
		return nil, m.getErr
	}
	// Return copies to prevent test mutation
	result := make([]*types.AccessPolicy, len(m.policies))
	copy(result, m.policies)
	return result, nil
}

func (m *mockAccessPolicyStorage) GetAccessPolicyByID(_ context.Context, id int64) (*types.AccessPolicy, error) {
	for _, p := range m.policies {
		if p.ID == id {
			return p, nil
		}
	}
	return nil, fmt.Errorf("policy %d not found", id)
}

func (m *mockAccessPolicyStorage) CreateAccessPolicy(_ context.Context, policy *types.AccessPolicy) error {
	if m.createErr != nil {
		return m.createErr
	}
	m.nextID++
	policy.ID = m.nextID
	m.policies = append(m.policies, policy)
	return nil
}

func (m *mockAccessPolicyStorage) UpdateAccessPolicy(_ context.Context, policy *types.AccessPolicy) error {
	for i, p := range m.policies {
		if p.ID == policy.ID {
			m.policies[i] = policy
			return nil
		}
	}
	return fmt.Errorf("policy %d not found", policy.ID)
}

func (m *mockAccessPolicyStorage) DeleteAccessPolicy(_ context.Context, id int64) error {
	for i, p := range m.policies {
		if p.ID == id {
			m.policies = append(m.policies[:i], m.policies[i+1:]...)
			return nil
		}
	}
	return fmt.Errorf("policy %d not found", id)
}

func newTestPolicy(id int64, name string, callerTags, targetTags []string, action string, priority int) *types.AccessPolicy {
	return &types.AccessPolicy{
		ID:         id,
		Name:       name,
		CallerTags: callerTags,
		TargetTags: targetTags,
		Action:     action,
		Priority:   priority,
		Enabled:    true,
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
	}
}

// ============================================================================
// EvaluateAccess — core policy evaluation
// ============================================================================

func TestEvaluateAccess_NoMatchReturnsNotMatched(t *testing.T) {
	svc := NewAccessPolicyService(&mockAccessPolicyStorage{})
	svc.policies = []*types.AccessPolicy{
		newTestPolicy(1, "finance_to_billing", []string{"finance"}, []string{"billing"}, "allow", 10),
	}

	result := svc.EvaluateAccess([]string{"support"}, []string{"billing"}, "get_balance", nil)
	assert.False(t, result.Matched)
	assert.False(t, result.Allowed)
	assert.Contains(t, result.Reason, "No access policy matched")
}

func TestEvaluateAccess_SimpleAllow(t *testing.T) {
	svc := NewAccessPolicyService(&mockAccessPolicyStorage{})
	svc.policies = []*types.AccessPolicy{
		newTestPolicy(1, "finance_to_billing", []string{"finance"}, []string{"billing"}, "allow", 10),
	}

	result := svc.EvaluateAccess([]string{"finance"}, []string{"billing"}, "charge", nil)
	assert.True(t, result.Matched)
	assert.True(t, result.Allowed)
	assert.Equal(t, "finance_to_billing", result.PolicyName)
	assert.Equal(t, int64(1), result.PolicyID)
}

func TestEvaluateAccess_SimpleDeny(t *testing.T) {
	svc := NewAccessPolicyService(&mockAccessPolicyStorage{})
	svc.policies = []*types.AccessPolicy{
		newTestPolicy(1, "block_support", []string{"support"}, []string{"admin"}, "deny", 10),
	}

	result := svc.EvaluateAccess([]string{"support"}, []string{"admin"}, "delete_user", nil)
	assert.True(t, result.Matched)
	assert.False(t, result.Allowed)
	assert.Contains(t, result.Reason, "explicitly denies")
}

func TestEvaluateAccess_DisabledPolicySkipped(t *testing.T) {
	svc := NewAccessPolicyService(&mockAccessPolicyStorage{})
	policy := newTestPolicy(1, "disabled_policy", []string{"finance"}, []string{"billing"}, "allow", 10)
	policy.Enabled = false
	svc.policies = []*types.AccessPolicy{policy}

	result := svc.EvaluateAccess([]string{"finance"}, []string{"billing"}, "charge", nil)
	assert.False(t, result.Matched)
}

func TestEvaluateAccess_PriorityOrdering(t *testing.T) {
	svc := NewAccessPolicyService(&mockAccessPolicyStorage{})
	svc.policies = []*types.AccessPolicy{
		newTestPolicy(1, "high_priority_deny", []string{"finance"}, []string{"billing"}, "deny", 100),
		newTestPolicy(2, "low_priority_allow", []string{"finance"}, []string{"billing"}, "allow", 10),
	}

	result := svc.EvaluateAccess([]string{"finance"}, []string{"billing"}, "charge", nil)
	assert.True(t, result.Matched)
	assert.False(t, result.Allowed)
	assert.Equal(t, "high_priority_deny", result.PolicyName)
}

func TestEvaluateAccess_EmptyCallerTagsWildcard(t *testing.T) {
	// Empty caller_tags on a policy means "match any caller"
	svc := NewAccessPolicyService(&mockAccessPolicyStorage{})
	svc.policies = []*types.AccessPolicy{
		newTestPolicy(1, "any_to_public", []string{}, []string{"public"}, "allow", 10),
	}

	result := svc.EvaluateAccess([]string{"random-tag"}, []string{"public"}, "get", nil)
	assert.True(t, result.Matched)
	assert.True(t, result.Allowed)
}

func TestEvaluateAccess_EmptyTargetTagsWildcard(t *testing.T) {
	// Empty target_tags on a policy means "match any target"
	svc := NewAccessPolicyService(&mockAccessPolicyStorage{})
	svc.policies = []*types.AccessPolicy{
		newTestPolicy(1, "admin_to_any", []string{"admin"}, []string{}, "allow", 10),
	}

	result := svc.EvaluateAccess([]string{"admin"}, []string{"whatever"}, "anything", nil)
	assert.True(t, result.Matched)
	assert.True(t, result.Allowed)
}

func TestEvaluateAccess_TagsAreCaseInsensitive(t *testing.T) {
	svc := NewAccessPolicyService(&mockAccessPolicyStorage{})
	svc.policies = []*types.AccessPolicy{
		newTestPolicy(1, "case_test", []string{"Finance"}, []string{"Billing"}, "allow", 10),
	}

	result := svc.EvaluateAccess([]string{"FINANCE"}, []string{"billing"}, "charge", nil)
	assert.True(t, result.Matched)
	assert.True(t, result.Allowed)
}

// ============================================================================
// Function allow/deny lists
// ============================================================================

func TestEvaluateAccess_AllowFunctionList(t *testing.T) {
	svc := NewAccessPolicyService(&mockAccessPolicyStorage{})
	policy := newTestPolicy(1, "allow_specific", []string{"finance"}, []string{"billing"}, "allow", 10)
	policy.AllowFunctions = []string{"charge_*", "get_*"}
	svc.policies = []*types.AccessPolicy{policy}

	t.Run("allowed function matches prefix", func(t *testing.T) {
		result := svc.EvaluateAccess([]string{"finance"}, []string{"billing"}, "charge_customer", nil)
		assert.True(t, result.Matched)
		assert.True(t, result.Allowed)
	})

	t.Run("disallowed function skips policy", func(t *testing.T) {
		result := svc.EvaluateAccess([]string{"finance"}, []string{"billing"}, "delete_account", nil)
		assert.False(t, result.Matched) // Policy doesn't match, falls through
	})

	t.Run("empty function name skips policy with allow list", func(t *testing.T) {
		result := svc.EvaluateAccess([]string{"finance"}, []string{"billing"}, "", nil)
		assert.False(t, result.Matched)
	})
}

func TestEvaluateAccess_DenyFunctionList(t *testing.T) {
	svc := NewAccessPolicyService(&mockAccessPolicyStorage{})
	policy := newTestPolicy(1, "deny_specific", []string{"finance"}, []string{"billing"}, "allow", 10)
	policy.DenyFunctions = []string{"delete_*", "admin_*"}
	svc.policies = []*types.AccessPolicy{policy}

	t.Run("denied function is blocked", func(t *testing.T) {
		result := svc.EvaluateAccess([]string{"finance"}, []string{"billing"}, "delete_user", nil)
		assert.True(t, result.Matched)
		assert.False(t, result.Allowed)
		assert.Contains(t, result.Reason, "denied")
	})

	t.Run("non-denied function allowed", func(t *testing.T) {
		result := svc.EvaluateAccess([]string{"finance"}, []string{"billing"}, "charge_customer", nil)
		assert.True(t, result.Matched)
		assert.True(t, result.Allowed)
	})
}

func TestEvaluateAccess_DenyTakesPrecedenceOverAllow(t *testing.T) {
	// A function in both allow and deny lists should be denied
	svc := NewAccessPolicyService(&mockAccessPolicyStorage{})
	policy := newTestPolicy(1, "mixed", []string{"finance"}, []string{"billing"}, "allow", 10)
	policy.AllowFunctions = []string{"*"}
	policy.DenyFunctions = []string{"delete_*"}
	svc.policies = []*types.AccessPolicy{policy}

	result := svc.EvaluateAccess([]string{"finance"}, []string{"billing"}, "delete_user", nil)
	assert.True(t, result.Matched)
	assert.False(t, result.Allowed)
}

// ============================================================================
// Constraint evaluation
// ============================================================================

func TestEvaluateAccess_NumericConstraints(t *testing.T) {
	svc := NewAccessPolicyService(&mockAccessPolicyStorage{})
	policy := newTestPolicy(1, "constrained", []string{"finance"}, []string{"billing"}, "allow", 10)
	policy.Constraints = map[string]types.AccessConstraint{
		"amount": {Operator: "<=", Value: float64(10000)},
	}
	svc.policies = []*types.AccessPolicy{policy}

	t.Run("within limit", func(t *testing.T) {
		result := svc.EvaluateAccess([]string{"finance"}, []string{"billing"}, "charge",
			map[string]any{"amount": float64(5000)})
		assert.True(t, result.Matched)
		assert.True(t, result.Allowed)
	})

	t.Run("at limit", func(t *testing.T) {
		result := svc.EvaluateAccess([]string{"finance"}, []string{"billing"}, "charge",
			map[string]any{"amount": float64(10000)})
		assert.True(t, result.Matched)
		assert.True(t, result.Allowed)
	})

	t.Run("over limit", func(t *testing.T) {
		result := svc.EvaluateAccess([]string{"finance"}, []string{"billing"}, "charge",
			map[string]any{"amount": float64(15000)})
		assert.True(t, result.Matched)
		assert.False(t, result.Allowed)
		assert.Contains(t, result.Reason, "Constraint violation")
	})
}

func TestEvaluateAccess_AllNumericOperators(t *testing.T) {
	tests := []struct {
		operator string
		value    float64
		param    float64
		expect   bool
	}{
		{"<=", 100, 50, true},
		{"<=", 100, 100, true},
		{"<=", 100, 101, false},
		{">=", 100, 150, true},
		{">=", 100, 100, true},
		{">=", 100, 99, false},
		{"<", 100, 99, true},
		{"<", 100, 100, false},
		{">", 100, 101, true},
		{">", 100, 100, false},
		{"==", 100, 100, true},
		{"==", 100, 99, false},
		{"!=", 100, 99, true},
		{"!=", 100, 100, false},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("%v%s%v", tc.param, tc.operator, tc.value), func(t *testing.T) {
			svc := NewAccessPolicyService(&mockAccessPolicyStorage{})
			policy := newTestPolicy(1, "op_test", []string{}, []string{}, "allow", 10)
			policy.Constraints = map[string]types.AccessConstraint{
				"val": {Operator: tc.operator, Value: tc.value},
			}
			svc.policies = []*types.AccessPolicy{policy}

			result := svc.EvaluateAccess([]string{"any"}, []string{"any"}, "fn",
				map[string]any{"val": tc.param})
			assert.Equal(t, tc.expect, result.Allowed,
				"expected %v %s %v = %v", tc.param, tc.operator, tc.value, tc.expect)
		})
	}
}

func TestEvaluateAccess_StringEqualityConstraints(t *testing.T) {
	svc := NewAccessPolicyService(&mockAccessPolicyStorage{})
	policy := newTestPolicy(1, "string_test", []string{}, []string{}, "allow", 10)
	policy.Constraints = map[string]types.AccessConstraint{
		"region": {Operator: "==", Value: "us-east"},
	}
	svc.policies = []*types.AccessPolicy{policy}

	t.Run("matching string", func(t *testing.T) {
		result := svc.EvaluateAccess([]string{"any"}, []string{"any"}, "fn",
			map[string]any{"region": "us-east"})
		assert.True(t, result.Allowed)
	})

	t.Run("non-matching string", func(t *testing.T) {
		result := svc.EvaluateAccess([]string{"any"}, []string{"any"}, "fn",
			map[string]any{"region": "eu-west"})
		assert.False(t, result.Allowed)
	})
}

func TestEvaluateAccess_MissingParameterFailsClosed(t *testing.T) {
	svc := NewAccessPolicyService(&mockAccessPolicyStorage{})
	policy := newTestPolicy(1, "constrained", []string{}, []string{}, "allow", 10)
	policy.Constraints = map[string]types.AccessConstraint{
		"amount": {Operator: "<=", Value: float64(10000)},
	}
	svc.policies = []*types.AccessPolicy{policy}

	result := svc.EvaluateAccess([]string{"any"}, []string{"any"}, "fn",
		map[string]any{"other_param": float64(5000)})
	assert.True(t, result.Matched)
	assert.False(t, result.Allowed)
	assert.Contains(t, result.Reason, "not found in input")
}

func TestEvaluateAccess_NilInputParamsFailsClosed(t *testing.T) {
	svc := NewAccessPolicyService(&mockAccessPolicyStorage{})
	policy := newTestPolicy(1, "constrained", []string{}, []string{}, "allow", 10)
	policy.Constraints = map[string]types.AccessConstraint{
		"amount": {Operator: "<=", Value: float64(10000)},
	}
	svc.policies = []*types.AccessPolicy{policy}

	result := svc.EvaluateAccess([]string{"any"}, []string{"any"}, "fn", nil)
	assert.True(t, result.Matched)
	assert.False(t, result.Allowed)
	assert.Contains(t, result.Reason, "no input parameters provided")
}

func TestEvaluateAccess_IntegerParameterConversion(t *testing.T) {
	svc := NewAccessPolicyService(&mockAccessPolicyStorage{})
	policy := newTestPolicy(1, "int_test", []string{}, []string{}, "allow", 10)
	policy.Constraints = map[string]types.AccessConstraint{
		"count": {Operator: "<=", Value: float64(10)},
	}
	svc.policies = []*types.AccessPolicy{policy}

	// Go int, int32, int64 should all convert to float64 for comparison
	for _, val := range []any{int(5), int32(5), int64(5), float32(5)} {
		result := svc.EvaluateAccess([]string{"any"}, []string{"any"}, "fn",
			map[string]any{"count": val})
		assert.True(t, result.Allowed, "should accept %T(%v)", val, val)
	}
}

func TestEvaluateAccess_NonNumericWithOrderingOperatorFailsClosed(t *testing.T) {
	// Strings with < or > operators should fail closed (not comparable)
	svc := NewAccessPolicyService(&mockAccessPolicyStorage{})
	policy := newTestPolicy(1, "fail_closed_test", []string{}, []string{}, "allow", 10)
	policy.Constraints = map[string]types.AccessConstraint{
		"name": {Operator: "<=", Value: "some_string"},
	}
	svc.policies = []*types.AccessPolicy{policy}

	result := svc.EvaluateAccess([]string{"any"}, []string{"any"}, "fn",
		map[string]any{"name": "other_string"})
	assert.True(t, result.Matched)
	assert.False(t, result.Allowed)
}

// ============================================================================
// Tag wildcard patterns
// ============================================================================

func TestEvaluateAccess_WildcardTagPattern(t *testing.T) {
	svc := NewAccessPolicyService(&mockAccessPolicyStorage{})
	svc.policies = []*types.AccessPolicy{
		newTestPolicy(1, "fin_wildcard", []string{"fin*"}, []string{"billing"}, "allow", 10),
	}

	t.Run("matches prefix", func(t *testing.T) {
		result := svc.EvaluateAccess([]string{"finance"}, []string{"billing"}, "charge", nil)
		assert.True(t, result.Matched)
		assert.True(t, result.Allowed)
	})

	t.Run("doesn't match different prefix", func(t *testing.T) {
		result := svc.EvaluateAccess([]string{"support"}, []string{"billing"}, "charge", nil)
		assert.False(t, result.Matched)
	})
}

func TestEvaluateAccess_StarWildcardMatchesAll(t *testing.T) {
	svc := NewAccessPolicyService(&mockAccessPolicyStorage{})
	svc.policies = []*types.AccessPolicy{
		newTestPolicy(1, "star_wildcard", []string{"*"}, []string{"*"}, "allow", 10),
	}

	result := svc.EvaluateAccess([]string{"anything"}, []string{"whatever"}, "fn", nil)
	assert.True(t, result.Matched)
	assert.True(t, result.Allowed)
}

// ============================================================================
// Initialize — loading and sorting
// ============================================================================

func TestInitialize_SortsByPriorityDescThenIDDesc(t *testing.T) {
	storage := &mockAccessPolicyStorage{
		policies: []*types.AccessPolicy{
			newTestPolicy(3, "low", []string{"a"}, []string{"a"}, "allow", 1),
			newTestPolicy(1, "high", []string{"a"}, []string{"a"}, "deny", 100),
			newTestPolicy(2, "medium", []string{"a"}, []string{"a"}, "allow", 50),
		},
	}
	svc := NewAccessPolicyService(storage)
	err := svc.Initialize(context.Background())
	require.NoError(t, err)

	// Verify policies are sorted: high(100), medium(50), low(1)
	assert.Equal(t, "high", svc.policies[0].Name)
	assert.Equal(t, "medium", svc.policies[1].Name)
	assert.Equal(t, "low", svc.policies[2].Name)
}

func TestInitialize_SamePriorityDeterministicByID(t *testing.T) {
	storage := &mockAccessPolicyStorage{
		policies: []*types.AccessPolicy{
			newTestPolicy(5, "later", []string{}, []string{}, "deny", 10),
			newTestPolicy(2, "earlier", []string{}, []string{}, "allow", 10),
		},
	}
	svc := NewAccessPolicyService(storage)
	err := svc.Initialize(context.Background())
	require.NoError(t, err)

	// Same priority → lower ID first (stable, deterministic)
	assert.Equal(t, "earlier", svc.policies[0].Name)
	assert.Equal(t, "later", svc.policies[1].Name)
}

func TestInitialize_StorageError(t *testing.T) {
	storage := &mockAccessPolicyStorage{getErr: fmt.Errorf("db down")}
	svc := NewAccessPolicyService(storage)
	err := svc.Initialize(context.Background())
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to load access policies")
}

// ============================================================================
// AddPolicy / UpdatePolicy / RemovePolicy — CRUD with cache refresh
// ============================================================================

func TestAddPolicy_ValidatesAction(t *testing.T) {
	storage := &mockAccessPolicyStorage{}
	svc := NewAccessPolicyService(storage)

	_, err := svc.AddPolicy(context.Background(), &types.AccessPolicyRequest{
		Name:       "bad",
		CallerTags: []string{"a"},
		TargetTags: []string{"b"},
		Action:     "maybe",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid policy action")
}

func TestAddPolicy_ValidatesConstraintOperator(t *testing.T) {
	storage := &mockAccessPolicyStorage{}
	svc := NewAccessPolicyService(storage)

	_, err := svc.AddPolicy(context.Background(), &types.AccessPolicyRequest{
		Name:       "bad_op",
		CallerTags: []string{"a"},
		TargetTags: []string{"b"},
		Action:     "allow",
		Constraints: map[string]types.AccessConstraint{
			"amount": {Operator: "~=", Value: 100},
		},
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid constraint operator")
}

func TestAddPolicy_SuccessAndCacheRefresh(t *testing.T) {
	storage := &mockAccessPolicyStorage{}
	svc := NewAccessPolicyService(storage)

	policy, err := svc.AddPolicy(context.Background(), &types.AccessPolicyRequest{
		Name:       "test_policy",
		CallerTags: []string{"finance"},
		TargetTags: []string{"billing"},
		Action:     "allow",
		Priority:   10,
	})
	require.NoError(t, err)
	assert.Equal(t, "test_policy", policy.Name)
	assert.True(t, policy.Enabled)

	// Verify the cache was refreshed (policy is now in the in-memory list)
	result := svc.EvaluateAccess([]string{"finance"}, []string{"billing"}, "fn", nil)
	assert.True(t, result.Matched)
	assert.True(t, result.Allowed)
}

func TestRemovePolicy_RemovesFromCacheAndStorage(t *testing.T) {
	storage := &mockAccessPolicyStorage{
		policies: []*types.AccessPolicy{
			newTestPolicy(1, "to_remove", []string{"a"}, []string{"b"}, "allow", 10),
		},
	}
	svc := NewAccessPolicyService(storage)
	require.NoError(t, svc.Initialize(context.Background()))

	// Verify it exists
	result := svc.EvaluateAccess([]string{"a"}, []string{"b"}, "fn", nil)
	assert.True(t, result.Matched)

	// Remove
	err := svc.RemovePolicy(context.Background(), 1)
	require.NoError(t, err)

	// Verify it's gone
	result = svc.EvaluateAccess([]string{"a"}, []string{"b"}, "fn", nil)
	assert.False(t, result.Matched)
}

// ============================================================================
// Internal helpers
// ============================================================================

func TestToFloat64_VariousTypes(t *testing.T) {
	tests := []struct {
		input    any
		expected float64
		ok       bool
	}{
		{float64(42), 42, true},
		{float32(42), 42, true},
		{int(42), 42, true},
		{int64(42), 42, true},
		{int32(42), 42, true},
		{"not a number", 0, false},
		{true, 0, false},
		{nil, 0, false},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("%T(%v)", tc.input, tc.input), func(t *testing.T) {
			val, ok := toFloat64(tc.input)
			assert.Equal(t, tc.ok, ok)
			if ok {
				assert.Equal(t, tc.expected, val)
			}
		})
	}
}

func TestFunctionMatchesAny_Patterns(t *testing.T) {
	tests := []struct {
		fn       string
		patterns []string
		expect   bool
	}{
		{"charge_customer", []string{"charge_*"}, true},
		{"get_balance", []string{"get_*", "query_*"}, true},
		{"delete_user", []string{"get_*", "query_*"}, false},
		{"any_function", []string{"*"}, true},
		{"charge_customer", []string{"charge_customer"}, true},
		{"CHARGE_CUSTOMER", []string{"charge_customer"}, true}, // case insensitive
	}

	for _, tc := range tests {
		t.Run(tc.fn, func(t *testing.T) {
			assert.Equal(t, tc.expect, functionMatchesAny(tc.fn, tc.patterns))
		})
	}
}

func TestTagsIntersect(t *testing.T) {
	tests := []struct {
		name       string
		policyTags []string
		agentTags  []string
		expect     bool
	}{
		{"empty policy tags = wildcard", nil, []string{"anything"}, true},
		{"exact match", []string{"finance"}, []string{"finance", "internal"}, true},
		{"no match", []string{"admin"}, []string{"finance", "internal"}, false},
		{"wildcard pattern", []string{"fin*"}, []string{"finance"}, true},
		{"star matches all", []string{"*"}, []string{"finance"}, true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.expect, tagsIntersect(tc.policyTags, tc.agentTags))
		})
	}
}
