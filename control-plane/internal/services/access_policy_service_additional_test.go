package services

import (
	"context"
	"errors"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestAccessPolicyServiceUpdateListAndGet(t *testing.T) {
	storage := &mockAccessPolicyStorage{
		policies: []*types.AccessPolicy{
			newTestPolicy(1, "old", []string{"caller"}, []string{"target"}, "allow", 1),
		},
	}
	service := NewAccessPolicyService(storage)

	updated, err := service.UpdatePolicy(context.Background(), 1, &types.AccessPolicyRequest{
		Name:           "new",
		Description:    "updated",
		CallerTags:     []string{"caller", "extra"},
		TargetTags:     []string{"target"},
		AllowFunctions: []string{"read:*"},
		DenyFunctions:  []string{"write:*"},
		Constraints:    map[string]types.AccessConstraint{"limit": {Operator: "<=", Value: 10}},
		Action:         "deny",
		Priority:       5,
	})
	require.NoError(t, err)
	require.Equal(t, int64(1), updated.ID)
	require.Equal(t, "new", updated.Name)
	require.Equal(t, "deny", updated.Action)
	require.NotNil(t, updated.Description)
	require.Equal(t, "updated", *updated.Description)
	require.NotEmpty(t, service.policies)

	all, err := service.ListPolicies(context.Background())
	require.NoError(t, err)
	require.Len(t, all, 1)

	policy, err := service.GetPolicyByID(context.Background(), 1)
	require.NoError(t, err)
	require.Equal(t, "new", policy.Name)
}

func TestAccessPolicyServiceUpdatePolicyErrors(t *testing.T) {
	t.Run("validation and not found", func(t *testing.T) {
		service := NewAccessPolicyService(&mockAccessPolicyStorage{})

		_, err := service.UpdatePolicy(context.Background(), 1, &types.AccessPolicyRequest{
			Name:   "bad",
			Action: "invalid",
		})
		require.ErrorContains(t, err, "invalid policy action")

		_, err = service.UpdatePolicy(context.Background(), 1, &types.AccessPolicyRequest{
			Name:   "missing",
			Action: "allow",
		})
		require.ErrorContains(t, err, "access policy not found")
	})

	t.Run("storage update failure", func(t *testing.T) {
		storage := &mockAccessPolicyStorage{
			policies: []*types.AccessPolicy{
				newTestPolicy(1, "existing", []string{"caller"}, []string{"target"}, "allow", 1),
			},
		}
		service := NewAccessPolicyService(storage)

		storage.policies = nil
		_, err := service.UpdatePolicy(context.Background(), 1, &types.AccessPolicyRequest{
			Name:   "updated",
			Action: "allow",
		})
		require.ErrorContains(t, err, "access policy not found")
	})

	t.Run("cache reload failure after update", func(t *testing.T) {
		storage := &mockAccessPolicyStorage{
			policies: []*types.AccessPolicy{
				newTestPolicy(1, "existing", []string{"caller"}, []string{"target"}, "allow", 1),
			},
		}
		service := NewAccessPolicyService(storage)
		storage.getErr = errors.New("reload failed")

		_, err := service.UpdatePolicy(context.Background(), 1, &types.AccessPolicyRequest{
			Name:   "updated",
			Action: "allow",
		})
		require.ErrorContains(t, err, "policy updated but cache reload failed")
	})
}

func TestAccessPolicyServiceRemovePolicy(t *testing.T) {
	storage := &mockAccessPolicyStorage{
		policies: []*types.AccessPolicy{
			newTestPolicy(1, "existing", []string{"caller"}, []string{"target"}, "allow", 1),
		},
	}
	service := NewAccessPolicyService(storage)

	require.NoError(t, service.RemovePolicy(context.Background(), 1))
	require.Empty(t, storage.policies)

	storage = &mockAccessPolicyStorage{
		policies: []*types.AccessPolicy{
			newTestPolicy(1, "existing", []string{"caller"}, []string{"target"}, "allow", 1),
		},
		getErr: errors.New("reload failed"),
	}
	service = NewAccessPolicyService(storage)
	err := service.RemovePolicy(context.Background(), 1)
	require.ErrorContains(t, err, "policy deleted but cache reload failed")
}
