package handlers

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	// Import sources to register them.
	_ "github.com/Agent-Field/agentfield/control-plane/internal/sources/all"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestSourceOfTruth_StickyPauseSurvivesReregistration is the headline §5.3
// guarantee: when an operator pauses a code-managed trigger via the API,
// the next agent re-registration must NOT silently flip it back to enabled.
// Without this, "pause this misbehaving Stripe webhook" would require a
// code deploy + agent restart, which is exactly the 2am incident multiplier
// we're trying to avoid.
func TestSourceOfTruth_StickyPauseSurvivesReregistration(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	node := &types.AgentNode{
		ID:              "node-sticky-pause",
		BaseURL:         "http://localhost:9999",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners: []types.ReasonerDefinition{{
			ID: "handle_payment",
			Triggers: []types.TriggerBinding{{
				Source:       "stripe",
				EventTypes:   []string{"payment_intent.succeeded"},
				SecretEnvVar: "STRIPE_SECRET",
				Config:       json.RawMessage(`{}`),
				CodeOrigin:   "agent.py:42",
			}},
		}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	// First registration creates the trigger row enabled=true.
	entries := upsertCodeManagedTriggers(ctx, provider, node)
	require.Len(t, entries, 1)
	triggerID := entries[0].TriggerID

	created, err := provider.GetTrigger(ctx, triggerID)
	require.NoError(t, err)
	require.True(t, created.Enabled)
	require.False(t, created.ManualOverrideEnabled)
	require.False(t, created.Orphaned)
	require.NotNil(t, created.LastRegisteredAt)
	require.Equal(t, "agent.py:42", created.CodeOrigin)

	// Operator pauses via the API path.
	require.NoError(t, provider.SetTriggerOverride(ctx, triggerID, true, false))

	paused, err := provider.GetTrigger(ctx, triggerID)
	require.NoError(t, err)
	require.False(t, paused.Enabled, "pause should disable")
	require.True(t, paused.ManualOverrideEnabled, "override flag should be set")

	// Agent restarts and re-registers (same bindings, would normally try to
	// set Enabled=true again).
	originalRegisteredAt := *paused.LastRegisteredAt
	time.Sleep(10 * time.Millisecond) // ensure last_registered_at advances
	upsertCodeManagedTriggers(ctx, provider, node)

	after, err := provider.GetTrigger(ctx, triggerID)
	require.NoError(t, err)
	assert.False(t, after.Enabled, "STICKY: re-registration must not flip enabled back on")
	assert.True(t, after.ManualOverrideEnabled, "override survives re-registration")
	require.NotNil(t, after.LastRegisteredAt)
	assert.True(t, after.LastRegisteredAt.After(originalRegisteredAt), "last_registered_at advances")

	// Operator resumes — clears override, re-enables.
	require.NoError(t, provider.SetTriggerOverride(ctx, triggerID, false, true))

	resumed, err := provider.GetTrigger(ctx, triggerID)
	require.NoError(t, err)
	assert.True(t, resumed.Enabled)
	assert.False(t, resumed.ManualOverrideEnabled)
	assert.Nil(t, resumed.ManualOverrideAt)
}

// TestSourceOfTruth_OrphanFlowOnDecoratorRemoval covers §5.4. The agent
// re-registers without a previously-declared binding (decorator removed in
// user code). The row stays — events stop dispatching but history is
// preserved — and the UI sees orphaned=true so the operator can decide to
// convert-to-ui or delete.
func TestSourceOfTruth_OrphanFlowOnDecoratorRemoval(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	nodeWithStripe := &types.AgentNode{
		ID:              "node-orphan",
		BaseURL:         "http://localhost:9999",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners: []types.ReasonerDefinition{{
			ID: "handle_payment",
			Triggers: []types.TriggerBinding{{
				Source:       "stripe",
				EventTypes:   []string{"payment_intent.succeeded"},
				SecretEnvVar: "STRIPE_SECRET",
				Config:       json.RawMessage(`{}`),
			}},
		}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, nodeWithStripe))

	entries := upsertCodeManagedTriggers(ctx, provider, nodeWithStripe)
	require.Len(t, entries, 1)
	triggerID := entries[0].TriggerID

	// Agent code refactor — Stripe binding removed, reasoner stays.
	nodeWithoutStripe := &types.AgentNode{
		ID:              "node-orphan",
		BaseURL:         "http://localhost:9999",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners: []types.ReasonerDefinition{{
			ID:       "handle_payment",
			Triggers: []types.TriggerBinding{}, // decorator removed
		}},
	}
	upsertCodeManagedTriggers(ctx, provider, nodeWithoutStripe)

	orphaned, err := provider.GetTrigger(ctx, triggerID)
	require.NoError(t, err)
	assert.True(t, orphaned.Orphaned, "trigger row preserved with orphaned=true")
	assert.Equal(t, types.ManagedByCode, orphaned.ManagedBy, "still code-managed until operator converts")

	// Operator converts to UI-managed so they can edit/delete via the UI
	// without the next registration recreating it.
	require.NoError(t, provider.ConvertTriggerToUIManaged(ctx, triggerID))

	converted, err := provider.GetTrigger(ctx, triggerID)
	require.NoError(t, err)
	assert.Equal(t, types.ManagedByUI, converted.ManagedBy)
	assert.False(t, converted.Orphaned, "orphan flag cleared after conversion")
}

// TestSourceOfTruth_ReregistrationClearsOrphanWhenBindingReturns confirms
// that bringing a decorator BACK in code clears the orphan flag — we don't
// want a stale orphan badge stuck on a re-introduced trigger.
func TestSourceOfTruth_ReregistrationClearsOrphanWhenBindingReturns(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	node := &types.AgentNode{
		ID:              "node-orphan-restored",
		BaseURL:         "http://localhost:9999",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners: []types.ReasonerDefinition{{
			ID: "handle_payment",
			Triggers: []types.TriggerBinding{{
				Source:       "stripe",
				EventTypes:   []string{"payment_intent.succeeded"},
				SecretEnvVar: "STRIPE_SECRET",
				Config:       json.RawMessage(`{}`),
			}},
		}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))
	entries := upsertCodeManagedTriggers(ctx, provider, node)
	triggerID := entries[0].TriggerID

	// Remove binding → orphaned.
	noBindings := *node
	noBindings.Reasoners = []types.ReasonerDefinition{{ID: "handle_payment", Triggers: []types.TriggerBinding{}}}
	upsertCodeManagedTriggers(ctx, provider, &noBindings)

	orphaned, _ := provider.GetTrigger(ctx, triggerID)
	require.True(t, orphaned.Orphaned)

	// Restore binding → orphan flag cleared on re-upsert.
	upsertCodeManagedTriggers(ctx, provider, node)

	restored, err := provider.GetTrigger(ctx, triggerID)
	require.NoError(t, err)
	assert.False(t, restored.Orphaned, "re-declared binding clears orphan flag")
}
