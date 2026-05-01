package services

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func dispatcherTrigger(targetURL string, reasoners []types.ReasonerDefinition) (*types.AgentNode, *types.Trigger, *types.InboundEvent) {
	node := &types.AgentNode{
		ID:              "dispatch-node",
		BaseURL:         targetURL,
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       reasoners,
	}
	trig := &types.Trigger{
		ID:             "dispatch-trigger",
		SourceName:     "generic_bearer",
		TargetNodeID:   node.ID,
		TargetReasoner: "handle",
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		Config:         json.RawMessage(`{}`),
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	ev := &types.InboundEvent{
		ID:                "dispatch-event",
		TriggerID:         trig.ID,
		SourceName:        trig.SourceName,
		EventType:         "push",
		RawPayload:        json.RawMessage(`{"raw":true}`),
		NormalizedPayload: json.RawMessage(`{"normalized":true}`),
		IdempotencyKey:    "idem-dispatch",
		Status:            types.InboundEventStatusReceived,
		ReceivedAt:        time.Now().UTC(),
	}
	return node, trig, ev
}

func storeDispatchFixture(t *testing.T, node *types.AgentNode, trig *types.Trigger, ev *types.InboundEvent) (context.Context, interface {
	RegisterAgent(context.Context, *types.AgentNode) error
	CreateTrigger(context.Context, *types.Trigger) error
	InsertInboundEvent(context.Context, *types.InboundEvent) error
	GetInboundEvent(context.Context, string) (*types.InboundEvent, error)
}, *TriggerDispatcher) {
	t.Helper()
	provider, ctx := setupTestStorage(t)
	if node != nil {
		require.NoError(t, provider.RegisterAgent(ctx, node))
	}
	if trig != nil {
		require.NoError(t, provider.CreateTrigger(ctx, trig))
	}
	if ev != nil {
		require.NoError(t, provider.InsertInboundEvent(ctx, ev))
	}
	return ctx, provider, NewTriggerDispatcher(provider, nil)
}

func TestTriggerDispatcherFailureBranches(t *testing.T) {
	t.Run("nil inputs are ignored", func(t *testing.T) {
		provider, ctx := setupTestStorage(t)
		NewTriggerDispatcher(provider, nil).DispatchEvent(ctx, nil, nil)
	})

	t.Run("missing node marks event failed", func(t *testing.T) {
		_, trig, ev := dispatcherTrigger("http://127.0.0.1:1", []types.ReasonerDefinition{{ID: "handle"}})
		ctx, provider, dispatcher := storeDispatchFixture(t, nil, trig, ev)
		dispatcher.DispatchEvent(ctx, trig, ev)
		stored, err := provider.GetInboundEvent(ctx, ev.ID)
		require.NoError(t, err)
		require.Equal(t, types.InboundEventStatusFailed, stored.Status)
		require.Contains(t, stored.ErrorMessage, "target node")
	})

	t.Run("inactive node marks event failed", func(t *testing.T) {
		node, trig, ev := dispatcherTrigger("http://127.0.0.1:1", []types.ReasonerDefinition{{ID: "handle"}})
		node.HealthStatus = types.HealthStatusInactive
		ctx, provider, dispatcher := storeDispatchFixture(t, node, trig, ev)
		dispatcher.DispatchEvent(ctx, trig, ev)
		stored, err := provider.GetInboundEvent(ctx, ev.ID)
		require.NoError(t, err)
		require.Equal(t, types.InboundEventStatusFailed, stored.Status)
		require.Contains(t, stored.ErrorMessage, "unreachable")
	})

	t.Run("missing reasoner marks event failed", func(t *testing.T) {
		node, trig, ev := dispatcherTrigger("http://127.0.0.1:1", []types.ReasonerDefinition{{ID: "other"}})
		ctx, provider, dispatcher := storeDispatchFixture(t, node, trig, ev)
		dispatcher.DispatchEvent(ctx, trig, ev)
		stored, err := provider.GetInboundEvent(ctx, ev.ID)
		require.NoError(t, err)
		require.Equal(t, types.InboundEventStatusFailed, stored.Status)
		require.Contains(t, stored.ErrorMessage, "reasoner")
	})

	t.Run("bad target URL marks event failed", func(t *testing.T) {
		node, trig, ev := dispatcherTrigger("http://[::1", []types.ReasonerDefinition{{ID: "handle"}})
		ctx, provider, dispatcher := storeDispatchFixture(t, node, trig, ev)
		dispatcher.DispatchEvent(ctx, trig, ev)
		stored, err := provider.GetInboundEvent(ctx, ev.ID)
		require.NoError(t, err)
		require.Equal(t, types.InboundEventStatusFailed, stored.Status)
		require.Contains(t, stored.ErrorMessage, "build request")
	})
}

func TestTriggerDispatcherHTTPStatusAndSuccessBranches(t *testing.T) {
	t.Run("agent error response marks event failed", func(t *testing.T) {
		target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "nope", http.StatusBadGateway)
		}))
		defer target.Close()

		node, trig, ev := dispatcherTrigger(target.URL, []types.ReasonerDefinition{{ID: "handle"}})
		ctx, provider, dispatcher := storeDispatchFixture(t, node, trig, ev)
		dispatcher.DispatchEvent(ctx, trig, ev)
		stored, err := provider.GetInboundEvent(ctx, ev.ID)
		require.NoError(t, err)
		require.Equal(t, types.InboundEventStatusFailed, stored.Status)
		require.Contains(t, stored.ErrorMessage, "502")
	})

	t.Run("successful dispatch records workflow and preserves parent VC", func(t *testing.T) {
		var got map[string]any
		var workflowID string
		target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			workflowID = r.Header.Get("X-Workflow-ID")
			require.Equal(t, "parent-vc", r.Header.Get("X-Parent-VC-ID"))
			require.Equal(t, "dispatch-trigger", r.Header.Get("X-Trigger-ID"))
			require.NoError(t, json.NewDecoder(r.Body).Decode(&got))
			w.WriteHeader(http.StatusOK)
		}))
		defer target.Close()

		node, trig, ev := dispatcherTrigger(target.URL, []types.ReasonerDefinition{{ID: "handle"}})
		ev.VCID = "parent-vc"
		ctx, provider, dispatcher := storeDispatchFixture(t, node, trig, ev)
		dispatcher.DispatchEvent(ctx, trig, ev)
		stored, err := provider.GetInboundEvent(ctx, ev.ID)
		require.NoError(t, err)
		require.Equal(t, types.InboundEventStatusDispatched, stored.Status)
		require.Equal(t, "parent-vc", stored.VCID)
		require.Equal(t, workflowID, stored.DispatchedWorkflowID)
		require.Equal(t, map[string]any{"normalized": true}, got["event"])
		require.Contains(t, got, "_meta")
	})
}
