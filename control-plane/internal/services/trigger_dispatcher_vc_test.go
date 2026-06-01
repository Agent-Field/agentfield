package services

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestDispatcher_MintsTriggerEventVC_AndSetsParentHeader exercises the full
// happy path: dispatcher receives a trigger + inbound event, mints the
// trigger event VC, sets X-Parent-VC-ID on the outbound request, and writes
// the VC ID back onto the inbound event row.
func TestDispatcher_MintsTriggerEventVC_AndSetsParentHeader(t *testing.T) {
	vcService, didService, provider, ctx := setupVCTestEnvironment(t)

	// Register a node + reasoner so the dispatcher can find a target.
	regResp, err := didService.RegisterAgent(&types.DIDRegistrationRequest{
		AgentNodeID: "node-trigger-dispatch",
		Reasoners:   []types.ReasonerDefinition{{ID: "handle_payment"}},
	})
	require.NoError(t, err)
	require.True(t, regResp.Success)

	var (
		mu             sync.Mutex
		gotParentVCID  string
		gotTriggerID   string
		gotSourceName  string
		gotEventType   string
		receivedReq    bool
	)
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		receivedReq = true
		gotParentVCID = r.Header.Get("X-Parent-VC-ID")
		gotTriggerID = r.Header.Get("X-Trigger-ID")
		gotSourceName = r.Header.Get("X-Source-Name")
		gotEventType = r.Header.Get("X-Event-Type")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer target.Close()

	node := &types.AgentNode{
		ID:              "node-trigger-dispatch",
		BaseURL:         target.URL,
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "handle_payment"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	trig := &types.Trigger{
		ID:             "trg_dispatch_001",
		SourceName:     "stripe",
		TargetNodeID:   "node-trigger-dispatch",
		TargetReasoner: "handle_payment",
		ManagedBy:      types.ManagedByCode,
		Enabled:        true,
		Config:         json.RawMessage(`{}`),
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	ev := &types.InboundEvent{
		ID:                "evt_dispatch_001",
		TriggerID:         trig.ID,
		SourceName:        "stripe",
		EventType:         "payment_intent.succeeded",
		RawPayload:        json.RawMessage(`{"id":"evt_dispatch_001"}`),
		NormalizedPayload: json.RawMessage(`{"id":"evt_dispatch_001"}`),
		IdempotencyKey:    "evt_dispatch_001",
		Status:            types.InboundEventStatusReceived,
		ReceivedAt:        time.Now().UTC(),
	}
	require.NoError(t, provider.InsertInboundEvent(ctx, ev))

	dispatcher := NewTriggerDispatcher(provider, vcService)
	dispatcher.DispatchEvent(ctx, trig, ev)

	mu.Lock()
	defer mu.Unlock()
	require.True(t, receivedReq, "target reasoner should have been invoked")
	assert.NotEmpty(t, gotParentVCID, "X-Parent-VC-ID must be set when DID is enabled")
	assert.Equal(t, trig.ID, gotTriggerID)
	assert.Equal(t, "stripe", gotSourceName)
	assert.Equal(t, "payment_intent.succeeded", gotEventType)

	// Confirm the inbound event row got the VC ID written back.
	stored, err := provider.GetInboundEvent(ctx, ev.ID)
	require.NoError(t, err)
	require.NotNil(t, stored)
	assert.Equal(t, gotParentVCID, stored.VCID, "inbound_events.vc_id must match the trigger event VC ID propagated to the reasoner")
	assert.Equal(t, types.InboundEventStatusDispatched, stored.Status)

	// And the VC must be retrievable via storage.
	storedVC, err := provider.GetExecutionVC(ctx, gotParentVCID)
	require.NoError(t, err)
	require.NotNil(t, storedVC)
}

// TestDispatcher_DIDDisabled_DispatchesWithoutVC confirms triggers still work
// when DID isn't configured. Dispatcher mints no VC, sends no parent header,
// and the event row's vc_id stays empty — but dispatch succeeds.
func TestDispatcher_DIDDisabled_DispatchesWithoutVC(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	var (
		mu          sync.Mutex
		hadParentVC bool
		invoked     bool
	)
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		invoked = true
		hadParentVC = r.Header.Get("X-Parent-VC-ID") != ""
		w.WriteHeader(http.StatusOK)
	}))
	defer target.Close()

	node := &types.AgentNode{
		ID:              "node-no-did",
		BaseURL:         target.URL,
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "do_thing"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	trig := &types.Trigger{
		ID:             "trg_no_did",
		SourceName:     "generic_hmac",
		TargetNodeID:   "node-no-did",
		TargetReasoner: "do_thing",
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		Config:         json.RawMessage(`{}`),
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	ev := &types.InboundEvent{
		ID:         "evt_no_did",
		TriggerID:  trig.ID,
		SourceName: "generic_hmac",
		RawPayload: json.RawMessage(`{}`),
		Status:     types.InboundEventStatusReceived,
		ReceivedAt: time.Now().UTC(),
	}
	require.NoError(t, provider.InsertInboundEvent(ctx, ev))

	// VCService nil simulates "DID not wired" — dispatcher must handle this
	// without panicking and without setting the parent header.
	disabledCfg := &config.DIDConfig{Enabled: false}
	disabledSvc := &VCService{config: disabledCfg}
	dispatcher := NewTriggerDispatcher(provider, disabledSvc)
	dispatcher.DispatchEvent(ctx, trig, ev)

	mu.Lock()
	defer mu.Unlock()
	assert.True(t, invoked, "dispatch must succeed without DID")
	assert.False(t, hadParentVC, "no X-Parent-VC-ID when DID is disabled")

	stored, err := provider.GetInboundEvent(ctx, ev.ID)
	require.NoError(t, err)
	assert.Empty(t, stored.VCID, "vc_id stays empty when DID is disabled")
	assert.Equal(t, types.InboundEventStatusDispatched, stored.Status)
}

// TestDispatcher_Replay_ReusesOriginalVC confirms a replay does NOT mint a
// fresh trigger event VC — it reuses the original event's VC so the chain
// still terminates at the original signed inbound payload's evidence.
func TestDispatcher_Replay_ReusesOriginalVC(t *testing.T) {
	vcService, didService, provider, ctx := setupVCTestEnvironment(t)

	regResp, err := didService.RegisterAgent(&types.DIDRegistrationRequest{
		AgentNodeID: "node-replay",
		Reasoners:   []types.ReasonerDefinition{{ID: "do_thing"}},
	})
	require.NoError(t, err)
	require.True(t, regResp.Success)

	var (
		mu            sync.Mutex
		gotParentVCID string
	)
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		gotParentVCID = r.Header.Get("X-Parent-VC-ID")
		w.WriteHeader(http.StatusOK)
	}))
	defer target.Close()

	require.NoError(t, provider.RegisterAgent(ctx, &types.AgentNode{
		ID:              "node-replay",
		BaseURL:         target.URL,
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "do_thing"}},
	}))

	trig := &types.Trigger{
		ID:             "trg_replay",
		SourceName:     "stripe",
		TargetNodeID:   "node-replay",
		TargetReasoner: "do_thing",
		ManagedBy:      types.ManagedByCode,
		Enabled:        true,
		Config:         json.RawMessage(`{}`),
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	ev := &types.InboundEvent{
		ID:         "evt_replay_origin",
		TriggerID:  trig.ID,
		SourceName: "stripe",
		EventType:  "payment_intent.succeeded",
		RawPayload: json.RawMessage(`{"id":"evt_replay_origin"}`),
		Status:     types.InboundEventStatusReceived,
		ReceivedAt: time.Now().UTC(),
	}
	require.NoError(t, provider.InsertInboundEvent(ctx, ev))

	dispatcher := NewTriggerDispatcher(provider, vcService)

	// First dispatch — fresh VC minted.
	dispatcher.DispatchEvent(ctx, trig, ev)
	mu.Lock()
	originalVCID := gotParentVCID
	gotParentVCID = ""
	mu.Unlock()
	require.NotEmpty(t, originalVCID)

	// Replay — synthetic event row carrying the original event's VC ID.
	// Mirrors what the ReplayEvent handler produces.
	replay := &types.InboundEvent{
		ID:                "evt_replay_clone",
		TriggerID:         trig.ID,
		SourceName:        ev.SourceName,
		EventType:         ev.EventType,
		RawPayload:        ev.RawPayload,
		NormalizedPayload: ev.NormalizedPayload,
		Status:            types.InboundEventStatusReplayed,
		ReceivedAt:        time.Now().UTC(),
		VCID:              originalVCID, // preserved from the original event
	}
	require.NoError(t, provider.InsertInboundEvent(ctx, replay))

	dispatcher.DispatchEvent(ctx, trig, replay)
	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, originalVCID, gotParentVCID, "replay must reuse the original trigger event VC, not mint a new one")
}
