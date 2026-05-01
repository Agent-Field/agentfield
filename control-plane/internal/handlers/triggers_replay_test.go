package handlers

import (
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/Agent-Field/agentfield/control-plane/internal/sources/all"
)

// TestReplayEvent_StoresReplayOfPointer locks in that the ReplayEvent
// handler stamps the new event's ReplayOf field with the original event's
// ID and returns it in the response body.
//
// Regression target: before the fix the replay row was indistinguishable
// from a fresh delivery, so UI consumers had no way to navigate from the
// replay back to the signed original. The "did this run come from a real
// webhook or a button-click?" question was un-answerable.
func TestReplayEvent_StoresReplayOfPointer(t *testing.T) {
	provider, dispatcher, ctx := setupAPIContractTestEnv(t)
	h := NewTriggerHandlers(provider, dispatcher, nil)
	r := triggerCoverageRouter(h)

	trig := &types.Trigger{
		ID:             "replay-trigger",
		SourceName:     "stripe",
		Config:         json.RawMessage(`{}`),
		SecretEnvVar:   "STRIPE_TEST_SECRET",
		TargetNodeID:   "replay-node",
		TargetReasoner: "handle_payment",
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	original := &types.InboundEvent{
		ID:                "original-event-1",
		TriggerID:         trig.ID,
		SourceName:        trig.SourceName,
		EventType:         "payment_intent.succeeded",
		RawPayload:        json.RawMessage(`{"id":"evt_1"}`),
		NormalizedPayload: json.RawMessage(`{"id":"evt_1"}`),
		IdempotencyKey:    "evt_1",
		Status:            types.InboundEventStatusReceived,
		ReceivedAt:        time.Now().UTC(),
	}
	require.NoError(t, provider.InsertInboundEvent(ctx, original))

	rec := serveJSON(r, http.MethodPost,
		"/api/v1/triggers/"+trig.ID+"/events/"+original.ID+"/replay", nil)
	require.Equalf(t, http.StatusAccepted, rec.Code, "body=%s", rec.Body.String())

	var body struct {
		EventID  string `json:"event_id"`
		ReplayOf string `json:"replay_of"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
	require.NotEmpty(t, body.EventID, "replay must mint a new event id")
	require.NotEqual(t, original.ID, body.EventID, "replay must not reuse the original id")
	assert.Equal(t, original.ID, body.ReplayOf, "response.replay_of must be the original event's id")

	// Storage row carries the same back-pointer.
	persisted, err := provider.GetInboundEvent(ctx, body.EventID)
	require.NoError(t, err)
	assert.Equal(t, original.ID, persisted.ReplayOf,
		"persisted replay row must carry ReplayOf back to the original")
	assert.Empty(t, persisted.IdempotencyKey,
		"replay row must clear idempotency_key so providers' dedup index "+
			"doesn't reject the re-dispatch")
	assert.Equal(t, types.InboundEventStatusReplayed, persisted.Status,
		"replay row should boot in status=replayed")
}
