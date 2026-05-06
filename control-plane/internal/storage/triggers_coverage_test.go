package storage

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

func testTrigger(id, source, node, reasoner string) *types.Trigger {
	now := time.Now().UTC()
	return &types.Trigger{
		ID:             id,
		SourceName:     source,
		Config:         json.RawMessage(`{"k":"v"}`),
		SecretEnvVar:   "TRIGGER_SECRET",
		TargetNodeID:   node,
		TargetReasoner: reasoner,
		EventTypes:     []string{"push"},
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		CreatedAt:      now,
		UpdatedAt:      now,
	}
}

func TestLocalStorageTriggerLifecycleCoverage(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	require.EqualError(t, ls.CreateTrigger(ctx, nil), "nil trigger")
	require.EqualError(t, ls.UpdateTrigger(ctx, nil), "nil trigger")
	require.EqualError(t, ls.InsertInboundEvent(ctx, nil), "nil event")

	alpha := testTrigger("trigger-alpha", "generic_bearer", "node-a", "handle")
	beta := testTrigger("trigger-beta", "generic_hmac", "node-b", "handle")
	beta.Config = nil
	beta.EventTypes = nil
	require.NoError(t, ls.CreateTrigger(ctx, alpha))
	require.NoError(t, ls.CreateTrigger(ctx, beta))

	got, err := ls.GetTrigger(ctx, alpha.ID)
	require.NoError(t, err)
	require.Equal(t, alpha.ID, got.ID)
	require.JSONEq(t, string(alpha.Config), string(got.Config))

	filtered, err := ls.ListTriggers(ctx, "node-a", "generic_bearer")
	require.NoError(t, err)
	require.Len(t, filtered, 1)
	require.Equal(t, alpha.ID, filtered[0].ID)

	betaGot, err := ls.GetTrigger(ctx, beta.ID)
	require.NoError(t, err)
	require.JSONEq(t, `{}`, string(betaGot.Config))
	require.Empty(t, betaGot.EventTypes)

	alpha.Enabled = false
	alpha.SecretEnvVar = "OTHER_SECRET"
	require.NoError(t, ls.UpdateTrigger(ctx, alpha))
	updated, err := ls.GetTrigger(ctx, alpha.ID)
	require.NoError(t, err)
	require.False(t, updated.Enabled)
	require.Equal(t, "OTHER_SECRET", updated.SecretEnvVar)

	require.NoError(t, ls.DeleteTrigger(ctx, beta.ID))
	_, err = ls.GetTrigger(ctx, beta.ID)
	require.Error(t, err)
}

func TestLocalStorageCodeManagedTriggerCoverage(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	_, err := ls.UpsertCodeManagedTrigger(ctx, nil)
	require.EqualError(t, err, "nil trigger")
	_, err = ls.UpsertCodeManagedTrigger(ctx, &types.Trigger{
		SourceName:     "stripe",
		TargetNodeID:   "node-code",
		TargetReasoner: "handle",
		Enabled:        true,
	})
	require.EqualError(t, err, "code-managed trigger requires caller-supplied ID for inserts")

	code := testTrigger("code-trigger", "stripe", "node-code", "handle")
	code.ManagedBy = types.ManagedByCode
	code.CodeOrigin = "agent.go:10"
	id, err := ls.UpsertCodeManagedTrigger(ctx, code)
	require.NoError(t, err)
	require.Equal(t, code.ID, id)

	stored, err := ls.GetTrigger(ctx, id)
	require.NoError(t, err)
	require.Equal(t, types.ManagedByCode, stored.ManagedBy)
	require.NotNil(t, stored.LastRegisteredAt)
	require.False(t, stored.Orphaned)
	require.Equal(t, "agent.go:10", stored.CodeOrigin)

	require.NoError(t, ls.SetTriggerOverride(ctx, id, true, false))
	paused, err := ls.GetTrigger(ctx, id)
	require.NoError(t, err)
	require.True(t, paused.ManualOverrideEnabled)
	require.False(t, paused.Enabled)
	require.NotNil(t, paused.ManualOverrideAt)

	redeclared := testTrigger("ignored-new-id", "stripe", "node-code", "handle")
	redeclared.ManagedBy = types.ManagedByCode
	redeclared.Enabled = true
	redeclared.CodeOrigin = "agent.go:20"
	id2, err := ls.UpsertCodeManagedTrigger(ctx, redeclared)
	require.NoError(t, err)
	require.Equal(t, id, id2)
	after, err := ls.GetTrigger(ctx, id)
	require.NoError(t, err)
	require.False(t, after.Enabled, "manual pause survives re-registration")
	require.True(t, after.ManualOverrideEnabled)
	require.Equal(t, "agent.go:20", after.CodeOrigin)

	kept := testTrigger("kept-trigger", "github", "node-code", "handle")
	kept.ManagedBy = types.ManagedByCode
	_, err = ls.UpsertCodeManagedTrigger(ctx, kept)
	require.NoError(t, err)
	require.NoError(t, ls.MarkOrphanedTriggers(ctx, "node-code", []string{"github:handle"}))

	orphaned, err := ls.GetTrigger(ctx, id)
	require.NoError(t, err)
	require.True(t, orphaned.Orphaned)
	keptAfter, err := ls.GetTrigger(ctx, kept.ID)
	require.NoError(t, err)
	require.False(t, keptAfter.Orphaned)

	require.NoError(t, ls.ConvertTriggerToUIManaged(ctx, id))
	converted, err := ls.GetTrigger(ctx, id)
	require.NoError(t, err)
	require.Equal(t, types.ManagedByUI, converted.ManagedBy)
	require.False(t, converted.Orphaned)
	require.ErrorIs(t, ls.ConvertTriggerToUIManaged(ctx, "missing-trigger"), gorm.ErrRecordNotFound)

	require.NoError(t, ls.SetTriggerOverride(ctx, kept.ID, false, true))
	resumed, err := ls.GetTrigger(ctx, kept.ID)
	require.NoError(t, err)
	require.False(t, resumed.ManualOverrideEnabled)
	require.Nil(t, resumed.ManualOverrideAt)
	require.True(t, resumed.Enabled)
}

func TestLocalStorageInboundEventCoverage(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	trigger := testTrigger("event-trigger", "generic_bearer", "node", "handle")
	require.NoError(t, ls.CreateTrigger(ctx, trigger))

	exists, err := ls.InboundEventExistsByIdempotency(ctx, "generic_bearer", "")
	require.NoError(t, err)
	require.False(t, exists)

	payload := json.RawMessage(`{"ok":true}`)
	event := &types.InboundEvent{
		ID:                "event-1",
		TriggerID:         trigger.ID,
		SourceName:        trigger.SourceName,
		EventType:         "push",
		RawPayload:        payload,
		NormalizedPayload: payload,
		IdempotencyKey:    "idem-1",
		Status:            types.InboundEventStatusReceived,
	}
	require.NoError(t, ls.InsertInboundEvent(ctx, event))

	exists, err = ls.InboundEventExistsByIdempotency(ctx, trigger.SourceName, "idem-1")
	require.NoError(t, err)
	require.True(t, exists)

	got, err := ls.GetInboundEvent(ctx, event.ID)
	require.NoError(t, err)
	require.Equal(t, event.ID, got.ID)
	require.False(t, got.ReceivedAt.IsZero())

	events, err := ls.ListInboundEvents(ctx, trigger.ID, 0)
	require.NoError(t, err)
	require.Len(t, events, 1)

	require.NoError(t, ls.MarkInboundEventProcessed(ctx, event.ID, types.InboundEventStatusFailed, "boom", "vc-1"))
	processed, err := ls.GetInboundEvent(ctx, event.ID)
	require.NoError(t, err)
	require.Equal(t, types.InboundEventStatusFailed, processed.Status)
	require.Equal(t, "boom", processed.ErrorMessage)
	require.Equal(t, "vc-1", processed.VCID)

	require.NoError(t, ls.SetInboundEventDispatchedWorkflow(ctx, event.ID, "wf-1"))
	byWorkflow, err := ls.GetInboundEventByWorkflowID(ctx, "wf-1")
	require.NoError(t, err)
	require.Equal(t, event.ID, byWorkflow.ID)
	none, err := ls.GetInboundEventByWorkflowID(ctx, "")
	require.NoError(t, err)
	require.Nil(t, none)

	require.NoError(t, ls.InsertInboundEvent(ctx, &types.InboundEvent{
		ID:         "event-2",
		TriggerID:  trigger.ID,
		SourceName: trigger.SourceName,
		EventType:  "push",
		Status:     types.InboundEventStatusDispatched,
		ReceivedAt: time.Now().UTC(),
	}))
	metrics, err := ls.TriggerMetrics(ctx)
	require.NoError(t, err)
	require.Equal(t, 1, metrics.TotalTriggers)
	require.Equal(t, 1, metrics.EnabledTriggers)
	require.GreaterOrEqual(t, metrics.Events24h, 2)
	require.Greater(t, metrics.DispatchSuccessRate24h, 0.0)

	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	_, err = ls.ListTriggers(cancelled, "", "")
	require.Error(t, err)
}

func TestLocalStorageTriggerErrorBranchesCoverage(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	cancelled, cancel := context.WithCancel(ctx)
	cancel()

	require.Error(t, ls.CreateTrigger(cancelled, testTrigger("cancel-create", "generic_bearer", "node", "handle")))
	_, err := ls.GetTrigger(cancelled, "missing")
	require.Error(t, err)
	_, err = ls.ListTriggers(cancelled, "", "")
	require.Error(t, err)
	require.Error(t, ls.UpdateTrigger(cancelled, testTrigger("cancel-update", "generic_bearer", "node", "handle")))
	require.Error(t, ls.DeleteTrigger(cancelled, "missing"))
	_, err = ls.UpsertCodeManagedTrigger(cancelled, testTrigger("cancel-upsert", "github", "node", "handle"))
	require.Error(t, err)
	require.Error(t, ls.MarkOrphanedTriggers(cancelled, "node", nil))
	require.Error(t, ls.SetTriggerOverride(cancelled, "missing", true, false))
	require.Error(t, ls.ConvertTriggerToUIManaged(cancelled, "missing"))
	require.Error(t, ls.InsertInboundEvent(cancelled, &types.InboundEvent{ID: "cancel-event"}))
	_, err = ls.InboundEventExistsByIdempotency(cancelled, "source", "idem")
	require.Error(t, err)
	_, err = ls.GetInboundEvent(cancelled, "missing")
	require.Error(t, err)
	_, err = ls.ListInboundEvents(cancelled, "missing", 1)
	require.Error(t, err)
	require.Error(t, ls.MarkInboundEventProcessed(cancelled, "missing", types.InboundEventStatusFailed, "nope", ""))
	require.Error(t, ls.SetInboundEventDispatchedWorkflow(cancelled, "missing", "wf"))

	gormDB, err := ls.gormWithContext(ctx)
	require.NoError(t, err)
	now := time.Now().UTC()
	require.NoError(t, gormDB.Create(&TriggerModel{
		ID:             "bad-event-types",
		SourceName:     "generic_bearer",
		ConfigJSON:     "{}",
		TargetNodeID:   "node",
		TargetReasoner: "handle",
		EventTypes:     "{",
		ManagedBy:      string(types.ManagedByUI),
		Enabled:        true,
		CreatedAt:      now,
		UpdatedAt:      now,
	}).Error)
	_, err = ls.GetTrigger(ctx, "bad-event-types")
	require.Error(t, err)
	_, err = ls.ListTriggers(ctx, "", "")
	require.Error(t, err)
}

func TestLocalStorageStoreExecutionVCRecordCoverage(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	require.EqualError(t, ls.StoreExecutionVCRecord(ctx, nil), "execution VC is nil")

	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	require.Error(t, ls.StoreExecutionVCRecord(cancelled, &types.ExecutionVC{VCID: "cancel-vc"}))

	parentID := "parent-vc"
	triggerID := "trigger-vc"
	sourceName := "github"
	eventType := "push"
	eventID := "event-vc"
	created := time.Now().UTC().Truncate(time.Second)
	vc := &types.ExecutionVC{
		VCID:         "vc-record",
		ExecutionID:  "exec-vc-record",
		WorkflowID:   "wf-vc-record",
		SessionID:    "session-vc-record",
		IssuerDID:    "did:issuer",
		TargetDID:    "did:target",
		CallerDID:    "did:caller",
		VCDocument:   json.RawMessage(`{"vc":true}`),
		Signature:    "sig",
		StorageURI:   "file://vc",
		DocumentSize: 11,
		InputHash:    "in",
		OutputHash:   "out",
		Status:       "verified",
		CreatedAt:    created,
		ParentVCID:   &parentID,
		TriggerID:    &triggerID,
		SourceName:   &sourceName,
		EventType:    &eventType,
		EventID:      &eventID,
	}
	require.NoError(t, ls.StoreExecutionVCRecord(ctx, vc))

	stored, err := ls.GetExecutionVC(ctx, vc.VCID)
	require.NoError(t, err)
	require.Equal(t, types.ExecutionVCKindExecution, stored.Kind)
	require.Equal(t, parentID, *stored.ParentVCID)
	require.Equal(t, triggerID, *stored.TriggerID)
	require.Equal(t, sourceName, *stored.SourceName)
	require.Equal(t, eventType, *stored.EventType)
	require.Equal(t, eventID, *stored.EventID)

	vc.Status = "updated"
	vc.Kind = types.ExecutionVCKindTriggerEvent
	vc.Signature = "sig2"
	require.NoError(t, ls.StoreExecutionVCRecord(ctx, vc))
	updated, err := ls.GetExecutionVC(ctx, vc.VCID)
	require.NoError(t, err)
	require.Equal(t, "updated", updated.Status)
	require.Equal(t, types.ExecutionVCKindTriggerEvent, updated.Kind)
}
