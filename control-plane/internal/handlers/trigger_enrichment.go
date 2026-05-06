package handlers

import (
	"context"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// TriggerForExecution traces an execution back through its VC chain to find
// the trigger event that originated it, if any. Returns nil when the
// execution was not webhook-triggered, when the parent VC isn't a
// trigger_event, or when any storage lookup fails.
//
// The traversal is:
//
//	execution_id → execution VC → parent VC → inbound event → trigger row
//
// All errors are swallowed (returns nil). Trigger metadata is best-effort
// presentation data — it must never block the surrounding response.
//
// The same logic was inlined inside ExecutionHandler.enrichExecutionWithTrigger;
// extracting it here lets the runs-list and run-dag handlers populate
// WorkflowRunSummary.Trigger / DAGResponse.Trigger from the same source.
func TriggerForExecution(
	ctx context.Context,
	store storage.StorageProvider,
	executionID string,
) *types.TriggerEventMetadata {
	if store == nil || executionID == "" {
		return nil
	}

	vcs, err := store.ListExecutionVCs(ctx, types.VCFilters{ExecutionID: &executionID, Limit: 1})
	if err != nil || len(vcs) == 0 || vcs[0] == nil {
		return nil
	}
	vc := vcs[0]
	if vc.ParentVCID == nil || *vc.ParentVCID == "" {
		return nil
	}

	parentVC, err := store.GetExecutionVC(ctx, *vc.ParentVCID)
	if err != nil || parentVC == nil {
		return nil
	}
	if parentVC.Kind != types.ExecutionVCKindTriggerEvent {
		return nil
	}
	if parentVC.EventID == nil || *parentVC.EventID == "" {
		return nil
	}

	inboundEvent, err := store.GetInboundEvent(ctx, *parentVC.EventID)
	if err != nil || inboundEvent == nil {
		return nil
	}

	trigger, err := store.GetTrigger(ctx, inboundEvent.TriggerID)
	if err != nil || trigger == nil {
		return nil
	}

	return &types.TriggerEventMetadata{
		TriggerID:      trigger.ID,
		SourceName:     trigger.SourceName,
		EventType:      inboundEvent.EventType,
		EventID:        inboundEvent.ID,
		ReceivedAt:     inboundEvent.ReceivedAt.Format(time.RFC3339),
		IdempotencyKey: inboundEvent.IdempotencyKey,
	}
}

// TriggerForRun looks up the trigger that originated a run, trying the
// direct dispatcher-recorded workflow_id mapping first and falling back to
// the VC chain anchored at the root execution. The direct mapping works
// without DID being wired; the VC path is the canonical provenance.
func TriggerForRun(
	ctx context.Context,
	store storage.StorageProvider,
	workflowID, rootExecutionID string,
) *types.TriggerEventMetadata {
	if meta := triggerByDispatchedWorkflow(ctx, store, workflowID); meta != nil {
		return meta
	}
	return TriggerForExecution(ctx, store, rootExecutionID)
}

// triggerByDispatchedWorkflow does a direct lookup using the workflow ID
// the dispatcher stamped onto the inbound event row at dispatch time.
func triggerByDispatchedWorkflow(
	ctx context.Context,
	store storage.StorageProvider,
	workflowID string,
) *types.TriggerEventMetadata {
	if store == nil || workflowID == "" {
		return nil
	}
	ev, err := store.GetInboundEventByWorkflowID(ctx, workflowID)
	if err != nil || ev == nil {
		return nil
	}
	trigger, err := store.GetTrigger(ctx, ev.TriggerID)
	if err != nil || trigger == nil {
		return nil
	}
	return &types.TriggerEventMetadata{
		TriggerID:      trigger.ID,
		SourceName:     trigger.SourceName,
		EventType:      ev.EventType,
		EventID:        ev.ID,
		ReceivedAt:     ev.ReceivedAt.Format(time.RFC3339),
		IdempotencyKey: ev.IdempotencyKey,
	}
}
