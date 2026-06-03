package handlers

import (
	"context"
	"encoding/json"
	"sync"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/sources"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/internal/utils"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// triggerSourceManager is the singleton SourceManager wired in from server.go
// at startup. Kept package-level so the RegisterNodeHandler closure factory
// (whose signature is shared with other call sites) does not need to grow yet
// another parameter just to start loop-kind triggers on first registration.
var (
	triggerSourceManagerMu sync.RWMutex
	triggerSourceManager   *services.SourceManager
)

// SetTriggerSourceManager wires the source manager so code-managed loop
// triggers (e.g. @on_schedule cron declarations) start running immediately
// when an agent registers, without waiting for a server restart.
func SetTriggerSourceManager(m *services.SourceManager) {
	triggerSourceManagerMu.Lock()
	defer triggerSourceManagerMu.Unlock()
	triggerSourceManager = m
}

func getTriggerSourceManager() *services.SourceManager {
	triggerSourceManagerMu.RLock()
	defer triggerSourceManagerMu.RUnlock()
	return triggerSourceManager
}

// triggerSummaryEntry describes one upserted code-managed trigger to echo back
// to the SDK so it can print the public webhook URL at startup.
type triggerSummaryEntry struct {
	ReasonerID string `json:"reasoner_id"`
	Source     string `json:"source"`
	TriggerID  string `json:"trigger_id"`
}

// upsertCodeManagedTriggers walks each reasoner's declared TriggerBindings and
// idempotently writes a code-managed Trigger row. Validation failures are
// logged and skipped — registration must not fail because of a single bad
// trigger declaration. Returns one entry per successfully upserted binding.
//
// After upserting all bindings, this also calls MarkOrphanedTriggers so any
// previously-declared trigger whose decorator was removed in user code gets
// flagged orphaned (events stop dispatching but row + history are preserved
// for the operator to convert-to-ui or delete via the UI).
func upsertCodeManagedTriggers(ctx context.Context, store storage.StorageProvider, node *types.AgentNode) []triggerSummaryEntry {
	if node == nil || len(node.Reasoners) == 0 {
		return nil
	}
	out := make([]triggerSummaryEntry, 0)
	declaredKeys := make([]string, 0)
	for _, reasoner := range node.Reasoners {
		for _, binding := range reasoner.Triggers {
			src, ok := sources.Get(binding.Source)
			if !ok {
				logger.Logger.Warn().
					Str("node_id", node.ID).
					Str("reasoner_id", reasoner.ID).
					Str("source", binding.Source).
					Msg("trigger registration: skipping binding with unknown source")
				continue
			}
			cfg := binding.Config
			if len(cfg) == 0 {
				cfg = json.RawMessage("{}")
			}
			if err := src.Validate(cfg); err != nil {
				logger.Logger.Warn().
					Err(err).
					Str("node_id", node.ID).
					Str("reasoner_id", reasoner.ID).
					Str("source", binding.Source).
					Msg("trigger registration: invalid config")
				continue
			}
			t := &types.Trigger{
				ID:             utils.GenerateExecutionID(),
				SourceName:     binding.Source,
				Config:         cfg,
				SecretEnvVar:   binding.SecretEnvVar,
				TargetNodeID:   node.ID,
				TargetReasoner: reasoner.ID,
				EventTypes:     binding.EventTypes,
				ManagedBy:      types.ManagedByCode,
				Enabled:        true,
				CodeOrigin:     binding.CodeOrigin,
			}
			id, err := store.UpsertCodeManagedTrigger(ctx, t)
			if err != nil {
				logger.Logger.Warn().
					Err(err).
					Str("node_id", node.ID).
					Str("reasoner_id", reasoner.ID).
					Msg("trigger registration: upsert failed")
				continue
			}
			t.ID = id
			declaredKeys = append(declaredKeys, binding.Source+":"+reasoner.ID)
			// Start loop-kind triggers right away so cron schedules begin
			// firing on first registration, not on the next server restart.
			if mgr := getTriggerSourceManager(); mgr != nil && src.Kind() == sources.KindLoop {
				if err := mgr.Start(t); err != nil {
					logger.Logger.Warn().
						Err(err).
						Str("trigger_id", id).
						Msg("trigger registration: failed to start loop source")
				}
			}
			out = append(out, triggerSummaryEntry{
				ReasonerID: reasoner.ID,
				Source:     binding.Source,
				TriggerID:  id,
			})
		}
	}
	// Flag previously-declared bindings now missing from this registration
	// as orphaned. Best-effort — a failure here doesn't block registration
	// (the agent can still run; the operator just won't see drift signal).
	if err := store.MarkOrphanedTriggers(ctx, node.ID, declaredKeys); err != nil {
		logger.Logger.Warn().
			Err(err).
			Str("node_id", node.ID).
			Msg("trigger registration: orphan-mark failed")
	}
	return out
}
