package services

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/sources"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/internal/utils"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// SourceManager owns the lifecycle of LoopSource (cron, polling) trigger
// instances. It runs one goroutine per enabled loop trigger, listening for
// emitted events and routing them through the same persistence + dispatch
// pipeline as the public HTTP ingest path.
//
// CRUD handlers call Start/Stop when triggers are created, updated, deleted,
// or toggled. Server startup calls LoadAll. Server shutdown calls StopAll.
type SourceManager struct {
	storage    storage.StorageProvider
	dispatcher *TriggerDispatcher

	mu      sync.Mutex
	running map[string]context.CancelFunc
	wg      sync.WaitGroup
}

// NewSourceManager wires a manager to storage and a dispatcher.
func NewSourceManager(storage storage.StorageProvider, dispatcher *TriggerDispatcher) *SourceManager {
	return &SourceManager{
		storage:    storage,
		dispatcher: dispatcher,
		running:    make(map[string]context.CancelFunc),
	}
}

// LoadAll boots every enabled loop trigger currently in storage. Called once
// at server startup so a restart resumes existing schedules without UI action.
func (m *SourceManager) LoadAll(ctx context.Context) error {
	triggers, err := m.storage.ListTriggers(ctx, "", "")
	if err != nil {
		return fmt.Errorf("source manager: list triggers: %w", err)
	}
	for _, t := range triggers {
		if !t.Enabled {
			continue
		}
		s, ok := sources.Get(t.SourceName)
		if !ok {
			logger.Logger.Warn().
				Str("trigger_id", t.ID).
				Str("source", t.SourceName).
				Msg("source manager: skipping trigger with unknown source")
			continue
		}
		if s.Kind() != sources.KindLoop {
			continue
		}
		if err := m.Start(t); err != nil {
			logger.Logger.Warn().
				Err(err).
				Str("trigger_id", t.ID).
				Msg("source manager: failed to start loop trigger")
		}
	}
	return nil
}

// Start spawns a goroutine for one loop trigger. No-op if a goroutine is
// already running for the trigger ID; callers that need a config refresh
// should Stop then Start.
func (m *SourceManager) Start(t *types.Trigger) error {
	if t == nil {
		return errors.New("nil trigger")
	}
	src, ok := sources.Get(t.SourceName)
	if !ok {
		return sources.ErrUnknownSource{Name: t.SourceName}
	}
	loop, ok := src.(sources.LoopSource)
	if !ok {
		return sources.ErrSourceKindMismatch{Name: t.SourceName, Want: "loop"}
	}

	m.mu.Lock()
	if _, exists := m.running[t.ID]; exists {
		m.mu.Unlock()
		return nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	m.running[t.ID] = cancel
	m.wg.Add(1)
	m.mu.Unlock()

	go func() {
		defer m.wg.Done()
		secret := ""
		if t.SecretEnvVar != "" {
			secret = os.Getenv(t.SecretEnvVar)
		}
		// Capture the trigger ID/source for the closure; if storage is updated
		// during the run we'll fetch the freshest target node + reasoner inside
		// dispatcher.DispatchEvent.
		triggerID := t.ID
		sourceName := t.SourceName
		emit := func(ev sources.Event) {
			m.handleEmit(ctx, triggerID, sourceName, ev)
		}
		if err := loop.Run(ctx, t.Config, secret, emit); err != nil && !errors.Is(err, context.Canceled) {
			logger.Logger.Warn().
				Err(err).
				Str("trigger_id", t.ID).
				Str("source", t.SourceName).
				Msg("source manager: loop source exited with error")
		}
	}()
	return nil
}

// Stop cancels the goroutine for a trigger and waits for it to exit. Safe to
// call when the trigger is not running.
func (m *SourceManager) Stop(triggerID string) {
	m.mu.Lock()
	cancel, ok := m.running[triggerID]
	if ok {
		delete(m.running, triggerID)
	}
	m.mu.Unlock()
	if ok {
		cancel()
	}
}

// StopAll cancels every running goroutine and waits for shutdown.
func (m *SourceManager) StopAll() {
	m.mu.Lock()
	cancels := make([]context.CancelFunc, 0, len(m.running))
	for _, c := range m.running {
		cancels = append(cancels, c)
	}
	m.running = map[string]context.CancelFunc{}
	m.mu.Unlock()
	for _, c := range cancels {
		c()
	}
	m.wg.Wait()
}

// handleEmit persists the emitted event and hands it to the dispatcher. It
// re-loads the trigger from storage on each emit so config changes (target
// reasoner change, disable) take effect at the next tick without a restart.
func (m *SourceManager) handleEmit(ctx context.Context, triggerID, sourceName string, ev sources.Event) {
	t, err := m.storage.GetTrigger(ctx, triggerID)
	if err != nil {
		logger.Logger.Warn().
			Err(err).
			Str("trigger_id", triggerID).
			Msg("source manager: trigger lookup failed during emit")
		return
	}
	if !t.Enabled {
		return
	}

	// Idempotency check before insert so we never persist the same scheduled
	// fire twice if the manager bounces during a tick window.
	if ev.IdempotencyKey != "" {
		exists, err := m.storage.InboundEventExistsByIdempotency(ctx, sourceName, ev.IdempotencyKey)
		if err == nil && exists {
			return
		}
	}

	raw := ev.Raw
	if len(raw) == 0 {
		raw = json.RawMessage("{}")
	}
	normalized := ev.Normalized
	if len(normalized) == 0 {
		normalized = raw
	}

	stored := &types.InboundEvent{
		ID:                utils.GenerateExecutionID(),
		TriggerID:         triggerID,
		SourceName:        sourceName,
		EventType:         ev.Type,
		RawPayload:        raw,
		NormalizedPayload: normalized,
		IdempotencyKey:    ev.IdempotencyKey,
		Status:            types.InboundEventStatusReceived,
		ReceivedAt:        ev.ReceivedAt,
	}
	if stored.ReceivedAt.IsZero() {
		stored.ReceivedAt = time.Now().UTC()
	}
	if err := m.storage.InsertInboundEvent(ctx, stored); err != nil {
		logger.Logger.Warn().
			Err(err).
			Str("trigger_id", triggerID).
			Msg("source manager: persist inbound event failed")
		return
	}
	m.dispatcher.DispatchEvent(ctx, t, stored)
}
