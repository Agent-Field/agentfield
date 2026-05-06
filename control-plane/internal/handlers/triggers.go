package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/sources"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/internal/utils"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// TriggerHandlers groups the HTTP layer for the trigger plugin system. Public
// ingest (`POST /sources/:trigger_id`) is registered separately from the
// authenticated CRUD/UI endpoints because they have different auth posture.
type TriggerHandlers struct {
	storage       storage.StorageProvider
	dispatcher    *services.TriggerDispatcher
	sourceManager *services.SourceManager
}

// NewTriggerHandlers wires the dependencies for trigger HTTP endpoints.
func NewTriggerHandlers(s storage.StorageProvider, d *services.TriggerDispatcher, m *services.SourceManager) *TriggerHandlers {
	return &TriggerHandlers{storage: s, dispatcher: d, sourceManager: m}
}

// ingestRequest is the parsed shape of an inbound trigger POST. We accept any
// content but pass it through to the Source as raw bytes so signature
// verification can run over the exact wire bytes.
type triggerCreateRequest struct {
	SourceName     string          `json:"source_name"`
	Config         json.RawMessage `json:"config"`
	SecretEnvVar   string          `json:"secret_env_var"`
	TargetNodeID   string          `json:"target_node_id"`
	TargetReasoner string          `json:"target_reasoner"`
	EventTypes     []string        `json:"event_types"`
	Enabled        *bool           `json:"enabled"`
}

// IngestSourceHandler is the public ingest endpoint at /sources/:trigger_id.
//
// It looks up the trigger, resolves the Source from the registry, runs the
// Source's verification, persists every emitted event with idempotency
// checking, and hands each event to the dispatcher. The endpoint always
// returns 200 once persistence succeeds — dispatch failures update the
// event row's status but do not propagate back to the provider, since
// providers retry on non-2xx and we already have the event durably stored.
func (h *TriggerHandlers) IngestSourceHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		triggerID := c.Param("trigger_id")
		if triggerID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "trigger_id is required"})
			return
		}

		trig, err := h.storage.GetTrigger(ctx, triggerID)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "trigger not found"})
			return
		}
		if !trig.Enabled {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "trigger disabled"})
			return
		}

		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "failed to read request body"})
			return
		}

		raw := &sources.RawRequest{
			Headers: c.Request.Header,
			Body:    body,
			URL:     c.Request.URL,
			Method:  c.Request.Method,
		}

		secret := ""
		if trig.SecretEnvVar != "" {
			secret = os.Getenv(trig.SecretEnvVar)
		}

		events, err := sources.HandleHTTP(ctx, trig.SourceName, raw, trig.Config, secret)
		if err != nil {
			// Verification failures and unknown sources both return 401 to
			// match webhook-provider expectations: a 4xx means "don't retry".
			logger.Logger.Warn().
				Err(err).
				Str("trigger_id", triggerID).
				Str("source", trig.SourceName).
				Msg("trigger ingest verification failed")
			c.JSON(http.StatusUnauthorized, gin.H{"error": err.Error()})
			return
		}

		// Track per-event outcomes so the response can tell the operator
		// *why* received=0 happened (event-type mismatch vs. idempotency
		// dedup). Without this, sending a `pull_request` to an `issues`-only
		// trigger silently returned 0 with no explanation, which forced
		// operators to read CP logs to diagnose a misconfigured webhook.
		type filteredEntry struct {
			EventType string `json:"event_type"`
			Reason    string `json:"reason"`
		}
		filtered := make([]filteredEntry, 0)
		duplicates := 0
		persisted := 0
		for _, ev := range events {
			if !triggerEventTypeMatches(trig.EventTypes, ev.Type) {
				accepted := strings.Join(trig.EventTypes, ",")
				if accepted == "" {
					accepted = "(none)"
				}
				filtered = append(filtered, filteredEntry{
					EventType: ev.Type,
					Reason:    "event_type not in trigger filter [" + accepted + "]",
				})
				continue
			}
			if ev.IdempotencyKey != "" {
				exists, err := h.storage.InboundEventExistsByIdempotency(ctx, trig.SourceName, ev.IdempotencyKey)
				if err == nil && exists {
					duplicates++
					continue
				}
			}
			rawPayload := ev.Raw
			if len(rawPayload) == 0 {
				rawPayload = json.RawMessage(body)
			}
			normalized := ev.Normalized
			if len(normalized) == 0 {
				normalized = rawPayload
			}
			stored := &types.InboundEvent{
				ID:                utils.GenerateExecutionID(),
				TriggerID:         trig.ID,
				SourceName:        trig.SourceName,
				EventType:         ev.Type,
				RawPayload:        rawPayload,
				NormalizedPayload: normalized,
				IdempotencyKey:    ev.IdempotencyKey,
				Status:            types.InboundEventStatusReceived,
				ReceivedAt:        ev.ReceivedAt,
			}
			if stored.ReceivedAt.IsZero() {
				stored.ReceivedAt = time.Now().UTC()
			}
			if err := h.storage.InsertInboundEvent(ctx, stored); err != nil {
				logger.Logger.Error().
					Err(err).
					Str("trigger_id", triggerID).
					Msg("failed to persist inbound event")
				continue
			}
			persisted++
			// Async dispatch — provider has waited long enough.
			go h.dispatcher.DispatchEvent(context.Background(), trig, stored)
		}

		// Status string is structured so operators don't need to parse
		// counts: "ok" means at least one event was persisted; "filtered"
		// means every event was rejected by the type filter; "duplicate"
		// means every event was a known idempotency dup; "no_events" is
		// an HTTP-200-but-Source-extracted-zero edge case (e.g. Slack
		// URL verification handshake). 200 is preserved across the board
		// so providers don't retry — the response body carries the nuance.
		status := "ok"
		switch {
		case persisted > 0:
			status = "ok"
		case len(filtered) > 0 && duplicates == 0:
			status = "filtered"
		case duplicates > 0 && len(filtered) == 0:
			status = "duplicate"
		case len(filtered) == 0 && duplicates == 0:
			status = "no_events"
		default:
			// Mixed filtered + duplicate — both informative, no canonical
			// single-word summary; fall back to "filtered" since that's the
			// case operators most often need to debug.
			status = "filtered"
		}
		resp := gin.H{
			"status":     status,
			"received":   persisted,
			"duplicates": duplicates,
		}
		if len(filtered) > 0 {
			resp["filtered"] = filtered
		}
		c.JSON(http.StatusOK, resp)
	}
}

// CreateTrigger handles POST /api/v1/triggers (UI-managed instances only).
func (h *TriggerHandlers) CreateTrigger() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req triggerCreateRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		if req.SourceName == "" || req.TargetNodeID == "" || req.TargetReasoner == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "source_name, target_node_id, and target_reasoner are required"})
			return
		}
		src, ok := sources.Get(req.SourceName)
		if !ok {
			c.JSON(http.StatusBadRequest, gin.H{"error": "unknown source: " + req.SourceName})
			return
		}
		cfg := req.Config
		if len(cfg) == 0 {
			cfg = json.RawMessage("{}")
		}
		if err := src.Validate(cfg); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		if src.SecretRequired() && req.SecretEnvVar == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "secret_env_var is required for source " + req.SourceName})
			return
		}

		// Validate accepts_webhook flag on target reasoner
		agent, err := h.storage.GetAgent(c.Request.Context(), req.TargetNodeID)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "target node not found: " + req.TargetNodeID})
			return
		}

		// Find the reasoner in the agent's registered reasoners
		var reasoner *types.ReasonerDefinition
		if agent.Reasoners != nil {
			for i := range agent.Reasoners {
				if agent.Reasoners[i].ID == req.TargetReasoner {
					reasoner = &agent.Reasoners[i]
					break
				}
			}
		}

		if reasoner == nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "target reasoner not found: " + req.TargetReasoner})
			return
		}

		// Check accepts_webhook flag
		if reasoner.AcceptsWebhook != nil && *reasoner.AcceptsWebhook == "false" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "target reasoner has accepts_webhook=false; it explicitly does not accept webhook triggers"})
			return
		}
		if reasoner.AcceptsWebhook != nil && *reasoner.AcceptsWebhook == "warn" || reasoner.AcceptsWebhook == nil {
			logger.Logger.Warn().
				Str("target_node_id", req.TargetNodeID).
				Str("target_reasoner", req.TargetReasoner).
				Msg("UI trigger created for reasoner with accepts_webhook=warn (default); operator should verify reasoner is webhook-compatible")
		}

		enabled := true
		if req.Enabled != nil {
			enabled = *req.Enabled
		}
		now := time.Now().UTC()
		t := &types.Trigger{
			ID:             utils.GenerateExecutionID(),
			SourceName:     req.SourceName,
			Config:         cfg,
			SecretEnvVar:   req.SecretEnvVar,
			TargetNodeID:   req.TargetNodeID,
			TargetReasoner: req.TargetReasoner,
			EventTypes:     req.EventTypes,
			ManagedBy:      types.ManagedByUI,
			Enabled:        enabled,
			CreatedAt:      now,
			UpdatedAt:      now,
		}
		if err := h.storage.CreateTrigger(c.Request.Context(), t); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		if enabled && src.Kind() == sources.KindLoop && h.sourceManager != nil {
			if err := h.sourceManager.Start(t); err != nil {
				logger.Logger.Warn().Err(err).Str("trigger_id", t.ID).Msg("failed to start loop source")
			}
		}
		c.JSON(http.StatusCreated, t)
	}
}

// TriggerListItem wraps a Trigger with per-row 24h activity stats so the
// Active triggers table can render a small "Activity" column without an
// N+1 fetch storm from the client. The stats are computed by walking the
// trigger's recent inbound events (capped at MaxEventsForStats) — cheap on
// SQLite for typical operator deployments.
type TriggerListItem struct {
	*types.Trigger
	EventCount24h      int        `json:"event_count_24h"`
	DispatchSuccess24h int        `json:"dispatch_success_24h"`
	DispatchFailed24h  int        `json:"dispatch_failed_24h"`
	LastEventAt        *time.Time `json:"last_event_at,omitempty"`
	// DispatchBuckets24h is a 24-element histogram. Index 0 = oldest hour
	// (24h ago), index 23 = current hour. Lets the UI render a sparkline
	// without a separate aggregation endpoint.
	DispatchBuckets24h [24]int `json:"dispatch_buckets_24h"`
}

// MaxEventsForStats caps how many recent events we consider when computing
// per-trigger 24h stats. 1000 events / 24h = 41 events/h average — well
// above what an operator surface needs to display, and bounds the worst
// case on busy triggers.
const MaxEventsForStats = 1000

// ListTriggers handles GET /api/v1/triggers.
//
// Returns each trigger with rolling 24h stats (event count, success /
// failure split, last event timestamp, hourly histogram for sparkline)
// so the operator surface can render activity without per-row fetches.
func (h *TriggerHandlers) ListTriggers() gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		nodeID := c.Query("target_node_id")
		source := c.Query("source_name")
		ts, err := h.storage.ListTriggers(ctx, nodeID, source)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		now := time.Now().UTC()
		windowStart := now.Add(-24 * time.Hour)
		items := make([]TriggerListItem, 0, len(ts))
		for _, t := range ts {
			item := TriggerListItem{Trigger: t}
			events, err := h.storage.ListInboundEvents(ctx, t.ID, MaxEventsForStats)
			if err == nil {
				populateTriggerStats(&item, events, now, windowStart)
			}
			items = append(items, item)
		}
		c.JSON(http.StatusOK, gin.H{"triggers": items})
	}
}

// populateTriggerStats fills the 24h stats fields on item from events.
// Events older than windowStart are ignored. Bucket index 23 is the
// current hour; index 0 is 23 hours before that.
func populateTriggerStats(
	item *TriggerListItem,
	events []*types.InboundEvent,
	now, windowStart time.Time,
) {
	for _, ev := range events {
		if ev == nil {
			continue
		}
		if item.LastEventAt == nil || ev.ReceivedAt.After(*item.LastEventAt) {
			ts := ev.ReceivedAt
			item.LastEventAt = &ts
		}
		if ev.ReceivedAt.Before(windowStart) {
			continue
		}
		item.EventCount24h++
		switch ev.Status {
		case types.InboundEventStatusDispatched, types.InboundEventStatusReplayed:
			item.DispatchSuccess24h++
		case types.InboundEventStatusFailed:
			item.DispatchFailed24h++
		}
		// Hour offset: 0 = oldest hour in window, 23 = current hour.
		offset := int(ev.ReceivedAt.Sub(windowStart) / time.Hour)
		if offset < 0 {
			offset = 0
		}
		if offset > 23 {
			offset = 23
		}
		item.DispatchBuckets24h[offset]++
	}
}

// GetTrigger handles GET /api/v1/triggers/:trigger_id.
func (h *TriggerHandlers) GetTrigger() gin.HandlerFunc {
	return func(c *gin.Context) {
		t, err := h.storage.GetTrigger(c.Request.Context(), c.Param("trigger_id"))
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "trigger not found"})
			return
		}
		c.JSON(http.StatusOK, t)
	}
}

// UpdateTrigger handles PUT /api/v1/triggers/:trigger_id. Code-managed
// triggers cannot be edited from the UI; their config is sourced from agent
// registration. The handler rejects writes against them.
func (h *TriggerHandlers) UpdateTrigger() gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		id := c.Param("trigger_id")
		existing, err := h.storage.GetTrigger(ctx, id)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "trigger not found"})
			return
		}
		if existing.ManagedBy == types.ManagedByCode {
			c.JSON(http.StatusForbidden, gin.H{"error": "code-managed trigger cannot be edited via UI"})
			return
		}
		var req triggerCreateRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		if req.SourceName != "" {
			existing.SourceName = req.SourceName
		}
		if len(req.Config) > 0 {
			existing.Config = req.Config
		}
		if req.SecretEnvVar != "" {
			existing.SecretEnvVar = req.SecretEnvVar
		}
		if req.TargetNodeID != "" {
			existing.TargetNodeID = req.TargetNodeID
		}
		if req.TargetReasoner != "" {
			existing.TargetReasoner = req.TargetReasoner
		}
		if req.EventTypes != nil {
			existing.EventTypes = req.EventTypes
		}
		if req.Enabled != nil {
			existing.Enabled = *req.Enabled
		}
		existing.UpdatedAt = time.Now().UTC()

		src, ok := sources.Get(existing.SourceName)
		if !ok {
			c.JSON(http.StatusBadRequest, gin.H{"error": "unknown source: " + existing.SourceName})
			return
		}
		if err := src.Validate(existing.Config); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		if err := h.storage.UpdateTrigger(ctx, existing); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		// Restart the loop runner if the source is loop-kind so config changes
		// take effect immediately without waiting for the next emit cycle.
		if h.sourceManager != nil && src.Kind() == sources.KindLoop {
			h.sourceManager.Stop(existing.ID)
			if existing.Enabled {
				if err := h.sourceManager.Start(existing); err != nil {
					logger.Logger.Warn().Err(err).Str("trigger_id", existing.ID).Msg("failed to restart loop source")
				}
			}
		}

		c.JSON(http.StatusOK, existing)
	}
}

// DeleteTrigger handles DELETE /api/v1/triggers/:trigger_id. Rejects deletion
// of code-managed triggers — they're owned by agent code and re-created on
// the next registration anyway.
func (h *TriggerHandlers) DeleteTrigger() gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		id := c.Param("trigger_id")
		existing, err := h.storage.GetTrigger(ctx, id)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "trigger not found"})
			return
		}
		if existing.ManagedBy == types.ManagedByCode {
			c.JSON(http.StatusForbidden, gin.H{"error": "code-managed trigger cannot be deleted via UI"})
			return
		}
		if h.sourceManager != nil {
			h.sourceManager.Stop(id)
		}
		if err := h.storage.DeleteTrigger(ctx, id); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"status": "deleted"})
	}
}

// PauseTrigger handles POST /api/v1/triggers/:trigger_id/pause. Sets the
// sticky-pause override and disables the trigger atomically. For code-managed
// rows this means the next agent re-registration will NOT silently re-enable
// it.
func (h *TriggerHandlers) PauseTrigger() gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		id := c.Param("trigger_id")
		if _, err := h.storage.GetTrigger(ctx, id); err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "trigger not found"})
			return
		}
		if err := h.storage.SetTriggerOverride(ctx, id, true, false); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		// If the trigger is a loop source, stop its goroutine immediately —
		// pause should be operationally instant, not "wait for next registration".
		if h.sourceManager != nil {
			h.sourceManager.Stop(id)
		}
		c.JSON(http.StatusOK, gin.H{"status": "paused"})
	}
}

// ResumeTrigger handles POST /api/v1/triggers/:trigger_id/resume. Clears the
// sticky-pause override and re-enables the trigger so the row tracks code
// again. For loop sources, restarts the goroutine.
func (h *TriggerHandlers) ResumeTrigger() gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		id := c.Param("trigger_id")
		trig, err := h.storage.GetTrigger(ctx, id)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "trigger not found"})
			return
		}
		if err := h.storage.SetTriggerOverride(ctx, id, false, true); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		// Restart loop source if applicable. Re-fetch so the goroutine sees
		// enabled=true and the cleared override.
		if h.sourceManager != nil {
			if src, ok := sources.Get(trig.SourceName); ok && src.Kind() == sources.KindLoop {
				updated, _ := h.storage.GetTrigger(ctx, id)
				if updated != nil {
					_ = h.sourceManager.Start(updated)
				}
			}
		}
		c.JSON(http.StatusOK, gin.H{"status": "resumed"})
	}
}

// ConvertTriggerToUI handles POST /api/v1/triggers/:trigger_id/convert-to-ui.
// Flips an orphaned code-managed trigger to UI-managed so the operator can
// edit/delete it via the UI without the next agent registration recreating
// it. Only valid on rows where managed_by='code' AND orphaned=true.
func (h *TriggerHandlers) ConvertTriggerToUI() gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		id := c.Param("trigger_id")
		trig, err := h.storage.GetTrigger(ctx, id)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "trigger not found"})
			return
		}
		if trig.ManagedBy != types.ManagedByCode {
			c.JSON(http.StatusBadRequest, gin.H{"error": "trigger is already UI-managed"})
			return
		}
		if !trig.Orphaned {
			c.JSON(http.StatusBadRequest, gin.H{"error": "trigger is not orphaned; only orphaned code-managed rows can be converted"})
			return
		}
		if err := h.storage.ConvertTriggerToUIManaged(ctx, id); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"status": "converted", "managed_by": string(types.ManagedByUI)})
	}
}

// ListTriggerEvents handles GET /api/v1/triggers/:trigger_id/events.
func (h *TriggerHandlers) ListTriggerEvents() gin.HandlerFunc {
	return func(c *gin.Context) {
		id := c.Param("trigger_id")
		limit := 100
		if v := c.Query("limit"); v != "" {
			if n, err := strconv.Atoi(v); err == nil {
				limit = n
			}
		}
		events, err := h.storage.ListInboundEvents(c.Request.Context(), id, limit)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"events": events})
	}
}

// ReplayEvent handles POST /api/v1/triggers/:trigger_id/events/:event_id/replay.
// It re-dispatches a stored event without re-verifying the original signature
// — the assumption is that anyone with UI access has authority to replay.
func (h *TriggerHandlers) ReplayEvent() gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		eventID := c.Param("event_id")
		ev, err := h.storage.GetInboundEvent(ctx, eventID)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "event not found"})
			return
		}
		trig, err := h.storage.GetTrigger(ctx, ev.TriggerID)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "trigger not found"})
			return
		}
		// Mint a new event row so the replay has its own audit trail. Keep the
		// idempotency key cleared so providers' dedup index doesn't reject it,
		// and stamp ReplayOf with the original event's ID so the UI / audit
		// walkers can navigate from "this replay" back to the signed delivery.
		replayed := &types.InboundEvent{
			ID:                utils.GenerateExecutionID(),
			TriggerID:         trig.ID,
			SourceName:        trig.SourceName,
			EventType:         ev.EventType,
			RawPayload:        ev.RawPayload,
			NormalizedPayload: ev.NormalizedPayload,
			IdempotencyKey:    "",
			Status:            types.InboundEventStatusReplayed,
			ReceivedAt:        time.Now().UTC(),
			ReplayOf:          ev.ID,
		}
		if err := h.storage.InsertInboundEvent(ctx, replayed); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		go h.dispatcher.DispatchEvent(context.Background(), trig, replayed)
		c.JSON(http.StatusAccepted, gin.H{
			"event_id":  replayed.ID,
			"replay_of": ev.ID,
		})
	}
}

// ListSources handles GET /api/v1/sources — the catalog endpoint the UI uses
// to populate the "new trigger" form.
func ListSourcesHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"sources": sources.List()})
	}
}

// triggerEventTypeMatches reports whether an inbound event matches one of the
// trigger's configured filters. An empty filter list means "match all".
// GetSource handles GET /api/v1/sources/:name. Returns one Source plugin's
// full metadata: name, kind, secret_required, config_schema.
func (h *TriggerHandlers) GetSource() gin.HandlerFunc {
	return func(c *gin.Context) {
		name := c.Param("name")
		if name == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "source name is required"})
			return
		}
		src, ok := sources.Get(name)
		if !ok {
			c.JSON(http.StatusNotFound, gin.H{"error": "source not found"})
			return
		}
		c.JSON(http.StatusOK, gin.H{
			"name":             src.Name(),
			"kind":             src.Kind().String(),
			"secret_required":  src.SecretRequired(),
			"config_schema":    src.ConfigSchema(),
		})
	}
}

// GetTriggerEvent handles GET /api/v1/triggers/:trigger_id/events/:event_id.
// Returns a single InboundEvent row with full raw + normalized payloads.
// Validates that the event's TriggerID matches the path to prevent enumeration.
func (h *TriggerHandlers) GetTriggerEvent() gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		triggerID := c.Param("trigger_id")
		eventID := c.Param("event_id")

		if triggerID == "" || eventID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "trigger_id and event_id are required"})
			return
		}

		ev, err := h.storage.GetInboundEvent(ctx, eventID)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "event not found"})
			return
		}

		// Prevent enumeration: event must belong to the specified trigger.
		if ev.TriggerID != triggerID {
			c.JSON(http.StatusNotFound, gin.H{"error": "event not found"})
			return
		}

		c.JSON(http.StatusOK, ev)
	}
}

// GetSecretStatus handles GET /api/v1/triggers/:trigger_id/secret-status.
// Returns { env_var: string, set: bool } indicating whether the trigger's
// secret environment variable is set on the control plane host.
func (h *TriggerHandlers) GetSecretStatus() gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		id := c.Param("trigger_id")

		trig, err := h.storage.GetTrigger(ctx, id)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "trigger not found"})
			return
		}

		envVar := trig.SecretEnvVar
		if envVar == "" {
			// Source doesn't require a secret
			c.JSON(http.StatusOK, gin.H{
				"env_var": "",
				"set":     true,
			})
			return
		}

		_, set := os.LookupEnv(envVar)
		c.JSON(http.StatusOK, gin.H{
			"env_var": envVar,
			"set":     set,
		})
	}
}

// testEventPayload is the optional request body for POST /triggers/:trigger_id/test.
type testEventPayload struct {
	Payload   json.RawMessage `json:"payload"`
	EventType string          `json:"event_type"`
}

// TestTrigger handles POST /api/v1/triggers/:trigger_id/test. Operator-initiated
// synthetic event. Manually persists and dispatches a test event.
// Returns { event_id, status }. If source doesn't support synthetic signing,
// returns 501 with a descriptive message. For v1, only generic_hmac and
// generic_bearer are supported; other sources require a captured fixture.
func (h *TriggerHandlers) TestTrigger() gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		id := c.Param("trigger_id")

		trig, err := h.storage.GetTrigger(ctx, id)
		if err != nil {
			c.JSON(http.StatusNotFound, gin.H{"error": "trigger not found"})
			return
		}

		var payload testEventPayload
		if err := c.ShouldBindJSON(&payload); err != nil {
			// Allow empty body (use default payload)
			payload = testEventPayload{
				Payload:   json.RawMessage("{}"),
				EventType: "test",
			}
		}

		if len(payload.Payload) == 0 {
			payload.Payload = json.RawMessage("{}")
		}
		if payload.EventType == "" {
			payload.EventType = "test"
		}

		// Check if source supports synthetic test events
		src, ok := sources.Get(trig.SourceName)
		if !ok {
			c.JSON(http.StatusNotFound, gin.H{"error": "source not found"})
			return
		}

		// For v1, we support generic_hmac and generic_bearer. Other sources
		// like Stripe, GitHub, Cron are not yet supported for synthetic events.
		// FIXME: Implement synthetic signing for Stripe, GitHub, and other HTTP sources.
		if trig.SourceName != "generic_hmac" && trig.SourceName != "generic_bearer" {
			c.JSON(http.StatusNotImplemented, gin.H{
				"error": fmt.Sprintf("synthetic test events not yet supported for source %q — use a captured fixture instead", trig.SourceName),
			})
			return
		}

		// Get the secret
		secret := ""
		if trig.SecretEnvVar != "" {
			secret = os.Getenv(trig.SecretEnvVar)
			if src.SecretRequired() && secret == "" {
				c.JSON(http.StatusBadRequest, gin.H{
					"error": fmt.Sprintf("secret environment variable %q is not set", trig.SecretEnvVar),
				})
				return
			}
		}

		// Build the synthetic event and manually dispatch it
		// (skips signature verification since we trust the operator).
		syntheticEvent := &types.InboundEvent{
			ID:                utils.GenerateExecutionID(),
			TriggerID:         trig.ID,
			SourceName:        trig.SourceName,
			EventType:         payload.EventType,
			RawPayload:        payload.Payload,
			NormalizedPayload: payload.Payload,
			IdempotencyKey:    "", // No dedup for synthetic events
			Status:            types.InboundEventStatusReceived,
			ReceivedAt:        time.Now().UTC(),
		}

		if err := h.storage.InsertInboundEvent(ctx, syntheticEvent); err != nil {
			logger.Logger.Error().
				Err(err).
				Str("trigger_id", id).
				Msg("failed to persist synthetic test event")
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to create test event"})
			return
		}

		// Dispatch asynchronously
		go h.dispatcher.DispatchEvent(context.Background(), trig, syntheticEvent)

		c.JSON(http.StatusAccepted, gin.H{
			"event_id": syntheticEvent.ID,
			"status":   "accepted",
		})
	}
}

func triggerEventTypeMatches(filters []string, eventType string) bool {
	if len(filters) == 0 {
		return true
	}
	for _, f := range filters {
		if f == "*" || f == eventType {
			return true
		}
		// Allow prefix-match on event family, e.g. "pull_request" matches
		// "pull_request.opened".
		if strings.HasSuffix(f, ".*") && strings.HasPrefix(eventType, strings.TrimSuffix(f, ".*")+".") {
			return true
		}
		if !strings.Contains(f, ".") && strings.HasPrefix(eventType, f+".") {
			return true
		}
	}
	return false
}


// GetTriggerMetrics handles GET /api/v1/triggers/metrics — dashboard tile stats
// (global aggregate, not per-trigger).
func (h *TriggerHandlers) GetTriggerMetrics() gin.HandlerFunc {
	return func(c *gin.Context) {
		metrics, err := h.storage.TriggerMetrics(c.Request.Context())
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, metrics)
	}
}
