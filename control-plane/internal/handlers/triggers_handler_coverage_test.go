package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"

	_ "github.com/Agent-Field/agentfield/control-plane/internal/sources/all"
)

func triggerCoverageRouter(h *TriggerHandlers) *gin.Engine {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.POST("/sources/:trigger_id", h.IngestSourceHandler())
	r.POST("/api/v1/triggers", h.CreateTrigger())
	r.GET("/api/v1/triggers", h.ListTriggers())
	r.GET("/api/v1/triggers/:trigger_id", h.GetTrigger())
	r.PUT("/api/v1/triggers/:trigger_id", h.UpdateTrigger())
	r.DELETE("/api/v1/triggers/:trigger_id", h.DeleteTrigger())
	r.POST("/api/v1/triggers/:trigger_id/pause", h.PauseTrigger())
	r.POST("/api/v1/triggers/:trigger_id/resume", h.ResumeTrigger())
	r.POST("/api/v1/triggers/:trigger_id/convert-to-ui", h.ConvertTriggerToUI())
	r.GET("/api/v1/triggers/:trigger_id/events", h.ListTriggerEvents())
	r.GET("/api/v1/triggers/:trigger_id/events/:event_id", h.GetTriggerEvent())
	r.POST("/api/v1/triggers/:trigger_id/events/:event_id/replay", h.ReplayEvent())
	r.GET("/api/v1/triggers/:trigger_id/secret-status", h.GetSecretStatus())
	r.POST("/api/v1/triggers/:trigger_id/test", h.TestTrigger())
	r.GET("/api/v1/sources", ListSourcesHandler())
	r.GET("/api/v1/sources/:name", h.GetSource())
	r.GET("/api/v1/triggers/metrics", h.GetTriggerMetrics())
	return r
}

func serveJSON(r http.Handler, method, path string, body any) *httptest.ResponseRecorder {
	var reader *bytes.Reader
	if body == nil {
		reader = bytes.NewReader(nil)
	} else {
		raw, _ := json.Marshal(body)
		reader = bytes.NewReader(raw)
	}
	req := httptest.NewRequest(method, path, reader)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	return rec
}

func TestTriggerHandlersCRUDAndOperationalEndpoints(t *testing.T) {
	provider, dispatcher, ctx := setupAPIContractTestEnv(t)
	h := NewTriggerHandlers(provider, dispatcher, nil)
	r := triggerCoverageRouter(h)

	flagTrue := "true"
	require.NoError(t, provider.RegisterAgent(ctx, &types.AgentNode{
		ID:              "node-create",
		BaseURL:         "http://127.0.0.1:1",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners: []types.ReasonerDefinition{{
			ID:             "handle",
			AcceptsWebhook: &flagTrue,
		}},
	}))

	create := serveJSON(r, http.MethodPost, "/api/v1/triggers", map[string]any{
		"source_name":     "generic_bearer",
		"config":          map[string]any{},
		"target_node_id":  "node-create",
		"target_reasoner": "handle",
		"event_types":     []string{"push", "issue.*"},
		"secret_env_var":  "BEARER_SECRET",
		"enabled":         true,
	})
	require.Equal(t, http.StatusCreated, create.Code, create.Body.String())

	var created types.Trigger
	require.NoError(t, json.Unmarshal(create.Body.Bytes(), &created))
	require.NotEmpty(t, created.ID)

	list := serveJSON(r, http.MethodGet, "/api/v1/triggers?target_node_id=node-create&source_name=generic_bearer", nil)
	require.Equal(t, http.StatusOK, list.Code, list.Body.String())
	require.Contains(t, list.Body.String(), created.ID)

	get := serveJSON(r, http.MethodGet, "/api/v1/triggers/"+created.ID, nil)
	require.Equal(t, http.StatusOK, get.Code, get.Body.String())

	update := serveJSON(r, http.MethodPut, "/api/v1/triggers/"+created.ID, map[string]any{
		"event_types": []string{"push"},
		"enabled":     false,
	})
	require.Equal(t, http.StatusOK, update.Code, update.Body.String())

	pause := serveJSON(r, http.MethodPost, "/api/v1/triggers/"+created.ID+"/pause", nil)
	require.Equal(t, http.StatusOK, pause.Code, pause.Body.String())
	resume := serveJSON(r, http.MethodPost, "/api/v1/triggers/"+created.ID+"/resume", nil)
	require.Equal(t, http.StatusOK, resume.Code, resume.Body.String())

	noSecret := serveJSON(r, http.MethodGet, "/api/v1/triggers/"+created.ID+"/secret-status", nil)
	require.Equal(t, http.StatusOK, noSecret.Code, noSecret.Body.String())

	t.Setenv("BEARER_SECRET", "token")
	testEvent := serveJSON(r, http.MethodPost, "/api/v1/triggers/"+created.ID+"/test", map[string]any{
		"payload":    map[string]any{"hello": "world"},
		"event_type": "push",
	})
	require.Equal(t, http.StatusAccepted, testEvent.Code, testEvent.Body.String())

	events := serveJSON(r, http.MethodGet, "/api/v1/triggers/"+created.ID+"/events?limit=10", nil)
	require.Equal(t, http.StatusOK, events.Code, events.Body.String())

	metrics := serveJSON(r, http.MethodGet, "/api/v1/triggers/metrics", nil)
	require.Equal(t, http.StatusOK, metrics.Code, metrics.Body.String())

	del := serveJSON(r, http.MethodDelete, "/api/v1/triggers/"+created.ID, nil)
	require.Equal(t, http.StatusOK, del.Code, del.Body.String())
}

func TestTriggerHandlersCodeManagedAndEventEndpoints(t *testing.T) {
	provider, dispatcher, ctx := setupAPIContractTestEnv(t)
	h := NewTriggerHandlers(provider, dispatcher, nil)
	r := triggerCoverageRouter(h)

	trig := &types.Trigger{
		ID:             "code-managed-trigger",
		SourceName:     "stripe",
		Config:         json.RawMessage(`{}`),
		TargetNodeID:   "node-code",
		TargetReasoner: "handle",
		ManagedBy:      types.ManagedByCode,
		Enabled:        true,
		Orphaned:       true,
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	for _, tc := range []struct {
		method string
		path   string
		want   int
		body   any
	}{
		{http.MethodPut, "/api/v1/triggers/code-managed-trigger", http.StatusForbidden, map[string]any{"enabled": false}},
		{http.MethodDelete, "/api/v1/triggers/code-managed-trigger", http.StatusForbidden, nil},
		{http.MethodPost, "/api/v1/triggers/code-managed-trigger/convert-to-ui", http.StatusOK, nil},
	} {
		rec := serveJSON(r, tc.method, tc.path, tc.body)
		require.Equal(t, tc.want, rec.Code, rec.Body.String())
	}

	event := &types.InboundEvent{
		ID:                "event-to-replay",
		TriggerID:         trig.ID,
		SourceName:        trig.SourceName,
		EventType:         "payment_intent.succeeded",
		RawPayload:        json.RawMessage(`{"id":"evt"}`),
		NormalizedPayload: json.RawMessage(`{"id":"evt"}`),
		Status:            types.InboundEventStatusReceived,
		ReceivedAt:        time.Now().UTC(),
	}
	require.NoError(t, provider.InsertInboundEvent(ctx, event))

	detail := serveJSON(r, http.MethodGet, "/api/v1/triggers/"+trig.ID+"/events/"+event.ID, nil)
	require.Equal(t, http.StatusOK, detail.Code, detail.Body.String())

	replay := serveJSON(r, http.MethodPost, "/api/v1/triggers/"+trig.ID+"/events/"+event.ID+"/replay", nil)
	require.Equal(t, http.StatusAccepted, replay.Code, replay.Body.String())

	unsupportedTest := serveJSON(r, http.MethodPost, "/api/v1/triggers/"+trig.ID+"/test", nil)
	require.Equal(t, http.StatusNotImplemented, unsupportedTest.Code, unsupportedTest.Body.String())
}

func TestTriggerHandlersValidationAndIngestBranches(t *testing.T) {
	provider, dispatcher, ctx := setupAPIContractTestEnv(t)
	h := NewTriggerHandlers(provider, dispatcher, nil)
	r := triggerCoverageRouter(h)

	require.Equal(t, http.StatusMovedPermanently, serveJSON(r, http.MethodGet, "/api/v1/sources/", nil).Code)
	require.Equal(t, http.StatusOK, serveJSON(r, http.MethodGet, "/api/v1/sources", nil).Code)
	require.Equal(t, http.StatusNotFound, serveJSON(r, http.MethodGet, "/api/v1/sources/missing", nil).Code)

	missingFields := serveJSON(r, http.MethodPost, "/api/v1/triggers", map[string]any{"source_name": "generic_bearer"})
	require.Equal(t, http.StatusBadRequest, missingFields.Code)

	flagFalse := "false"
	require.NoError(t, provider.RegisterAgent(ctx, &types.AgentNode{
		ID:              "node-reject",
		BaseURL:         "http://127.0.0.1:1",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners: []types.ReasonerDefinition{{
			ID:             "handle",
			AcceptsWebhook: &flagFalse,
		}},
	}))
	rejected := serveJSON(r, http.MethodPost, "/api/v1/triggers", map[string]any{
		"source_name":     "generic_bearer",
		"config":          map[string]any{},
		"target_node_id":  "node-reject",
		"target_reasoner": "handle",
		"secret_env_var":  "TOKEN",
	})
	require.Equal(t, http.StatusBadRequest, rejected.Code, rejected.Body.String())

	trig := &types.Trigger{
		ID:             "ingest-trigger",
		SourceName:     "generic_bearer",
		Config:         json.RawMessage(`{}`),
		SecretEnvVar:   "INGEST_TOKEN",
		TargetNodeID:   "node-reject",
		TargetReasoner: "handle",
		EventTypes:     []string{"push"},
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))
	require.NoError(t, provider.SetTriggerOverride(ctx, trig.ID, false, false))

	disabled := serveJSON(r, http.MethodPost, "/sources/"+trig.ID, map[string]any{"x": "y"})
	require.Equal(t, http.StatusServiceUnavailable, disabled.Code, disabled.Body.String())

	trig.Enabled = true
	require.NoError(t, provider.UpdateTrigger(ctx, trig))
	unauthorized := serveJSON(r, http.MethodPost, "/sources/"+trig.ID, map[string]any{"x": "y"})
	require.Equal(t, http.StatusUnauthorized, unauthorized.Code, unauthorized.Body.String())

	require.NoError(t, provider.DeleteTrigger(ctx, trig.ID))
	missing := serveJSON(r, http.MethodPost, "/sources/"+trig.ID, map[string]any{"x": "y"})
	require.Equal(t, http.StatusNotFound, missing.Code, missing.Body.String())
}

func TestTriggerHandlersCreateUpdateValidationBranches(t *testing.T) {
	provider, dispatcher, ctx := setupAPIContractTestEnv(t)
	h := NewTriggerHandlers(provider, dispatcher, nil)
	r := triggerCoverageRouter(h)

	require.Equal(t, http.StatusBadRequest, serveJSON(r, http.MethodPost, "/api/v1/triggers", nil).Code)
	require.Equal(t, http.StatusBadRequest, serveJSON(r, http.MethodPost, "/api/v1/triggers", map[string]any{
		"source_name":     "missing_source",
		"target_node_id":  "node",
		"target_reasoner": "handle",
	}).Code)
	require.Equal(t, http.StatusBadRequest, serveJSON(r, http.MethodPost, "/api/v1/triggers", map[string]any{
		"source_name":     "cron",
		"config":          map[string]any{"expression": "* * *"},
		"target_node_id":  "node",
		"target_reasoner": "handle",
	}).Code)
	require.Equal(t, http.StatusBadRequest, serveJSON(r, http.MethodPost, "/api/v1/triggers", map[string]any{
		"source_name":     "generic_hmac",
		"target_node_id":  "node",
		"target_reasoner": "handle",
	}).Code)
	require.Equal(t, http.StatusNotFound, serveJSON(r, http.MethodPost, "/api/v1/triggers", map[string]any{
		"source_name":     "cron",
		"config":          map[string]any{"expression": "* * * * *"},
		"target_node_id":  "missing-node",
		"target_reasoner": "handle",
	}).Code)

	warn := "warn"
	require.NoError(t, provider.RegisterAgent(ctx, &types.AgentNode{
		ID:              "node-warn",
		BaseURL:         "http://127.0.0.1:1",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners: []types.ReasonerDefinition{
			{ID: "warn", AcceptsWebhook: &warn},
			{ID: "default"},
		},
	}))
	require.Equal(t, http.StatusNotFound, serveJSON(r, http.MethodPost, "/api/v1/triggers", map[string]any{
		"source_name":     "cron",
		"config":          map[string]any{"expression": "* * * * *"},
		"target_node_id":  "node-warn",
		"target_reasoner": "missing",
	}).Code)

	createdWarn := serveJSON(r, http.MethodPost, "/api/v1/triggers", map[string]any{
		"source_name":     "cron",
		"config":          map[string]any{"expression": "* * * * *"},
		"target_node_id":  "node-warn",
		"target_reasoner": "warn",
	})
	require.Equal(t, http.StatusCreated, createdWarn.Code, createdWarn.Body.String())
	var warnTrigger types.Trigger
	require.NoError(t, json.Unmarshal(createdWarn.Body.Bytes(), &warnTrigger))
	require.True(t, warnTrigger.Enabled)

	createdDefault := serveJSON(r, http.MethodPost, "/api/v1/triggers", map[string]any{
		"source_name":     "cron",
		"config":          map[string]any{"expression": "*/5 * * * *"},
		"target_node_id":  "node-warn",
		"target_reasoner": "default",
		"enabled":         false,
	})
	require.Equal(t, http.StatusCreated, createdDefault.Code, createdDefault.Body.String())
	var defaultTrigger types.Trigger
	require.NoError(t, json.Unmarshal(createdDefault.Body.Bytes(), &defaultTrigger))
	require.False(t, defaultTrigger.Enabled)

	require.Equal(t, http.StatusNotFound, serveJSON(r, http.MethodPut, "/api/v1/triggers/missing", map[string]any{"enabled": true}).Code)
	require.Equal(t, http.StatusBadRequest, serveJSON(r, http.MethodPut, "/api/v1/triggers/"+defaultTrigger.ID, nil).Code)
	require.Equal(t, http.StatusBadRequest, serveJSON(r, http.MethodPut, "/api/v1/triggers/"+defaultTrigger.ID, map[string]any{"source_name": "missing_source"}).Code)
	require.Equal(t, http.StatusBadRequest, serveJSON(r, http.MethodPut, "/api/v1/triggers/"+defaultTrigger.ID, map[string]any{
		"source_name": "cron",
		"config":      map[string]any{"expression": "* * *"},
	}).Code)

	update := serveJSON(r, http.MethodPut, "/api/v1/triggers/"+defaultTrigger.ID, map[string]any{
		"source_name":     "cron",
		"config":          map[string]any{"expression": "0 * * * *"},
		"target_node_id":  "node-warn",
		"target_reasoner": "warn",
		"event_types":     []string{"tick"},
		"enabled":         true,
	})
	require.Equal(t, http.StatusOK, update.Code, update.Body.String())
	require.Equal(t, http.StatusNotFound, serveJSON(r, http.MethodDelete, "/api/v1/triggers/missing", nil).Code)
	require.Equal(t, http.StatusNotFound, serveJSON(r, http.MethodPost, "/api/v1/triggers/missing/pause", nil).Code)
	require.Equal(t, http.StatusNotFound, serveJSON(r, http.MethodPost, "/api/v1/triggers/missing/resume", nil).Code)
	require.Equal(t, http.StatusNotFound, serveJSON(r, http.MethodPost, "/api/v1/triggers/missing/convert-to-ui", nil).Code)
	require.Equal(t, http.StatusBadRequest, serveJSON(r, http.MethodPost, "/api/v1/triggers/"+defaultTrigger.ID+"/convert-to-ui", nil).Code)
}

func TestTriggerHandlersIngestSuccessMismatchAndDuplicate(t *testing.T) {
	provider, dispatcher, ctx := setupAPIContractTestEnv(t)
	h := NewTriggerHandlers(provider, dispatcher, nil)
	r := triggerCoverageRouter(h)

	trig := &types.Trigger{
		ID:             "ingest-success",
		SourceName:     "generic_bearer",
		Config:         json.RawMessage(`{"event_type_header":"X-Event-Type","idempotency_header":"X-Delivery"}`),
		SecretEnvVar:   "INGEST_SUCCESS_TOKEN",
		TargetNodeID:   "missing-node",
		TargetReasoner: "handle",
		EventTypes:     []string{"push"},
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))
	t.Setenv("INGEST_SUCCESS_TOKEN", "token")

	post := func(eventType, delivery string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, "/sources/"+trig.ID, bytes.NewBufferString(`{"ok":true}`))
		req.Header.Set("Authorization", "Bearer token")
		req.Header.Set("X-Event-Type", eventType)
		req.Header.Set("X-Delivery", delivery)
		rec := httptest.NewRecorder()
		r.ServeHTTP(rec, req)
		return rec
	}

	mismatch := post("issue.opened", "delivery-mismatch")
	require.Equal(t, http.StatusOK, mismatch.Code, mismatch.Body.String())
	require.Contains(t, mismatch.Body.String(), `"received":0`)

	first := post("push", "delivery-1")
	require.Equal(t, http.StatusOK, first.Code, first.Body.String())
	require.Contains(t, first.Body.String(), `"received":1`)

	duplicate := post("push", "delivery-1")
	require.Equal(t, http.StatusOK, duplicate.Code, duplicate.Body.String())
	require.Contains(t, duplicate.Body.String(), `"received":0`)

	events, err := provider.ListInboundEvents(ctx, trig.ID, 10)
	require.NoError(t, err)
	require.Len(t, events, 1)
	require.Equal(t, "push", events[0].EventType)
}

func TestTriggerHandlersEventSecretAndMatcherBranches(t *testing.T) {
	provider, dispatcher, ctx := setupAPIContractTestEnv(t)
	h := NewTriggerHandlers(provider, dispatcher, nil)
	r := triggerCoverageRouter(h)

	trigger := &types.Trigger{
		ID:             "event-branches",
		SourceName:     "generic_bearer",
		Config:         json.RawMessage(`{}`),
		SecretEnvVar:   "MISSING_SECRET_STATUS",
		TargetNodeID:   "node",
		TargetReasoner: "handle",
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trigger))
	require.Equal(t, http.StatusOK, serveJSON(r, http.MethodGet, "/api/v1/triggers/"+trigger.ID+"/secret-status", nil).Code)
	require.Equal(t, http.StatusNotFound, serveJSON(r, http.MethodGet, "/api/v1/triggers/missing/secret-status", nil).Code)
	require.Equal(t, http.StatusBadRequest, serveJSON(r, http.MethodPost, "/api/v1/triggers/"+trigger.ID+"/test", nil).Code)
	require.Equal(t, http.StatusNotFound, serveJSON(r, http.MethodPost, "/api/v1/triggers/missing/test", nil).Code)

	other := &types.InboundEvent{
		ID:         "other-event",
		TriggerID:  "other-trigger",
		SourceName: "generic_bearer",
		EventType:  "push",
		Status:     types.InboundEventStatusReceived,
		ReceivedAt: time.Now().UTC(),
	}
	require.NoError(t, provider.InsertInboundEvent(ctx, other))
	require.Equal(t, http.StatusNotFound, serveJSON(r, http.MethodGet, "/api/v1/triggers/"+trigger.ID+"/events/missing-event", nil).Code)
	require.Equal(t, http.StatusNotFound, serveJSON(r, http.MethodGet, "/api/v1/triggers/"+trigger.ID+"/events/"+other.ID, nil).Code)
	require.Equal(t, http.StatusNotFound, serveJSON(r, http.MethodPost, "/api/v1/triggers/"+trigger.ID+"/events/missing-event/replay", nil).Code)

	item := TriggerListItem{Trigger: trigger}
	now := time.Now().UTC().Truncate(time.Hour)
	populateTriggerStats(&item, []*types.InboundEvent{
		nil,
		{ID: "old", Status: types.InboundEventStatusDispatched, ReceivedAt: now.Add(-25 * time.Hour)},
		{ID: "ok", Status: types.InboundEventStatusDispatched, ReceivedAt: now.Add(-2 * time.Hour)},
		{ID: "replay", Status: types.InboundEventStatusReplayed, ReceivedAt: now.Add(-1 * time.Hour)},
		{ID: "failed", Status: types.InboundEventStatusFailed, ReceivedAt: now},
	}, now, now.Add(-24*time.Hour))
	require.Equal(t, 3, item.EventCount24h)
	require.Equal(t, 2, item.DispatchSuccess24h)
	require.Equal(t, 1, item.DispatchFailed24h)
	require.NotNil(t, item.LastEventAt)

	require.True(t, triggerEventTypeMatches(nil, "anything"))
	require.True(t, triggerEventTypeMatches([]string{"*"}, "anything"))
	require.True(t, triggerEventTypeMatches([]string{"pull_request.*"}, "pull_request.opened"))
	require.True(t, triggerEventTypeMatches([]string{"pull_request"}, "pull_request.closed"))
	require.False(t, triggerEventTypeMatches([]string{"push"}, "pull_request.opened"))
}

func TestUpsertCodeManagedTriggersBranches(t *testing.T) {
	provider, dispatcher, ctx := setupAPIContractTestEnv(t)
	require.Nil(t, upsertCodeManagedTriggers(ctx, provider, nil))
	require.Nil(t, upsertCodeManagedTriggers(ctx, provider, &types.AgentNode{ID: "node-empty"}))

	node := &types.AgentNode{
		ID:              "node-register-triggers",
		BaseURL:         "http://127.0.0.1:1",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners: []types.ReasonerDefinition{{
			ID: "handle",
			Triggers: []types.TriggerBinding{
				{Source: "missing_source"},
				{Source: "cron", Config: json.RawMessage(`{"expression":"bad"}`)},
				{Source: "generic_bearer", EventTypes: []string{"push"}, SecretEnvVar: "TOKEN", CodeOrigin: "agent.py:10"},
				{Source: "cron", Config: json.RawMessage(`{"expression":"* * * * *"}`), CodeOrigin: "agent.py:11"},
			},
		}},
	}
	manager := services.NewSourceManager(provider, dispatcher)
	SetTriggerSourceManager(manager)
	t.Cleanup(func() {
		manager.StopAll()
		SetTriggerSourceManager(nil)
	})

	entries := upsertCodeManagedTriggers(ctx, provider, node)
	require.Len(t, entries, 2)
	require.Equal(t, "handle", entries[0].ReasonerID)

	triggers, err := provider.ListTriggers(ctx, node.ID, "")
	require.NoError(t, err)
	require.Len(t, triggers, 2)
	for _, trig := range triggers {
		require.Equal(t, types.ManagedByCode, trig.ManagedBy)
		require.False(t, trig.Orphaned)
	}

	node.Reasoners[0].Triggers = []types.TriggerBinding{
		{Source: "generic_bearer", EventTypes: []string{"push"}, SecretEnvVar: "TOKEN", CodeOrigin: "agent.py:20"},
	}
	entries = upsertCodeManagedTriggers(ctx, provider, node)
	require.Len(t, entries, 1)
	after, err := provider.ListTriggers(ctx, node.ID, "")
	require.NoError(t, err)
	require.Len(t, after, 2)
	orphaned := 0
	for _, trig := range after {
		if trig.Orphaned {
			orphaned++
		}
	}
	require.Equal(t, 1, orphaned)
}

func TestTriggerEnrichmentDirectAndVCFallback(t *testing.T) {
	provider, _, ctx := setupAPIContractTestEnv(t)
	now := time.Now().UTC()
	trigger := &types.Trigger{
		ID:             "enrich-trigger",
		SourceName:     "github",
		Config:         json.RawMessage(`{}`),
		TargetNodeID:   "node",
		TargetReasoner: "handle",
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		CreatedAt:      now,
		UpdatedAt:      now,
	}
	require.NoError(t, provider.CreateTrigger(ctx, trigger))
	event := &types.InboundEvent{
		ID:                   "enrich-event",
		TriggerID:            trigger.ID,
		SourceName:           trigger.SourceName,
		EventType:            "push",
		RawPayload:           json.RawMessage(`{}`),
		NormalizedPayload:    json.RawMessage(`{}`),
		IdempotencyKey:       "idem",
		Status:               types.InboundEventStatusDispatched,
		ReceivedAt:           now,
		DispatchedWorkflowID: "wf-direct",
	}
	require.NoError(t, provider.InsertInboundEvent(ctx, event))

	direct := TriggerForRun(ctx, provider, "wf-direct", "missing-exec")
	require.NotNil(t, direct)
	require.Equal(t, trigger.ID, direct.TriggerID)
	require.Equal(t, event.ID, direct.EventID)
	require.Equal(t, "idem", direct.IdempotencyKey)

	parentID := "trigger-vc"
	require.NoError(t, provider.StoreExecutionVCRecord(ctx, &types.ExecutionVC{
		VCID:        parentID,
		ExecutionID: "trigger-event-exec",
		WorkflowID:  "wf-vc",
		SessionID:   "session",
		IssuerDID:   "did:issuer",
		TargetDID:   "did:target",
		CallerDID:   "did:caller",
		InputHash:   "in",
		OutputHash:  "out",
		Status:      "verified",
		VCDocument:  json.RawMessage(`{"vc":true}`),
		Signature:   "sig",
		CreatedAt:   now,
		Kind:        types.ExecutionVCKindTriggerEvent,
		TriggerID:   &trigger.ID,
		SourceName:  &trigger.SourceName,
		EventType:   &event.EventType,
		EventID:     &event.ID,
	}))
	require.NoError(t, provider.StoreExecutionVCRecord(ctx, &types.ExecutionVC{
		VCID:        "child-vc",
		ExecutionID: "exec-child",
		WorkflowID:  "wf-vc",
		SessionID:   "session",
		IssuerDID:   "did:issuer",
		TargetDID:   "did:target",
		CallerDID:   "did:caller",
		InputHash:   "in",
		OutputHash:  "out",
		Status:      "verified",
		VCDocument:  json.RawMessage(`{"vc":true}`),
		Signature:   "sig",
		CreatedAt:   now,
		ParentVCID:  &parentID,
		Kind:        types.ExecutionVCKindExecution,
	}))

	fromVC := TriggerForExecution(ctx, provider, "exec-child")
	require.NotNil(t, fromVC)
	require.Equal(t, trigger.ID, fromVC.TriggerID)
	require.Equal(t, event.EventType, fromVC.EventType)

	require.Nil(t, TriggerForRun(ctx, provider, "", ""))
	require.Nil(t, TriggerForExecution(ctx, provider, ""))
}
