package handlers

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/Agent-Field/agentfield/control-plane/internal/sources/all"
)

// TestIngest_FilteredEventTypeSurfacedInResponse pins the contract that
// when a webhook arrives with an event_type that's not in the trigger's
// EventTypes filter, the ingest endpoint:
//   - still returns 200 (so providers don't retry),
//   - returns received=0,
//   - returns status="filtered",
//   - returns a `filtered` array describing why each event was dropped,
//     including the actual event_type and the trigger's accepted list.
//
// Regression target: previously this returned `{"received":0,"status":"ok"}`
// with no diagnosable signal — operators had to read CP server logs to
// figure out their webhook target was misconfigured.
func TestIngest_FilteredEventTypeSurfacedInResponse(t *testing.T) {
	provider, dispatcher, ctx := setupAPIContractTestEnv(t)
	h := NewTriggerHandlers(provider, dispatcher, nil)
	r := triggerCoverageRouter(h)

	require.NoError(t, provider.RegisterAgent(ctx, &types.AgentNode{
		ID:              "filter-target",
		BaseURL:         "http://127.0.0.1:1",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "handle_inbound"}},
	}))

	const secret = "hmac_filter_secret"
	t.Setenv("HMAC_FILTER_SECRET", secret)

	trig := &types.Trigger{
		ID:             "trig-issues-only",
		SourceName:     "generic_hmac",
		Config:         json.RawMessage(`{"event_type_header":"X-Event-Type"}`),
		SecretEnvVar:   "HMAC_FILTER_SECRET",
		TargetNodeID:   "filter-target",
		TargetReasoner: "handle_inbound",
		EventTypes:     []string{"issues"},
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	body := []byte(`{"action":"opened","number":42}`)
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	sig := hex.EncodeToString(mac.Sum(nil))

	req := httptest.NewRequest(http.MethodPost,
		"/sources/"+trig.ID, strings.NewReader(string(body)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Signature", sig)
	req.Header.Set("X-Event-Type", "pull_request") // NOT in EventTypes filter

	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)

	require.Equalf(t, http.StatusOK, rec.Code, "body=%s", rec.Body.String())

	var resp struct {
		Status     string `json:"status"`
		Received   int    `json:"received"`
		Duplicates int    `json:"duplicates"`
		Filtered   []struct {
			EventType string `json:"event_type"`
			Reason    string `json:"reason"`
		} `json:"filtered"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))

	assert.Equal(t, 0, resp.Received, "filtered event must not count as received")
	assert.Equal(t, "filtered", resp.Status,
		"status must be 'filtered' (not 'ok') when every event was dropped by the type filter")
	assert.Equal(t, 0, resp.Duplicates, "no duplicates expected here")
	require.Len(t, resp.Filtered, 1, "every dropped event must be itemized")
	assert.Equal(t, "pull_request", resp.Filtered[0].EventType)
	assert.Contains(t, resp.Filtered[0].Reason, "issues",
		"reason must mention the trigger's accepted event_types so the "+
			"operator knows what filter their webhook is failing")

	// And no row was persisted.
	events, err := provider.ListInboundEvents(ctx, trig.ID, 10)
	require.NoError(t, err)
	assert.Empty(t, events, "filtered events must not be persisted")
}

// TestIngest_DuplicateEventSurfacedAsDuplicateStatus pins that idempotency
// dedup is reported as status=duplicate with a duplicates counter, so the
// operator can tell "this fired before" vs. "this didn't match".
func TestIngest_DuplicateEventSurfacedAsDuplicateStatus(t *testing.T) {
	provider, dispatcher, ctx := setupAPIContractTestEnv(t)
	h := NewTriggerHandlers(provider, dispatcher, nil)
	r := triggerCoverageRouter(h)

	require.NoError(t, provider.RegisterAgent(ctx, &types.AgentNode{
		ID:              "dup-target",
		BaseURL:         "http://127.0.0.1:1",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "handle_inbound"}},
	}))

	const secret = "hmac_dup_secret"
	t.Setenv("HMAC_DUP_SECRET", secret)

	trig := &types.Trigger{
		ID:             "trig-dup",
		SourceName:     "generic_hmac",
		Config:         json.RawMessage(`{"idempotency_header":"X-Idempotency-Key","event_type_header":"X-Event-Type"}`),
		SecretEnvVar:   "HMAC_DUP_SECRET",
		TargetNodeID:   "dup-target",
		TargetReasoner: "handle_inbound",
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	body := []byte(`{"order":"x"}`)
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	sig := hex.EncodeToString(mac.Sum(nil))

	fire := func() *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost,
			"/sources/"+trig.ID, strings.NewReader(string(body)))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Signature", sig)
		req.Header.Set("X-Event-Type", "order.created")
		req.Header.Set("X-Idempotency-Key", "stable-dedup-key-1")
		w := httptest.NewRecorder()
		r.ServeHTTP(w, req)
		return w
	}

	first := fire()
	require.Equalf(t, http.StatusOK, first.Code, "body=%s", first.Body.String())

	var firstBody struct {
		Status   string `json:"status"`
		Received int    `json:"received"`
	}
	require.NoError(t, json.Unmarshal(first.Body.Bytes(), &firstBody))
	assert.Equal(t, "ok", firstBody.Status)
	assert.Equal(t, 1, firstBody.Received)

	second := fire()
	require.Equalf(t, http.StatusOK, second.Code, "body=%s", second.Body.String())

	var secondBody struct {
		Status     string `json:"status"`
		Received   int    `json:"received"`
		Duplicates int    `json:"duplicates"`
	}
	require.NoError(t, json.Unmarshal(second.Body.Bytes(), &secondBody))
	assert.Equal(t, "duplicate", secondBody.Status,
		"duplicate-only response must surface as status='duplicate'")
	assert.Equal(t, 0, secondBody.Received)
	assert.Equal(t, 1, secondBody.Duplicates)
}

// TestIngest_HappyPathStillReturnsStatusOk pins backward compatibility:
// the existing "received >= 1" path must continue to return status="ok".
// Anything that depends on the previous response shape (status==ok ⇒ at
// least one event landed) keeps working.
func TestIngest_HappyPathStillReturnsStatusOk(t *testing.T) {
	provider, dispatcher, ctx := setupAPIContractTestEnv(t)
	h := NewTriggerHandlers(provider, dispatcher, nil)
	r := triggerCoverageRouter(h)

	require.NoError(t, provider.RegisterAgent(ctx, &types.AgentNode{
		ID:              "happy-target",
		BaseURL:         "http://127.0.0.1:1",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "handle_inbound"}},
	}))

	const secret = "hmac_happy_secret"
	t.Setenv("HMAC_HAPPY_SECRET", secret)

	trig := &types.Trigger{
		ID:             "trig-happy",
		SourceName:     "generic_hmac",
		Config:         json.RawMessage(`{}`),
		SecretEnvVar:   "HMAC_HAPPY_SECRET",
		TargetNodeID:   "happy-target",
		TargetReasoner: "handle_inbound",
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	body := []byte(`{"any":"thing"}`)
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	sig := hex.EncodeToString(mac.Sum(nil))

	req := httptest.NewRequest(http.MethodPost,
		"/sources/"+trig.ID, strings.NewReader(string(body)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Signature", sig)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	require.Equalf(t, http.StatusOK, w.Code, "body=%s", w.Body.String())

	var resp struct {
		Status     string `json:"status"`
		Received   int    `json:"received"`
		Duplicates int    `json:"duplicates"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	assert.Equal(t, "ok", resp.Status)
	assert.Equal(t, 1, resp.Received)
	assert.Equal(t, 0, resp.Duplicates)
}
