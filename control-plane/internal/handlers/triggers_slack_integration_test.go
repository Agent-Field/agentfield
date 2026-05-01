package handlers

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	// Import sources to register them
	_ "github.com/Agent-Field/agentfield/control-plane/internal/sources/slack"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// slackSignRequest creates a Slack HMAC-SHA256 signature over "v0:<timestamp>:<body>".
func slackSignRequest(body []byte, secret string, ts int64) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(fmt.Sprintf("v0:%d:", ts)))
	mac.Write(body)
	return "v0=" + hex.EncodeToString(mac.Sum(nil))
}

// setupSlackTestEnv creates storage and dispatcher for Slack webhook tests.
func setupSlackTestEnv(t *testing.T) (storage.StorageProvider, *services.TriggerDispatcher, context.Context) {
	t.Helper()

	ctx := context.Background()
	tempDir := t.TempDir()
	cfg := storage.StorageConfig{
		Mode: "local",
		Local: storage.LocalStorageConfig{
			DatabasePath: filepath.Join(tempDir, "agentfield.db"),
			KVStorePath:  filepath.Join(tempDir, "agentfield.bolt"),
		},
	}

	provider := storage.NewLocalStorage(storage.LocalStorageConfig{})
	require.NoError(t, provider.Initialize(ctx, cfg))

	t.Cleanup(func() {
		_ = provider.Close(ctx)
	})

	// VCService with DID disabled for testing
	disabledCfg := &config.DIDConfig{Enabled: false}
	vcService := services.NewVCService(disabledCfg, nil, provider)
	dispatcher := services.NewTriggerDispatcher(provider, vcService)

	return provider, dispatcher, ctx
}

// TestSlackIngest_AppMentionEvent tests the full ingest flow for an app_mention event.
func TestSlackIngest_AppMentionEvent(t *testing.T) {
	provider, dispatcher, ctx := setupSlackTestEnv(t)
	secret := "slack_test_signing_secret"

	// Set up fake target server
	var (
		mu            sync.Mutex
		targetCalled  bool
		gotSourceName string
		gotEventType  string
		gotEventID    string
	)
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		targetCalled = true
		gotSourceName = r.Header.Get("X-Source-Name")
		gotEventType = r.Header.Get("X-Event-Type")
		gotEventID = r.Header.Get("X-Event-ID")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer target.Close()

	// Register target node
	node := &types.AgentNode{
		ID:              "slack-target",
		BaseURL:         target.URL,
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "handle_mention"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	// Create trigger
	trig := &types.Trigger{
		ID:             "slack_mention_trigger",
		SourceName:     "slack",
		TargetNodeID:   "slack-target",
		TargetReasoner: "handle_mention",
		SecretEnvVar:   "SLACK_TEST_SECRET",
		EventTypes:     []string{"app_mention"},
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		Config:         json.RawMessage(`{}`),
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	// Set up Gin router with ingest handler
	gin.SetMode(gin.TestMode)
	r := gin.New()
	handlers := NewTriggerHandlers(provider, dispatcher, nil)
	r.POST("/sources/:trigger_id", handlers.IngestSourceHandler())

	// Prepare request
	ts := time.Now().Unix()
	body := []byte(`{"type":"event_callback","event_id":"Ev_mention_001","event":{"type":"app_mention","user":"U123","text":"@bot help"}}`)
	sig := slackSignRequest(body, secret, ts)
	t.Setenv("SLACK_TEST_SECRET", secret)

	req := httptest.NewRequest("POST", "/sources/slack_mention_trigger", strings.NewReader(string(body)))
	req.Header.Set("X-Slack-Request-Timestamp", strconv.FormatInt(ts, 10))
	req.Header.Set("X-Slack-Signature", sig)
	req.Header.Set("Content-Type", "application/json")

	// Execute
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	// Assertions
	assert.Equal(t, http.StatusOK, w.Code)
	var resp map[string]interface{}
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	assert.Equal(t, "ok", resp["status"])
	assert.Equal(t, float64(1), resp["received"])

	// Verify inbound event persisted with unwrapped event type
	events, err := provider.ListInboundEvents(ctx, "slack_mention_trigger", 10)
	require.NoError(t, err)
	require.Equal(t, 1, len(events))
	assert.Equal(t, "app_mention", events[0].EventType, "slack should unwrap event.type from event_callback")
	assert.Equal(t, "Ev_mention_001", events[0].IdempotencyKey)

	// Give dispatch time to complete
	time.Sleep(100 * time.Millisecond)

	mu.Lock()
	assert.True(t, targetCalled, "target reasoner should have been invoked")
	assert.Equal(t, "slack", gotSourceName)
	assert.Equal(t, "app_mention", gotEventType)
	assert.Equal(t, events[0].ID, gotEventID)
	mu.Unlock()
}

// TestSlackIngest_URLVerificationChallenge tests the special-case url_verification handshake.
// Note: Slack source emits url_verification as an event. Handler returns 200 before challenge is echoed back.
// FIXME: Handler should respond with the challenge value in the response body per Slack's protocol spec.
func TestSlackIngest_URLVerificationChallenge(t *testing.T) {
	provider, dispatcher, ctx := setupSlackTestEnv(t)
	secret := "slack_test_signing_secret"

	// Trigger configured to NOT receive url_verification events
	trig := &types.Trigger{
		ID:             "slack_urlverify_trigger",
		SourceName:     "slack",
		TargetNodeID:   "slack-target",
		TargetReasoner: "handle_verify",
		SecretEnvVar:   "SLACK_TEST_SECRET",
		EventTypes:     []string{"app_mention"}, // intentionally excludes url_verification
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		Config:         json.RawMessage(`{}`),
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	gin.SetMode(gin.TestMode)
	r := gin.New()
	handlers := NewTriggerHandlers(provider, dispatcher, nil)
	r.POST("/sources/:trigger_id", handlers.IngestSourceHandler())

	ts := time.Now().Unix()
	body := []byte(`{"type":"url_verification","challenge":"3eZbrw1aBm2rZgRNFdxV2595E9CY3gmdALWMmHkvFXO7tYXAYM8P"}`)
	sig := slackSignRequest(body, secret, ts)
	t.Setenv("SLACK_TEST_SECRET", secret)

	req := httptest.NewRequest("POST", "/sources/slack_urlverify_trigger", strings.NewReader(string(body)))
	req.Header.Set("X-Slack-Request-Timestamp", strconv.FormatInt(ts, 10))
	req.Header.Set("X-Slack-Signature", sig)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	// Should return 200 (Slack verification succeeds)
	assert.Equal(t, http.StatusOK, w.Code)
	var resp map[string]interface{}
	_ = json.Unmarshal(w.Body.Bytes(), &resp)

	// Event is emitted but filtered out by EventTypes check
	assert.Equal(t, float64(0), resp["received"], "url_verification should not match trigger event_types")

	// Verify no inbound event was persisted (filtered by event_types)
	events, err := provider.ListInboundEvents(ctx, "slack_urlverify_trigger", 10)
	require.NoError(t, err)
	assert.Equal(t, 0, len(events), "url_verification should be filtered by event_types")
}

// TestSlackIngest_TamperedBody tests rejection of a request with tampered body.
func TestSlackIngest_TamperedBody(t *testing.T) {
	provider, _, ctx := setupSlackTestEnv(t)
	secret := "slack_test_signing_secret"

	node := &types.AgentNode{
		ID:              "slack-tamper-target",
		BaseURL:         "http://localhost:9999",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "handle_tamper"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	trig := &types.Trigger{
		ID:             "slack_tamper_trigger",
		SourceName:     "slack",
		TargetNodeID:   "slack-tamper-target",
		TargetReasoner: "handle_tamper",
		SecretEnvVar:   "SLACK_TEST_SECRET",
		EventTypes:     []string{"app_mention"},
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		Config:         json.RawMessage(`{}`),
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	gin.SetMode(gin.TestMode)
	r := gin.New()
	handlers := NewTriggerHandlers(provider, nil, nil)
	r.POST("/sources/:trigger_id", handlers.IngestSourceHandler())

	ts := time.Now().Unix()
	// Sign correct body
	body := []byte(`{"type":"event_callback","event":{"type":"app_mention","text":"hello"}}`)
	sig := slackSignRequest(body, secret, ts)
	t.Setenv("SLACK_TEST_SECRET", secret)

	// But send tampered body
	tampered := []byte(`{"type":"event_callback","event":{"type":"app_mention","text":"hacked"}}`)
	req := httptest.NewRequest("POST", "/sources/slack_tamper_trigger", strings.NewReader(string(tampered)))
	req.Header.Set("X-Slack-Request-Timestamp", strconv.FormatInt(ts, 10))
	req.Header.Set("X-Slack-Signature", sig)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Contains(t, w.Body.String(), "signature mismatch")

	// Verify no event persisted
	events, err := provider.ListInboundEvents(ctx, "slack_tamper_trigger", 10)
	require.NoError(t, err)
	assert.Equal(t, 0, len(events))
}

// TestSlackIngest_ExpiredTimestamp tests rejection of a request with timestamp outside tolerance window.
func TestSlackIngest_ExpiredTimestamp(t *testing.T) {
	provider, _, ctx := setupSlackTestEnv(t)
	secret := "slack_test_signing_secret"

	node := &types.AgentNode{
		ID:              "slack-expired-target",
		BaseURL:         "http://localhost:9999",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "handle_expired"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	trig := &types.Trigger{
		ID:             "slack_expired_trigger",
		SourceName:     "slack",
		TargetNodeID:   "slack-expired-target",
		TargetReasoner: "handle_expired",
		SecretEnvVar:   "SLACK_TEST_SECRET",
		EventTypes:     []string{"app_mention"},
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		Config:         json.RawMessage(`{}`),
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	gin.SetMode(gin.TestMode)
	r := gin.New()
	handlers := NewTriggerHandlers(provider, nil, nil)
	r.POST("/sources/:trigger_id", handlers.IngestSourceHandler())

	// Timestamp 10 minutes in the past (Slack default tolerance is 5 minutes)
	ts := time.Now().Add(-10 * time.Minute).Unix()
	body := []byte(`{"type":"event_callback","event":{"type":"app_mention","text":"old"}}`)
	sig := slackSignRequest(body, secret, ts)
	t.Setenv("SLACK_TEST_SECRET", secret)

	req := httptest.NewRequest("POST", "/sources/slack_expired_trigger", strings.NewReader(string(body)))
	req.Header.Set("X-Slack-Request-Timestamp", strconv.FormatInt(ts, 10))
	req.Header.Set("X-Slack-Signature", sig)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Contains(t, w.Body.String(), "tolerance")

	// Verify no event persisted
	events, err := provider.ListInboundEvents(ctx, "slack_expired_trigger", 10)
	require.NoError(t, err)
	assert.Equal(t, 0, len(events))
}
