package handlers

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	// Import sources to register them
	_ "github.com/Agent-Field/agentfield/control-plane/internal/sources/github"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// githubSignRequest creates a GitHub HMAC-SHA256 signature over the body.
func githubSignRequest(body []byte, secret string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	return "sha256=" + hex.EncodeToString(mac.Sum(nil))
}

// setupGitHubTestEnv creates storage and dispatcher for GitHub webhook tests.
func setupGitHubTestEnv(t *testing.T) (storage.StorageProvider, *services.TriggerDispatcher, context.Context) {
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

// TestGitHubIngest_PullRequestOpened tests the full ingest flow for a pull_request.opened event.
func TestGitHubIngest_PullRequestOpened(t *testing.T) {
	provider, dispatcher, ctx := setupGitHubTestEnv(t)
	secret := "github_test_secret_pr"

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
		ID:              "github-target",
		BaseURL:         target.URL,
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "handle_pr"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	// Create trigger
	trig := &types.Trigger{
		ID:             "github_pr_trigger",
		SourceName:     "github",
		TargetNodeID:   "github-target",
		TargetReasoner: "handle_pr",
		SecretEnvVar:   "GITHUB_TEST_SECRET",
		EventTypes:     []string{"pull_request.opened"},
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
	body := []byte(`{"action":"opened","pull_request":{"id":1,"title":"Add features"}}`)
	sig := githubSignRequest(body, secret)
	t.Setenv("GITHUB_TEST_SECRET", secret)

	req := httptest.NewRequest("POST", "/sources/github_pr_trigger", strings.NewReader(string(body)))
	req.Header.Set("X-Hub-Signature-256", sig)
	req.Header.Set("X-GitHub-Event", "pull_request")
	req.Header.Set("X-GitHub-Delivery", "delivery-pr-001")
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

	// Verify inbound event persisted
	events, err := provider.ListInboundEvents(ctx, "github_pr_trigger", 10)
	require.NoError(t, err)
	require.Equal(t, 1, len(events))
	assert.Equal(t, "pull_request.opened", events[0].EventType)
	assert.Equal(t, "delivery-pr-001", events[0].IdempotencyKey)

	// Give dispatch time to complete
	time.Sleep(100 * time.Millisecond)

	mu.Lock()
	assert.True(t, targetCalled, "target reasoner should have been invoked")
	assert.Equal(t, "github", gotSourceName)
	assert.Equal(t, "pull_request.opened", gotEventType)
	assert.Equal(t, events[0].ID, gotEventID)
	mu.Unlock()
}

// TestGitHubIngest_BareEvent_NoAction tests an event without action field (e.g., ping).
func TestGitHubIngest_BareEvent_NoAction(t *testing.T) {
	provider, dispatcher, ctx := setupGitHubTestEnv(t)
	secret := "github_test_secret_ping"

	var (
		mu           sync.Mutex
		targetCalled bool
		gotEventType string
	)
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		targetCalled = true
		gotEventType = r.Header.Get("X-Event-Type")
		w.WriteHeader(http.StatusOK)
	}))
	defer target.Close()

	node := &types.AgentNode{
		ID:              "github-ping-target",
		BaseURL:         target.URL,
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "handle_ping"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	trig := &types.Trigger{
		ID:             "github_ping_trigger",
		SourceName:     "github",
		TargetNodeID:   "github-ping-target",
		TargetReasoner: "handle_ping",
		SecretEnvVar:   "GITHUB_TEST_SECRET",
		EventTypes:     []string{"ping"},
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

	body := []byte(`{"zen":"Approachable is better than simple."}`)
	sig := githubSignRequest(body, secret)
	t.Setenv("GITHUB_TEST_SECRET", secret)

	req := httptest.NewRequest("POST", "/sources/github_ping_trigger", strings.NewReader(string(body)))
	req.Header.Set("X-Hub-Signature-256", sig)
	req.Header.Set("X-GitHub-Event", "ping")
	req.Header.Set("X-GitHub-Delivery", "ping-001")

	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	var resp map[string]interface{}
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	assert.Equal(t, float64(1), resp["received"])

	events, err := provider.ListInboundEvents(ctx, "github_ping_trigger", 10)
	require.NoError(t, err)
	require.Equal(t, 1, len(events))
	assert.Equal(t, "ping", events[0].EventType, "bare event without action should use event type as-is")

	time.Sleep(100 * time.Millisecond)
	mu.Lock()
	assert.True(t, targetCalled)
	assert.Equal(t, "ping", gotEventType)
	mu.Unlock()
}

// TestGitHubIngest_TamperedBody tests rejection of a request with tampered body.
func TestGitHubIngest_TamperedBody(t *testing.T) {
	provider, _, ctx := setupGitHubTestEnv(t)
	secret := "github_test_secret_tampered"

	node := &types.AgentNode{
		ID:              "github-tamper-target",
		BaseURL:         "http://localhost:9999",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "handle_tamper"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	trig := &types.Trigger{
		ID:             "github_tamper_trigger",
		SourceName:     "github",
		TargetNodeID:   "github-tamper-target",
		TargetReasoner: "handle_tamper",
		SecretEnvVar:   "GITHUB_TEST_SECRET",
		EventTypes:     []string{"push"},
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

	// Sign correct body
	body := []byte(`{"action":"push"}`)
	sig := githubSignRequest(body, secret)
	t.Setenv("GITHUB_TEST_SECRET", secret)

	// But send tampered body
	tampered := []byte(`{"action":"force_delete"}`)
	req := httptest.NewRequest("POST", "/sources/github_tamper_trigger", strings.NewReader(string(tampered)))
	req.Header.Set("X-Hub-Signature-256", sig)
	req.Header.Set("X-GitHub-Event", "push")
	req.Header.Set("X-GitHub-Delivery", "tamper-001")

	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Contains(t, w.Body.String(), "signature mismatch")

	// Verify no event persisted
	events, err := provider.ListInboundEvents(ctx, "github_tamper_trigger", 10)
	require.NoError(t, err)
	assert.Equal(t, 0, len(events))
}

// TestGitHubIngest_MissingSignatureHeader tests rejection of a request without signature.
func TestGitHubIngest_MissingSignatureHeader(t *testing.T) {
	provider, _, ctx := setupGitHubTestEnv(t)

	node := &types.AgentNode{
		ID:              "github-nosig-target",
		BaseURL:         "http://localhost:9999",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "handle_nosig"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	trig := &types.Trigger{
		ID:             "github_nosig_trigger",
		SourceName:     "github",
		TargetNodeID:   "github-nosig-target",
		TargetReasoner: "handle_nosig",
		SecretEnvVar:   "GITHUB_TEST_SECRET",
		EventTypes:     []string{"push"},
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

	body := []byte(`{"action":"push"}`)
	t.Setenv("GITHUB_TEST_SECRET", "secret123")

	req := httptest.NewRequest("POST", "/sources/github_nosig_trigger", strings.NewReader(string(body)))
	req.Header.Set("X-GitHub-Event", "push")
	// Intentionally omit X-Hub-Signature-256

	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Contains(t, w.Body.String(), "missing")

	events, err := provider.ListInboundEvents(ctx, "github_nosig_trigger", 10)
	require.NoError(t, err)
	assert.Equal(t, 0, len(events))
}
