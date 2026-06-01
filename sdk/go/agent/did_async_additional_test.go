package agent

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/did"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func testDIDCredentials(t *testing.T) (string, string) {
	t.Helper()

	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)

	jwkJSON, err := json.Marshal(map[string]string{
		"kty": "OKP",
		"crv": "Ed25519",
		"d":   base64.RawURLEncoding.EncodeToString(priv.Seed()),
		"x":   base64.RawURLEncoding.EncodeToString(pub),
	})
	require.NoError(t, err)

	return "did:web:example.com:agents:test-agent", string(jwkJSON)
}

func TestInitializeDIDSystem_ExistingCredentialsEnableVCAndFillContext(t *testing.T) {
	agentDID, privateKeyJWK := testDIDCredentials(t)

	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "http://example.com",
		DID:           agentDID,
		PrivateKeyJWK: privateKeyJWK,
		VCEnabled:     true,
		Logger:        log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)

	require.NoError(t, a.initializeDIDSystem(context.Background()))
	require.NotNil(t, a.DIDManager())
	require.NotNil(t, a.VCGenerator())
	assert.True(t, a.DIDManager().IsRegistered())
	assert.Equal(t, agentDID, a.DIDManager().GetAgentDID())
	assert.True(t, a.VCGenerator().IsEnabled())

	execCtx := ExecutionContext{}
	a.fillDIDContext(&execCtx)
	assert.Equal(t, agentDID, execCtx.AgentNodeDID)

	assert.True(t, a.shouldGenerateVC(nil))
	disabled := false
	enabled := true
	assert.False(t, a.shouldGenerateVC(&Reasoner{VCEnabled: &disabled}))
	assert.True(t, a.shouldGenerateVC(&Reasoner{VCEnabled: &enabled}))
}

func TestInitializeDIDSystem_AutoRegistersWhenCredentialsMissing(t *testing.T) {
	agentDID, privateKeyJWK := testDIDCredentials(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/api/v1/did/register", r.URL.Path)
		require.Equal(t, http.MethodPost, r.Method)

		var req did.RegistrationRequest
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		assert.Equal(t, "node-1", req.AgentNodeID)
		require.Len(t, req.Reasoners, 1)
		assert.Equal(t, "demo", req.Reasoners[0].ID)

		w.Header().Set("Content-Type", "application/json")
		require.NoError(t, json.NewEncoder(w).Encode(did.RegistrationResponse{
			Success: true,
			IdentityPackage: did.DIDIdentityPackage{
				AgentDID: did.DIDIdentity{
					DID:           agentDID,
					PrivateKeyJWK: privateKeyJWK,
				},
				ReasonerDIDs: map[string]did.DIDIdentity{},
				SkillDIDs:    map[string]did.DIDIdentity{},
			},
		}))
	}))
	defer server.Close()

	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: server.URL,
		EnableDID:     true,
		VCEnabled:     true,
		Logger:        log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	a.RegisterReasoner("demo", func(ctx context.Context, input map[string]any) (any, error) {
		return nil, nil
	})

	require.NoError(t, a.initializeDIDSystem(context.Background()))
	require.NotNil(t, a.didManager)
	assert.True(t, a.didManager.IsRegistered())
	assert.Equal(t, agentDID, a.didManager.GetAgentDID())
	require.NotNil(t, a.vcGenerator)
	assert.True(t, a.vcGenerator.IsEnabled())
}

func TestMaybeGenerateVC_EmitsRequestWhenEnabled(t *testing.T) {
	requestCh := make(chan did.VCGenerationRequest, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/api/v1/execution/vc", r.URL.Path)
		var req did.VCGenerationRequest
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		requestCh <- req
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"vc_id":"vc-1","execution_id":"exec-1","workflow_id":"wf-1","issuer_did":"did:issuer","target_did":"did:target","status":"succeeded","created_at":"now"}`)
	}))
	defer server.Close()

	agentDID, privateKeyJWK := testDIDCredentials(t)
	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: server.URL,
		DID:           agentDID,
		PrivateKeyJWK: privateKeyJWK,
		VCEnabled:     true,
		Logger:        log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	require.NoError(t, a.initializeDIDSystem(context.Background()))

	a.maybeGenerateVC(ExecutionContext{
		ExecutionID: "exec-1",
		WorkflowID:  "wf-1",
		SessionID:   "session-1",
		TargetDID:   "did:target",
	}, map[string]any{"input": true}, map[string]any{"output": true}, "succeeded", "", 12, &Reasoner{Name: "demo"})

	select {
	case req := <-requestCh:
		assert.Equal(t, "exec-1", req.ExecutionContext.ExecutionID)
		assert.Equal(t, "wf-1", req.ExecutionContext.WorkflowID)
		assert.Equal(t, "session-1", req.ExecutionContext.SessionID)
		assert.Equal(t, agentDID, req.ExecutionContext.AgentNodeDID)
		assert.NotEmpty(t, req.ExecutionContext.Timestamp)
		assert.NotEmpty(t, req.InputData)
		assert.NotEmpty(t, req.OutputData)
		assert.Equal(t, "succeeded", req.Status)
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for VC request")
	}
}

func TestExecuteReasonerAsync_ReportsErrorAndPanicStatuses(t *testing.T) {
	statusCh := make(chan map[string]any, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Contains(t, r.URL.Path, "/api/v1/executions/")
		switch {
		case strings.HasSuffix(r.URL.Path, "/status"):
			var payload map[string]any
			require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
			statusCh <- payload
			w.WriteHeader(http.StatusNoContent)
		case strings.HasSuffix(r.URL.Path, "/logs"):
			w.WriteHeader(http.StatusNoContent)
		default:
			t.Fatalf("unexpected async callback path %s", r.URL.Path)
		}
	}))
	defer server.Close()

	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: server.URL,
		Logger:        log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	a.httpClient = server.Client()

	baseExecCtx := ExecutionContext{
		ExecutionID: "exec-1",
		RunID:       "run-1",
		WorkflowID:  "wf-1",
	}

	a.executeReasonerAsync(&Reasoner{
		Name: "returns-error",
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return nil, fmt.Errorf("boom")
		},
	}, map[string]any{"x": 1}, baseExecCtx)

	a.executeReasonerAsync(&Reasoner{
		Name: "panics",
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			panic("kaboom")
		},
	}, map[string]any{"x": 2}, ExecutionContext{
		ExecutionID: "exec-2",
		RunID:       "run-2",
		WorkflowID:  "wf-2",
	})

	var got []map[string]any
	for i := 0; i < 2; i++ {
		select {
		case payload := <-statusCh:
			got = append(got, payload)
		case <-time.After(2 * time.Second):
			t.Fatal("timed out waiting for async status")
		}
	}

	assert.ElementsMatch(t, []string{"failed", "failed"}, []string{got[0]["status"].(string), got[1]["status"].(string)})
	assert.Contains(t, fmt.Sprint(got[0]["error"])+fmt.Sprint(got[1]["error"]), "boom")
	assert.Contains(t, fmt.Sprint(got[0]["error"])+fmt.Sprint(got[1]["error"]), "kaboom")
}

func TestWaitForApproval_ReturnsCancellationError(t *testing.T) {
	a := &Agent{
		cfg:    Config{NodeID: "node-1"},
		logger: log.New(io.Discard, "", 0),
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := a.waitForApproval(ctx)
	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)
}

func TestStartLeaseLoop_PeriodicallyCallsMarkReady(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPatch && r.URL.Path == "/api/v1/nodes/node-1/status" {
			calls.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"lease_seconds":120}`)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	a, err := New(Config{
		NodeID:               "node-1",
		Version:              "1.0.0",
		AgentFieldURL:        server.URL,
		LeaseRefreshInterval: 10 * time.Millisecond,
		Logger:               log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	a.startLeaseLoop()

	require.Eventually(t, func() bool {
		return calls.Load() > 0
	}, 500*time.Millisecond, 10*time.Millisecond)

	close(a.stopLease)
}
