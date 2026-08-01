package agent

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/types"
	"github.com/stretchr/testify/require"
)

// The shared client is not the only thing that talks to the control plane —
// several paths build their own *http.Request. Those are the ones that used to
// drop the API key, which left an agent registered and running while every
// result it reported came back 401 and the run never finished.
//
// Each test below drives one of those paths against a recording server and
// asserts the credential actually left the process.

// credentialRecorder captures the credential headers of every request it sees.
type credentialRecorder struct {
	mu      sync.Mutex
	apiKeys []string
	auths   []string
	paths   []string
}

func (r *credentialRecorder) record(req *http.Request) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.apiKeys = append(r.apiKeys, req.Header.Get("X-API-Key"))
	r.auths = append(r.auths, req.Header.Get("Authorization"))
	r.paths = append(r.paths, req.URL.Path)
}

func (r *credentialRecorder) seen() ([]string, []string, []string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.apiKeys...),
		append([]string(nil), r.auths...),
		append([]string(nil), r.paths...)
}

// recordingControlPlane answers every request with an empty JSON object, which
// is enough for the fire-and-forget paths and for a single status callback.
func recordingControlPlane(t *testing.T) (*httptest.Server, *credentialRecorder) {
	t.Helper()
	rec := &credentialRecorder{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec.record(r)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{}`))
	}))
	t.Cleanup(srv.Close)
	return srv, rec
}

func newAgentForCredentialTest(t *testing.T, url, apiKey, token string) *Agent {
	t.Helper()
	t.Setenv("AGENTFIELD_API_KEY", "")
	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: url,
		APIKey:        apiKey,
		Token:         token,
	})
	require.NoError(t, err)
	return a
}

// The result of every asynchronous execution is reported over this callback.
// Without the key the control plane rejects it and the run hangs in "running"
// forever, which is how this was found.
func TestSendExecutionStatus_CarriesAPIKey(t *testing.T) {
	srv, rec := recordingControlPlane(t)
	a := newAgentForCredentialTest(t, srv.URL, "config-key", "")

	require.NoError(t, a.sendExecutionStatus("exec-1", map[string]any{"status": "succeeded"}))

	keys, _, paths := rec.seen()
	require.NotEmpty(t, keys)
	require.Contains(t, paths[0], "/api/v1/executions/exec-1/status")
	require.Equal(t, "config-key", keys[0])
}

// Token is a separate credential and must keep working on its own, so an agent
// that only sets Token is unaffected by the change.
func TestSendExecutionStatus_CarriesBearerTokenAlone(t *testing.T) {
	srv, rec := recordingControlPlane(t)
	a := newAgentForCredentialTest(t, srv.URL, "", "bearer-token")

	require.NoError(t, a.sendExecutionStatus("exec-1", map[string]any{"status": "succeeded"}))

	keys, auths, _ := rec.seen()
	require.Equal(t, "Bearer bearer-token", auths[0])
	require.Empty(t, keys[0])
}

// Both credentials may be configured at once; neither suppresses the other.
func TestSendExecutionStatus_CarriesBothCredentials(t *testing.T) {
	srv, rec := recordingControlPlane(t)
	a := newAgentForCredentialTest(t, srv.URL, "config-key", "bearer-token")

	require.NoError(t, a.sendExecutionStatus("exec-1", map[string]any{"status": "succeeded"}))

	keys, auths, _ := rec.seen()
	require.Equal(t, "config-key", keys[0])
	require.Equal(t, "Bearer bearer-token", auths[0])
}

// The default local setup has no credentials at all and must send none — this
// guards the out-of-the-box flow against an accidental empty header.
func TestSendExecutionStatus_SendsNoCredentialHeadersWhenUnset(t *testing.T) {
	srv, rec := recordingControlPlane(t)
	a := newAgentForCredentialTest(t, srv.URL, "", "")

	require.NoError(t, a.sendExecutionStatus("exec-1", map[string]any{"status": "succeeded"}))

	keys, auths, _ := rec.seen()
	require.Empty(t, keys[0])
	require.Empty(t, auths[0])
}

// Agent-to-agent calls submit and then poll; both legs are hand-built requests
// and both need the key or a cross-node call cannot complete.
func TestCallHeaders_CarryAPIKey(t *testing.T) {
	srv, _ := recordingControlPlane(t)
	a := newAgentForCredentialTest(t, srv.URL, "config-key", "")

	req, err := http.NewRequest(http.MethodPost, srv.URL+"/api/v1/execute/async/other.reasoner", nil)
	require.NoError(t, err)
	a.applyCallHeaders(req, ExecutionContext{RunID: "run-1"}, "run-1")

	require.Equal(t, "config-key", req.Header.Get("X-API-Key"))
}

// Workflow events feed the DAG the UI renders; dropping them silently loses
// observability against an authenticated control plane.
func TestSendWorkflowEvent_CarriesAPIKey(t *testing.T) {
	srv, rec := recordingControlPlane(t)
	a := newAgentForCredentialTest(t, srv.URL, "config-key", "")

	require.NoError(t, a.sendWorkflowEvent(types.WorkflowExecutionEvent{
		ExecutionID: "exec-1",
		Status:      "succeeded",
	}))

	keys, _, paths := rec.seen()
	require.NotEmpty(t, keys)
	require.Contains(t, paths[0], "/api/v1/workflow/executions/events")
	require.Equal(t, "config-key", keys[0])
}

// Notes are written from inside a reasoner and are the user's own audit trail.
func TestSendNote_CarriesAPIKey(t *testing.T) {
	srv, rec := recordingControlPlane(t)
	a := newAgentForCredentialTest(t, srv.URL, "config-key", "")

	// Note is fire-and-forget, so wait for the request to actually land.
	a.Note(context.Background(), "hello")
	require.Eventually(t, func() bool {
		keys, _, _ := rec.seen()
		return len(keys) > 0
	}, 5*time.Second, 10*time.Millisecond, "note request never reached the control plane")

	keys, _, paths := rec.seen()
	require.Contains(t, paths[0], "/api/v1/executions/note")
	require.Equal(t, "config-key", keys[0])
}

// Discovery is how an agent finds its peers; without the key it returns an
// auth error instead of the catalog.
func TestDiscover_CarriesAPIKey(t *testing.T) {
	srv, rec := recordingControlPlane(t)
	a := newAgentForCredentialTest(t, srv.URL, "config-key", "")

	_, err := a.Discover(context.Background())
	require.NoError(t, err)

	keys, _, paths := rec.seen()
	require.NotEmpty(t, keys)
	require.Contains(t, paths[0], "/api/v1/discovery/capabilities")
	require.Equal(t, "config-key", keys[0])
}

// A nil request must be tolerated rather than panicking, since the helper is
// called from several builders.
func TestApplyControlPlaneAuth_NilRequestIsSafe(t *testing.T) {
	srv, _ := recordingControlPlane(t)
	a := newAgentForCredentialTest(t, srv.URL, "config-key", "bearer-token")

	require.NotPanics(t, func() { a.applyControlPlaneAuth(nil) })
}

// Local verification refreshes policies and revocations from the control
// plane, and it authenticates with X-API-Key — so the dedicated API key is
// what it should be given.
func TestNew_LocalVerifierUsesAPIKey(t *testing.T) {
	srv, _ := recordingControlPlane(t)
	t.Setenv("AGENTFIELD_API_KEY", "")

	a, err := New(Config{
		NodeID:            "node-1",
		Version:           "1.0.0",
		AgentFieldURL:     srv.URL,
		APIKey:            "config-key",
		LocalVerification: true,
	})
	require.NoError(t, err)
	require.NotNil(t, a.localVerifier)
	require.Equal(t, "config-key", a.localVerifier.apiKey)
}

// Before APIKey existed, callers passed their key as Token and it reached the
// verifier that way. That has to keep working.
func TestNew_LocalVerifierFallsBackToToken(t *testing.T) {
	srv, _ := recordingControlPlane(t)
	t.Setenv("AGENTFIELD_API_KEY", "")

	a, err := New(Config{
		NodeID:            "node-1",
		Version:           "1.0.0",
		AgentFieldURL:     srv.URL,
		Token:             "legacy-token",
		LocalVerification: true,
	})
	require.NoError(t, err)
	require.NotNil(t, a.localVerifier)
	require.Equal(t, "legacy-token", a.localVerifier.apiKey)
}

// DID registration goes through its own client. Without the key an agent on an
// authenticated control plane comes up with no cryptographic identity.
func TestInitializeDIDSystem_CarriesAPIKey(t *testing.T) {
	srv, rec := recordingControlPlane(t)
	a := newAgentForCredentialTest(t, srv.URL, "config-key", "")
	a.cfg.EnableDID = true

	// Registration may fail against the stub response; only the credential on
	// the wire matters here.
	_ = a.initializeDIDSystem(context.Background())

	keys, _, _ := rec.seen()
	require.NotEmpty(t, keys, "DID client never called the control plane")
	require.Equal(t, "config-key", keys[0])
}
