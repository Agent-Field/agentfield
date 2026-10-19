package server

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type sessionAgentCall struct {
	path                     string
	headers                  http.Header
	body                     []byte
	authPresent, authMatched bool
}

type sessionToolFixture struct {
	server *AgentFieldServer
	calls  chan sessionAgentCall
}

func sessionLocalServer(t *testing.T, handler http.Handler) *httptest.Server {
	t.Helper()
	// httptest.NewServer listens on a random 127.0.0.1 port.
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return srv
}

func newSessionToolFixture(t *testing.T, cfg *config.Config) *sessionToolFixture {
	t.Helper()
	gin.SetMode(gin.TestMode)
	// Server Stop (exercised by other tests in this package) retires the
	// process-wide async pool; give each fixture a fresh one either way.
	handlers.ResetAsyncWorkerPoolForTesting()
	t.Cleanup(handlers.ResetAsyncWorkerPoolForTesting)
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("AGENTFIELD_UI_ENABLED", "false")
	ctx := context.Background()
	local := storage.LocalStorageConfig{DatabasePath: filepath.Join(home, "test.db"), KVStorePath: filepath.Join(home, "test.bolt")}
	store := storage.NewLocalStorage(local)
	require.NoError(t, store.Initialize(ctx, storage.StorageConfig{Mode: "local", Local: local}))
	t.Cleanup(func() { require.NoError(t, store.Close(ctx)) })
	calls := make(chan sessionAgentCall, 16)
	agent := sessionLocalServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			w.WriteHeader(400)
			return
		}
		headers := r.Header.Clone()
		present := headers.Get("Authorization") != ""
		matched := headers.Get("Authorization") == "Bearer "+cfg.Features.DID.Authorization.InternalToken
		headers.Del("Authorization")
		calls <- sessionAgentCall{r.URL.Path, headers, body, present, matched}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"answer":"ok"}`))
	}))
	require.NoError(t, store.RegisterAgent(ctx, &types.AgentNode{
		ID: "session-node", BaseURL: agent.URL, HealthStatus: types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady, RegisteredAt: time.Now(), LastHeartbeat: time.Now(),
		Reasoners:    []types.ReasonerDefinition{{ID: "echo", InputSchema: json.RawMessage(`{}`), OutputSchema: json.RawMessage(`{}`)}},
		ApprovedTags: []string{"session-target"},
	}))
	s := &AgentFieldServer{config: cfg, storage: store, Router: gin.New(), payloadStore: services.NewFilePayloadStore(filepath.Join(home, "payloads"))}
	s.accessPolicyService = services.NewAccessPolicyService(store)
	require.NoError(t, s.accessPolicyService.Initialize(ctx))
	s.didWebService = services.NewDIDWebService("local.test", nil, store)
	if cfg.Features.DID.Enabled {
		s.didService = services.NewDIDService(&cfg.Features.DID, nil, services.NewDIDRegistryWithStorage(store))
		require.NoError(t, s.didService.Initialize("session-test"))
		s.vcService = services.NewVCService(&cfg.Features.DID, s.didService, store)
		require.NoError(t, s.vcService.Initialize())
	}
	s.tagVCVerifier = services.NewTagVCVerifier(store, s.vcService)
	s.applyGlobalMiddleware()
	s.initRateLimiters()
	s.registerCoreRoutes(s.Router.Group("/api/v1"))
	t.Cleanup(s.cancelStreams)
	for _, limiter := range []*middleware.RateLimiterStore{s.rateLimitExecute, s.rateLimitGlobal, s.rateLimitDiscovery, s.rateLimitBulkStatus} {
		if limiter != nil {
			t.Cleanup(limiter.Stop)
		}
	}
	return &sessionToolFixture{s, calls}
}

func sessionToolConfig() *config.Config {
	cfg := &config.Config{}
	cfg.AgentField.ExecutionQueue.AgentCallTimeout = 3 * time.Second
	cfg.Features.DID.Authorization.InternalToken = "session-routing-test-credential"
	return cfg
}

func (f *sessionToolFixture) request(t *testing.T, path, body, host string, headers map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if host != "" {
		req.Host = host
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	f.server.Router.ServeHTTP(rec, req)
	return rec
}

func (f *sessionToolFixture) completed(t *testing.T, response *httptest.ResponseRecorder) (*types.Execution, sessionAgentCall) {
	t.Helper()
	require.Equal(t, http.StatusAccepted, response.Code, response.Body.String())
	var result struct {
		ExecutionID string `json:"execution_id"`
		RunID       string `json:"run_id"`
	}
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &result))
	require.NotEmpty(t, result.ExecutionID)
	require.NotEmpty(t, result.RunID)
	var execution *types.Execution
	require.Eventually(t, func() bool {
		var err error
		execution, err = f.server.storage.GetExecutionRecord(context.Background(), result.ExecutionID)
		return err == nil && execution != nil && execution.Status == types.ExecutionStatusSucceeded
	}, 5*time.Second, 5*time.Millisecond, "execution did not succeed")
	require.Equal(t, result.RunID, execution.RunID)
	select {
	case call := <-f.calls:
		return execution, call
	default:
		t.Fatal("completed execution did not dispatch to the registered agent")
		return nil, sessionAgentCall{}
	}
}

// C1, C2, C3, C8: the real registrar/global stack routes only to the registered
// agent, uses only the normal dispatch credential, and preserves execution context.
func TestSessionToolsRoutingAndContext(t *testing.T) {
	f := newSessionToolFixture(t, sessionToolConfig())
	var recorderCalls atomic.Int32
	recorder := sessionLocalServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		recorderCalls.Add(1)
		w.WriteHeader(http.StatusAccepted)
	}))
	hosts := []string{"arbitrary.invalid", strings.TrimPrefix(recorder.URL, "http://"), "localhost:1"}
	for i, host := range hosts {
		for _, bodyTarget := range []bool{true, false} {
			tool, body := "session-node.echo", `{"input":{"msg":"hi"}}`
			if bodyTarget {
				tool, body = "echo", `{"target":"session-node.echo","input":{"msg":"hi"}}`
			}
			runID := fmt.Sprintf("run-session-%d-%t", i, bodyTarget)
			rec := f.request(t, "/api/v1/session-instances/sess-path/tools/"+tool, body, host, map[string]string{
				"X-Session-ID": "caller-session", "X-Run-ID": runID, "X-Parent-Execution-ID": "parent-1", "X-Actor-ID": "actor-1",
				"Authorization": "Bearer caller-credential", "X-API-Key": "caller-key", "X-Admin-Token": "caller-admin", "X-Connector-Token": "caller-connector",
				"X-AgentField-Replay-Source-Run-ID": "missing-run", "X-AgentField-Replay-Before-Execution-ID": "missing-exec", "X-AgentField-Replay-Mode": "strict",
			})
			execution, call := f.completed(t, rec)
			require.Equal(t, "/reasoners/echo", call.path)
			require.JSONEq(t, `{"msg":"hi"}`, string(call.body))
			require.Equal(t, "sess-path", *execution.SessionID)
			require.Equal(t, runID, execution.RunID)
			require.Equal(t, "parent-1", *execution.ParentExecutionID)
			require.Equal(t, "actor-1", *execution.ActorID)
			for header, value := range map[string]string{"X-Session-ID": "sess-path", "X-Run-ID": runID, "X-Parent-Execution-ID": "parent-1", "X-Actor-ID": "actor-1"} {
				require.Equal(t, value, call.headers.Get(header))
			}
			require.True(t, call.authPresent, "Authorization header present: no")
			require.True(t, call.authMatched, "Authorization header matched configured token: no")
			for _, header := range []string{"X-API-Key", "X-Admin-Token", "X-Connector-Token", "X-DID-Signature", "X-AgentField-Replay-Source-Run-ID", "X-AgentField-Replay-Before-Execution-ID", "X-AgentField-Replay-Mode"} {
				require.Empty(t, call.headers.Get(header), header)
			}
		}
	}
	require.Zero(t, recorderCalls.Load(), "Host recorder must receive nothing")
	// C3: an empty body retains downstream input-null handling.
	_, call := f.completed(t, f.request(t, "/api/v1/session-instances/sess-path/tools/session-node.echo", "", "", nil))
	require.JSONEq(t, `{}`, string(call.body))
}

// C4: validation fails before any outbound request through the production route.
func TestSessionToolsValidation(t *testing.T) {
	f := newSessionToolFixture(t, sessionToolConfig())
	for _, tc := range []struct{ body, message string }{
		{"{", "invalid JSON body"},
		{`{"target":"echo"}`, "session tool target must be <node>.<reasoner>"},
	} {
		rec := f.request(t, "/api/v1/session-instances/sess/tools/echo", tc.body, "", nil)
		require.Equal(t, 400, rec.Code)
		expected, _ := json.Marshal(gin.H{"error": tc.message})
		require.JSONEq(t, string(expected), rec.Body.String())
	}
	require.Empty(t, f.calls)
}

// C5: session and direct execute use real local policy/DID services and give the
// same decisions for the resolved reasoner and normalized input.
func TestSessionToolsPermissionParity(t *testing.T) {
	for _, tc := range []struct {
		name, action, errorCode          string
		defaultDeny, pending, constraint bool
		status                           int
	}{
		{name: "deny", action: "deny", status: 403, errorCode: "access_denied"},
		{name: "allow", action: "allow", status: 202},
		{name: "unmatched default deny", defaultDeny: true, status: 403, errorCode: "no_policy_matched"},
		{name: "input constraint", action: "allow", constraint: true, status: 403, errorCode: "access_denied"},
		{name: "pending approval", action: "allow", pending: true, status: 503, errorCode: "agent_pending_approval"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := sessionToolConfig()
			cfg.Features.DID.Authorization.Enabled = true
			cfg.Features.DID.Authorization.DefaultDeny = tc.defaultDeny
			f := newSessionToolFixture(t, cfg)
			if tc.action != "" {
				policy := &types.AccessPolicyRequest{Name: "echo-policy", TargetTags: []string{"session-target"}, AllowFunctions: []string{"echo"}, Action: tc.action}
				if tc.constraint {
					policy.Constraints = map[string]types.AccessConstraint{"limit": {Operator: "<=", Value: 10}}
				}
				_, err := f.server.accessPolicyService.AddPolicy(context.Background(), policy)
				require.NoError(t, err)
			}
			if tc.pending {
				node, err := f.server.storage.GetAgent(context.Background(), "session-node")
				require.NoError(t, err)
				node.LifecycleStatus = types.AgentStatusPendingApproval
				require.NoError(t, f.server.storage.UpdateAgentLifecycleStatus(context.Background(), node.ID, node.LifecycleStatus))
			}
			var deniedBody string
			for _, path := range []string{"/api/v1/execute/async/session-node.echo", "/api/v1/session-instances/sess/tools/echo"} {
				rec := f.request(t, path, `{"target":"session-node.echo","input":{"limit":20}}`, "", nil)
				require.Equal(t, tc.status, rec.Code, rec.Body.String())
				if tc.status == 202 {
					f.completed(t, rec)
				} else {
					require.Contains(t, rec.Body.String(), tc.errorCode)
					if deniedBody != "" {
						require.JSONEq(t, deniedBody, rec.Body.String())
					}
					deniedBody = rec.Body.String()
				}
			}
			require.Empty(t, f.calls)
		})
	}
}

// C6: the original API key middleware runs once; Bearer API keys survive when
// the separate internal dispatch credential is configured.
func TestSessionToolsAPIKeys(t *testing.T) {
	cfg := sessionToolConfig()
	cfg.API.Auth.APIKey = "session-api-key"
	f := newSessionToolFixture(t, cfg)
	for _, headers := range []map[string]string{nil, {"X-API-Key": "session-api-key"}, {"Authorization": "Bearer session-api-key"}} {
		rec := f.request(t, "/api/v1/session-instances/sess/tools/session-node.echo", `{"input":{}}`, "", headers)
		if headers == nil {
			require.Equal(t, 401, rec.Code)
			require.Empty(t, f.calls)
		} else {
			_, call := f.completed(t, rec)
			require.True(t, call.authMatched, "Authorization header matched configured token: no")
			require.Empty(t, call.headers.Get("X-API-Key"))
		}
	}
}

// C7: real Ed25519 DIDAuth verifies the original session body. Permission checks
// resolve that verified caller's tags and prepare propagates that DID to the agent.
func TestSessionToolsVerifiedDID(t *testing.T) {
	cfg := sessionToolConfig()
	cfg.Features.DID.Enabled = true
	cfg.Features.DID.Authorization.Enabled = true
	cfg.Features.DID.Authorization.DIDAuthEnabled = true
	cfg.Features.DID.Authorization.DefaultDeny = true
	f := newSessionToolFixture(t, cfg)
	ctx := context.Background()
	require.NoError(t, f.server.storage.RegisterAgent(ctx, &types.AgentNode{ID: "caller", ApprovedTags: []string{"verified-caller"}}))
	public, private, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)
	jwk, _ := json.Marshal(map[string]string{"kty": "OKP", "crv": "Ed25519", "x": base64.RawURLEncoding.EncodeToString(public)})
	document, err := f.server.didWebService.CreateDIDDocument(ctx, "caller", jwk)
	require.NoError(t, err)
	issuer, err := f.server.didService.GetControlPlaneIssuerDID()
	require.NoError(t, err)
	vc := &types.AgentTagVCDocument{
		Context: []string{"https://www.w3.org/2018/credentials/v1"},
		Type:    []string{"VerifiableCredential", "AgentTagCredential"},
		ID:      "urn:agentfield:session-caller", Issuer: issuer, IssuanceDate: time.Now().UTC().Format(time.RFC3339),
		CredentialSubject: types.AgentTagVCCredentialSubject{ID: document.ID, AgentID: "caller", Permissions: types.AgentTagVCPermissions{Tags: []string{"verified-caller"}}},
	}
	vc.Proof, err = f.server.vcService.SignAgentTagVC(vc)
	require.NoError(t, err)
	vcBytes, err := json.Marshal(vc)
	require.NoError(t, err)
	require.NoError(t, f.server.storage.StoreAgentTagVC(ctx, "caller", document.ID, vc.ID, string(vcBytes), "", time.Now(), nil))

	_, err = f.server.accessPolicyService.AddPolicy(ctx, &types.AccessPolicyRequest{Name: "verified-only", CallerTags: []string{"verified-caller"}, TargetTags: []string{"session-target"}, AllowFunctions: []string{"echo"}, Action: "allow"})
	require.NoError(t, err)
	for i, path := range []string{"/api/v1/execute/async/session-node.echo", "/api/v1/session-instances/sess/tools/echo"} {
		body := `{"target":"session-node.echo","input":{"msg":"signed"}}`
		timestamp, nonce := fmt.Sprint(time.Now().Unix()), fmt.Sprintf("session-%d", i)
		hash := sha256.Sum256([]byte(body))
		signature := ed25519.Sign(private, []byte(fmt.Sprintf("%s:%s:%x", timestamp, nonce, hash)))
		rec := f.request(t, path, body, "", map[string]string{"X-Caller-DID": document.ID, "X-DID-Signature": base64.StdEncoding.EncodeToString(signature), "X-DID-Timestamp": timestamp, "X-DID-Nonce": nonce, "X-Caller-Agent-ID": "unverified-other"})
		_, call := f.completed(t, rec)
		require.Equal(t, document.ID, call.headers.Get("X-Caller-DID"))
		require.Equal(t, f.server.didWebService.GenerateDIDWeb("session-node"), call.headers.Get("X-Target-DID"))
		require.Empty(t, call.headers.Get("X-DID-Signature"))
	}
}

// C9: both routes consume the same limiter keyed by the original caller; a new
// caller has a separate bucket, and the limiter runs before permission checks.
func TestSessionToolsSharedExecuteRateLimit(t *testing.T) {
	for _, first := range []string{"/api/v1/execute/async/session-node.echo", "/api/v1/session-instances/sess/tools/session-node.echo"} {
		t.Run(first, func(t *testing.T) {
			cfg := sessionToolConfig()
			cfg.Features.DID.Authorization.Enabled = true
			cfg.Features.DID.Authorization.DefaultDeny = true
			cfg.AgentField.RateLimit.Enabled = true
			cfg.AgentField.RateLimit.ExecuteRPS = 0.001
			cfg.AgentField.RateLimit.ExecuteBurst = 1
			f := newSessionToolFixture(t, cfg)
			require.Equal(t, 403, f.request(t, first, `{"input":{}}`, "", nil).Code)
			for _, path := range []string{"/api/v1/execute/async/session-node.echo", "/api/v1/session-instances/sess/tools/session-node.echo"} {
				rec := f.request(t, path, `{"input":{}}`, "", nil)
				require.Equal(t, 429, rec.Code, rec.Body.String())
				require.NotEmpty(t, rec.Header().Get("Retry-After"))
			}
			require.Equal(t, 403, f.request(t, first, `{"input":{}}`, "", map[string]string{"X-Forwarded-For": "192.0.2.2"}).Code)

			require.Empty(t, f.calls)
		})
	}
}
