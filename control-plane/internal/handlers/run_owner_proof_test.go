package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/nodelink"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
)

func ownerProofRequest(method, path, owner, proof, body string) *http.Request {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if owner != "" {
		req.Header.Set(RunOwnerHeader, owner)
	}
	if proof != "" {
		req.Header.Set(RunOwnerProofHeader, proof)
	}
	return req
}

func mintTestOwnerProof(t *testing.T, signer *nodelink.ProofSigner, owner, run string, scopes ...string) string {
	t.Helper()
	token, _, err := signer.Mint(owner, scopes, run, 0)
	require.NoError(t, err)
	return token
}

func TestRootOwnerProofContractA3A4A5A6A7A7bA8A9A13A14(t *testing.T) {
	const owner = "my-app"
	var mu sync.Mutex
	var headers []http.Header
	node := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		headers = append(headers, r.Header.Clone())
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer node.Close()
	r := newRelayRoutesRig(t, node.URL)
	require.NoError(t, r.store.RegisterAgent(t.Context(), &types.AgentNode{
		ID: "executor", Version: "1.0.0", BaseURL: node.URL, HealthStatus: types.HealthStatusActive, LastHeartbeat: time.Now(),
		Reasoners: []types.ReasonerDefinition{{ID: "intake"}},
		Skills:    []types.SkillDefinition{{ID: "prepare"}},
	}))
	r.router.Use(middleware.GinLogger())
	r.router.POST("/api/v1/execute/:target", ExecuteHandler(r.store, services.NewFilePayloadStore(t.TempDir()), nil, time.Second, ""))
	r.router.POST("/api/v1/reasoners/:reasoner_id", ExecuteReasonerHandler(r.store))
	r.router.POST("/api/v1/skills/:skill_id", ExecuteSkillHandler(r.store))

	signer := nodelink.NewProofSigner([]byte("example signing key"))
	oldSigner := nodelink.NewProofSigner([]byte("example signing key"))
	oldSigner.Now = func() time.Time { return time.Now().Add(-2 * time.Hour) }
	proof := mintTestOwnerProof(t, signer, owner, "", nodelink.ScopeOwn)
	expired := mintTestOwnerProof(t, oldSigner, owner, "", nodelink.ScopeOwn)
	other := mintTestOwnerProof(t, signer, "other-app", "", nodelink.ScopeOwn)
	noScope := mintTestOwnerProof(t, signer, owner, "", nodelink.ScopeRead)
	parts := strings.Split(proof, ".")
	first := byte('A')
	if parts[2][0] == first {
		first = 'B'
	}
	tampered := strings.Join([]string{parts[0], parts[1], string(first) + parts[2][1:]}, ".")

	var logs bytes.Buffer
	oldLogger := logger.Logger
	logger.Logger = zerolog.New(zerolog.SyncWriter(&logs)).Level(zerolog.DebugLevel)
	t.Cleanup(func() { logger.Logger = oldLogger })
	t.Cleanup(func() { SetRunOwnerPolicy(nil) })

	for i, path := range []string{"/api/v1/execute/executor.intake", "/api/v1/reasoners/executor.intake", "/api/v1/skills/executor.prepare"} {
		for _, strict := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/strict=%t", path, strict), func(t *testing.T) {
				SetRunOwnerPolicy(&nodelink.OwnerPolicy{Signer: signer, RequireProof: strict})
				suffix := fmt.Sprintf("-%d-%t", i, strict)
				run := "bound" + suffix
				bound := mintTestOwnerProof(t, signer, owner, run, nodelink.ScopeOwn)
				missing := ""
				if strict {
					missing = "run_owner_proof_required"
				}
				tests := []struct {
					name, run, owner, proof, refusal string
					nested, omitRunHeader            bool
				}{
					{name: "unproven", run: "unproven" + suffix, owner: owner, refusal: missing},
					{name: "proven", run: "proven" + suffix, owner: owner, proof: proof},
					{name: "expired", run: "expired" + suffix, owner: owner, proof: expired, refusal: "run_owner_proof_expired"},
					{name: "wrong owner", run: "wrong-owner" + suffix, owner: owner, proof: other, refusal: "run_owner_proof_invalid"},
					{name: "lacks own", run: "no-scope" + suffix, owner: owner, proof: noScope, refusal: "run_owner_proof_invalid"},
					{name: "bad signature", run: "bad-signature" + suffix, owner: owner, proof: tampered, refusal: "run_owner_proof_invalid"},
					{name: "run bound", run: run, owner: owner, proof: bound},
					{name: "another run", run: "another" + suffix, owner: owner, proof: bound, refusal: "run_owner_proof_invalid"},
					{name: "no caller run", run: "implicit" + suffix, owner: owner, proof: bound, omitRunHeader: true, refusal: "run_owner_proof_invalid"},
					{name: "nested", run: run, owner: "other-app", proof: tampered, nested: true},
					{name: "operator", run: "operator" + suffix, proof: tampered},
				}
				for _, tc := range tests {
					t.Run(tc.name, func(t *testing.T) {
						req := ownerProofRequest(http.MethodPost, path, tc.owner, tc.proof, `{"input":{}}`)
						if !tc.omitRunHeader {
							req.Header.Set("X-Run-ID", tc.run)
						}
						req.Header.Set("X-Workflow-ID", tc.run)
						if tc.nested {
							parent, err := r.store.QueryExecutionRecords(t.Context(), types.ExecutionFilter{RunID: &tc.run})
							require.NoError(t, err)
							require.Len(t, parent, 1)
							req.Header.Set("X-Parent-Execution-ID", parent[0].ExecutionID)
						}
						before, err := r.store.QueryExecutionRecords(t.Context(), types.ExecutionFilter{})
						require.NoError(t, err)
						rec := httptest.NewRecorder()
						r.router.ServeHTTP(rec, req)
						records, err := r.store.QueryExecutionRecords(t.Context(), types.ExecutionFilter{})
						require.NoError(t, err)
						if tc.refusal != "" {
							requireOwnerHTTPRefusal(t, rec, tc.refusal)
							require.Len(t, records, len(before))
						} else {
							require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())
							require.Len(t, records, len(before)+1)
							want := tc.owner
							if tc.nested {
								want = owner
							}
							got, err := r.store.RunOwner(t.Context(), tc.run)
							require.NoError(t, err)
							require.Equal(t, want, got)
						}
						if tc.proof != "" {
							require.NotContains(t, rec.Body.String(), tc.proof)
							require.NotContains(t, fmt.Sprint(rec.Header()), tc.proof)
						}
					})
				}
			})
		}
	}

	SetRunOwnerPolicy(nil)
	req := ownerProofRequest(http.MethodPost, "/api/v1/execute/executor.intake", owner, tampered, `{"input":{}}`)
	rec := httptest.NewRecorder()
	r.router.ServeHTTP(rec, req)
	require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())
	mu.Lock()
	require.NotEmpty(t, headers)
	for _, header := range headers {
		require.Empty(t, header.Values(RunOwnerProofHeader))
	}
	mu.Unlock()
	SetRunOwnerPolicy(&nodelink.OwnerPolicy{Signer: signer, RequireProof: true})
	query := url.Values{RunOwnerProofHeader: {proof}}.Encode()
	req = ownerProofRequest(http.MethodPost, "/api/v1/execute/executor.intake?"+query, owner, "", `{"input":{}}`)
	rec = httptest.NewRecorder()
	r.router.ServeHTTP(rec, req)
	requireOwnerHTTPRefusal(t, rec, "run_owner_proof_required")
	for _, token := range []string{proof, expired, other, noScope, tampered} {
		require.NotContains(t, logs.String(), token)
	}
}

func requireOwnerHTTPRefusal(t *testing.T, rec *httptest.ResponseRecorder, code string) {
	t.Helper()
	require.Equal(t, http.StatusForbidden, rec.Code, rec.Body.String())
	var body map[string]string
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
	require.Len(t, body, 1)
	require.True(t, strings.HasPrefix(body["error"], code+": "), body)
}

func TestOwnerProofHTTPActionsContractA5A6A7A7bA9A14(t *testing.T) {
	node := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"accepted":true}`))
	}))
	defer node.Close()

	const owner, run, exec = "my-app", "my-run", "my-exec"
	signer := nodelink.NewProofSigner([]byte("example signing key"))
	t.Cleanup(func() { SetRunOwnerPolicy(nil) })
	for _, strict := range []bool{false, true} {
		t.Run(fmt.Sprintf("strict=%t", strict), func(t *testing.T) {
			r := newRelayRoutesRig(t, node.URL)
			SetRunOwnerPolicy(&nodelink.OwnerPolicy{Signer: signer, RequireProof: strict})
			r.seedRunning(run, exec, owner)
			r.router.POST("/api/v1/executions/:execution_id/request-approval", RequestApprovalHandler(r.store))
			r.router.POST("/api/v1/executions/:execution_id/approval-response", ResolveApprovalHandler(r.store, r.pipe))
			runID := run
			require.NoError(t, r.store.StoreWorkflowExecution(t.Context(), &types.WorkflowExecution{
				ExecutionID: exec, WorkflowID: run, RunID: &runID, AgentNodeID: "worker", ReasonerID: "intake",
				Status: types.ExecutionStatusRunning, StartedAt: time.Now(),
			}))
			base := "/api/v1/executions/" + exec
			post := func(path, body string) {
				t.Helper()
				code, out := r.doRaw(http.MethodPost, path, "", body)
				require.Equal(t, http.StatusOK, code, out)
			}
			post(base+"/request-approval", `{"kind":"question","question":"Continue?"}`)

			tokens := map[string]string{}
			for _, scope := range []string{nodelink.ScopeOwn, nodelink.ScopeAct, nodelink.ScopeRead} {
				tokens[scope] = mintTestOwnerProof(t, signer, owner, run, scope)
			}
			for _, action := range []struct {
				method, resource, body, scope string
			}{
				{http.MethodGet, "/relay", "", nodelink.ScopeRead},
				{http.MethodPost, "/relay/ack", `{"through":1}`, nodelink.ScopeAct},
				{http.MethodPost, "/steer", `{"message":"check this"}`, nodelink.ScopeAct},
				{http.MethodPost, "/approval-response", `{"decision":"approved"}`, nodelink.ScopeAct},
			} {
				t.Run(action.resource, func(t *testing.T) {
					do := func(path, owner, proof string) *httptest.ResponseRecorder {
						t.Helper()
						req := ownerProofRequest(action.method, path, owner, proof, action.body)
						rec := httptest.NewRecorder()
						r.router.ServeHTTP(rec, req)
						return rec
					}
					path := base + action.resource
					for scope, proof := range tokens {
						if scope != action.scope {
							requireOwnerHTTPRefusal(t, do(path, owner, proof), "run_owner_proof_invalid")
						}
					}
					wrongRun := mintTestOwnerProof(t, signer, owner, "another-run", action.scope)
					requireOwnerHTTPRefusal(t, do(path, owner, wrongRun), "run_owner_proof_invalid")
					wrongOwner := mintTestOwnerProof(t, signer, "other-app", run, action.scope)
					requireOwnerHTTPRefusal(t, do(path, owner, wrongOwner), "run_owner_proof_invalid")
					query := url.Values{RunOwnerProofHeader: {tokens[action.scope]}}.Encode()
					noHeader := do(path+"?"+query, owner, "")
					if strict {
						requireOwnerHTTPRefusal(t, noHeader, "run_owner_proof_required")
						requireOwnerHTTPRefusal(t, do(path, owner, ""), "run_owner_proof_required")
					} else {
						require.Equal(t, http.StatusOK, noHeader.Code, noHeader.Body.String())
						require.Equal(t, http.StatusOK, do(path, owner, "").Code)
					}
					proven := do(path, owner, tokens[action.scope])
					require.Equal(t, http.StatusOK, proven.Code, proven.Body.String())
					operator := do(path, "", "invalid")
					require.Equal(t, http.StatusOK, operator.Code, operator.Body.String())
				})
			}
		})
	}
}

func TestMintedOwnerProofWorksInHTTPContractA1A2A7(t *testing.T) {
	r := newLinkRig(t)
	signer := nodelink.NewProofSigner([]byte("example signing key"))
	policy := &nodelink.OwnerPolicy{Signer: signer, RequireProof: true}
	r.pipe.Owners = policy
	SetRunOwnerPolicy(policy)
	t.Cleanup(func() { SetRunOwnerPolicy(nil) })

	const owner, run, exec = "my-app", "my-run", "my-exec"
	require.NoError(t, r.store.RegisterAgent(t.Context(), &types.AgentNode{
		ID: owner, Version: "1.0.0", LastHeartbeat: time.Now(),
	}))
	r.running(run, exec, owner)
	app := r.dial(owner, nil, nil)
	seq := app.seq.Add(1)
	app.send(nodelink.Frame{T: nodelink.TDelegate, Seq: seq}, nodelink.DelegateUp{
		Scopes: []string{nodelink.ScopeAct}, Run: run,
	})
	var ok nodelink.OKBody
	f := app.next(nodelink.TOK, func(f nodelink.Frame) bool { return f.Seq == seq })
	require.NoError(t, json.Unmarshal(f.Body, &ok))
	require.Equal(t, http.StatusOK, ok.Code, ok.Error)
	require.Equal(t, 3600, ok.ExpiresIn)

	app.send(nodelink.Frame{T: nodelink.TSubscribe, Seq: app.seq.Add(1)}, nodelink.Subscribe{})
	app.execState(exec, types.ExecutionStatusRunning)
	code, body := r.do(http.MethodPost, "/api/v1/executions/"+exec+"/request-approval", "", map[string]any{
		"kind": "question", "question": "Continue?",
	})
	require.Equal(t, http.StatusOK, code, body)
	waiting := app.execState(exec, types.ExecutionStatusWaiting)
	require.NotNil(t, waiting.Approval)
	require.Equal(t, "Continue?", waiting.Approval.Question)

	req, err := http.NewRequest(http.MethodPost, r.srv.URL+"/api/v1/executions/"+exec+"/approval-response",
		strings.NewReader(`{"decision":"approved"}`))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(RunOwnerHeader, owner)
	req.Header.Set(RunOwnerProofHeader, ok.Token)
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)
}
