package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/client"
)

// newPauseAgent constructs a minimal Agent wired only with a PauseManager,
// mirroring newCancelAgent in cancel_test.go. Avoids the full New(cfg)
// bootstrap which drags in memory backends, AI clients, and DID subsystems we
// don't need here.
func newPauseAgent() *Agent {
	return &Agent{pauseManager: NewPauseManager()}
}

// ---------------------------------------------------------------------------
// ApprovalResult
// ---------------------------------------------------------------------------

func TestApprovalResult_ConvenienceHelpers(t *testing.T) {
	cases := []struct {
		decision         string
		approved         bool
		changesRequested bool
	}{
		{ApprovalApproved, true, false},
		{ApprovalRejected, false, false},
		{ApprovalRequestChanges, false, true},
		{ApprovalExpired, false, false},
		{ApprovalError, false, false},
		{ApprovalCancelled, false, false},
	}
	for _, tc := range cases {
		r := ApprovalResult{Decision: tc.decision}
		if r.Approved() != tc.approved {
			t.Errorf("decision %q: Approved()=%v want %v", tc.decision, r.Approved(), tc.approved)
		}
		if r.ChangesRequested() != tc.changesRequested {
			t.Errorf("decision %q: ChangesRequested()=%v want %v", tc.decision, r.ChangesRequested(), tc.changesRequested)
		}
	}
}

// ---------------------------------------------------------------------------
// PauseManager
// ---------------------------------------------------------------------------

func TestPauseManager_ResolveByApprovalRequestID(t *testing.T) {
	m := NewPauseManager()
	ch := m.Register("req-1", "exec-1")
	if got := m.PendingCount(); got != 1 {
		t.Fatalf("PendingCount=%d want 1", got)
	}

	if !m.Resolve("req-1", ApprovalResult{Decision: ApprovalApproved, Feedback: "lgtm"}) {
		t.Fatal("Resolve returned false for a registered pause")
	}

	result := <-ch
	if !result.Approved() || result.Feedback != "lgtm" {
		t.Fatalf("result=%+v want approved/lgtm", result)
	}
	if got := m.PendingCount(); got != 0 {
		t.Fatalf("PendingCount=%d want 0 after resolve", got)
	}
}

func TestPauseManager_ResolveByExecutionID_Fallback(t *testing.T) {
	m := NewPauseManager()
	ch := m.Register("req-2", "exec-2")

	if !m.ResolveByExecutionID("exec-2", ApprovalResult{Decision: ApprovalRejected}) {
		t.Fatal("ResolveByExecutionID returned false for a registered pause")
	}
	if got := (<-ch).Decision; got != ApprovalRejected {
		t.Fatalf("decision=%q want rejected", got)
	}
}

func TestPauseManager_RegisterIsIdempotent(t *testing.T) {
	m := NewPauseManager()
	a := m.Register("req-3", "exec-3")
	b := m.Register("req-3", "exec-3")
	if m.PendingCount() != 1 {
		t.Fatalf("PendingCount=%d want 1 after duplicate register", m.PendingCount())
	}
	// Both channels must be the same underlying channel: resolving once
	// delivers to a single receiver.
	m.Resolve("req-3", ApprovalResult{Decision: ApprovalApproved})
	if got := (<-a).Decision; got != ApprovalApproved {
		t.Fatalf("channel a decision=%q want approved", got)
	}
	select {
	case _, ok := <-b:
		if ok {
			t.Fatal("second channel should be the same (closed, no second value)")
		}
	default:
		t.Fatal("expected b to be the same closed channel as a")
	}
}

func TestPauseManager_ResolveUnknownReturnsFalse(t *testing.T) {
	m := NewPauseManager()
	if m.Resolve("nope", ApprovalResult{Decision: ApprovalApproved}) {
		t.Fatal("Resolve returned true for unknown request")
	}
	if m.ResolveByExecutionID("nope", ApprovalResult{Decision: ApprovalApproved}) {
		t.Fatal("ResolveByExecutionID returned true for unknown execution")
	}
}

func TestPauseManager_DoubleResolveReturnsFalse(t *testing.T) {
	m := NewPauseManager()
	ch := m.Register("req-dbl", "exec-dbl")
	if !m.Resolve("req-dbl", ApprovalResult{Decision: ApprovalApproved}) {
		t.Fatal("first Resolve returned false")
	}
	<-ch
	if m.Resolve("req-dbl", ApprovalResult{Decision: ApprovalRejected}) {
		t.Fatal("second Resolve returned true — must be idempotent")
	}
}

func TestPauseManager_CancelAll(t *testing.T) {
	m := NewPauseManager()
	c1 := m.Register("req-a", "exec-a")
	c2 := m.Register("req-b", "exec-b")

	m.CancelAll()
	if m.PendingCount() != 0 {
		t.Fatalf("PendingCount=%d want 0 after CancelAll", m.PendingCount())
	}
	if got := (<-c1).Decision; got != ApprovalCancelled {
		t.Fatalf("c1 decision=%q want cancelled", got)
	}
	if got := (<-c2).Decision; got != ApprovalCancelled {
		t.Fatalf("c2 decision=%q want cancelled", got)
	}
}

// Concurrent pauses must resolve independently — each waiter receives exactly
// its own result, with no cross-talk between approval_request_ids.
func TestPauseManager_ConcurrentPausesResolveIndependently(t *testing.T) {
	m := NewPauseManager()
	const n = 50

	chans := make([]<-chan ApprovalResult, n)
	for i := 0; i < n; i++ {
		chans[i] = m.Register("req-"+strconv.Itoa(i), "exec-"+strconv.Itoa(i))
	}
	if m.PendingCount() != n {
		t.Fatalf("PendingCount=%d want %d", m.PendingCount(), n)
	}

	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			m.Resolve("req-"+strconv.Itoa(i), ApprovalResult{
				Decision:          ApprovalApproved,
				Feedback:          "fb-" + strconv.Itoa(i),
				ApprovalRequestID: "req-" + strconv.Itoa(i),
			})
		}(i)
	}
	wg.Wait()

	for i := 0; i < n; i++ {
		got := <-chans[i]
		want := "fb-" + strconv.Itoa(i)
		if got.Feedback != want {
			t.Fatalf("waiter %d received feedback %q, want %q (cross-talk)", i, got.Feedback, want)
		}
	}
	if m.PendingCount() != 0 {
		t.Fatalf("PendingCount=%d want 0 after all resolved", m.PendingCount())
	}
}

// ---------------------------------------------------------------------------
// handleApprovalWebhook
// ---------------------------------------------------------------------------

func postApprovalWebhook(t *testing.T, h http.HandlerFunc, body map[string]any) (*httptest.ResponseRecorder, map[string]any) {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/webhooks/approval", bytes.NewReader(raw))
	rr := httptest.NewRecorder()
	h(rr, req)
	return rr, decodeJSON(t, rr.Body)
}

func TestApprovalWebhook_ResolvesAndReplies(t *testing.T) {
	a := newPauseAgent()
	pending := a.pauseManager.Register("req-webhook", "exec-webhook")

	rr, resp := postApprovalWebhook(t, a.handleApprovalWebhook, map[string]any{
		"execution_id":        "exec-webhook",
		"approval_request_id": "req-webhook",
		"decision":            "approved",
		"feedback":            "ship it",
		"response":            `{"reviewer":"alice"}`,
	})

	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d want 200", rr.Code)
	}
	if resp["status"] != "received" || resp["resolved"] != true {
		t.Fatalf("reply=%v want {status:received, resolved:true}", resp)
	}

	result := <-pending
	if !result.Approved() {
		t.Errorf("result not approved: %+v", result)
	}
	if result.Feedback != "ship it" {
		t.Errorf("feedback=%q want 'ship it'", result.Feedback)
	}
	if result.RawResponse["reviewer"] != "alice" {
		t.Errorf("rawResponse=%v want reviewer=alice", result.RawResponse)
	}
}

// Decision strings map straight through to ApprovalResult.Decision.
func TestApprovalWebhook_DecisionMapping(t *testing.T) {
	for _, decision := range []string{ApprovalApproved, ApprovalRejected, ApprovalRequestChanges, ApprovalExpired} {
		t.Run(decision, func(t *testing.T) {
			a := newPauseAgent()
			pending := a.pauseManager.Register("req-"+decision, "exec-"+decision)
			_, resp := postApprovalWebhook(t, a.handleApprovalWebhook, map[string]any{
				"execution_id":        "exec-" + decision,
				"approval_request_id": "req-" + decision,
				"decision":            decision,
			})
			if resp["resolved"] != true {
				t.Fatalf("resolved=%v want true", resp["resolved"])
			}
			if got := (<-pending).Decision; got != decision {
				t.Fatalf("decision=%q want %q", got, decision)
			}
		})
	}
}

func TestApprovalWebhook_FallbackByExecutionID(t *testing.T) {
	a := newPauseAgent()
	pending := a.pauseManager.Register("req-fallback", "exec-fallback")

	// No approval_request_id in the callback — must fall back to execution_id.
	_, resp := postApprovalWebhook(t, a.handleApprovalWebhook, map[string]any{
		"execution_id": "exec-fallback",
		"decision":     "request_changes",
	})
	if resp["resolved"] != true {
		t.Fatalf("resolved=%v want true", resp["resolved"])
	}
	if !(<-pending).ChangesRequested() {
		t.Fatal("expected changes-requested result")
	}
}

func TestApprovalWebhook_ResolvedFalseWhenNoMatch(t *testing.T) {
	a := newPauseAgent()
	_, resp := postApprovalWebhook(t, a.handleApprovalWebhook, map[string]any{
		"execution_id":        "unknown",
		"approval_request_id": "unknown",
		"decision":            "approved",
	})
	if resp["resolved"] != false {
		t.Fatalf("resolved=%v want false", resp["resolved"])
	}
}

// The `response` field may be a JSON-encoded string or an inline object; both
// must land in RawResponse. A non-JSON string is surfaced under "text".
func TestApprovalWebhook_RawResponseParsing(t *testing.T) {
	cases := []struct {
		name     string
		response any
		check    func(t *testing.T, raw map[string]any)
	}{
		{
			name:     "json-encoded-string",
			response: `{"note":"stringy"}`,
			check: func(t *testing.T, raw map[string]any) {
				if raw["note"] != "stringy" {
					t.Fatalf("raw=%v want note=stringy", raw)
				}
			},
		},
		{
			name:     "inline-object",
			response: map[string]any{"note": "inline object"},
			check: func(t *testing.T, raw map[string]any) {
				if raw["note"] != "inline object" {
					t.Fatalf("raw=%v want note='inline object'", raw)
				}
			},
		},
		{
			name:     "non-json-string",
			response: "just words",
			check: func(t *testing.T, raw map[string]any) {
				if raw["text"] != "just words" {
					t.Fatalf("raw=%v want text='just words'", raw)
				}
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := newPauseAgent()
			pending := a.pauseManager.Register("req-"+tc.name, "exec-"+tc.name)
			postApprovalWebhook(t, a.handleApprovalWebhook, map[string]any{
				"execution_id":        "exec-" + tc.name,
				"approval_request_id": "req-" + tc.name,
				"decision":            "approved",
				"response":            tc.response,
			})
			tc.check(t, (<-pending).RawResponse)
		})
	}
}

func TestApprovalWebhook_MissingFieldsReturns400(t *testing.T) {
	a := newPauseAgent()
	cases := []map[string]any{
		{"decision": "approved"},   // no execution_id
		{"execution_id": "exec-x"}, // no decision
		{},                         // neither
	}
	for _, body := range cases {
		rr, _ := postApprovalWebhook(t, a.handleApprovalWebhook, body)
		if rr.Code != http.StatusBadRequest {
			t.Fatalf("body %v: status=%d want 400", body, rr.Code)
		}
	}
}

func TestApprovalWebhook_RejectsGet(t *testing.T) {
	a := newPauseAgent()
	req := httptest.NewRequest(http.MethodGet, "/webhooks/approval", nil)
	rr := httptest.NewRecorder()
	a.handleApprovalWebhook(rr, req)
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status=%d want 405", rr.Code)
	}
}

// The approval webhook route is always wired into the mux (independent of any
// accepts_webhook trigger flag), matching the Python/TS SDKs where it is
// auto-installed for every agent. A POST with missing fields returns 400 (route
// present) rather than 404 (route absent).
func TestApprovalWebhook_RouteAlwaysRegisteredOnMux(t *testing.T) {
	a := newPauseAgent()
	srv := httptest.NewServer(a.Handler())
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/webhooks/approval", "application/json",
		bytes.NewReader([]byte(`{"decision":"approved"}`)))
	if err != nil {
		t.Fatalf("POST failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status=%d want 400 (route present but missing fields)", resp.StatusCode)
	}
}

// The control plane delivers the approval callback unauthenticated, so the
// /webhooks/approval route must bypass origin-token auth — otherwise the
// callback is rejected with 401 and the pause never resolves.
func TestApprovalWebhook_BypassesOriginAuth(t *testing.T) {
	a := newPauseAgent()
	a.cfg.RequireOriginAuth = true
	a.cfg.InternalToken = "secret-token"
	srv := httptest.NewServer(a.Handler())
	defer srv.Close()

	// No Authorization header — an ordinary /execute call would 401 here.
	resp, err := http.Post(srv.URL+"/webhooks/approval", "application/json",
		bytes.NewReader([]byte(`{"execution_id":"e","approval_request_id":"r","decision":"approved"}`)))
	if err != nil {
		t.Fatalf("POST failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized {
		t.Fatal("approval webhook was rejected by origin auth — must be exempt")
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status=%d want 200 (handler reached)", resp.StatusCode)
	}
}

// Likewise the route must bypass local DID verification — the CP callback
// carries no DID signature.
func TestApprovalWebhook_BypassesLocalVerification(t *testing.T) {
	a := newPauseAgent()
	// A verifier with no reachable control plane; the exemption must short
	// circuit before any DID/refresh logic runs, so no network call happens.
	a.localVerifier = NewLocalVerifier("http://127.0.0.1:0", time.Hour, "")
	a.logger = log.New(io.Discard, "", 0)
	srv := httptest.NewServer(a.Handler())
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/webhooks/approval", "application/json",
		bytes.NewReader([]byte(`{"execution_id":"e","approval_request_id":"r","decision":"approved"}`)))
	if err != nil {
		t.Fatalf("POST failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		t.Fatalf("approval webhook blocked by local verification (status=%d) — must be exempt", resp.StatusCode)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status=%d want 200 (handler reached)", resp.StatusCode)
	}
}

// ---------------------------------------------------------------------------
// Agent.Pause — end-to-end through the control-plane request-approval call and
// the webhook callback.
// ---------------------------------------------------------------------------

// mockControlPlane returns an httptest server that answers request-approval.
// If fireCallback is true it POSTs the given resolution to the callback_url it
// received, exercising the full pause/resolve loop.
func mockControlPlane(t *testing.T, status int, fireCallback bool, resolution map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req client.RequestApprovalRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if status >= 400 {
			w.WriteHeader(status)
			_, _ = w.Write([]byte(`{"error":"boom"}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"approval_request_id":  req.ApprovalRequestID,
			"approval_request_url": req.ApprovalRequestURL,
			"status":               "waiting",
		})
		if fireCallback && req.CallbackURL != "" {
			payload := map[string]any{"execution_id": "exec-e2e", "approval_request_id": req.ApprovalRequestID}
			for k, v := range resolution {
				payload[k] = v
			}
			body, _ := json.Marshal(payload)
			go func() {
				// Best-effort async callback, like the real control plane.
				_, _ = http.Post(req.CallbackURL, "application/json", bytes.NewReader(body))
			}()
		}
	}))
}

// buildPauseAgent wires a minimal Agent with a control-plane client and an
// HTTP server for the /webhooks/approval callback. Returns the agent and a
// cleanup func.
func buildPauseAgent(t *testing.T, cpURL string) (*Agent, func()) {
	t.Helper()
	c, err := client.New(cpURL)
	if err != nil {
		t.Fatalf("client.New: %v", err)
	}
	a := &Agent{
		pauseManager: NewPauseManager(),
		client:       c,
	}
	a.cfg.NodeID = "test-node"
	srv := httptest.NewServer(a.Handler())
	a.cfg.PublicURL = srv.URL
	return a, srv.Close
}

func TestPause_ResolvesOnWebhookCallback(t *testing.T) {
	cp := mockControlPlane(t, http.StatusOK, true, map[string]any{
		"decision": "approved",
		"feedback": "lgtm",
		"response": `{"reviewer":"bob"}`,
	})
	defer cp.Close()

	a, cleanup := buildPauseAgent(t, cp.URL)
	defer cleanup()

	result, err := a.Pause(context.Background(), PauseOptions{
		ApprovalRequestID: "req-e2e",
		ExecutionID:       "exec-e2e",
		Timeout:           5 * time.Second,
	})
	if err != nil {
		t.Fatalf("Pause returned error: %v", err)
	}
	if !result.Approved() {
		t.Fatalf("result=%+v want approved", result)
	}
	if result.Feedback != "lgtm" {
		t.Errorf("feedback=%q want lgtm", result.Feedback)
	}
	if result.RawResponse["reviewer"] != "bob" {
		t.Errorf("rawResponse=%v want reviewer=bob", result.RawResponse)
	}
	if a.pauseManager.PendingCount() != 0 {
		t.Errorf("PendingCount=%d want 0", a.pauseManager.PendingCount())
	}
}

func TestPause_ExpiryReturnsExpiredDecision(t *testing.T) {
	// CP accepts request-approval but never fires a callback.
	cp := mockControlPlane(t, http.StatusOK, false, nil)
	defer cp.Close()

	a, cleanup := buildPauseAgent(t, cp.URL)
	defer cleanup()

	result, err := a.Pause(context.Background(), PauseOptions{
		ApprovalRequestID: "req-expire",
		ExecutionID:       "exec-e2e",
		Timeout:           50 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("Pause returned error on timeout, want nil: %v", err)
	}
	if result.Decision != ApprovalExpired {
		t.Fatalf("decision=%q want expired", result.Decision)
	}
	if a.pauseManager.PendingCount() != 0 {
		t.Errorf("PendingCount=%d want 0 after expiry", a.pauseManager.PendingCount())
	}
}

func TestPause_ContextCancellation(t *testing.T) {
	cp := mockControlPlane(t, http.StatusOK, false, nil)
	defer cp.Close()

	a, cleanup := buildPauseAgent(t, cp.URL)
	defer cleanup()

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	result, err := a.Pause(ctx, PauseOptions{
		ApprovalRequestID: "req-cancel",
		ExecutionID:       "exec-e2e",
		Timeout:           10 * time.Second,
	})
	if err == nil {
		t.Fatalf("Pause returned nil error on cancellation, result=%+v", result)
	}
	if err != context.Canceled {
		t.Fatalf("err=%v want context.Canceled", err)
	}
	if a.pauseManager.PendingCount() != 0 {
		t.Errorf("PendingCount=%d want 0 after cancellation", a.pauseManager.PendingCount())
	}
}

func TestPause_RequestApprovalFailureReturnsError(t *testing.T) {
	cp := mockControlPlane(t, http.StatusInternalServerError, false, nil)
	defer cp.Close()

	a, cleanup := buildPauseAgent(t, cp.URL)
	defer cleanup()

	_, err := a.Pause(context.Background(), PauseOptions{
		ApprovalRequestID: "req-fail",
		ExecutionID:       "exec-e2e",
		Timeout:           time.Second,
	})
	if err == nil {
		t.Fatal("Pause returned nil error when control plane failed")
	}
	// The pending pause must have been cleaned up.
	if a.pauseManager.PendingCount() != 0 {
		t.Errorf("PendingCount=%d want 0 after CP failure", a.pauseManager.PendingCount())
	}
}

func TestPause_NoExecutionIDReturnsError(t *testing.T) {
	a := newPauseAgent()
	_, err := a.Pause(context.Background(), PauseOptions{ApprovalRequestID: "req-x"})
	if err == nil {
		t.Fatal("expected error when no execution_id is available")
	}
}

func TestPause_MissingApprovalRequestIDReturnsError(t *testing.T) {
	a := newPauseAgent()
	_, err := a.Pause(context.Background(), PauseOptions{ExecutionID: "exec-1"})
	if err == nil {
		t.Fatal("expected error when approval_request_id is empty")
	}
}

// Pause resolves the execution_id from the ExecutionContext on ctx when
// PauseOptions.ExecutionID is not set.
func TestPause_UsesExecutionIDFromContext(t *testing.T) {
	cp := mockControlPlane(t, http.StatusOK, true, map[string]any{"decision": "approved"})
	defer cp.Close()

	a, cleanup := buildPauseAgent(t, cp.URL)
	defer cleanup()

	ctx := contextWithExecution(context.Background(), ExecutionContext{ExecutionID: "exec-e2e"})
	result, err := a.Pause(ctx, PauseOptions{
		ApprovalRequestID: "req-ctx",
		Timeout:           5 * time.Second,
	})
	if err != nil {
		t.Fatalf("Pause error: %v", err)
	}
	if !result.Approved() {
		t.Fatalf("result=%+v want approved", result)
	}
}
