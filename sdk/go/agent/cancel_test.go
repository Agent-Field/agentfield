package agent

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// newCancelAgent constructs a minimal Agent suitable for unit-testing the
// cancel surface. Avoids the full New(cfg) bootstrap which drags in
// memory backends, AI clients, and DID subsystems we don't need here.
func newCancelAgent() *Agent {
	return &Agent{
		cancelFuncs: make(map[string]context.CancelFunc),
	}
}

func TestCancelExecution_TriggersContextDone(t *testing.T) {
	a := newCancelAgent()

	ctx, release := a.registerCancellableExecution(context.Background(), "exec-1")
	defer release()

	if a.CancelExecution("exec-1") != true {
		t.Fatal("CancelExecution returned false for active execution")
	}

	select {
	case <-ctx.Done():
		// expected — ctx cancelled
	case <-time.After(time.Second):
		t.Fatal("ctx was not cancelled")
	}
	if !errors.Is(ctx.Err(), context.Canceled) {
		t.Fatalf("ctx.Err() = %v, want context.Canceled", ctx.Err())
	}
}

func TestCancelExecution_UnknownIDReturnsFalse(t *testing.T) {
	a := newCancelAgent()
	if a.CancelExecution("never-registered") {
		t.Fatal("CancelExecution returned true for unknown id")
	}
	if a.CancelExecution("") {
		t.Fatal("CancelExecution returned true for empty id")
	}
}

func TestRegisterCancellableExecution_ReleaseDeregisters(t *testing.T) {
	a := newCancelAgent()
	_, release := a.registerCancellableExecution(context.Background(), "exec-2")
	release()

	a.cancelMu.Lock()
	_, present := a.cancelFuncs["exec-2"]
	a.cancelMu.Unlock()
	if present {
		t.Fatal("release did not deregister cancel func")
	}
	if a.CancelExecution("exec-2") {
		t.Fatal("CancelExecution returned true after release")
	}
}

func TestRegisterCancellableExecution_EmptyExecutionIDIsUntracked(t *testing.T) {
	a := newCancelAgent()
	ctx, release := a.registerCancellableExecution(context.Background(), "")
	defer release()
	a.cancelMu.Lock()
	registered := len(a.cancelFuncs)
	a.cancelMu.Unlock()
	if registered != 0 {
		t.Fatalf("expected 0 registrations for empty id, got %d", registered)
	}
	// ctx is still cancellable via release.
	release()
	select {
	case <-ctx.Done():
	case <-time.After(time.Second):
		t.Fatal("untracked ctx was not cancelled by release")
	}
}

func TestHandleInternalCancel_CancelsActiveExecution(t *testing.T) {
	a := newCancelAgent()
	ctx, release := a.registerCancellableExecution(context.Background(), "exec-active")
	defer release()

	req := httptest.NewRequest(http.MethodPost, "/_internal/executions/exec-active/cancel", nil)
	rr := httptest.NewRecorder()
	a.handleInternalCancel(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rr.Code)
	}
	body := decodeJSON(t, rr.Body)
	if body["cancelled"] != true {
		t.Fatalf("body cancelled = %v, want true", body["cancelled"])
	}
	if body["execution_id"] != "exec-active" {
		t.Fatalf("body execution_id = %v, want exec-active", body["execution_id"])
	}

	select {
	case <-ctx.Done():
	case <-time.After(time.Second):
		t.Fatal("ctx not cancelled after HTTP cancel")
	}
}

func TestHandleInternalCancel_UnknownExecutionReturns200WithReason(t *testing.T) {
	a := newCancelAgent()
	req := httptest.NewRequest(http.MethodPost, "/_internal/executions/exec-missing/cancel", nil)
	rr := httptest.NewRecorder()
	a.handleInternalCancel(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rr.Code)
	}
	body := decodeJSON(t, rr.Body)
	if body["cancelled"] != false {
		t.Fatalf("body cancelled = %v, want false", body["cancelled"])
	}
	if body["reason"] != "execution_not_active" {
		t.Fatalf("body reason = %v, want execution_not_active", body["reason"])
	}
}

func TestHandleInternalCancel_RejectsGet(t *testing.T) {
	a := newCancelAgent()
	req := httptest.NewRequest(http.MethodGet, "/_internal/executions/exec-1/cancel", nil)
	rr := httptest.NewRecorder()
	a.handleInternalCancel(rr, req)

	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rr.Code)
	}
}

func TestHandleInternalCancel_RejectsMalformedPath(t *testing.T) {
	a := newCancelAgent()
	cases := []string{
		"/_internal/executions//cancel",
		"/_internal/executions/has/slash/cancel",
	}
	for _, p := range cases {
		req := httptest.NewRequest(http.MethodPost, p, nil)
		rr := httptest.NewRecorder()
		a.handleInternalCancel(rr, req)
		if rr.Code != http.StatusNotFound {
			t.Fatalf("path %q: status = %d, want 404", p, rr.Code)
		}
	}
}

// End-to-end: long-running reasoner gets cancelled via the HTTP endpoint.
// Verifies the actual contract — the reasoner's ctx.Done() fires and the
// handler returns the cancellation error path.
func TestEndToEnd_HTTPCancelStopsLongRunningReasoner(t *testing.T) {
	a := newCancelAgent()

	started := make(chan struct{})
	done := make(chan error, 1)
	a.cfg.Logger = nil

	go func() {
		ctx, release := a.registerCancellableExecution(context.Background(), "exec-long")
		defer release()
		close(started)
		select {
		case <-ctx.Done():
			done <- ctx.Err()
		case <-time.After(5 * time.Second):
			done <- errors.New("reasoner finished without cancel — context never fired")
		}
	}()

	<-started

	req := httptest.NewRequest(http.MethodPost, "/_internal/executions/exec-long/cancel", nil)
	rr := httptest.NewRecorder()
	a.handleInternalCancel(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("cancel HTTP status = %d, want 200", rr.Code)
	}

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("reasoner exited with %v, want context.Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("reasoner did not exit after cancel")
	}
}

// Concurrent cancel/release: must not panic, must always end with the
// registry empty.
func TestRegisterCancellableExecution_ConcurrentReleaseAndCancel(t *testing.T) {
	a := newCancelAgent()
	const n = 50

	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		id := strings.Repeat("a", i+1)
		_, release := a.registerCancellableExecution(context.Background(), id)
		wg.Add(2)
		go func() { defer wg.Done(); release() }()
		go func() { defer wg.Done(); a.CancelExecution(id) }()
	}
	wg.Wait()

	a.cancelMu.Lock()
	leftover := len(a.cancelFuncs)
	a.cancelMu.Unlock()
	if leftover != 0 {
		t.Fatalf("leftover cancel registrations = %d, want 0", leftover)
	}
}

func decodeJSON(t *testing.T, r io.Reader) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.NewDecoder(r).Decode(&out); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	return out
}
