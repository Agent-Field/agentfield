package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
)

// registerCancellableExecution wraps `parent` with a child context that the
// SDK can cancel out-of-band when the control plane's cancel dispatcher
// fires. Returns the wrapped context and a release function that MUST be
// deferred — release deregisters the execution and cancels the inner
// context (no-op if cancel already fired). Empty executionID means
// "untracked", in which case nothing is registered and the context is just
// a cancellable wrapper of the parent.
func (a *Agent) registerCancellableExecution(parent context.Context, executionID string) (context.Context, func()) {
	ctx, cancel := context.WithCancel(parent)
	executionID = strings.TrimSpace(executionID)
	if executionID != "" {
		a.cancelMu.Lock()
		a.cancelFuncs[executionID] = cancel
		a.cancelMu.Unlock()
	}
	released := false
	release := func() {
		if released {
			return
		}
		released = true
		if executionID != "" {
			a.cancelMu.Lock()
			// Only delete our entry — a racing cancel that already
			// removed it shouldn't get a stale registration overwritten.
			if existing, ok := a.cancelFuncs[executionID]; ok {
				_ = existing
				delete(a.cancelFuncs, executionID)
			}
			a.cancelMu.Unlock()
		}
		cancel()
	}
	return ctx, release
}

// CancelExecution cancels an in-flight reasoner invocation by execution_id
// from inside the SDK. Mostly useful for tests and tools that drive the
// SDK directly; the production trigger is the HTTP cancel endpoint.
// Returns true if a matching execution was found and cancelled.
func (a *Agent) CancelExecution(executionID string) bool {
	executionID = strings.TrimSpace(executionID)
	if executionID == "" {
		return false
	}
	a.cancelMu.Lock()
	cancel, ok := a.cancelFuncs[executionID]
	if ok {
		delete(a.cancelFuncs, executionID)
	}
	a.cancelMu.Unlock()
	if !ok {
		return false
	}
	cancel()
	return true
}

// handleInternalCancel is the worker side of the cancel-signal transport
// shipped by the control plane. The dispatcher POSTs:
//
//	POST /_internal/executions/:execution_id/cancel
//
// We look up the matching cancel func and fire it. Reasoner code that
// honors ctx.Done() (idiomatic Go: net/http, database/sql, the official
// Anthropic SDK, etc.) will short-circuit. Reasoner code that doesn't
// honor cancellation finishes naturally and its output is discarded by
// the control plane (existing behaviour).
func (a *Agent) handleInternalCancel(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	// Path shape: /_internal/executions/{id}/cancel
	rest := strings.TrimPrefix(r.URL.Path, "/_internal/executions/")
	id := strings.TrimSuffix(rest, "/cancel")
	id = strings.TrimSpace(id)
	if id == "" || strings.Contains(id, "/") {
		http.NotFound(w, r)
		return
	}

	cancelled := a.CancelExecution(id)
	status := http.StatusOK
	body := map[string]any{
		"cancelled":    cancelled,
		"execution_id": id,
	}
	if !cancelled {
		// Not active locally — could be already finished, or never
		// dispatched here. 200 with the marker keeps the dispatcher's
		// best-effort logic happy and avoids spurious warning logs.
		body["reason"] = "execution_not_active"
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}
