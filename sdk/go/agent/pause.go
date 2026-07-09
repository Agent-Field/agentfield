package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/client"
)

// Approval decision values carried on an ApprovalResult. They mirror the
// Python SDK's ApprovalResult.decision strings so a reasoner authored against
// either SDK reads the same values.
const (
	// ApprovalApproved is set when the human approved the request.
	ApprovalApproved = "approved"
	// ApprovalRejected is set when the human rejected the request.
	ApprovalRejected = "rejected"
	// ApprovalRequestChanges is set when the human asked for changes.
	ApprovalRequestChanges = "request_changes"
	// ApprovalExpired is set when the pause timed out without a resolution.
	ApprovalExpired = "expired"
	// ApprovalError is set when the control plane could not be notified.
	ApprovalError = "error"
	// ApprovalCancelled is set when the pause was cancelled (e.g. on shutdown).
	ApprovalCancelled = "cancelled"
)

// ApprovalResult is the outcome of a paused execution once a human (or an
// upstream service) resolves it. It mirrors the Python SDK's ApprovalResult
// dataclass field-for-field: decision, feedback, execution_id,
// approval_request_id and raw_response.
type ApprovalResult struct {
	// Decision is one of the Approval* constants ("approved", "rejected",
	// "request_changes", "expired", "error", "cancelled") or any custom value
	// the control plane forwarded.
	Decision string `json:"decision"`
	// Feedback is a free-form human message accompanying the decision.
	Feedback string `json:"feedback"`
	// ExecutionID is the execution that was paused.
	ExecutionID string `json:"execution_id"`
	// ApprovalRequestID is the ID of the external approval request.
	ApprovalRequestID string `json:"approval_request_id"`
	// RawResponse is the parsed `response` payload from the callback, if any.
	RawResponse map[string]any `json:"raw_response,omitempty"`
}

// Approved reports whether the human approved the request.
func (r ApprovalResult) Approved() bool { return r.Decision == ApprovalApproved }

// ChangesRequested reports whether the human asked for changes rather than
// approving or rejecting.
func (r ApprovalResult) ChangesRequested() bool { return r.Decision == ApprovalRequestChanges }

// pendingPause holds the buffered channel that delivers the ApprovalResult to
// the blocked Pause caller. The channel is buffered (size 1) and closed after
// a single send so Resolve never blocks and a late Resolve cannot double-send.
type pendingPause struct {
	ch       chan ApprovalResult
	resolved bool
}

// PauseManager is a concurrency-safe registry of pending execution pauses,
// keyed by approval_request_id and resolved by the /webhooks/approval callback.
// It mirrors the Python SDK's _PauseManager and the TypeScript SDK's
// PauseManager.
type PauseManager struct {
	mu            sync.Mutex
	pending       map[string]*pendingPause
	execToRequest map[string]string
}

// NewPauseManager constructs an empty PauseManager.
func NewPauseManager() *PauseManager {
	return &PauseManager{
		pending:       make(map[string]*pendingPause),
		execToRequest: make(map[string]string),
	}
}

// Register creates a pending pause for approvalRequestID and returns a
// receive-only channel that yields exactly one ApprovalResult when the pause
// resolves. Registration is idempotent: a second Register for the same
// approvalRequestID returns the existing channel rather than replacing it, so
// a fast callback that arrives before the caller blocks is never lost.
// executionID, when non-empty, is recorded for fallback resolution.
func (m *PauseManager) Register(approvalRequestID, executionID string) <-chan ApprovalResult {
	m.mu.Lock()
	defer m.mu.Unlock()
	if existing, ok := m.pending[approvalRequestID]; ok {
		return existing.ch
	}
	p := &pendingPause{ch: make(chan ApprovalResult, 1)}
	m.pending[approvalRequestID] = p
	if executionID != "" {
		m.execToRequest[executionID] = approvalRequestID
	}
	return p.ch
}

// Resolve resolves the pending pause identified by approvalRequestID with the
// given result. It returns true if a waiter was found and resolved, false
// otherwise (unknown or already-resolved request).
func (m *PauseManager) Resolve(approvalRequestID string, result ApprovalResult) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.resolveLocked(approvalRequestID, result)
}

// resolveLocked resolves a pending pause; the caller must hold m.mu.
func (m *PauseManager) resolveLocked(approvalRequestID string, result ApprovalResult) bool {
	p, ok := m.pending[approvalRequestID]
	if !ok {
		return false
	}
	delete(m.pending, approvalRequestID)
	for eid, rid := range m.execToRequest {
		if rid == approvalRequestID {
			delete(m.execToRequest, eid)
			break
		}
	}
	if p.resolved {
		return false
	}
	p.resolved = true
	p.ch <- result // buffered (cap 1) — never blocks
	close(p.ch)
	return true
}

// ResolveByExecutionID resolves a pending pause by execution_id, used as a
// fallback when the callback omits the approval_request_id. Returns true if a
// waiter was found.
func (m *PauseManager) ResolveByExecutionID(executionID string, result ApprovalResult) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	rid, ok := m.execToRequest[executionID]
	if !ok {
		return false
	}
	return m.resolveLocked(rid, result)
}

// CancelAll resolves every pending pause with a cancelled result. Used on
// shutdown so a reasoner blocked in Pause does not hang the process forever.
func (m *PauseManager) CancelAll() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for rid, p := range m.pending {
		if !p.resolved {
			p.resolved = true
			p.ch <- ApprovalResult{
				Decision:          ApprovalCancelled,
				Feedback:          "agent shutting down",
				ApprovalRequestID: rid,
			}
			close(p.ch)
		}
	}
	m.pending = make(map[string]*pendingPause)
	m.execToRequest = make(map[string]string)
}

// PendingCount returns the number of currently-pending pauses. Useful for tests.
func (m *PauseManager) PendingCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.pending)
}

// PauseOptions configures Agent.Pause. It mirrors the keyword arguments of the
// Python SDK's Agent.pause().
type PauseOptions struct {
	// ApprovalRequestID is the ID of the approval request the agent already
	// created on an external service. Required.
	ApprovalRequestID string
	// ApprovalRequestURL is where a human can review the request. Optional.
	ApprovalRequestURL string
	// ExpiresInHours is passed to the control plane and, when Timeout is zero,
	// also bounds how long Pause blocks. Defaults to 72 when <= 0.
	ExpiresInHours int
	// Timeout bounds how long Pause blocks. Zero defaults to
	// ExpiresInHours hours.
	Timeout time.Duration
	// ExecutionID overrides the current execution. Defaults to the execution
	// carried on ctx.
	ExecutionID string
}

// Pause pauses the current execution for external approval.
//
// It transitions the execution to "waiting" on the control plane (via
// client.RequestApproval), then blocks until the approval webhook callback
// resolves it, the context is cancelled, or the timeout elapses. The agent is
// responsible for creating the approval request on an external service and
// passing the resulting ApprovalRequestID.
//
// On timeout Pause returns an ApprovalResult with Decision=="expired" (not an
// error), mirroring the Python SDK. On context cancellation it returns the
// context error. If the control plane cannot be notified it returns an error
// after resolving the pending pause with Decision=="error".
//
// The agent must be serving (a reachable PublicURL) and connected to a control
// plane, since the callback URL is required for resolution.
func (a *Agent) Pause(ctx context.Context, opts PauseOptions) (*ApprovalResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	executionID := strings.TrimSpace(opts.ExecutionID)
	if executionID == "" {
		executionID = executionContextFrom(ctx).ExecutionID
	}
	if executionID == "" {
		return nil, errors.New("no execution_id available — cannot pause")
	}
	if strings.TrimSpace(opts.ApprovalRequestID) == "" {
		return nil, errors.New("approval_request_id is required — cannot pause")
	}
	if a.client == nil {
		return nil, errors.New("agent is not connected to a control plane — cannot pause")
	}

	// The callback URL must match the /webhooks/approval route registered on
	// the agent's mux and be reachable from the control plane, so it is built
	// from PublicURL (the same base URL advertised at registration).
	base := strings.TrimSuffix(strings.TrimSpace(a.cfg.PublicURL), "/")
	if base == "" {
		return nil, errors.New("agent has no public URL — cannot build approval callback URL")
	}
	callbackURL := base + "/webhooks/approval"

	expiresInHours := opts.ExpiresInHours
	if expiresInHours <= 0 {
		expiresInHours = 72
	}

	// Register the pending pause BEFORE telling the control plane, so a fast
	// callback that arrives before RequestApproval returns is not missed.
	future := a.pauseManager.Register(opts.ApprovalRequestID, executionID)

	if _, err := a.client.RequestApproval(ctx, a.cfg.NodeID, executionID, client.RequestApprovalRequest{
		ApprovalRequestID:  opts.ApprovalRequestID,
		ApprovalRequestURL: opts.ApprovalRequestURL,
		CallbackURL:        callbackURL,
		ExpiresInHours:     expiresInHours,
	}); err != nil {
		// Clean up the pending pause if we could not even notify the CP.
		a.pauseManager.Resolve(opts.ApprovalRequestID, ApprovalResult{
			Decision:          ApprovalError,
			Feedback:          "failed to notify control plane",
			ExecutionID:       executionID,
			ApprovalRequestID: opts.ApprovalRequestID,
		})
		return nil, fmt.Errorf("pause: request approval: %w", err)
	}

	a.Note(ctx, fmt.Sprintf("Execution paused — waiting for approval %s", opts.ApprovalRequestID), "approval", "waiting")

	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = time.Duration(expiresInHours) * time.Hour
	}
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	select {
	case result := <-future:
		return &result, nil
	case <-timer.C:
		// Timeout is a normal outcome — return an "expired" result rather than
		// erroring, matching the Python SDK. Resolve the pending pause so a
		// late callback does not leak a resolved-but-unawaited entry.
		expired := ApprovalResult{
			Decision:          ApprovalExpired,
			Feedback:          "timed out waiting for approval",
			ExecutionID:       executionID,
			ApprovalRequestID: opts.ApprovalRequestID,
		}
		a.pauseManager.Resolve(opts.ApprovalRequestID, expired)
		return &expired, nil
	case <-ctx.Done():
		// Cooperative cancellation — drop the pending pause and surface the
		// context error to the caller (idiomatic Go).
		a.pauseManager.Resolve(opts.ApprovalRequestID, ApprovalResult{
			Decision:          ApprovalCancelled,
			Feedback:          "context cancelled",
			ExecutionID:       executionID,
			ApprovalRequestID: opts.ApprovalRequestID,
		})
		return nil, ctx.Err()
	}
}

// handleApprovalWebhook is the worker side of the approval callback transport.
// The control plane POSTs here (via the callback_url registered when the
// execution paused) once a human resolves the approval. The body carries
// {execution_id, decision, feedback, approval_request_id, response, new_status};
// we resolve the matching pending pause — first by approval_request_id, then by
// execution_id as a fallback — and reply {status, resolved}. This matches the
// Python SDK's /webhooks/approval route and the TypeScript SDK's
// installApprovalWebhookRoute.
func (a *Agent) handleApprovalWebhook(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()

	var body struct {
		ExecutionID       string          `json:"execution_id"`
		Decision          string          `json:"decision"`
		Feedback          string          `json:"feedback"`
		ApprovalRequestID string          `json:"approval_request_id"`
		Response          json.RawMessage `json:"response"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil && !errors.Is(err, io.EOF) {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": "invalid JSON"})
		return
	}

	if body.ExecutionID == "" || body.Decision == "" {
		writeJSON(w, http.StatusBadRequest, map[string]any{
			"error": "execution_id and decision are required",
		})
		return
	}

	result := ApprovalResult{
		Decision:          body.Decision,
		Feedback:          body.Feedback,
		ExecutionID:       body.ExecutionID,
		ApprovalRequestID: body.ApprovalRequestID,
		RawResponse:       parseRawApprovalResponse(body.Response),
	}

	resolved := false
	if body.ApprovalRequestID != "" {
		resolved = a.pauseManager.Resolve(body.ApprovalRequestID, result)
	}
	if !resolved && body.ExecutionID != "" {
		resolved = a.pauseManager.ResolveByExecutionID(body.ExecutionID, result)
	}

	writeJSON(w, http.StatusOK, map[string]any{"status": "received", "resolved": resolved})
}

// parseRawApprovalResponse parses the `response` field of an approval callback,
// which the control plane may deliver as either a JSON object or a JSON-encoded
// string. A non-JSON string is surfaced under a "text" key rather than dropped.
// Mirrors the TypeScript SDK's parseRawResponse and the Python SDK's handling.
func parseRawApprovalResponse(raw json.RawMessage) map[string]any {
	if len(raw) == 0 {
		return nil
	}
	// Direct object (or JSON null → nil).
	var obj map[string]any
	if err := json.Unmarshal(raw, &obj); err == nil {
		return obj
	}
	// A JSON-encoded string that may itself contain a JSON object.
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		if strings.TrimSpace(s) == "" {
			return nil
		}
		var inner map[string]any
		if err := json.Unmarshal([]byte(s), &inner); err == nil {
			return inner
		}
		return map[string]any{"text": s}
	}
	return nil
}
