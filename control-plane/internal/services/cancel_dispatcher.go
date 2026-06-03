package services

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
)

// CancelDispatcher is the bridge between the in-process execution event bus
// and remote SDK workers. The control plane already publishes
// ExecutionCancelledEvent on the bus whenever an execution is cancelled
// (per-execution cancel handler, cancel-tree, future sources). This service
// subscribes to that bus and POSTs a per-execution cancel notification to
// the worker that owns the execution, giving the worker's SDK a chance to
// raise CancelledError / abort an AbortController / cancel a context into
// the user's still-running reasoner code.
//
// Transport choice: HTTP callback to a well-known path on the worker
// (BaseURL/_internal/executions/:exec_id/cancel). Workers already expose
// HTTP servers that the control plane dispatches into; reusing that channel
// avoids a second long-lived connection per worker. Older SDK versions
// without the cancel endpoint will return 404 — that's a no-op and
// backwards-compatible.
//
// Best-effort delivery: short timeout, no retries. The control plane
// already records the cancelled status in storage; the SDK callback only
// exists to give the user's in-flight code a chance to short-circuit.
// Missed deliveries fall back to the existing behaviour (in-flight work
// finishes naturally and its output is discarded).
type CancelDispatcher struct {
	store         storage.StorageProvider
	bus           *events.ExecutionEventBus
	httpClient    *http.Client
	cancelPath    string
	internalToken string

	subscriberID string

	stopMu sync.Mutex
	stopCh chan struct{}
	wg     sync.WaitGroup
}

// CancelDispatcherConfig configures a CancelDispatcher.
type CancelDispatcherConfig struct {
	// Bus is the execution event bus to subscribe to. If nil, defaults to
	// events.GlobalExecutionEventBus.
	Bus *events.ExecutionEventBus
	// HTTPClient is the client used to call worker cancel endpoints.
	// If nil, a sane default with a short timeout is used.
	HTTPClient *http.Client
	// CancelPathTemplate is the worker-side path the dispatcher will POST
	// to. The literal substring ":execution_id" is replaced with the actual
	// execution id at dispatch time. Defaults to
	// "/_internal/executions/:execution_id/cancel".
	CancelPathTemplate string
	// SubscriberID identifies this dispatcher's subscription on the bus.
	// Defaults to "cancel-dispatcher".
	SubscriberID string
	// InternalToken is sent as Authorization: Bearer on the cancel
	// callback so workers running with RequireOriginAuth=true accept it.
	// Matches the existing reasoner-dispatch auth contract.
	InternalToken string
}

// NewCancelDispatcher constructs a dispatcher with sensible defaults.
func NewCancelDispatcher(store storage.StorageProvider, cfg CancelDispatcherConfig) *CancelDispatcher {
	bus := cfg.Bus
	if bus == nil {
		bus = events.GlobalExecutionEventBus
	}
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 5 * time.Second}
	}
	path := cfg.CancelPathTemplate
	if strings.TrimSpace(path) == "" {
		path = "/_internal/executions/:execution_id/cancel"
	}
	subscriberID := cfg.SubscriberID
	if strings.TrimSpace(subscriberID) == "" {
		subscriberID = "cancel-dispatcher"
	}
	return &CancelDispatcher{
		store:         store,
		bus:           bus,
		httpClient:    httpClient,
		cancelPath:    path,
		subscriberID:  subscriberID,
		internalToken: cfg.InternalToken,
	}
}

// Start begins consuming the event bus. It returns immediately after
// launching the consumer goroutine; Stop() must be called for clean
// shutdown. Calling Start twice without an intervening Stop is a no-op.
func (d *CancelDispatcher) Start(ctx context.Context) {
	d.stopMu.Lock()
	if d.stopCh != nil {
		d.stopMu.Unlock()
		return
	}
	stop := make(chan struct{})
	d.stopCh = stop
	d.stopMu.Unlock()

	ch := d.bus.Subscribe(d.subscriberID)

	d.wg.Add(1)
	go func() {
		defer d.wg.Done()
		defer d.bus.Unsubscribe(d.subscriberID)

		for {
			select {
			case <-stop:
				return
			case <-ctx.Done():
				return
			case ev, ok := <-ch:
				if !ok {
					return
				}
				if ev.Type != events.ExecutionCancelledEvent {
					continue
				}
				d.dispatch(ctx, ev)
			}
		}
	}()
}

// Stop tears down the subscription and waits for the consumer goroutine
// to exit. Safe to call multiple times.
func (d *CancelDispatcher) Stop() {
	d.stopMu.Lock()
	stop := d.stopCh
	d.stopCh = nil
	d.stopMu.Unlock()
	if stop == nil {
		return
	}
	close(stop)
	d.wg.Wait()
}

// dispatch fires the cancel callback for a single execution. All errors
// here are best-effort: we log and continue. The control plane's record
// of the cancelled status is the source of truth — a failed callback
// means the worker just doesn't get a chance to short-circuit.
func (d *CancelDispatcher) dispatch(ctx context.Context, ev events.ExecutionEvent) {
	executionID := strings.TrimSpace(ev.ExecutionID)
	if executionID == "" {
		return
	}
	agentNodeID := strings.TrimSpace(ev.AgentNodeID)
	if agentNodeID == "" {
		logger.Logger.Debug().Str("execution_id", executionID).Msg("cancel-dispatcher: event missing agent_node_id, skipping")
		return
	}

	agent, err := d.store.GetAgent(ctx, agentNodeID)
	if err != nil {
		logger.Logger.Warn().Err(err).Str("agent_node_id", agentNodeID).Str("execution_id", executionID).Msg("cancel-dispatcher: failed to look up agent")
		return
	}
	if agent == nil {
		logger.Logger.Debug().Str("agent_node_id", agentNodeID).Str("execution_id", executionID).Msg("cancel-dispatcher: agent not registered, skipping")
		return
	}
	baseURL := strings.TrimRight(strings.TrimSpace(agent.BaseURL), "/")
	if baseURL == "" {
		logger.Logger.Debug().Str("agent_node_id", agentNodeID).Str("execution_id", executionID).Msg("cancel-dispatcher: agent has no base_url (likely serverless), skipping")
		return
	}

	// reason is best-effort metadata pulled from the publisher's data map.
	reason := ""
	if m, ok := ev.Data.(map[string]interface{}); ok {
		if r, ok := m["reason"].(string); ok {
			reason = r
		}
	}

	body, _ := json.Marshal(map[string]interface{}{
		"execution_id": executionID,
		"workflow_id":  ev.WorkflowID,
		"reason":       reason,
		"emitted_at":   ev.Timestamp.UTC().Format(time.RFC3339Nano),
	})

	url := baseURL + strings.ReplaceAll(d.cancelPath, ":execution_id", executionID)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		logger.Logger.Warn().Err(err).Str("url", url).Msg("cancel-dispatcher: build request failed")
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Execution-ID", executionID)
	req.Header.Set("X-Workflow-ID", ev.WorkflowID)
	req.Header.Set("X-AgentField-Source", "cancel-dispatcher")
	if d.internalToken != "" {
		req.Header.Set("Authorization", "Bearer "+d.internalToken)
	}

	resp, err := d.httpClient.Do(req)
	if err != nil {
		// Don't escalate timeouts/connection refused — the worker may have
		// already exited or be unreachable. Workers reachable next time
		// will pick up future events.
		var netErr net.Error
		if errors.As(err, &netErr) && netErr.Timeout() {
			logger.Logger.Debug().Str("url", url).Msg("cancel-dispatcher: callback timed out")
		} else {
			logger.Logger.Debug().Err(err).Str("url", url).Msg("cancel-dispatcher: callback failed (best-effort)")
		}
		return
	}
	defer resp.Body.Close()

	switch {
	case resp.StatusCode == http.StatusNotFound:
		// Worker doesn't speak the cancel protocol yet (older SDK). No-op.
	case resp.StatusCode >= 400:
		logger.Logger.Debug().Int("status", resp.StatusCode).Str("url", url).Msg("cancel-dispatcher: worker rejected cancel callback")
	default:
		logger.Logger.Debug().Int("status", resp.StatusCode).Str("execution_id", executionID).Msg("cancel-dispatcher: cancel callback delivered")
	}
}
