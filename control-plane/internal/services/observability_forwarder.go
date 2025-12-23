package services

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/google/uuid"
)

// ObservabilityWebhookStore defines storage operations for observability webhook config.
type ObservabilityWebhookStore interface {
	GetObservabilityWebhook(ctx context.Context) (*types.ObservabilityWebhookConfig, error)
	AddToDeadLetterQueue(ctx context.Context, event *types.ObservabilityEvent, errorMessage string, retryCount int) error
	GetDeadLetterQueueCount(ctx context.Context) (int64, error)
	GetDeadLetterQueue(ctx context.Context, limit, offset int) ([]types.ObservabilityDeadLetterEntry, error)
	DeleteFromDeadLetterQueue(ctx context.Context, ids []int64) error
	ClearDeadLetterQueue(ctx context.Context) error
}

// ObservabilityForwarder subscribes to all event buses and forwards events to configured webhook.
type ObservabilityForwarder interface {
	Start(ctx context.Context) error
	Stop(ctx context.Context) error
	ReloadConfig(ctx context.Context) error
	GetStatus() types.ObservabilityForwarderStatus
	Redrive(ctx context.Context) types.ObservabilityRedriveResponse
}

// ObservabilityForwarderConfig holds configuration for the forwarder.
type ObservabilityForwarderConfig struct {
	BatchSize         int           // Max events per batch (default: 10)
	BatchTimeout      time.Duration // Max time to wait before sending batch (default: 1s)
	HTTPTimeout       time.Duration // HTTP request timeout (default: 10s)
	MaxAttempts       int           // Max retry attempts (default: 3)
	RetryBackoff      time.Duration // Initial backoff (default: 1s)
	MaxRetryBackoff   time.Duration // Max backoff (default: 30s)
	WorkerCount       int           // Number of parallel workers (default: 2)
	QueueSize         int           // Internal queue size (default: 1000)
	ResponseBodyLimit int           // Max response body to capture (default: 16KB)
}

type observabilityForwarder struct {
	store  ObservabilityWebhookStore
	cfg    ObservabilityForwarderConfig
	client *http.Client

	// Runtime state
	mu         sync.RWMutex
	webhookCfg *types.ObservabilityWebhookConfig

	// Event collection
	eventQueue chan types.ObservabilityEvent

	// Lifecycle
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	// Metrics
	forwarded   atomic.Int64
	dropped     atomic.Int64
	lastForward atomic.Pointer[time.Time]
	lastError   atomic.Pointer[string]
}

// NewObservabilityForwarder creates a new observability forwarder.
func NewObservabilityForwarder(store ObservabilityWebhookStore, cfg ObservabilityForwarderConfig) ObservabilityForwarder {
	normalized := normalizeObservabilityConfig(cfg)
	return &observabilityForwarder{
		store: store,
		cfg:   normalized,
		client: &http.Client{
			Timeout: normalized.HTTPTimeout,
		},
	}
}

func normalizeObservabilityConfig(cfg ObservabilityForwarderConfig) ObservabilityForwarderConfig {
	result := cfg
	if result.BatchSize <= 0 {
		result.BatchSize = 10
	}
	if result.BatchTimeout <= 0 {
		result.BatchTimeout = time.Second
	}
	if result.HTTPTimeout <= 0 {
		result.HTTPTimeout = 10 * time.Second
	}
	if result.MaxAttempts <= 0 {
		result.MaxAttempts = 3
	}
	if result.RetryBackoff <= 0 {
		result.RetryBackoff = time.Second
	}
	if result.MaxRetryBackoff <= 0 {
		result.MaxRetryBackoff = 30 * time.Second
	}
	if result.WorkerCount <= 0 {
		result.WorkerCount = 2
	}
	if result.QueueSize <= 0 {
		result.QueueSize = 1000
	}
	if result.ResponseBodyLimit <= 0 {
		result.ResponseBodyLimit = 16 * 1024
	}
	return result
}

// Start initializes the forwarder and begins listening to event buses.
func (f *observabilityForwarder) Start(ctx context.Context) error {
	if f.store == nil {
		return fmt.Errorf("observability forwarder requires a store")
	}

	// Load initial config
	if err := f.ReloadConfig(ctx); err != nil {
		logger.Logger.Warn().Err(err).Msg("failed to load initial observability webhook config")
	}

	f.eventQueue = make(chan types.ObservabilityEvent, f.cfg.QueueSize)
	f.ctx, f.cancel = context.WithCancel(ctx)

	// Start batch workers
	for i := 0; i < f.cfg.WorkerCount; i++ {
		f.wg.Add(1)
		go f.batchWorker()
	}

	// Subscribe to event buses
	f.wg.Add(3)
	go f.subscribeExecutionEvents()
	go f.subscribeNodeEvents()
	go f.subscribeReasonerEvents()

	logger.Logger.Info().Msg("observability forwarder started")
	return nil
}

// Stop gracefully shuts down the forwarder.
func (f *observabilityForwarder) Stop(ctx context.Context) error {
	if f.cancel == nil {
		return nil
	}
	f.cancel()

	done := make(chan struct{})
	go func() {
		f.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		logger.Logger.Info().Msg("observability forwarder stopped")
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// ReloadConfig reloads webhook configuration from storage.
func (f *observabilityForwarder) ReloadConfig(ctx context.Context) error {
	cfg, err := f.store.GetObservabilityWebhook(ctx)
	if err != nil {
		return fmt.Errorf("failed to load observability webhook config: %w", err)
	}

	f.mu.Lock()
	f.webhookCfg = cfg
	f.mu.Unlock()

	if cfg != nil && cfg.Enabled {
		logger.Logger.Info().Str("url", cfg.URL).Msg("observability webhook configured")
	} else {
		logger.Logger.Debug().Msg("observability webhook not configured or disabled")
	}

	return nil
}

// GetStatus returns the current forwarder status.
func (f *observabilityForwarder) GetStatus() types.ObservabilityForwarderStatus {
	f.mu.RLock()
	cfg := f.webhookCfg
	f.mu.RUnlock()

	status := types.ObservabilityForwarderStatus{
		EventsForwarded: f.forwarded.Load(),
		EventsDropped:   f.dropped.Load(),
	}

	if f.eventQueue != nil {
		status.QueueDepth = len(f.eventQueue)
	}

	if cfg != nil && cfg.Enabled {
		status.Enabled = true
		status.WebhookURL = cfg.URL
	}

	if lastFwd := f.lastForward.Load(); lastFwd != nil {
		status.LastForwardedAt = lastFwd
	}

	if lastErr := f.lastError.Load(); lastErr != nil {
		status.LastError = lastErr
	}

	// Get DLQ count from storage
	if f.store != nil {
		if count, err := f.store.GetDeadLetterQueueCount(context.Background()); err == nil {
			status.DeadLetterCount = count
		}
	}

	return status
}

// Redrive attempts to resend all events in the dead letter queue.
func (f *observabilityForwarder) Redrive(ctx context.Context) types.ObservabilityRedriveResponse {
	f.mu.RLock()
	cfg := f.webhookCfg
	f.mu.RUnlock()

	if cfg == nil || !cfg.Enabled || cfg.URL == "" {
		return types.ObservabilityRedriveResponse{
			Success: false,
			Message: "webhook not configured or disabled",
		}
	}

	// Get all DLQ entries (in batches of 100)
	var processed, failed int
	var successfulIDs []int64
	offset := 0
	batchSize := 100

	for {
		entries, err := f.store.GetDeadLetterQueue(ctx, batchSize, offset)
		if err != nil {
			return types.ObservabilityRedriveResponse{
				Success:   false,
				Message:   fmt.Sprintf("failed to read dead letter queue: %v", err),
				Processed: processed,
				Failed:    failed,
			}
		}

		if len(entries) == 0 {
			break
		}

		// Process each entry
		for _, entry := range entries {
			// Reconstruct the event
			event := types.ObservabilityEvent{
				EventType:   entry.EventType,
				EventSource: entry.EventSource,
				Timestamp:   entry.EventTimestamp.Format(time.RFC3339),
				Data:        json.RawMessage(entry.Payload),
			}

			// Try to parse the payload back to interface{}
			var data interface{}
			if err := json.Unmarshal([]byte(entry.Payload), &data); err == nil {
				event.Data = data
			}

			// Create a single-event batch
			batch := types.ObservabilityEventBatch{
				BatchID:    uuid.New().String(),
				EventCount: 1,
				Events:     []types.ObservabilityEvent{event},
				Timestamp:  time.Now().UTC().Format(time.RFC3339),
			}

			body, err := json.Marshal(batch)
			if err != nil {
				failed++
				continue
			}

			// Try to send with retries
			var sendErr error
			for attempt := 0; attempt < f.cfg.MaxAttempts; attempt++ {
				if attempt > 0 {
					backoff := f.computeBackoff(attempt)
					select {
					case <-ctx.Done():
						return types.ObservabilityRedriveResponse{
							Success:   false,
							Message:   "redrive cancelled",
							Processed: processed,
							Failed:    failed,
						}
					case <-time.After(backoff):
					}
				}

				sendErr = f.doSend(cfg, body)
				if sendErr == nil {
					break
				}
			}

			if sendErr != nil {
				failed++
				logger.Logger.Warn().Err(sendErr).Int64("dlq_id", entry.ID).Msg("failed to redrive event")
			} else {
				processed++
				successfulIDs = append(successfulIDs, entry.ID)
				f.forwarded.Add(1)
				now := time.Now().UTC()
				f.lastForward.Store(&now)
			}
		}

		// Delete successfully processed entries
		if len(successfulIDs) > 0 {
			if err := f.store.DeleteFromDeadLetterQueue(ctx, successfulIDs); err != nil {
				logger.Logger.Error().Err(err).Int("count", len(successfulIDs)).Msg("failed to delete redriven entries from DLQ")
			}
			successfulIDs = successfulIDs[:0]
		}

		offset += batchSize
	}

	message := fmt.Sprintf("redrove %d events", processed)
	if failed > 0 {
		message = fmt.Sprintf("redrove %d events, %d failed", processed, failed)
	}

	return types.ObservabilityRedriveResponse{
		Success:   failed == 0,
		Message:   message,
		Processed: processed,
		Failed:    failed,
	}
}

// subscribeExecutionEvents listens to the execution event bus.
func (f *observabilityForwarder) subscribeExecutionEvents() {
	defer f.wg.Done()

	subscriberID := fmt.Sprintf("observability-forwarder-execution-%s", uuid.New().String()[:8])
	ch := events.GlobalExecutionEventBus.Subscribe(subscriberID)
	defer events.GlobalExecutionEventBus.Unsubscribe(subscriberID)

	for {
		select {
		case <-f.ctx.Done():
			return
		case event, ok := <-ch:
			if !ok {
				return
			}
			f.enqueueEvent(f.transformExecutionEvent(event))
		}
	}
}

// subscribeNodeEvents listens to the node event bus.
func (f *observabilityForwarder) subscribeNodeEvents() {
	defer f.wg.Done()

	subscriberID := fmt.Sprintf("observability-forwarder-node-%s", uuid.New().String()[:8])
	ch := events.GlobalNodeEventBus.Subscribe(subscriberID)
	defer events.GlobalNodeEventBus.Unsubscribe(subscriberID)

	for {
		select {
		case <-f.ctx.Done():
			return
		case event, ok := <-ch:
			if !ok {
				return
			}
			// Skip heartbeat events - they're just keep-alives, not useful for observability
			if event.Type == events.NodeHeartbeat {
				continue
			}
			f.enqueueEvent(f.transformNodeEvent(event))
		}
	}
}

// subscribeReasonerEvents listens to the reasoner event bus.
func (f *observabilityForwarder) subscribeReasonerEvents() {
	defer f.wg.Done()

	subscriberID := fmt.Sprintf("observability-forwarder-reasoner-%s", uuid.New().String()[:8])
	ch := events.GlobalReasonerEventBus.Subscribe(subscriberID)
	defer events.GlobalReasonerEventBus.Unsubscribe(subscriberID)

	for {
		select {
		case <-f.ctx.Done():
			return
		case event, ok := <-ch:
			if !ok {
				return
			}
			// Skip heartbeat events - they're just keep-alives, not useful for observability
			if event.Type == events.Heartbeat {
				continue
			}
			f.enqueueEvent(f.transformReasonerEvent(event))
		}
	}
}

// enqueueEvent adds an event to the queue, dropping if full.
func (f *observabilityForwarder) enqueueEvent(event types.ObservabilityEvent) {
	// Check if webhook is configured and enabled
	f.mu.RLock()
	cfg := f.webhookCfg
	f.mu.RUnlock()

	if cfg == nil || !cfg.Enabled {
		return
	}

	select {
	case f.eventQueue <- event:
		// Event queued successfully
	default:
		// Queue full, drop event
		f.dropped.Add(1)
		logger.Logger.Warn().Str("event_type", event.EventType).Msg("observability event dropped: queue full")
	}
}

// batchWorker collects events and sends them in batches.
func (f *observabilityForwarder) batchWorker() {
	defer f.wg.Done()

	batch := make([]types.ObservabilityEvent, 0, f.cfg.BatchSize)
	timer := time.NewTimer(f.cfg.BatchTimeout)
	defer timer.Stop()

	flushBatch := func() {
		if len(batch) == 0 {
			return
		}

		// Copy batch for sending
		toSend := make([]types.ObservabilityEvent, len(batch))
		copy(toSend, batch)
		batch = batch[:0]

		f.sendBatch(toSend)
	}

	for {
		select {
		case <-f.ctx.Done():
			// Flush remaining events before exit
			flushBatch()
			return

		case event, ok := <-f.eventQueue:
			if !ok {
				flushBatch()
				return
			}
			batch = append(batch, event)
			if len(batch) >= f.cfg.BatchSize {
				flushBatch()
				// Reset timer after flush
				if !timer.Stop() {
					select {
					case <-timer.C:
					default:
					}
				}
				timer.Reset(f.cfg.BatchTimeout)
			}

		case <-timer.C:
			flushBatch()
			timer.Reset(f.cfg.BatchTimeout)
		}
	}
}

// sendBatch sends a batch of events to the configured webhook.
func (f *observabilityForwarder) sendBatch(events []types.ObservabilityEvent) {
	if len(events) == 0 {
		return
	}

	f.mu.RLock()
	cfg := f.webhookCfg
	f.mu.RUnlock()

	if cfg == nil || !cfg.Enabled || cfg.URL == "" {
		return
	}

	batch := types.ObservabilityEventBatch{
		BatchID:    uuid.New().String(),
		EventCount: len(events),
		Events:     events,
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
	}

	body, err := json.Marshal(batch)
	if err != nil {
		logger.Logger.Error().Err(err).Msg("failed to marshal observability event batch")
		return
	}

	// Retry logic
	var lastErr error
	for attempt := 0; attempt < f.cfg.MaxAttempts; attempt++ {
		if attempt > 0 {
			backoff := f.computeBackoff(attempt)
			select {
			case <-f.ctx.Done():
				return
			case <-time.After(backoff):
			}
		}

		err := f.doSend(cfg, body)
		if err == nil {
			// Success
			now := time.Now().UTC()
			f.lastForward.Store(&now)
			f.forwarded.Add(int64(len(events)))
			return
		}
		lastErr = err
	}

	// All attempts failed - write to dead letter queue
	if lastErr != nil {
		errStr := lastErr.Error()
		f.lastError.Store(&errStr)
		f.dropped.Add(int64(len(events)))

		// Write each event to DLQ
		for i := range events {
			if err := f.store.AddToDeadLetterQueue(context.Background(), &events[i], errStr, f.cfg.MaxAttempts); err != nil {
				logger.Logger.Error().Err(err).Str("event_type", events[i].EventType).Msg("failed to add event to dead letter queue")
			}
		}

		logger.Logger.Warn().Err(lastErr).Int("event_count", len(events)).Msg("failed to deliver observability events, added to DLQ")
	}
}

// doSend performs the actual HTTP request.
func (f *observabilityForwarder) doSend(cfg *types.ObservabilityWebhookConfig, body []byte) error {
	ctx, cancel := context.WithTimeout(f.ctx, f.cfg.HTTPTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.URL, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "AgentField-Observability/1.0")

	// Custom headers
	for key, value := range cfg.Headers {
		if key != "" {
			req.Header.Set(key, value)
		}
	}

	// HMAC signature
	if cfg.Secret != nil && *cfg.Secret != "" {
		req.Header.Set("X-AgentField-Signature", generateObservabilitySignature(*cfg.Secret, body))
	}

	resp, err := f.client.Do(req)
	if err != nil {
		return fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	// Read response body (limited)
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, int64(f.cfg.ResponseBodyLimit)))

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return fmt.Errorf("non-2xx response: %d", resp.StatusCode)
	}

	return nil
}

// computeBackoff calculates exponential backoff duration.
func (f *observabilityForwarder) computeBackoff(attempt int) time.Duration {
	if attempt <= 0 {
		attempt = 1
	}
	backoff := f.cfg.RetryBackoff * time.Duration(1<<uint(attempt-1))
	if backoff > f.cfg.MaxRetryBackoff {
		backoff = f.cfg.MaxRetryBackoff
	}
	return backoff
}

// Event transformers

func (f *observabilityForwarder) transformExecutionEvent(e events.ExecutionEvent) types.ObservabilityEvent {
	data := map[string]interface{}{
		"execution_id":  e.ExecutionID,
		"workflow_id":   e.WorkflowID,
		"agent_node_id": e.AgentNodeID,
		"status":        e.Status,
	}
	if e.Data != nil {
		data["payload"] = e.Data
	}

	return types.ObservabilityEvent{
		EventType:   string(e.Type),
		EventSource: "execution",
		Timestamp:   e.Timestamp.Format(time.RFC3339),
		Data:        data,
	}
}

func (f *observabilityForwarder) transformNodeEvent(e events.NodeEvent) types.ObservabilityEvent {
	data := map[string]interface{}{
		"node_id": e.NodeID,
		"status":  e.Status,
	}
	if e.OldStatus != nil {
		data["old_status"] = e.OldStatus
	}
	if e.NewStatus != nil {
		data["new_status"] = e.NewStatus
	}
	if e.Source != "" {
		data["source"] = e.Source
	}
	if e.Reason != "" {
		data["reason"] = e.Reason
	}
	if e.Data != nil {
		data["payload"] = e.Data
	}

	return types.ObservabilityEvent{
		EventType:   string(e.Type),
		EventSource: "node",
		Timestamp:   e.Timestamp.Format(time.RFC3339),
		Data:        data,
	}
}

func (f *observabilityForwarder) transformReasonerEvent(e events.ReasonerEvent) types.ObservabilityEvent {
	data := map[string]interface{}{
		"reasoner_id": e.ReasonerID,
		"node_id":     e.NodeID,
		"status":      e.Status,
	}
	if e.Data != nil {
		data["payload"] = e.Data
	}

	return types.ObservabilityEvent{
		EventType:   string(e.Type),
		EventSource: "reasoner",
		Timestamp:   e.Timestamp.Format(time.RFC3339),
		Data:        data,
	}
}

func generateObservabilitySignature(secret string, body []byte) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	return "sha256=" + hex.EncodeToString(mac.Sum(nil))
}
