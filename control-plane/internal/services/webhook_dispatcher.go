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
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

type WebhookStore interface {
	GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error)
	GetExecutionWebhook(ctx context.Context, executionID string) (*types.ExecutionWebhook, error)
	TryMarkExecutionWebhookInFlight(ctx context.Context, executionID string, now time.Time) (bool, error)
	UpdateExecutionWebhookState(ctx context.Context, executionID string, update types.ExecutionWebhookStateUpdate) error
	StoreExecutionWebhookEvent(ctx context.Context, event *types.ExecutionWebhookEvent) error
	ListDueExecutionWebhooks(ctx context.Context, limit int) ([]*types.ExecutionWebhook, error)
	GetAgent(ctx context.Context, id string) (*types.AgentNode, error)
}

type WebhookDispatcher interface {
	Start(ctx context.Context) error
	Stop(ctx context.Context) error
	Notify(ctx context.Context, executionID string) error
}

type WebhookDispatcherConfig struct {
	Timeout           time.Duration
	MaxAttempts       int
	RetryBackoff      time.Duration
	MaxRetryBackoff   time.Duration
	PollInterval      time.Duration
	PollBatchSize     int
	WorkerCount       int
	QueueSize         int
	ResponseBodyLimit int
}

type webhookDispatcher struct {
	store  WebhookStore
	cfg    WebhookDispatcherConfig
	client *http.Client

	once   sync.Once
	xctx   context.Context
	cancel context.CancelFunc

	jobs chan webhookJob
	wg   sync.WaitGroup
}

type webhookJob struct {
	ExecutionID string
}

func NewWebhookDispatcher(store WebhookStore, cfg WebhookDispatcherConfig) WebhookDispatcher {
	normalized := normalizeWebhookConfig(cfg)
	return &webhookDispatcher{
		store: store,
		cfg:   normalized,
		client: &http.Client{
			Timeout: normalized.Timeout,
		},
	}
}

func normalizeWebhookConfig(cfg WebhookDispatcherConfig) WebhookDispatcherConfig {
	result := cfg
	if result.Timeout <= 0 {
		result.Timeout = 10 * time.Second
	}
	if result.MaxAttempts <= 0 {
		result.MaxAttempts = 5
	}
	if result.RetryBackoff <= 0 {
		result.RetryBackoff = 5 * time.Second
	}
	if result.MaxRetryBackoff <= 0 {
		result.MaxRetryBackoff = 5 * time.Minute
	}
	if result.PollInterval <= 0 {
		result.PollInterval = 5 * time.Second
	}
	if result.PollBatchSize <= 0 {
		result.PollBatchSize = 64
	}
	if result.WorkerCount <= 0 {
		result.WorkerCount = 4
	}
	if result.QueueSize <= 0 {
		result.QueueSize = 256
	}
	if result.ResponseBodyLimit <= 0 {
		result.ResponseBodyLimit = 16 * 1024
	}
	return result
}

func (d *webhookDispatcher) Start(ctx context.Context) error {
	var startErr error
	d.once.Do(func() {
		if d.store == nil {
			startErr = fmt.Errorf("webhook dispatcher requires a store")
			return
		}
		d.jobs = make(chan webhookJob, d.cfg.QueueSize)
		d.xctx, d.cancel = context.WithCancel(ctx)
		for i := 0; i < d.cfg.WorkerCount; i++ {
			d.wg.Add(1)
			go d.worker()
		}
		d.wg.Add(1)
		go d.poller()
		// Warm start: resume any pending deliveries immediately.
		d.scanDue()
	})
	return startErr
}

func (d *webhookDispatcher) Stop(ctx context.Context) error {
	if d.cancel == nil {
		return nil
	}
	d.cancel()

	done := make(chan struct{})
	go func() {
		d.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (d *webhookDispatcher) Notify(ctx context.Context, executionID string) error {
	if executionID == "" {
		return nil
	}
	if d.xctx == nil {
		return fmt.Errorf("webhook dispatcher has not been started")
	}

	opCtx, cancel := context.WithTimeout(ctx, d.cfg.Timeout)
	defer cancel()

	webhook, err := d.store.GetExecutionWebhook(opCtx, executionID)
	if err != nil {
		return err
	}
	if webhook == nil {
		return nil
	}
	if webhook.Status == types.ExecutionWebhookStatusDelivered || webhook.Status == types.ExecutionWebhookStatusFailed {
		return nil
	}

	now := time.Now().UTC()
	if webhook.Status != types.ExecutionWebhookStatusPending || (webhook.NextAttemptAt != nil && webhook.NextAttemptAt.After(now)) {
		update := types.ExecutionWebhookStateUpdate{
			Status:       types.ExecutionWebhookStatusPending,
			AttemptCount: webhook.AttemptCount,
			NextAttemptAt: func() *time.Time {
				t := now
				return &t
			}(),
		}
		if webhook.LastAttemptAt != nil {
			t := webhook.LastAttemptAt.UTC()
			update.LastAttemptAt = &t
		}
		if webhook.LastError != nil {
			errCopy := *webhook.LastError
			update.LastError = &errCopy
		}
		if err := d.store.UpdateExecutionWebhookState(opCtx, executionID, update); err != nil {
			logger.Logger.Warn().Err(err).Str("execution_id", executionID).Msg("failed to refresh webhook schedule")
		}
	}

	scheduledTime := now.Add(d.cfg.Timeout)
	ok, err := d.store.TryMarkExecutionWebhookInFlight(opCtx, executionID, scheduledTime)
	if err != nil {
		return err
	}
	if !ok {
		return nil
	}

	select {
	case <-d.xctx.Done():
		return d.xctx.Err()
	case d.jobs <- webhookJob{ExecutionID: executionID}:
		return nil
	}
}

func (d *webhookDispatcher) poller() {
	defer d.wg.Done()
	ticker := time.NewTicker(d.cfg.PollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-d.xctx.Done():
			return
		case <-ticker.C:
			d.scanDue()
		}
	}
}

func (d *webhookDispatcher) scanDue() {
	if d.xctx == nil {
		return
	}

	opCtx, cancel := context.WithTimeout(d.xctx, d.cfg.Timeout)
	defer cancel()

	due, err := d.store.ListDueExecutionWebhooks(opCtx, d.cfg.PollBatchSize)
	if err != nil {
		logger.Logger.Warn().Err(err).Msg("failed to list due execution webhooks")
		return
	}
	now := time.Now().UTC()
	for _, webhook := range due {
		if webhook == nil {
			continue
		}
		ok, err := d.store.TryMarkExecutionWebhookInFlight(opCtx, webhook.ExecutionID, now)
		if err != nil {
			logger.Logger.Warn().Err(err).Str("execution_id", webhook.ExecutionID).Msg("failed to mark webhook in flight")
			continue
		}
		if !ok {
			continue
		}
		select {
		case <-d.xctx.Done():
			return
		case d.jobs <- webhookJob{ExecutionID: webhook.ExecutionID}:
		}
	}
}

func (d *webhookDispatcher) worker() {
	defer d.wg.Done()
	for {
		select {
		case <-d.xctx.Done():
			return
		case job := <-d.jobs:
			d.process(job)
		}
	}
}

func (d *webhookDispatcher) process(job webhookJob) {
	ctx, cancel := context.WithTimeout(d.xctx, d.cfg.Timeout)
	defer cancel()

	exec, err := d.store.GetExecutionRecord(ctx, job.ExecutionID)
	if err != nil {
		logger.Logger.Warn().Err(err).Str("execution_id", job.ExecutionID).Msg("failed to load execution for webhook")
		return
	}
	if exec == nil {
		logger.Logger.Warn().Str("execution_id", job.ExecutionID).Msg("execution not found for webhook delivery")
		return
	}

	webhook, err := d.store.GetExecutionWebhook(ctx, job.ExecutionID)
	if err != nil {
		logger.Logger.Warn().Err(err).Str("execution_id", job.ExecutionID).Msg("failed to load webhook registration")
		return
	}
	if webhook == nil {
		return
	}

	eventType := determineWebhookEvent(exec.Status)
	payload := d.buildPayload(ctx, exec, eventType)

	body, err := json.Marshal(payload)
	if err != nil {
		logger.Logger.Error().Err(err).Str("execution_id", job.ExecutionID).Msg("failed to encode webhook payload")
		return
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, webhook.URL, bytes.NewReader(body))
	if err != nil {
		logger.Logger.Warn().Err(err).Str("execution_id", job.ExecutionID).Msg("failed to build webhook request")
		return
	}
	req.Header.Set("Content-Type", "application/json")
	for key, value := range webhook.Headers {
		trimmedKey := strings.TrimSpace(key)
		if trimmedKey == "" {
			continue
		}
		req.Header.Set(trimmedKey, value)
	}
	if webhook.Secret != nil {
		req.Header.Set("X-AgentField-Signature", generateWebhookSignature(*webhook.Secret, body))
	}

	var (
		httpStatus   *int
		responseBody *string
		attemptErr   error
	)

	resp, err := d.client.Do(req)
	if err != nil {
		attemptErr = err
	} else {
		defer resp.Body.Close()
		statusCode := resp.StatusCode
		httpStatus = &statusCode
		limited := io.LimitReader(resp.Body, int64(d.cfg.ResponseBodyLimit))
		buf, _ := io.ReadAll(limited)
		if len(buf) > 0 {
			bodyCopy := string(buf)
			responseBody = &bodyCopy
		}
		if statusCode < http.StatusOK || statusCode >= http.StatusMultipleChoices {
			attemptErr = fmt.Errorf("non-2xx response: %d", statusCode)
		}
	}

	statusLabel := "delivered"
	var errorMessage *string
	if attemptErr != nil {
		statusLabel = "failed"
		errCopy := attemptErr.Error()
		errorMessage = &errCopy
	}

	event := &types.ExecutionWebhookEvent{
		ExecutionID:  webhook.ExecutionID,
		EventType:    eventType,
		Status:       statusLabel,
		HTTPStatus:   httpStatus,
		Payload:      body,
		ResponseBody: responseBody,
		ErrorMessage: errorMessage,
		CreatedAt:    time.Now().UTC(),
	}
	if err := d.store.StoreExecutionWebhookEvent(ctx, event); err != nil {
		logger.Logger.Warn().Err(err).Str("execution_id", webhook.ExecutionID).Msg("failed to record webhook delivery attempt")
	}

	attemptCount := webhook.AttemptCount + 1
	now := time.Now().UTC()

	update := types.ExecutionWebhookStateUpdate{
		AttemptCount:  attemptCount,
		LastAttemptAt: &now,
	}

	if attemptErr != nil {
		if attemptCount >= d.cfg.MaxAttempts {
			update.Status = types.ExecutionWebhookStatusFailed
		} else {
			update.Status = types.ExecutionWebhookStatusPending
			next := now.Add(d.computeBackoff(attemptCount))
			update.NextAttemptAt = &next
		}
		update.LastError = errorMessage
	} else {
		update.Status = types.ExecutionWebhookStatusDelivered
	}

	if err := d.store.UpdateExecutionWebhookState(ctx, webhook.ExecutionID, update); err != nil {
		logger.Logger.Error().Err(err).Str("execution_id", webhook.ExecutionID).Msg("failed to update webhook state")
	}
}

func (d *webhookDispatcher) buildPayload(ctx context.Context, exec *types.Execution, eventType string) types.ExecutionWebhookPayload {
	targetType := d.resolveTargetType(ctx, exec)
	payload := types.ExecutionWebhookPayload{
		Event:       eventType,
		ExecutionID: exec.ExecutionID,
		RunID:       exec.RunID,
		Status:      exec.Status,
		Target:      fmt.Sprintf("%s.%s", exec.NodeID, exec.ReasonerID),
		TargetType:  targetType,
		DurationMS:  exec.DurationMS,
	}

	if exec.CompletedAt != nil {
		payload.Timestamp = exec.CompletedAt.UTC().Format(time.RFC3339)
	} else {
		payload.Timestamp = time.Now().UTC().Format(time.RFC3339)
	}

	if eventType == types.WebhookEventExecutionCompleted {
		payload.Result = decodeExecutionPayload(exec.ResultPayload)
	} else {
		payload.ErrorMessage = exec.ErrorMessage
	}

	return payload
}

func (d *webhookDispatcher) resolveTargetType(ctx context.Context, exec *types.Execution) string {
	agent, err := d.store.GetAgent(ctx, exec.NodeID)
	if err != nil || agent == nil {
		return "reasoner"
	}
	for _, reasoner := range agent.Reasoners {
		if reasoner.ID == exec.ReasonerID {
			return "reasoner"
		}
	}
	for _, skill := range agent.Skills {
		if skill.ID == exec.ReasonerID {
			return "skill"
		}
	}
	return "reasoner"
}

func (d *webhookDispatcher) computeBackoff(attempt int) time.Duration {
	if attempt <= 0 {
		attempt = 1
	}
	backoff := d.cfg.RetryBackoff * time.Duration(1<<uint(attempt-1))
	if backoff > d.cfg.MaxRetryBackoff {
		backoff = d.cfg.MaxRetryBackoff
	}
	return backoff
}

func determineWebhookEvent(status string) string {
	switch types.NormalizeExecutionStatus(status) {
	case string(types.ExecutionStatusSucceeded):
		return types.WebhookEventExecutionCompleted
	default:
		return types.WebhookEventExecutionFailed
	}
}

func decodeExecutionPayload(raw json.RawMessage) interface{} {
	if len(raw) == 0 {
		return nil
	}
	var v interface{}
	if err := json.Unmarshal(raw, &v); err == nil {
		return v
	}
	return string(raw)
}

func generateWebhookSignature(secret string, body []byte) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	return "sha256=" + hex.EncodeToString(mac.Sum(nil))
}
