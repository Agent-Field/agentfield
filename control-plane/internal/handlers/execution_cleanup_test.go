package handlers

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/rs/zerolog"
)

type cleanupResponse struct {
	count int
	err   error
}

type cleanupCall struct {
	ctx             context.Context
	retentionPeriod time.Duration
	batchSize       int
}

type markStaleCall struct {
	ctx        context.Context
	staleAfter time.Duration
	limit      int
}

type cleanupStoreMock struct {
	storage.StorageProvider

	mu sync.Mutex

	cleanupCalls     []cleanupCall
	cleanupResponses []cleanupResponse

	markStaleCalls     []markStaleCall
	markStaleResponses []cleanupResponse

	markStaleWfCalls     []markStaleCall
	markStaleWfResponses []cleanupResponse

	retryStaleCalls []markStaleCall
	retryStaleIDs   [][]string
	retryStaleErrs  []error
}

func (m *cleanupStoreMock) CleanupOldExecutions(ctx context.Context, retentionPeriod time.Duration, batchSize int) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	callIndex := len(m.cleanupCalls)
	m.cleanupCalls = append(m.cleanupCalls, cleanupCall{
		ctx:             ctx,
		retentionPeriod: retentionPeriod,
		batchSize:       batchSize,
	})

	if callIndex < len(m.cleanupResponses) {
		return m.cleanupResponses[callIndex].count, m.cleanupResponses[callIndex].err
	}

	return 0, nil
}

func (m *cleanupStoreMock) MarkStaleExecutions(ctx context.Context, staleAfter time.Duration, limit int) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	callIndex := len(m.markStaleCalls)
	m.markStaleCalls = append(m.markStaleCalls, markStaleCall{
		ctx:        ctx,
		staleAfter: staleAfter,
		limit:      limit,
	})

	if callIndex < len(m.markStaleResponses) {
		return m.markStaleResponses[callIndex].count, m.markStaleResponses[callIndex].err
	}

	return 0, nil
}

func (m *cleanupStoreMock) getCleanupCalls() []cleanupCall {
	m.mu.Lock()
	defer m.mu.Unlock()

	out := make([]cleanupCall, len(m.cleanupCalls))
	copy(out, m.cleanupCalls)
	return out
}

func (m *cleanupStoreMock) getMarkStaleCalls() []markStaleCall {
	m.mu.Lock()
	defer m.mu.Unlock()

	out := make([]markStaleCall, len(m.markStaleCalls))
	copy(out, m.markStaleCalls)
	return out
}

func (m *cleanupStoreMock) MarkStaleWorkflowExecutions(ctx context.Context, staleAfter time.Duration, limit int) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	callIndex := len(m.markStaleWfCalls)
	m.markStaleWfCalls = append(m.markStaleWfCalls, markStaleCall{
		ctx:        ctx,
		staleAfter: staleAfter,
		limit:      limit,
	})

	if callIndex < len(m.markStaleWfResponses) {
		return m.markStaleWfResponses[callIndex].count, m.markStaleWfResponses[callIndex].err
	}

	return 0, nil
}

func (m *cleanupStoreMock) getMarkStaleWfCalls() []markStaleCall {
	m.mu.Lock()
	defer m.mu.Unlock()

	out := make([]markStaleCall, len(m.markStaleWfCalls))
	copy(out, m.markStaleWfCalls)
	return out
}

func (m *cleanupStoreMock) RetryStaleWorkflowExecutions(ctx context.Context, staleAfter time.Duration, _ int, limit int) ([]string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	callIndex := len(m.retryStaleCalls)
	m.retryStaleCalls = append(m.retryStaleCalls, markStaleCall{
		ctx:        ctx,
		staleAfter: staleAfter,
		limit:      limit,
	})

	var ids []string
	if callIndex < len(m.retryStaleIDs) {
		ids = m.retryStaleIDs[callIndex]
	}
	var err error
	if callIndex < len(m.retryStaleErrs) {
		err = m.retryStaleErrs[callIndex]
	}
	return ids, err
}

func (m *cleanupStoreMock) getRetryStaleCalls() []markStaleCall {
	m.mu.Lock()
	defer m.mu.Unlock()

	out := make([]markStaleCall, len(m.retryStaleCalls))
	copy(out, m.retryStaleCalls)
	return out
}

type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (sb *syncBuffer) Write(p []byte) (n int, err error) {
	sb.mu.Lock()
	defer sb.mu.Unlock()
	return sb.buf.Write(p)
}

func (sb *syncBuffer) String() string {
	sb.mu.Lock()
	defer sb.mu.Unlock()
	return sb.buf.String()
}

func setupExecutionCleanupTestLogger(t *testing.T) *syncBuffer {
	t.Helper()

	sb := &syncBuffer{}
	previousLogger := logger.Logger

	logger.Logger = zerolog.New(sb).Level(zerolog.DebugLevel).With().Timestamp().Logger()
	t.Cleanup(func() {
		logger.Logger = previousLogger
	})

	return sb
}

func testExecutionCleanupConfig(batchSize int) config.ExecutionCleanupConfig {
	return config.ExecutionCleanupConfig{
		Enabled:               true,
		RetentionPeriod:       24 * time.Hour,
		CleanupInterval:       time.Hour,
		BatchSize:             batchSize,
		StaleExecutionTimeout: 30 * time.Minute,
	}
}

func TestExecutionCleanupService_PerformCleanup_CleansStaleExecutionsInBatches(t *testing.T) {
	logBuffer := setupExecutionCleanupTestLogger(t)
	store := &cleanupStoreMock{
		markStaleResponses: []cleanupResponse{{count: 4}},
		cleanupResponses: []cleanupResponse{
			{count: 3},
			{count: 1},
		},
	}

	cfg := testExecutionCleanupConfig(3)
	service := NewExecutionCleanupService(store, cfg)
	service.performCleanup(context.Background())

	markCalls := store.getMarkStaleCalls()
	if len(markCalls) != 1 {
		t.Fatalf("expected 1 mark stale call, got %d", len(markCalls))
	}
	if markCalls[0].staleAfter != cfg.StaleExecutionTimeout {
		t.Fatalf("expected stale timeout %v, got %v", cfg.StaleExecutionTimeout, markCalls[0].staleAfter)
	}
	if markCalls[0].limit != cfg.BatchSize {
		t.Fatalf("expected stale call limit %d, got %d", cfg.BatchSize, markCalls[0].limit)
	}

	cleanupCalls := store.getCleanupCalls()
	if len(cleanupCalls) != 2 {
		t.Fatalf("expected 2 cleanup calls, got %d", len(cleanupCalls))
	}
	for i, call := range cleanupCalls {
		if call.retentionPeriod != cfg.RetentionPeriod {
			t.Fatalf("call %d retention period mismatch: expected %v, got %v", i, cfg.RetentionPeriod, call.retentionPeriod)
		}
		if call.batchSize != cfg.BatchSize {
			t.Fatalf("call %d batch size mismatch: expected %d, got %d", i, cfg.BatchSize, call.batchSize)
		}
	}

	totalCleaned, lastCleanupTime, lastErr := service.GetMetrics()
	if totalCleaned != 4 {
		t.Fatalf("expected total cleaned to be 4, got %d", totalCleaned)
	}
	if lastCleanupTime.IsZero() {
		t.Fatal("expected last cleanup time to be set")
	}
	if lastErr != nil {
		t.Fatalf("expected last cleanup error to be nil, got %v", lastErr)
	}

	logs := logBuffer.String()
	if !strings.Contains(logs, "marked stale executions as timed out") {
		t.Fatalf("expected stale-marking log to be present, got logs: %s", logs)
	}
	if !strings.Contains(logs, "Execution cleanup completed") {
		t.Fatalf("expected completion log to be present, got logs: %s", logs)
	}
}

func TestExecutionCleanupService_PerformCleanup_SkipsStaleMarkWhenTimeoutIsDisabled(t *testing.T) {
	logBuffer := setupExecutionCleanupTestLogger(t)
	store := &cleanupStoreMock{
		cleanupResponses: []cleanupResponse{{count: 0}},
	}

	cfg := testExecutionCleanupConfig(10)
	cfg.StaleExecutionTimeout = 0

	service := NewExecutionCleanupService(store, cfg)
	service.performCleanup(context.Background())

	if len(store.getMarkStaleCalls()) != 0 {
		t.Fatalf("expected no mark stale calls when timeout is disabled, got %d", len(store.getMarkStaleCalls()))
	}
	if len(store.getCleanupCalls()) != 1 {
		t.Fatalf("expected 1 cleanup call, got %d", len(store.getCleanupCalls()))
	}

	totalCleaned, lastCleanupTime, lastErr := service.GetMetrics()
	if totalCleaned != 0 {
		t.Fatalf("expected total cleaned to be 0, got %d", totalCleaned)
	}
	if lastCleanupTime.IsZero() {
		t.Fatal("expected last cleanup time to be set")
	}
	if lastErr != nil {
		t.Fatalf("expected last cleanup error to be nil, got %v", lastErr)
	}

	if !strings.Contains(logBuffer.String(), "Execution cleanup completed - no executions to clean") {
		t.Fatalf("expected no-work cleanup log to be present, got logs: %s", logBuffer.String())
	}
}

func TestExecutionCleanupService_PerformCleanup_NoStaleExecutions(t *testing.T) {
	logBuffer := setupExecutionCleanupTestLogger(t)
	store := &cleanupStoreMock{
		markStaleResponses: []cleanupResponse{{count: 0}},
		cleanupResponses:   []cleanupResponse{{count: 0}},
	}

	cfg := testExecutionCleanupConfig(8)
	service := NewExecutionCleanupService(store, cfg)
	service.performCleanup(context.Background())

	if len(store.getMarkStaleCalls()) != 1 {
		t.Fatalf("expected one mark stale call, got %d", len(store.getMarkStaleCalls()))
	}
	if len(store.getCleanupCalls()) != 1 {
		t.Fatalf("expected one cleanup call, got %d", len(store.getCleanupCalls()))
	}

	totalCleaned, lastCleanupTime, lastErr := service.GetMetrics()
	if totalCleaned != 0 {
		t.Fatalf("expected total cleaned to be 0, got %d", totalCleaned)
	}
	if lastCleanupTime.IsZero() {
		t.Fatal("expected last cleanup time to be set")
	}
	if lastErr != nil {
		t.Fatalf("expected last cleanup error to be nil, got %v", lastErr)
	}

	logs := logBuffer.String()
	if strings.Contains(logs, "marked stale executions as timed out") {
		t.Fatalf("did not expect stale-marked log when stale count is zero, got logs: %s", logs)
	}
	if !strings.Contains(logs, "Execution cleanup completed - no executions to clean") {
		t.Fatalf("expected no-work cleanup log to be present, got logs: %s", logs)
	}
}

func TestExecutionCleanupService_PerformCleanup_AllExecutionsStale(t *testing.T) {
	logBuffer := setupExecutionCleanupTestLogger(t)
	store := &cleanupStoreMock{
		markStaleResponses: []cleanupResponse{{count: 6}},
		cleanupResponses:   []cleanupResponse{{count: 0}},
	}

	cfg := testExecutionCleanupConfig(6)
	service := NewExecutionCleanupService(store, cfg)
	service.performCleanup(context.Background())

	if len(store.getMarkStaleCalls()) != 1 {
		t.Fatalf("expected one mark stale call, got %d", len(store.getMarkStaleCalls()))
	}
	if len(store.getCleanupCalls()) != 1 {
		t.Fatalf("expected one cleanup call, got %d", len(store.getCleanupCalls()))
	}

	// Service metrics track deleted old executions, not stale-mark updates.
	totalCleaned, lastCleanupTime, lastErr := service.GetMetrics()
	if totalCleaned != 0 {
		t.Fatalf("expected total cleaned to be 0 when only stale executions are marked, got %d", totalCleaned)
	}
	if lastCleanupTime.IsZero() {
		t.Fatal("expected last cleanup time to be set")
	}
	if lastErr != nil {
		t.Fatalf("expected last cleanup error to be nil, got %v", lastErr)
	}

	logs := logBuffer.String()
	if !strings.Contains(logs, "marked stale executions as timed out") {
		t.Fatalf("expected stale-marked log to be present, got logs: %s", logs)
	}
	if !strings.Contains(logs, "Execution cleanup completed - no executions to clean") {
		t.Fatalf("expected no-work cleanup log to be present, got logs: %s", logs)
	}
}

func TestExecutionCleanupService_PerformCleanup_AccumulatesMetricsAcrossRuns(t *testing.T) {
	store := &cleanupStoreMock{
		cleanupResponses: []cleanupResponse{
			{count: 2},
			{count: 1},
			{count: 1},
		},
	}

	cfg := testExecutionCleanupConfig(2)
	cfg.StaleExecutionTimeout = 0

	service := NewExecutionCleanupService(store, cfg)
	service.performCleanup(context.Background())
	service.performCleanup(context.Background())

	totalCleaned, lastCleanupTime, lastErr := service.GetMetrics()
	if totalCleaned != 4 {
		t.Fatalf("expected total cleaned to be 4 after two runs, got %d", totalCleaned)
	}
	if lastCleanupTime.IsZero() {
		t.Fatal("expected last cleanup time to be set")
	}
	if lastErr != nil {
		t.Fatalf("expected last cleanup error to be nil, got %v", lastErr)
	}
}

func TestExecutionCleanupService_PerformCleanup_StoresErrorWhenCleanupFails(t *testing.T) {
	logBuffer := setupExecutionCleanupTestLogger(t)
	cleanupErr := errors.New("cleanup database failure")
	store := &cleanupStoreMock{
		cleanupResponses: []cleanupResponse{
			{count: 2},
			{err: cleanupErr},
		},
	}

	cfg := testExecutionCleanupConfig(2)
	cfg.StaleExecutionTimeout = 0

	service := NewExecutionCleanupService(store, cfg)
	service.performCleanup(context.Background())

	if len(store.getCleanupCalls()) != 2 {
		t.Fatalf("expected 2 cleanup calls before failure, got %d", len(store.getCleanupCalls()))
	}

	totalCleaned, lastCleanupTime, lastErr := service.GetMetrics()
	if totalCleaned != 0 {
		t.Fatalf("expected total cleaned metric to remain 0 on failed run, got %d", totalCleaned)
	}
	if lastCleanupTime.IsZero() {
		t.Fatal("expected last cleanup time to be set on failure")
	}
	if !errors.Is(lastErr, cleanupErr) {
		t.Fatalf("expected last error %v, got %v", cleanupErr, lastErr)
	}

	if !strings.Contains(logBuffer.String(), "Failed to cleanup old executions") {
		t.Fatalf("expected failure log to be present, got logs: %s", logBuffer.String())
	}
}

func TestExecutionCleanupService_PerformCleanup_ContinuesWhenMarkStaleFails(t *testing.T) {
	logBuffer := setupExecutionCleanupTestLogger(t)
	markErr := errors.New("mark stale failed")
	store := &cleanupStoreMock{
		markStaleResponses: []cleanupResponse{{err: markErr}},
		cleanupResponses:   []cleanupResponse{{count: 0}},
	}

	cfg := testExecutionCleanupConfig(5)
	service := NewExecutionCleanupService(store, cfg)
	service.performCleanup(context.Background())

	if len(store.getMarkStaleCalls()) != 1 {
		t.Fatalf("expected 1 mark stale call, got %d", len(store.getMarkStaleCalls()))
	}
	if len(store.getCleanupCalls()) != 1 {
		t.Fatalf("expected cleanup to continue after stale-marking failure")
	}

	totalCleaned, lastCleanupTime, lastErr := service.GetMetrics()
	if totalCleaned != 0 {
		t.Fatalf("expected total cleaned to be 0, got %d", totalCleaned)
	}
	if lastCleanupTime.IsZero() {
		t.Fatal("expected last cleanup time to be set")
	}
	if lastErr != nil {
		t.Fatalf("expected last cleanup error to be nil after successful cleanup, got %v", lastErr)
	}

	if !strings.Contains(logBuffer.String(), "failed to mark stale executions as timed out") {
		t.Fatalf("expected stale-mark failure log to be present, got logs: %s", logBuffer.String())
	}
}

func TestExecutionCleanupService_PerformCleanup_StopsWhenContextIsCancelled(t *testing.T) {
	logBuffer := setupExecutionCleanupTestLogger(t)
	store := &cleanupStoreMock{
		cleanupResponses: []cleanupResponse{{count: 2}},
	}

	cfg := testExecutionCleanupConfig(2)
	cfg.StaleExecutionTimeout = 0

	service := NewExecutionCleanupService(store, cfg)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	service.performCleanup(ctx)

	if len(store.getCleanupCalls()) != 1 {
		t.Fatalf("expected cleanup to stop after first batch on cancellation, got %d calls", len(store.getCleanupCalls()))
	}

	totalCleaned, lastCleanupTime, lastErr := service.GetMetrics()
	if totalCleaned != 0 {
		t.Fatalf("expected total cleaned metric to remain 0 when run is cancelled, got %d", totalCleaned)
	}
	if !lastCleanupTime.IsZero() {
		t.Fatalf("expected last cleanup time to remain zero on cancellation, got %v", lastCleanupTime)
	}
	if lastErr != nil {
		t.Fatalf("expected last cleanup error to remain nil on cancellation, got %v", lastErr)
	}

	if !strings.Contains(logBuffer.String(), "Execution cleanup cancelled") {
		t.Fatalf("expected cancellation log to be present, got logs: %s", logBuffer.String())
	}
}

func TestExecutionCleanupService_PerformCleanup_MarksStaleWorkflowExecutions(t *testing.T) {
	logBuffer := setupExecutionCleanupTestLogger(t)
	store := &cleanupStoreMock{
		markStaleResponses:   []cleanupResponse{{count: 1}},
		markStaleWfResponses: []cleanupResponse{{count: 3}},
		cleanupResponses:     []cleanupResponse{{count: 0}},
	}

	cfg := testExecutionCleanupConfig(10)
	service := NewExecutionCleanupService(store, cfg)
	service.performCleanup(context.Background())

	markCalls := store.getMarkStaleCalls()
	if len(markCalls) != 1 {
		t.Fatalf("expected 1 mark stale call, got %d", len(markCalls))
	}

	wfCalls := store.getMarkStaleWfCalls()
	if len(wfCalls) != 1 {
		t.Fatalf("expected 1 mark stale workflow call, got %d", len(wfCalls))
	}
	if wfCalls[0].staleAfter != cfg.StaleExecutionTimeout {
		t.Fatalf("expected stale timeout %v, got %v", cfg.StaleExecutionTimeout, wfCalls[0].staleAfter)
	}
	if wfCalls[0].limit != cfg.BatchSize {
		t.Fatalf("expected batch size %d, got %d", cfg.BatchSize, wfCalls[0].limit)
	}

	logs := logBuffer.String()
	if !strings.Contains(logs, "marked stale executions as timed out") {
		t.Fatalf("expected stale execution log, got logs: %s", logs)
	}
	if !strings.Contains(logs, "marked stale workflow executions as timed out") {
		t.Fatalf("expected stale workflow execution log, got logs: %s", logs)
	}
}

func TestExecutionCleanupService_PerformCleanup_ContinuesWhenMarkStaleWorkflowFails(t *testing.T) {
	logBuffer := setupExecutionCleanupTestLogger(t)
	store := &cleanupStoreMock{
		markStaleResponses:   []cleanupResponse{{count: 0}},
		markStaleWfResponses: []cleanupResponse{{err: errors.New("workflow stale failed")}},
		cleanupResponses:     []cleanupResponse{{count: 0}},
	}

	cfg := testExecutionCleanupConfig(5)
	service := NewExecutionCleanupService(store, cfg)
	service.performCleanup(context.Background())

	// Cleanup should still proceed despite workflow stale-marking failure
	if len(store.getCleanupCalls()) != 1 {
		t.Fatalf("expected cleanup to continue after workflow stale-marking failure")
	}

	logs := logBuffer.String()
	if !strings.Contains(logs, "failed to mark stale workflow executions as timed out") {
		t.Fatalf("expected workflow stale-mark failure log, got logs: %s", logs)
	}
}

func TestExecutionCleanupService_PerformCleanup_RetriesStaleWorkflowExecutionsBeforeTimingOut(t *testing.T) {
	logBuffer := setupExecutionCleanupTestLogger(t)
	store := &cleanupStoreMock{
		retryStaleIDs:        [][]string{{"exec-1"}},
		markStaleResponses:   []cleanupResponse{{count: 0}},
		markStaleWfResponses: []cleanupResponse{{count: 0}},
		cleanupResponses:     []cleanupResponse{{count: 0}},
	}

	cfg := testExecutionCleanupConfig(5)
	cfg.MaxRetries = 2

	service := NewExecutionCleanupService(store, cfg)
	service.performCleanup(context.Background())

	if len(store.getRetryStaleCalls()) != 1 {
		t.Fatalf("expected 1 retry stale call, got %d", len(store.getRetryStaleCalls()))
	}
	if len(store.getMarkStaleCalls()) != 1 {
		t.Fatalf("expected 1 stale execution mark call, got %d", len(store.getMarkStaleCalls()))
	}
	if len(store.getMarkStaleWfCalls()) != 1 {
		t.Fatalf("expected 1 stale workflow mark call, got %d", len(store.getMarkStaleWfCalls()))
	}
	if !strings.Contains(logBuffer.String(), "retried stale workflow executions") {
		t.Fatalf("expected retry log, got logs: %s", logBuffer.String())
	}
}

func TestExecutionCleanupService_CleanupLoop_StopsOnContextCancellation(t *testing.T) {
	logBuffer := setupExecutionCleanupTestLogger(t)
	store := &cleanupStoreMock{}
	cfg := testExecutionCleanupConfig(5)

	service := NewExecutionCleanupService(store, cfg)
	ctx, cancel := context.WithCancel(context.Background())
	if err := service.Start(ctx); err != nil {
		t.Fatalf("expected start to succeed, got error: %v", err)
	}

	// Yield so the cleanup goroutine enters its select loop before cancellation
	time.Sleep(10 * time.Millisecond)

	cancel()

	stopDone := make(chan error, 1)
	go func() {
		stopDone <- service.Stop()
	}()

	select {
	case err := <-stopDone:
		if err != nil {
			t.Fatalf("expected stop to succeed, got error: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("service stop timed out after context cancellation")
	}

	if len(store.getCleanupCalls()) != 0 {
		t.Fatalf("expected no cleanup calls before cancellation, got %d", len(store.getCleanupCalls()))
	}

	logs := logBuffer.String()
	if !strings.Contains(logs, "Execution cleanup loop stopped due to context cancellation") &&
		!strings.Contains(logs, "Execution cleanup loop stopped") {
		t.Fatalf("expected cleanup-loop stop log to be present, got logs: %s", logs)
	}
}
