package handlers

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

// waitForCleanupCalls polls until the loop has issued at least n cleanup calls.
func waitForCleanupCalls(t *testing.T, store *cleanupStoreMock, n int) {
	t.Helper()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if len(store.getCleanupCalls()) >= n {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("expected at least %d cleanup calls, got %d", n, len(store.getCleanupCalls()))
}

// The constructor must keep production on the 30s delay.
func TestExecutionCleanupService_DefaultsToProductionInitialDelay(t *testing.T) {
	svc := NewExecutionCleanupService(&cleanupStoreMock{}, testExecutionCleanupConfig(10))

	if svc.initialDelay != defaultInitialCleanupDelay {
		t.Fatalf("expected initial delay %v, got %v", defaultInitialCleanupDelay, svc.initialDelay)
	}
	if defaultInitialCleanupDelay != 30*time.Second {
		t.Fatalf("production initial delay changed: got %v", defaultInitialCleanupDelay)
	}
}

// cleanupLoop's first pass fires off initialDelay, not the ticker.
func TestExecutionCleanupService_CleanupLoop_RunsInitialCleanup(t *testing.T) {
	setupExecutionCleanupTestLogger(t)

	store := &cleanupStoreMock{}
	cfg := testExecutionCleanupConfig(10)
	cfg.CleanupInterval = time.Hour // ticker must not be what fires

	svc := NewExecutionCleanupService(store, cfg)
	svc.initialDelay = 5 * time.Millisecond

	if err := svc.Start(context.Background()); err != nil {
		t.Fatalf("start: %v", err)
	}
	waitForCleanupCalls(t, store, 1)
	if err := svc.Stop(); err != nil {
		t.Fatalf("stop: %v", err)
	}
}

// cleanupLoop keeps running on the ticker after the initial pass.
func TestExecutionCleanupService_CleanupLoop_RunsOnTicker(t *testing.T) {
	setupExecutionCleanupTestLogger(t)

	store := &cleanupStoreMock{}
	cfg := testExecutionCleanupConfig(10)
	cfg.CleanupInterval = 5 * time.Millisecond

	svc := NewExecutionCleanupService(store, cfg)
	svc.initialDelay = time.Hour // initial timer must not be what fires

	if err := svc.Start(context.Background()); err != nil {
		t.Fatalf("start: %v", err)
	}
	waitForCleanupCalls(t, store, 2)
	if err := svc.Stop(); err != nil {
		t.Fatalf("stop: %v", err)
	}
}

// A failing retry pass is logged and does not abort the cleanup.
func TestExecutionCleanupService_PerformCleanup_ContinuesWhenRetryStaleFails(t *testing.T) {
	logBuffer := setupExecutionCleanupTestLogger(t)

	store := &cleanupStoreMock{
		retryStaleErrs:   []error{errors.New("retry boom")},
		cleanupResponses: []cleanupResponse{{count: 2}},
	}
	cfg := testExecutionCleanupConfig(10)
	cfg.MaxRetries = 3

	svc := NewExecutionCleanupService(store, cfg)
	svc.performCleanup(context.Background())

	if got := len(store.getRetryStaleCalls()); got != 1 {
		t.Fatalf("expected 1 retry call, got %d", got)
	}
	if got := len(store.getCleanupCalls()); got != 1 {
		t.Fatalf("retry failure should not stop the delete pass, got %d cleanup calls", got)
	}

	totalCleaned, _, lastErr := svc.GetMetrics()
	if totalCleaned != 2 {
		t.Fatalf("expected 2 cleaned, got %d", totalCleaned)
	}
	if lastErr != nil {
		t.Fatalf("retry failure must not be recorded as a cleanup error, got %v", lastErr)
	}
	if logs := logBuffer.String(); !strings.Contains(logs, "failed to retry stale workflow executions") {
		t.Fatalf("expected retry failure log, got: %s", logs)
	}
}
