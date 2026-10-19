package observability

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

func TestUsageOutboxRetriesRestartDedupAndPrivacy(t *testing.T) {
	home := t.TempDir()
	cfg := config.TelemetryConfig{Endpoint: "https://example.invalid", InstallID: "test-install"}
	s, err := NewTelemetryService(cfg, home, "local", "v1.2.3")
	if err != nil {
		t.Fatal(err)
	}
	s.ctx = context.Background()
	row := &types.ExecutionUsage{Source: "llm", RoutingProvider: "openrouter", Provider: "anthropic", Model: "private/customer/deepseek-v4", InputTokens: 10, OutputTokens: 5, TotalTokens: 15}
	s.recordUsage("secret-execution", 0, row)
	s.recordUsage("secret-execution", 0, row)
	files, _ := os.ReadDir(s.usageOutbox)
	if len(files) != 1 {
		t.Fatalf("pending files=%d", len(files))
	}
	data, _ := os.ReadFile(filepath.Join(s.usageOutbox, files[0].Name()))
	if strings.Contains(string(data), "secret-execution") || strings.Contains(string(data), "private/customer") {
		t.Fatal("raw identifiers leaked")
	}
	s.sender = func(context.Context, string, time.Duration, TelemetryEvent) error { return errors.New("offline") }
	s.flushUsage()
	again, _ := NewTelemetryService(cfg, home, "local", "v1.2.3")
	again.ctx = context.Background()
	sent := 0
	again.sender = func(_ context.Context, _ string, _ time.Duration, e TelemetryEvent) error {
		sent++
		if e.Properties["routing_provider"] != "openrouter" || e.Properties["model_family"] != "deepseek" {
			t.Fatalf("props=%v", e.Properties)
		}
		return nil
	}
	again.flushUsage()
	again.recordUsage("secret-execution", 0, row)
	again.flushUsage()
	if sent != 1 {
		t.Fatalf("sent=%d", sent)
	}
}

func TestMissingUsageOmitsTokensAndLegacyRouteIsUnknown(t *testing.T) {
	s, _ := NewTelemetryService(config.TelemetryConfig{Endpoint: "x", InstallID: "test"}, t.TempDir(), "local", "v1")
	s.recordUsage("execution", 0, &types.ExecutionUsage{Source: "llm", Provider: "openrouter", Model: "custom-secret", UsageStatus: "missing", TotalTokens: 123})
	files, _ := os.ReadDir(s.usageOutbox)
	data, _ := os.ReadFile(filepath.Join(s.usageOutbox, files[0].Name()))
	var e TelemetryEvent
	_ = json.Unmarshal(data, &e)
	if e.Properties["routing_provider"] != "unknown" || e.Properties["model_family"] != "other" {
		t.Fatal(e.Properties)
	}
	if _, ok := e.Properties["total_tokens"]; ok {
		t.Fatal("missing receipt was counted")
	}
}

func TestUsageModelFamilyBoundedAndTokenTotalAuthoritative(t *testing.T) {
	for model, want := range map[string]string{"qwen/qwen3-coder": "qwen", "deepseek/deepseek-v4.1": "deepseek", "private/custom-claude-secret": "other", "openrouter/anthropic/claude-4": "claude"} {
		if got := usageModelFamily(model); got != want {
			t.Fatalf("%q: got %q want %q", model, got, want)
		}
	}
	s, _ := NewTelemetryService(config.TelemetryConfig{Endpoint: "x", InstallID: "test"}, t.TempDir(), "local", "v1")
	s.recordUsage("sum", 0, &types.ExecutionUsage{Source: "llm", InputTokens: 10, OutputTokens: 5, TotalTokens: 99})
	files, _ := os.ReadDir(s.usageOutbox)
	data, _ := os.ReadFile(filepath.Join(s.usageOutbox, files[0].Name()))
	var e TelemetryEvent
	_ = json.Unmarshal(data, &e)
	if e.Properties["total_tokens"] != float64(15) {
		t.Fatal(e.Properties)
	}
}

func TestUsageTelemetryExcludesHarnessRollupsAndUnknownSources(t *testing.T) {
	s, _ := NewTelemetryService(config.TelemetryConfig{Endpoint: "x", InstallID: "test"}, t.TempDir(), "local", "v1")
	for i, source := range []string{"harness", "", "parent_rollup"} {
		s.recordUsage("parent", i, &types.ExecutionUsage{Source: source, InputTokens: 100, OutputTokens: 20})
	}
	if _, err := os.Stat(s.usageOutbox); !os.IsNotExist(err) {
		t.Fatal("non-LLM usage was queued")
	}
	s.recordUsage("child", 0, &types.ExecutionUsage{Source: "llm", InputTokens: 10, OutputTokens: 2})
	files, _ := os.ReadDir(s.usageOutbox)
	if len(files) != 1 {
		t.Fatalf("expected one owned SDK call, got %d", len(files))
	}
}

func TestUsageAcknowledgementsSeparateFromPendingAndExpire(t *testing.T) {
	s, _ := NewTelemetryService(config.TelemetryConfig{Endpoint: "x", InstallID: "test"}, t.TempDir(), "local", "v1")
	s.ctx = context.Background()
	s.sender = func(context.Context, string, time.Duration, TelemetryEvent) error { return nil }
	row := &types.ExecutionUsage{Source: "llm", InputTokens: 3, OutputTokens: 1}
	s.recordUsage("done", 0, row)
	s.flushUsage()
	ackDir := filepath.Join(s.usageOutbox, "ack")
	files, err := os.ReadDir(ackDir)
	if err != nil || len(files) != 1 {
		t.Fatalf("ack files=%v error=%v", files, err)
	}
	s.recordUsage("pending", 0, row)
	pending, _ := filepath.Glob(filepath.Join(s.usageOutbox, "*.json"))
	if len(pending) != 1 {
		t.Fatalf("pending=%v", pending)
	}
	old := filepath.Join(ackDir, "expired.sent")
	if err = os.WriteFile(old, []byte("{}"), 0600); err != nil {
		t.Fatal(err)
	}
	when := time.Now().Add(-usageRetention - time.Hour)
	if err = os.Chtimes(old, when, when); err != nil {
		t.Fatal(err)
	}
	s.cleanupUsageAcknowledgements()
	if _, err = os.Stat(old); !os.IsNotExist(err) {
		t.Fatal("expired acknowledgement survived")
	}
	s.recordUsage("done", 0, row) // recent acknowledgement still prevents replay
	pending, _ = filepath.Glob(filepath.Join(s.usageOutbox, "*.json"))
	if len(pending) != 1 {
		t.Fatalf("duplicate pending=%v", pending)
	}
}

func TestUsageRejectsInvalidReceiptsAndOutboxFailures(t *testing.T) {
	s, err := NewTelemetryService(config.TelemetryConfig{Endpoint: "x", InstallID: "test"}, t.TempDir(), "local", "v1")
	if err != nil {
		t.Fatal(err)
	}
	for i, row := range []*types.ExecutionUsage{
		{Source: "llm", InputTokens: -1},
		{Source: "llm", OutputTokens: usageSafeInteger + 1},
		{Source: "llm", InputTokens: usageSafeInteger, OutputTokens: 1},
	} {
		s.recordUsage("invalid", i, row)
	}
	s.recordUsage("", 0, &types.ExecutionUsage{Source: "llm"})
	s.recordUsage("nil", 0, nil)
	if _, err = os.Stat(s.usageOutbox); !os.IsNotExist(err) {
		t.Fatal("invalid receipts queued")
	}
	if usageProvider("unbounded-vendor") != "other" || usageModelFamily("") != "unknown" {
		t.Fatal("unbounded dimensions")
	}
	// A blocked durable directory must not turn recording into an execution error.
	s.usageOutbox = filepath.Join(t.TempDir(), "blocked")
	if err = os.WriteFile(s.usageOutbox, []byte("file"), 0600); err != nil {
		t.Fatal(err)
	}
	s.recordUsage("durability-error", 0, &types.ExecutionUsage{Source: "llm", InputTokens: 1})
	if err = s.storeUsage(TelemetryEvent{EventID: "blocked"}); err == nil {
		t.Fatal("storage failure was hidden")
	}
	if err = syncUsageDirectory(filepath.Join(t.TempDir(), "absent")); err == nil {
		t.Fatal("absent directory sync succeeded")
	}
}

func TestUsageOutboxCapacitySerializationAndRestartCount(t *testing.T) {
	s, _ := NewTelemetryService(config.TelemetryConfig{Endpoint: "x", InstallID: "test"}, t.TempDir(), "local", "v1")
	if err := os.MkdirAll(s.usageOutbox, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(s.usageOutbox, "existing.json"), []byte("{}"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := s.storeUsage(TelemetryEvent{EventID: "new"}); err != nil {
		t.Fatal(err)
	}
	if s.usagePending != 2 {
		t.Fatalf("restart count=%d", s.usagePending)
	}
	if err := s.storeUsage(TelemetryEvent{EventID: "malformed", Properties: map[string]interface{}{"invalid": make(chan int)}}); err == nil {
		t.Fatal("unsupported receipt serialized")
	}
	s.usagePending = usageOutboxLimit
	if err := s.storeUsage(TelemetryEvent{EventID: "overflow"}); err == nil {
		t.Fatal("outbox accepted excess pending receipt")
	}
	if _, err := os.Stat(filepath.Join(s.usageOutbox, "overflow.json")); !os.IsNotExist(err) {
		t.Fatal("overflow event written")
	}
}

func TestUsageFlushExpiresPendingSkipsCorruptionAndHonorsCancellation(t *testing.T) {
	s, _ := NewTelemetryService(config.TelemetryConfig{Endpoint: "x", InstallID: "test"}, t.TempDir(), "local", "v1")
	s.ctx = context.Background()
	s.recordUsage("expired", 0, &types.ExecutionUsage{Source: "llm", InputTokens: 1})
	files, _ := filepath.Glob(filepath.Join(s.usageOutbox, "*.json"))
	old := time.Now().Add(-usageRetention - time.Hour)
	if err := os.Chtimes(files[0], old, old); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(s.usageOutbox, "broken.json"), []byte("{broken"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(s.usageOutbox, "unrelated.txt"), []byte("ignore"), 0600); err != nil {
		t.Fatal(err)
	}
	calls := 0
	s.sender = func(context.Context, string, time.Duration, TelemetryEvent) error { calls++; return nil }
	s.flushUsage()
	if calls != 0 || s.usagePending != 0 {
		t.Fatalf("expired/corrupt receipts delivered calls=%d pending=%d", calls, s.usagePending)
	}
	if _, err := os.Stat(files[0]); !os.IsNotExist(err) {
		t.Fatal("expired pending receipt retained")
	}
	s.recordUsage("cancelled", 0, &types.ExecutionUsage{Source: "llm", InputTokens: 1})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	s.ctx = ctx
	s.flushUsage()
	if calls != 0 {
		t.Fatal("cancelled worker sent receipt")
	}
}

func TestUsageAcknowledgementFailurePreservesPendingForRetry(t *testing.T) {
	s, _ := NewTelemetryService(config.TelemetryConfig{Endpoint: "x", InstallID: "test"}, t.TempDir(), "local", "v1")
	s.ctx = context.Background()
	s.recordUsage("ack-failure", 0, &types.ExecutionUsage{Source: "llm", InputTokens: 1})
	ackDir := filepath.Join(s.usageOutbox, "ack")
	if err := os.WriteFile(ackDir, []byte("blocked"), 0600); err != nil {
		t.Fatal(err)
	}
	calls := 0
	s.sender = func(context.Context, string, time.Duration, TelemetryEvent) error { calls++; return nil }
	s.flushUsage()
	if s.usagePending != 1 {
		t.Fatal("failed acknowledgement dropped pending receipt")
	}
	if err := os.Remove(ackDir); err != nil {
		t.Fatal(err)
	}
	s.flushUsage()
	if calls != 2 || s.usagePending != 0 {
		t.Fatalf("retry calls=%d pending=%d", calls, s.usagePending)
	}
}

func TestUsageWorkerLifecycleWakeAndOptOut(t *testing.T) {
	disabled := false
	home := t.TempDir()
	disabledService, err := NewTelemetryService(config.TelemetryConfig{Enabled: &disabled, Endpoint: "x", InstallID: "test"}, home, "local", "v1")
	if err != nil || disabledService != nil {
		t.Fatal("optout created telemetry")
	}
	disabledService.Start(context.Background())
	disabledService.Stop()
	disabledService.recordUsage("disabled", 0, &types.ExecutionUsage{Source: "llm"})
	if _, err = os.Stat(filepath.Join(home, "telemetry")); !os.IsNotExist(err) {
		t.Fatal("optout wrote telemetry")
	}
	s, _ := NewTelemetryService(config.TelemetryConfig{Endpoint: "x", InstallID: "test"}, t.TempDir(), "local", "v1")
	delivered := make(chan string, 4)
	s.sender = func(_ context.Context, _ string, _ time.Duration, event TelemetryEvent) error {
		if event.EventName == "usage_delta" {
			delivered <- event.EventID
		}
		return nil
	}
	s.Start(context.Background())
	defer s.Stop()
	events.PublishUsage("worker", 0, &types.ExecutionUsage{Source: "llm", InputTokens: 2})
	select {
	case <-delivered:
	case <-time.After(3 * time.Second):
		t.Fatal("worker did not deliver queued receipt")
	}
	s.Stop()
	events.PublishUsage("after-stop", 0, &types.ExecutionUsage{Source: "llm", InputTokens: 2})
	pending, _ := filepath.Glob(filepath.Join(s.usageOutbox, "*.json"))
	if len(pending) != 0 {
		t.Fatal("unsubscribed observer queued usage")
	}
}

func TestAcknowledgementCleanupKeepsNewestBoundedHistory(t *testing.T) {
	s, _ := NewTelemetryService(config.TelemetryConfig{Endpoint: "x", InstallID: "test"}, t.TempDir(), "local", "v1")
	dir := filepath.Join(s.usageOutbox, "ack")
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	oldest := filepath.Join(dir, "oldest.sent")
	for i := 0; i <= usageOutboxLimit; i++ {
		path := filepath.Join(dir, fmt.Sprintf("receipt-%05d.sent", i))
		if i == 0 {
			path = oldest
		}
		if err := os.WriteFile(path, []byte("{}"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	old := time.Now().Add(-time.Hour)
	if err := os.Chtimes(oldest, old, old); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "ignore.txt"), []byte("keep"), 0600); err != nil {
		t.Fatal(err)
	}
	s.cleanupUsageAcknowledgements()
	files, _ := filepath.Glob(filepath.Join(dir, "*.sent"))
	if len(files) != usageOutboxLimit {
		t.Fatalf("history count=%d", len(files))
	}
	if _, err := os.Stat(oldest); !os.IsNotExist(err) {
		t.Fatal("oldest acknowledgement survived capacity cleanup")
	}
}
