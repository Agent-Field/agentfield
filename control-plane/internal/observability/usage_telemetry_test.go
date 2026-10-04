package observability

import (
	"context"
	"encoding/json"
	"errors"
	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
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
