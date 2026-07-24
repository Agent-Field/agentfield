package observability

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

func TestTelemetryServiceDisabled(t *testing.T) {
	disabled := false
	svc, err := NewTelemetryService(config.TelemetryConfig{
		Enabled:  &disabled,
		Endpoint: "https://agentfield.ai/api/oss/telemetry",
	}, t.TempDir(), "local", "test")
	if err != nil {
		t.Fatalf("NewTelemetryService error: %v", err)
	}
	if svc != nil {
		t.Fatal("expected nil service when telemetry disabled")
	}
}

func TestTelemetryServiceInstallIDPersistsAndHashes(t *testing.T) {
	dir := t.TempDir()
	cfg := config.TelemetryConfig{
		Endpoint:      "https://agentfield.ai/api/oss/telemetry",
		InstallIDPath: filepath.Join(dir, "install_id"),
		Timeout:       time.Millisecond,
	}

	first, err := NewTelemetryService(cfg, dir, "postgres", "test")
	if err != nil {
		t.Fatalf("first NewTelemetryService error: %v", err)
	}
	second, err := NewTelemetryService(cfg, dir, "postgres", "test")
	if err != nil {
		t.Fatalf("second NewTelemetryService error: %v", err)
	}
	if first.installHash != second.installHash {
		t.Fatalf("expected stable install hash, got %q and %q", first.installHash, second.installHash)
	}
	if len(first.installHash) != 64 {
		t.Fatalf("expected sha256 hex hash, got %q", first.installHash)
	}
}

func TestTelemetryServiceDefaultInstallIDPathUsesAgentFieldHome(t *testing.T) {
	dir := t.TempDir()
	cfg := config.TelemetryConfig{
		Endpoint: "https://agentfield.ai/api/oss/telemetry",
		Timeout:  time.Millisecond,
	}

	if _, err := NewTelemetryService(cfg, dir, "local", "test"); err != nil {
		t.Fatalf("NewTelemetryService error: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "telemetry", "install_id")); err != nil {
		t.Fatalf("expected install ID under AgentField home: %v", err)
	}
}

func TestTelemetryServiceSanitizesEventData(t *testing.T) {
	svc := &TelemetryService{
		installHash: "a",
		eventIDKey:  eventIdentityKey("private-install-id"),
		runtimeName: "docker",
		version:     "test",
		storageMode: "postgres",
		queue:       make(chan TelemetryEvent, 10),
	}

	svc.handleExecutionEvent(events.ExecutionEvent{
		Type:   events.ExecutionFailed,
		Status: "failed",
		Data: map[string]interface{}{
			"target_type":       "reasoner",
			"execution_mode":    "async",
			"duration_ms":       int64(1200),
			"failure_category":  "agent_restart_orphaned: raw execution text",
			"is_root_execution": false,
			"workflow_depth":    3,
			"transition_source": "status_callback",
			"context":           map[string]interface{}{"prompt": "do not send"},
			"session_id":        "do-not-send",
			"actor_id":          "do-not-send",
			"raw_error_message": "do-not-send",
		},
	})

	event := <-svc.queue
	if event.EventName != "execution_failed" {
		t.Fatalf("unexpected event %q", event.EventName)
	}
	if _, ok := event.Properties["context"]; ok {
		t.Fatal("context leaked into telemetry properties")
	}
	if _, ok := event.Properties["session_id"]; ok {
		t.Fatal("session_id leaked into telemetry properties")
	}
	if got := event.Properties["failure_category"]; got != "agent_restart_orphaned" {
		t.Fatalf("unexpected failure category %v", got)
	}
	if got := event.Properties["duration_bucket_ms"]; got != "1000-4999" {
		t.Fatalf("unexpected duration bucket %v", got)
	}
	if event.Properties["is_root_execution"] != false ||
		event.Properties["workflow_depth_bucket"] != "3-5" ||
		event.Properties["transition_source"] != "status_callback" {
		t.Fatalf("unexpected lifecycle dimensions: %#v", event.Properties)
	}
	if event.SchemaVersion != telemetrySchemaVersion {
		t.Fatalf("unexpected schema version %d", event.SchemaVersion)
	}
	if len(event.EventID) != 64 {
		t.Fatalf("expected opaque sha256 event ID, got %q", event.EventID)
	}
	encoded, err := json.Marshal(event)
	if err != nil {
		t.Fatalf("marshal telemetry event: %v", err)
	}
	if strings.Contains(string(encoded), "private-install-id") ||
		strings.Contains(string(encoded), "raw execution text") {
		t.Fatalf("private identity material leaked into payload: %s", encoded)
	}
}

func TestTelemetryExecutionStarted(t *testing.T) {
	svc := &TelemetryService{
		installHash: "install-hash",
		eventIDKey:  eventIdentityKey("private-install-id"),
		runtimeName: "binary",
		version:     "test",
		queue:       make(chan TelemetryEvent, 1),
	}
	svc.handleExecutionEvent(events.ExecutionEvent{
		Type:   events.ExecutionStarted,
		Status: "running",
		Data: map[string]interface{}{
			"target_type":       "reasoner",
			"execution_mode":    "sync",
			"is_root_execution": true,
			"workflow_depth":    0,
		},
	})

	got := <-svc.queue
	if got.EventName != "execution_started" ||
		got.Properties["target_type"] != "reasoner" ||
		got.Properties["execution_mode"] != "sync" ||
		got.Properties["is_root_execution"] != true ||
		got.Properties["workflow_depth_bucket"] != "0" {
		t.Fatalf("unexpected started event: %#v", got)
	}
}

func TestTelemetryExecutionTerminalOutcomes(t *testing.T) {
	tests := []struct {
		name      string
		event     events.ExecutionEvent
		eventName string
		status    string
	}{
		{
			name:      "completed",
			event:     events.ExecutionEvent{Type: events.ExecutionCompleted, ExecutionID: "exec-1", Status: "succeeded"},
			eventName: "execution_completed",
			status:    "succeeded",
		},
		{
			name:      "failed",
			event:     events.ExecutionEvent{Type: events.ExecutionFailed, ExecutionID: "exec-2", Status: "failed"},
			eventName: "execution_failed",
			status:    "failed",
		},
		{
			name:      "cancelled",
			event:     events.ExecutionEvent{Type: events.ExecutionCancelledEvent, ExecutionID: "exec-3", Status: "cancelled"},
			eventName: "execution_cancelled",
			status:    "cancelled",
		},
		{
			name:      "timeout from failed publisher",
			event:     events.ExecutionEvent{Type: events.ExecutionFailed, ExecutionID: "exec-4", Status: "timeout"},
			eventName: "execution_timed_out",
			status:    "timeout",
		},
		{
			name:      "timeout from update",
			event:     events.ExecutionEvent{Type: events.ExecutionUpdated, ExecutionID: "exec-5", Status: "timed_out"},
			eventName: "execution_timed_out",
			status:    "timeout",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := &TelemetryService{
				installHash: "install-hash",
				eventIDKey:  eventIdentityKey("private-install-id"),
				runtimeName: "binary",
				version:     "test",
				queue:       make(chan TelemetryEvent, 1),
			}
			svc.handleExecutionEvent(tt.event)
			got := <-svc.queue
			if got.EventName != tt.eventName {
				t.Fatalf("event name = %q, want %q", got.EventName, tt.eventName)
			}
			if got.Properties["status"] != tt.status {
				t.Fatalf("status = %q, want %q", got.Properties["status"], tt.status)
			}
			if got.Properties["outcome"] != tt.status {
				t.Fatalf("outcome = %q, want %q", got.Properties["outcome"], tt.status)
			}
			if got.EventID == "" {
				t.Fatal("expected terminal event ID")
			}
		})
	}
}

func TestTelemetryExecutionEventIdentityIsStableAndOpaque(t *testing.T) {
	svc := &TelemetryService{
		installHash: "install-hash",
		eventIDKey:  eventIdentityKey("private-install-id"),
		runtimeName: "binary",
		version:     "test",
		queue:       make(chan TelemetryEvent, 2),
	}
	event := events.ExecutionEvent{
		Type:        events.ExecutionFailed,
		ExecutionID: "private-execution-id",
		WorkflowID:  "private-workflow-id",
		Status:      "failed",
	}
	svc.handleExecutionEvent(event)
	svc.handleExecutionEvent(event)
	first, second := <-svc.queue, <-svc.queue

	if first.EventID != second.EventID {
		t.Fatalf("event identity is not stable: %q != %q", first.EventID, second.EventID)
	}
	encoded, err := json.Marshal(first)
	if err != nil {
		t.Fatalf("marshal telemetry event: %v", err)
	}
	for _, privateValue := range []string{"private-install-id", "private-execution-id", "private-workflow-id"} {
		if strings.Contains(string(encoded), privateValue) {
			t.Fatalf("%q leaked into telemetry payload: %s", privateValue, encoded)
		}
	}
}

func TestTelemetryExecutionEventIdentityDoesNotCollapseMissingIDs(t *testing.T) {
	svc := &TelemetryService{
		installHash: "install-hash",
		eventIDKey:  eventIdentityKey("private-install-id"),
		runtimeName: "binary",
		version:     "test",
		queue:       make(chan TelemetryEvent, 2),
	}
	event := events.ExecutionEvent{Type: events.ExecutionFailed, Status: "failed"}
	svc.handleExecutionEvent(event)
	svc.handleExecutionEvent(event)
	first, second := <-svc.queue, <-svc.queue

	if first.EventID == second.EventID {
		t.Fatalf("events without execution IDs collided: %q", first.EventID)
	}
}

func TestTelemetryTimeoutEventIdentityDoesNotCollapseRepeatedTransitions(t *testing.T) {
	svc := &TelemetryService{
		installHash: "install-hash",
		eventIDKey:  eventIdentityKey("private-install-id"),
		runtimeName: "binary",
		version:     "test",
		queue:       make(chan TelemetryEvent, 2),
	}
	event := events.ExecutionEvent{
		Type:        events.ExecutionUpdated,
		ExecutionID: "same-execution-id",
		Status:      "timeout",
	}
	svc.handleExecutionEvent(event)
	svc.handleExecutionEvent(event)
	first, second := <-svc.queue, <-svc.queue

	if first.EventID == second.EventID {
		t.Fatalf("repeated timeout transitions collided: %q", first.EventID)
	}
}

func TestTelemetryFailureCategoriesMatchRuntimeVocabulary(t *testing.T) {
	for _, category := range []string{
		"agent_error",
		"agent_timeout",
		"agent_unreachable",
		"bad_response",
		"concurrency_limit",
		"llm_unavailable",
		"node_unavailable",
		"target_not_found",
	} {
		if got := errorCategory(category + ": private details"); got != category {
			t.Fatalf("errorCategory(%q) = %q", category, got)
		}
	}
}

func TestTelemetrySanitizesIdentityAdjacentMetadata(t *testing.T) {
	got := sanitizeProperties(map[string]interface{}{
		"agent_version":   "private-build-name",
		"deployment_type": "customer-specific-deployment",
		"storage_mode":    "customer-database-name",
		"sdk_version":     "customer-specific-version",
	})
	if _, ok := got["agent_version"]; ok {
		t.Fatal("agent version must not be sent")
	}
	if got["deployment_type"] != "unknown" {
		t.Fatalf("deployment type was not normalized: %#v", got)
	}
	if got["storage_mode"] != "unknown" {
		t.Fatalf("storage mode was not normalized: %#v", got)
	}
	if _, ok := got["sdk_version"]; ok {
		t.Fatalf("unbounded SDK version must not be sent: %#v", got)
	}

	valid := sanitizeProperties(map[string]interface{}{
		"deployment_type":   "SERVERLESS",
		"storage_mode":      "POSTGRES",
		"sdk_version":       "v1.2.3-beta.1+build.2",
		"target_type":       "SKILL",
		"execution_mode":    "SYNC",
		"transition_source": "private-callback-name",
	})
	if valid["deployment_type"] != "serverless" ||
		valid["storage_mode"] != "postgres" ||
		valid["sdk_version"] != "v1.2.3-beta.1+build.2" ||
		valid["target_type"] != "skill" ||
		valid["execution_mode"] != "sync" ||
		valid["transition_source"] != "unknown" {
		t.Fatalf("valid bounded metadata was not preserved: %#v", valid)
	}
}

func TestTelemetryServiceNodeRegistrationSDKMetadata(t *testing.T) {
	svc := &TelemetryService{
		installHash: "a",
		runtimeName: "docker",
		version:     "test",
		storageMode: "postgres",
		queue:       make(chan TelemetryEvent, 10),
	}

	svc.handleNodeEvent(events.NodeEvent{
		Type: events.NodeRegistered,
		Data: &types.AgentNode{
			Version:        "1.2.3",
			DeploymentType: "long_running",
			Reasoners:      []types.ReasonerDefinition{{ID: "one"}, {ID: "two"}},
			Metadata: types.AgentMetadata{
				Deployment: &types.DeploymentMetadata{
					Platform: "python",
					Tags:     map[string]string{"sdk_version": "0.1.82"},
				},
			},
		},
	})

	sdkEvent := <-svc.queue
	nodeEvent := <-svc.queue
	if sdkEvent.EventName != "sdk_used" {
		t.Fatalf("expected sdk_used first, got %q", sdkEvent.EventName)
	}
	if sdkEvent.Properties["sdk_language"] != "python" || sdkEvent.Properties["sdk_version"] != "0.1.82" {
		t.Fatalf("unexpected sdk properties: %#v", sdkEvent.Properties)
	}
	if nodeEvent.EventName != "node_registered" {
		t.Fatalf("expected node_registered second, got %q", nodeEvent.EventName)
	}
	if nodeEvent.Properties["reasoner_count_bucket"] != "2-5" {
		t.Fatalf("unexpected node properties: %#v", nodeEvent.Properties)
	}
}

func TestTelemetrySenderNonBlockingFailure(t *testing.T) {
	svc := &TelemetryService{
		cfg:         config.TelemetryConfig{Endpoint: "http://127.0.0.1:1"},
		installHash: "a",
		runtimeName: "binary",
		version:     "test",
		timeout:     time.Millisecond,
		sender: func(context.Context, string, time.Duration, TelemetryEvent) error {
			return context.Canceled
		},
		queue: make(chan TelemetryEvent, 1),
	}
	svc.ctx, svc.cancel = context.WithCancel(context.Background())
	svc.cancel()
	svc.Enqueue("control_plane_started", map[string]interface{}{"secret": "drop", "go_os": "linux"})
	event := <-svc.queue
	if _, ok := event.Properties["secret"]; ok {
		t.Fatal("secret property leaked")
	}
}
