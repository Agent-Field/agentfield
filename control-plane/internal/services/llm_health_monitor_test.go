package services

import (
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
)

func TestLLMHealthMonitor_CircuitBreaker_ClosedToOpen(t *testing.T) {
	// Simulate an endpoint that always fails
	failServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer failServer.Close()

	cfg := config.LLMHealthConfig{
		Enabled:          true,
		CheckInterval:    100 * time.Millisecond,
		CheckTimeout:     1 * time.Second,
		FailureThreshold: 3,
		RecoveryTimeout:  500 * time.Millisecond,
		HalfOpenMaxProbes: 2,
		Endpoints: []config.LLMEndpoint{
			{Name: "test-llm", URL: failServer.URL + "/health"},
		},
	}

	monitor := NewLLMHealthMonitor(cfg, nil)

	// Initially healthy (fail-open until first check)
	if !monitor.IsEndpointHealthy("test-llm") {
		t.Fatal("expected endpoint to be healthy before any checks")
	}

	// Run checks until circuit opens (3 failures)
	for i := 0; i < 4; i++ {
		monitor.checkAllEndpoints()
	}

	// Circuit should be open now
	if monitor.IsEndpointHealthy("test-llm") {
		t.Fatal("expected endpoint to be unhealthy after consecutive failures")
	}

	status, ok := monitor.GetStatus("test-llm")
	if !ok {
		t.Fatal("expected to get status for test-llm")
	}
	if status.CircuitState != CircuitOpen {
		t.Fatalf("expected circuit state to be open, got %s", status.CircuitState)
	}
	if status.ConsecutiveFailures < 3 {
		t.Fatalf("expected at least 3 consecutive failures, got %d", status.ConsecutiveFailures)
	}
	if status.LastError == "" {
		t.Fatal("expected last_error to be set")
	}
}

func TestLLMHealthMonitor_CircuitBreaker_OpenToHalfOpenToClosed(t *testing.T) {
	var healthy atomic.Bool
	healthy.Store(false)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if healthy.Load() {
			w.WriteHeader(http.StatusOK)
		} else {
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	defer server.Close()

	cfg := config.LLMHealthConfig{
		Enabled:          true,
		CheckInterval:    50 * time.Millisecond,
		CheckTimeout:     1 * time.Second,
		FailureThreshold: 2,
		RecoveryTimeout:  100 * time.Millisecond,
		HalfOpenMaxProbes: 2,
		Endpoints: []config.LLMEndpoint{
			{Name: "test-llm", URL: server.URL + "/health"},
		},
	}

	monitor := NewLLMHealthMonitor(cfg, nil)

	// Drive circuit to open state
	for i := 0; i < 3; i++ {
		monitor.checkAllEndpoints()
	}
	if monitor.IsEndpointHealthy("test-llm") {
		t.Fatal("expected circuit to be open")
	}

	// Wait for recovery timeout
	time.Sleep(150 * time.Millisecond)

	// Now make the server healthy
	healthy.Store(true)

	// First check should transition to half-open and probe
	monitor.checkAllEndpoints()
	status, _ := monitor.GetStatus("test-llm")
	if status.CircuitState != CircuitHalfOpen {
		t.Fatalf("expected half_open state, got %s", status.CircuitState)
	}

	// Second successful probe should close the circuit
	monitor.checkAllEndpoints()
	status, _ = monitor.GetStatus("test-llm")
	if status.CircuitState != CircuitClosed {
		t.Fatalf("expected closed state after recovery, got %s", status.CircuitState)
	}
	if !monitor.IsEndpointHealthy("test-llm") {
		t.Fatal("expected endpoint to be healthy after recovery")
	}
}

func TestLLMHealthMonitor_HalfOpenFailureReopens(t *testing.T) {
	var healthy atomic.Bool
	healthy.Store(false)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if healthy.Load() {
			w.WriteHeader(http.StatusOK)
		} else {
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	defer server.Close()

	cfg := config.LLMHealthConfig{
		Enabled:          true,
		CheckInterval:    50 * time.Millisecond,
		CheckTimeout:     1 * time.Second,
		FailureThreshold: 2,
		RecoveryTimeout:  100 * time.Millisecond,
		HalfOpenMaxProbes: 2,
		Endpoints: []config.LLMEndpoint{
			{Name: "test-llm", URL: server.URL + "/health"},
		},
	}

	monitor := NewLLMHealthMonitor(cfg, nil)

	// Drive to open
	for i := 0; i < 3; i++ {
		monitor.checkAllEndpoints()
	}

	// Wait for recovery timeout
	time.Sleep(150 * time.Millisecond)

	// Keep server unhealthy — half-open probe should fail and re-open
	monitor.checkAllEndpoints()
	status, _ := monitor.GetStatus("test-llm")
	if status.CircuitState != CircuitOpen {
		t.Fatalf("expected circuit to re-open after half-open failure, got %s", status.CircuitState)
	}
}

func TestLLMHealthMonitor_FailOpenWhenDisabled(t *testing.T) {
	cfg := config.LLMHealthConfig{
		Enabled: false,
	}
	monitor := NewLLMHealthMonitor(cfg, nil)

	// Should always return healthy when disabled
	if !monitor.IsEndpointHealthy("anything") {
		t.Fatal("expected fail-open when monitoring is disabled")
	}
	if !monitor.IsAnyEndpointHealthy() {
		t.Fatal("expected fail-open when monitoring is disabled")
	}
}

func TestLLMHealthMonitor_FailOpenForUnknownEndpoint(t *testing.T) {
	cfg := config.LLMHealthConfig{
		Enabled: true,
		Endpoints: []config.LLMEndpoint{
			{Name: "known", URL: "http://localhost:1/health"},
		},
	}
	monitor := NewLLMHealthMonitor(cfg, nil)

	// Unknown endpoint should fail-open
	if !monitor.IsEndpointHealthy("unknown") {
		t.Fatal("expected fail-open for unknown endpoint")
	}
}

func TestLLMHealthMonitor_MultipleEndpoints(t *testing.T) {
	healthyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer healthyServer.Close()

	failServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer failServer.Close()

	cfg := config.LLMHealthConfig{
		Enabled:          true,
		CheckInterval:    100 * time.Millisecond,
		CheckTimeout:     1 * time.Second,
		FailureThreshold: 2,
		RecoveryTimeout:  1 * time.Second,
		Endpoints: []config.LLMEndpoint{
			{Name: "healthy", URL: healthyServer.URL + "/health"},
			{Name: "failing", URL: failServer.URL + "/health"},
		},
	}

	monitor := NewLLMHealthMonitor(cfg, nil)

	// Run enough checks to open the failing endpoint's circuit
	for i := 0; i < 3; i++ {
		monitor.checkAllEndpoints()
	}

	// One healthy endpoint should be enough
	if !monitor.IsAnyEndpointHealthy() {
		t.Fatal("expected at least one endpoint healthy")
	}
	if !monitor.IsEndpointHealthy("healthy") {
		t.Fatal("expected healthy endpoint to be healthy")
	}
	if monitor.IsEndpointHealthy("failing") {
		t.Fatal("expected failing endpoint to be unhealthy")
	}

	statuses := monitor.GetAllStatuses()
	if len(statuses) != 2 {
		t.Fatalf("expected 2 statuses, got %d", len(statuses))
	}
}

func TestLLMHealthMonitor_GetAllStatuses(t *testing.T) {
	cfg := config.LLMHealthConfig{
		Enabled: true,
		Endpoints: []config.LLMEndpoint{
			{Name: "ep1", URL: "http://localhost:1/health"},
			{Name: "ep2", URL: "http://localhost:2/health"},
		},
	}
	monitor := NewLLMHealthMonitor(cfg, nil)

	statuses := monitor.GetAllStatuses()
	if len(statuses) != 2 {
		t.Fatalf("expected 2 statuses, got %d", len(statuses))
	}

	for _, s := range statuses {
		if s.CircuitState != CircuitClosed {
			t.Fatalf("expected initial state to be closed, got %s", s.CircuitState)
		}
	}
}
