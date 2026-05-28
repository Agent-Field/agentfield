package handlers

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
)

func TestCheckExecutionPreconditions_RejectsRequestedUnhealthyEndpoint(t *testing.T) {
	healthyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer healthyServer.Close()

	failingServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer failingServer.Close()

	monitor := services.NewLLMHealthMonitor(config.LLMHealthConfig{
		Enabled:          true,
		CheckInterval:    10 * time.Millisecond,
		CheckTimeout:     100 * time.Millisecond,
		FailureThreshold: 1,
		RecoveryTimeout:  time.Second,
		Endpoints: []config.LLMEndpoint{
			{Name: "healthy", URL: healthyServer.URL},
			{Name: "failing", URL: failingServer.URL},
		},
	}, nil)
	go monitor.Start()
	defer monitor.Stop()
	SetLLMHealthMonitor(monitor)
	defer SetLLMHealthMonitor(nil)

	time.Sleep(40 * time.Millisecond)

	err := CheckExecutionPreconditions("agent-1", "failing")
	if err == nil {
		t.Fatal("expected endpoint-specific precondition error")
	}
	if !strings.Contains(err.Error(), `LLM backend "failing" unavailable`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestCheckExecutionPreconditions_FailsClosedWhenBackendUnknownAndOneEndpointIsDown(t *testing.T) {
	healthyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer healthyServer.Close()

	failingServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer failingServer.Close()

	monitor := services.NewLLMHealthMonitor(config.LLMHealthConfig{
		Enabled:          true,
		CheckInterval:    10 * time.Millisecond,
		CheckTimeout:     100 * time.Millisecond,
		FailureThreshold: 1,
		RecoveryTimeout:  time.Second,
		Endpoints: []config.LLMEndpoint{
			{Name: "healthy", URL: healthyServer.URL},
			{Name: "failing", URL: failingServer.URL},
		},
	}, nil)
	go monitor.Start()
	defer monitor.Stop()
	SetLLMHealthMonitor(monitor)
	defer SetLLMHealthMonitor(nil)

	time.Sleep(40 * time.Millisecond)

	err := CheckExecutionPreconditions("agent-1", "")
	if err == nil {
		t.Fatal("expected fail-closed error when backend is ambiguous")
	}
	if !strings.Contains(err.Error(), "request backend could not be determined") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestCheckExecutionPreconditions_AllowsHealthySingleEndpointWhenBackendUnknown(t *testing.T) {
	healthyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer healthyServer.Close()

	monitor := services.NewLLMHealthMonitor(config.LLMHealthConfig{
		Enabled:          true,
		CheckInterval:    10 * time.Millisecond,
		CheckTimeout:     100 * time.Millisecond,
		FailureThreshold: 1,
		RecoveryTimeout:  time.Second,
		Endpoints: []config.LLMEndpoint{
			{Name: "healthy", URL: healthyServer.URL},
		},
	}, nil)
	go monitor.Start()
	defer monitor.Stop()
	SetLLMHealthMonitor(monitor)
	defer SetLLMHealthMonitor(nil)

	time.Sleep(30 * time.Millisecond)

	if err := CheckExecutionPreconditions("agent-1", ""); err != nil {
		t.Fatalf("expected healthy single-endpoint config to pass, got %v", err)
	}
}
