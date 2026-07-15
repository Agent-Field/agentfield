package server

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"syscall"
	"testing"
	"time"
)

// wrappedBindErr mimics the error Start() returns for a failed bind: the raw
// syscall.EADDRINUSE, wrapped the way net/os wrap it and then again by Start's
// fmt.Errorf("...: %w", err).
func wrappedBindErr() error {
	inner := &net.OpError{
		Op:  "listen",
		Net: "tcp",
		Err: &os.SyscallError{Syscall: "bind", Err: syscall.EADDRINUSE},
	}
	return fmt.Errorf("failed to start HTTP server on :8080: %w", inner)
}

func TestIsAddrInUse(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"raw eaddrinuse", syscall.EADDRINUSE, true},
		{"wrapped bind error", wrappedBindErr(), true},
		{"other syscall error", syscall.ECONNREFUSED, false},
		{"unrelated error", errors.New("database path is empty"), false},
		{"nil", nil, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isAddrInUse(tt.err); got != tt.want {
				t.Errorf("isAddrInUse(%v) = %v, want %v", tt.err, got, tt.want)
			}
		})
	}
}

func TestDecideExitClean(t *testing.T) {
	probeCalls := func(counter *int, result bool) func() bool {
		return func() bool {
			*counter++
			return result
		}
	}

	// The error shape an incumbent actually produces in local mode: server
	// *creation* dies on the incumbent's BoltDB file lock long before any bind.
	boltLockErr := errors.New(
		"failed to initialize local storage: failed to open BoltDB database: timeout")

	tests := []struct {
		name          string
		startErr      error
		probeResult   bool
		wantExitClean bool
		wantProbed    bool // was the probe consulted at all?
	}{
		{
			name:          "port in use, healthy AgentField owns it -> exit clean",
			startErr:      wrappedBindErr(),
			probeResult:   true,
			wantExitClean: true,
			wantProbed:    true,
		},
		{
			name:          "storage lock timeout, healthy AgentField owns it -> exit clean",
			startErr:      boltLockErr,
			probeResult:   true,
			wantExitClean: true,
			wantProbed:    true,
		},
		{
			name:          "port in use, foreign/unhealthy service -> fatal",
			startErr:      wrappedBindErr(),
			probeResult:   false,
			wantExitClean: false,
			wantProbed:    true,
		},
		{
			name:          "storage lock timeout, no healthy incumbent -> fatal",
			startErr:      boltLockErr,
			probeResult:   false,
			wantExitClean: false,
			wantProbed:    true,
		},
		{
			name:          "genuine startup bug, no healthy incumbent -> fatal",
			startErr:      errors.New("failed to create AgentField server"),
			probeResult:   false,
			wantExitClean: false,
			wantProbed:    true,
		},
		{
			name:          "nil error -> nothing failed, never probes",
			startErr:      nil,
			probeResult:   true,
			wantExitClean: false,
			wantProbed:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			got := decideExitClean(tt.startErr, probeCalls(&calls, tt.probeResult))
			if got != tt.wantExitClean {
				t.Errorf("decideExitClean = %v, want %v", got, tt.wantExitClean)
			}
			if probed := calls > 0; probed != tt.wantProbed {
				t.Errorf("probe consulted = %v, want %v", probed, tt.wantProbed)
			}
		})
	}
}

func TestProbeHealthyAgentField(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
		want   bool
	}{
		{
			name:   "healthy AgentField payload",
			status: http.StatusOK,
			body:   `{"status":"healthy","version":"1.0.0","checks":{"storage":{"status":"healthy"}}}`,
			want:   true,
		},
		{
			name:   "unhealthy AgentField payload (503) -> not exit-clean",
			status: http.StatusServiceUnavailable,
			body:   `{"status":"unhealthy","version":"1.0.0","checks":{"storage":{"status":"unhealthy"}}}`,
			want:   false,
		},
		{
			name:   "foreign service returning bare status -> rejected (no version/checks)",
			status: http.StatusOK,
			body:   `{"status":"healthy"}`,
			want:   false,
		},
		{
			name:   "foreign service, unrelated JSON -> rejected",
			status: http.StatusOK,
			body:   `{"ok":true,"service":"grafana"}`,
			want:   false,
		},
		{
			name:   "non-JSON body -> rejected",
			status: http.StatusOK,
			body:   `<html>hello</html>`,
			want:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(tt.body))
			}))
			defer srv.Close()

			if got := probeHealthyAgentField(srv.URL+"/health", healthProbeTimeout); got != tt.want {
				t.Errorf("probeHealthyAgentField = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestProbeHealthyAgentField_Unreachable(t *testing.T) {
	// Nothing listening on this port -> connection refused, not a healthy owner.
	if probeHealthyAgentField("http://127.0.0.1:1/health", 500*time.Millisecond) {
		t.Error("expected unreachable probe to return false")
	}
}
