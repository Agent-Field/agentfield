package server

import (
	"context"
	"net"
	"net/http"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/stretchr/testify/require"
)

func TestStopGracefulShutdownOnEmptyServer(t *testing.T) {
	// Stop() should not panic on a zero-value server (all fields nil)
	s := &AgentFieldServer{}
	err := s.Stop()
	require.NoError(t, err)
}

func TestStopGracefulShutdownWithHTTPServer(t *testing.T) {
	// Create a minimal server with an httpServer that's already listening
	cfg := &config.Config{}
	cfg.AgentField.ShutdownTimeout = 2 * time.Second

	srv := &AgentFieldServer{
		config: cfg,
		httpServer: &http.Server{
			Addr: ":0", // random port
		},
	}

	// Start listening in background
	ln, err := (&net.ListenConfig{}).Listen(context.Background(), "tcp", ":0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	srv.httpServer.Addr = ln.Addr().String()

	go func() {
		_ = srv.httpServer.Serve(ln)
	}()

	// Give server a moment to start
	time.Sleep(50 * time.Millisecond)

	// Stop should shut down gracefully
	err = srv.Stop()
	require.NoError(t, err)
}

func TestStopHTTPServerShutdownTimeout(t *testing.T) {
	// Test that a very short timeout causes force close
	cfg := &config.Config{}
	cfg.AgentField.ShutdownTimeout = 1 * time.Nanosecond // impossibly short

	srv := &AgentFieldServer{
		config: cfg,
		httpServer: &http.Server{
			Addr: ":0",
		},
	}

	// Start listening with a handler that holds connections open
	ln, err := (&net.ListenConfig{}).Listen(context.Background(), "tcp", ":0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	srv.httpServer.Addr = ln.Addr().String()
	srv.httpServer.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Second) // simulate long-running request
		w.WriteHeader(200)
	})

	go func() {
		_ = srv.httpServer.Serve(ln)
	}()
	time.Sleep(50 * time.Millisecond)

	// Make a request that will be in-flight during shutdown
	go func() {
		client := &http.Client{Timeout: 10 * time.Second}
		_, _ = client.Get("http://" + ln.Addr().String() + "/")
	}()
	time.Sleep(20 * time.Millisecond)

	// Stop with impossibly short timeout — should return error
	err = srv.Stop()
	require.Error(t, err)
}
