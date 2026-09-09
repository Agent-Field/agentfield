package server

import (
	"bufio"
	"context"
	"net"
	"net/http"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestStopCancelsSSEBeforeHTTPDrain(t *testing.T) {
	gin.SetMode(gin.TestMode)
	streamCtx, cancelStreams := context.WithCancel(context.Background())
	router := gin.New()
	router.GET("/events", handlers.WithStreamContext(streamCtx, func(c *gin.Context) {
		c.Header("Content-Type", "text/event-stream")
		_, _ = c.Writer.Write([]byte("event: connected\ndata: {}\n\n"))
		c.Writer.Flush()
		<-c.Request.Context().Done()
	}))
	ln, err := (&net.ListenConfig{}).Listen(context.Background(), "tcp", "127.0.0.1:0")
	require.NoError(t, err)
	httpServer := &http.Server{Handler: router}
	srv := &AgentFieldServer{
		config:        &config.Config{},
		httpServer:    httpServer,
		streamCtx:     streamCtx,
		cancelStreams: cancelStreams,
	}
	go func() { _ = httpServer.Serve(ln) }()

	resp, err := http.Get("http://" + ln.Addr().String() + "/events")
	require.NoError(t, err)
	defer resp.Body.Close()
	line, err := bufio.NewReader(resp.Body).ReadString('\n')
	require.NoError(t, err)
	require.Equal(t, "event: connected\n", line)

	started := time.Now()
	require.NoError(t, srv.Stop())
	require.Less(t, time.Since(started), time.Second)
}

func TestStopAllowsNonStreamingRequestToCompleteWithinBudget(t *testing.T) {
	requestStarted := make(chan struct{})
	releaseRequest := make(chan struct{})
	httpServer := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		close(requestStarted)
		<-releaseRequest
		w.WriteHeader(http.StatusNoContent)
	})}
	ln, err := (&net.ListenConfig{}).Listen(context.Background(), "tcp", "127.0.0.1:0")
	require.NoError(t, err)
	srv := &AgentFieldServer{
		config:     &config.Config{AgentField: config.AgentFieldConfig{ShutdownTimeout: time.Second}},
		httpServer: httpServer,
	}
	go func() { _ = httpServer.Serve(ln) }()
	requestDone := make(chan error, 1)
	go func() {
		resp, requestErr := http.Get("http://" + ln.Addr().String())
		if requestErr == nil {
			_ = resp.Body.Close()
		}
		requestDone <- requestErr
	}()
	<-requestStarted
	go func() {
		time.Sleep(50 * time.Millisecond)
		close(releaseRequest)
	}()

	require.NoError(t, srv.Stop())
	require.NoError(t, <-requestDone)
}

func TestAsyncDrainContextGetsFreshMinimumBudget(t *testing.T) {
	expired, cancelExpired := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancelExpired()

	ctx, cancel := asyncDrainContext(expired)
	defer cancel()
	deadline, ok := ctx.Deadline()
	require.True(t, ok)
	require.GreaterOrEqual(t, time.Until(deadline), 4900*time.Millisecond)
	require.NoError(t, ctx.Err())
}

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
