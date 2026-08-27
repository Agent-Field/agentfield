package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
)

// captureGinWriters points gin's plaintext output at in-memory buffers and
// restores the originals when the test finishes.
func captureGinWriters(t *testing.T) *bytes.Buffer {
	t.Helper()
	prevOut, prevErr := gin.DefaultWriter, gin.DefaultErrorWriter
	var buf bytes.Buffer
	gin.DefaultWriter = &buf
	gin.DefaultErrorWriter = &buf
	t.Cleanup(func() {
		gin.DefaultWriter = prevOut
		gin.DefaultErrorWriter = prevErr
	})
	return &buf
}

// captureServerLogger points the global zerolog logger at a buffer.
func captureServerLogger(t *testing.T, level zerolog.Level) *bytes.Buffer {
	t.Helper()
	prev := logger.Logger
	var buf bytes.Buffer
	logger.Logger = zerolog.New(&buf).With().Timestamp().Logger().Level(level)
	t.Cleanup(func() { logger.Logger = prev })
	return &buf
}

// TestRouterDoesNotWriteGinPlaintextRequestLines covers contract C4: a request
// served by the control plane's router produces no gin [GIN] plaintext line,
// only the structured http_request event.
func TestRouterDoesNotWriteGinPlaintextRequestLines(t *testing.T) {
	gin.SetMode(gin.TestMode)
	ginOut := captureGinWriters(t)
	logOut := captureServerLogger(t, zerolog.DebugLevel)

	router := newRouter()
	router.GET("/healthz", func(c *gin.Context) { c.String(http.StatusOK, "ok") })

	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/healthz", nil))

	require.Equal(t, http.StatusOK, recorder.Code)
	require.Empty(t, ginOut.String(), "gin must not write its own plaintext request log")

	var entry map[string]interface{}
	require.NoError(t, json.Unmarshal(bytes.TrimSpace(logOut.Bytes()), &entry))
	require.Equal(t, "http_request", entry["message"])
	require.Equal(t, "/healthz", entry["path"])
}

// TestRouterRecoversFromHandlerPanic covers contract C5: dropping
// gin.Default() must not drop panic recovery.
func TestRouterRecoversFromHandlerPanic(t *testing.T) {
	gin.SetMode(gin.TestMode)
	captureGinWriters(t)
	captureServerLogger(t, zerolog.DebugLevel)

	router := newRouter()
	router.GET("/boom", func(c *gin.Context) { panic("kaboom") })

	recorder := httptest.NewRecorder()
	require.NotPanics(t, func() {
		router.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/boom", nil))
	})
	require.Equal(t, http.StatusInternalServerError, recorder.Code)
}

// TestRouterLogsHandlerPanicAsError covers the review fix: Recovery is inner
// to GinLogger, so a panicking handler still emits a structured http_request
// at error/500. Recovery's plaintext stack on DefaultErrorWriter is pre-existing
// and is not asserted here.
func TestRouterLogsHandlerPanicAsError(t *testing.T) {
	gin.SetMode(gin.TestMode)
	captureGinWriters(t)
	logOut := captureServerLogger(t, zerolog.DebugLevel)

	router := newRouter()
	router.GET("/boom", func(c *gin.Context) { panic("kaboom") })

	recorder := httptest.NewRecorder()
	require.NotPanics(t, func() {
		router.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/boom", nil))
	})
	require.Equal(t, http.StatusInternalServerError, recorder.Code)

	lines := bytes.Split(bytes.TrimSpace(logOut.Bytes()), []byte("\n"))
	require.Len(t, lines, 1, "expected exactly one structured log line")

	var entry map[string]interface{}
	require.NoError(t, json.Unmarshal(lines[0], &entry))
	require.Equal(t, "http_request", entry["message"])
	require.Equal(t, float64(http.StatusInternalServerError), entry["status"])
	require.Equal(t, "error", entry["level"])
}

// corsRestrictedServer builds the router the way the control plane does —
// newRouter() followed by applyGlobalMiddleware() — with CORS narrowed to a
// single allowed origin, and registers one API route on it. The returned
// buffer holds the structured log and is emptied once construction is done, so
// only per-request output is asserted on.
func corsRestrictedServer(t *testing.T, level zerolog.Level) (*AgentFieldServer, *bytes.Buffer) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	captureGinWriters(t)
	logOut := captureServerLogger(t, level)

	cfg := &config.Config{}
	cfg.API.CORS.AllowedOrigins = []string{"http://allowed.example"}
	srv := &AgentFieldServer{Router: newRouter(), config: cfg}
	srv.applyGlobalMiddleware()
	srv.Router.GET("/api/v1/health", func(c *gin.Context) { c.String(http.StatusOK, "ok") })

	logOut.Reset()
	return srv, logOut
}

// requestLogEntries decodes the structured http_request events written to buf.
func requestLogEntries(t *testing.T, buf *bytes.Buffer) []map[string]interface{} {
	t.Helper()
	entries := []map[string]interface{}{}
	for _, line := range bytes.Split(bytes.TrimSpace(buf.Bytes()), []byte("\n")) {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var entry map[string]interface{}
		require.NoError(t, json.Unmarshal(line, &entry), "log line is not JSON: %s", line)
		if entry["message"] == "http_request" {
			entries = append(entries, entry)
		}
	}
	return entries
}

// TestRouterLogsCORSRejectedRequest covers contract V1: a request CORS rejects
// because of a disallowed Origin is answered with 403 AND recorded as exactly
// one structured http_request line at warn. Before the logger was moved ahead
// of CORS, gin's abort meant the request reached no logging middleware at all
// and was invisible at every level.
func TestRouterLogsCORSRejectedRequest(t *testing.T) {
	srv, logOut := corsRestrictedServer(t, zerolog.InfoLevel)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/health", nil)
	req.Header.Set("Origin", "http://evil.example")
	recorder := httptest.NewRecorder()
	srv.Router.ServeHTTP(recorder, req)

	require.Equal(t, http.StatusForbidden, recorder.Code, "CORS must still reject the origin")

	entries := requestLogEntries(t, logOut)
	require.Len(t, entries, 1, "a CORS-rejected request must be logged exactly once, got: %s", logOut.String())
	require.Equal(t, "warn", entries[0]["level"])
	require.Equal(t, float64(http.StatusForbidden), entries[0]["status"])
	require.Equal(t, "/api/v1/health", entries[0]["path"])
	require.Equal(t, http.MethodGet, entries[0]["method"])
}

// TestRouterLogsCORSRejectedPreflight covers contract V2: a preflight from a
// disallowed origin is rejected with 403 and logged once at warn, same as a
// simple request.
func TestRouterLogsCORSRejectedPreflight(t *testing.T) {
	srv, logOut := corsRestrictedServer(t, zerolog.InfoLevel)

	req := httptest.NewRequest(http.MethodOptions, "/api/v1/health", nil)
	req.Header.Set("Origin", "http://evil.example")
	req.Header.Set("Access-Control-Request-Method", http.MethodGet)
	recorder := httptest.NewRecorder()
	srv.Router.ServeHTTP(recorder, req)

	require.Equal(t, http.StatusForbidden, recorder.Code)

	entries := requestLogEntries(t, logOut)
	require.Len(t, entries, 1, "a rejected preflight must be logged exactly once, got: %s", logOut.String())
	require.Equal(t, "warn", entries[0]["level"])
	require.Equal(t, float64(http.StatusForbidden), entries[0]["status"])
	require.Equal(t, http.MethodOptions, entries[0]["method"])
}

// TestRouterLogsCORSAnsweredPreflight covers contract V3: a preflight CORS
// answers itself never reaches a route handler, but it is still one request
// and still produces exactly one structured line (at debug, since it succeeds).
func TestRouterLogsCORSAnsweredPreflight(t *testing.T) {
	srv, logOut := corsRestrictedServer(t, zerolog.DebugLevel)

	req := httptest.NewRequest(http.MethodOptions, "/api/v1/health", nil)
	req.Header.Set("Origin", "http://allowed.example")
	req.Header.Set("Access-Control-Request-Method", http.MethodGet)
	recorder := httptest.NewRecorder()
	srv.Router.ServeHTTP(recorder, req)

	require.Equal(t, http.StatusNoContent, recorder.Code)
	require.Equal(t, "http://allowed.example", recorder.Header().Get("Access-Control-Allow-Origin"))

	entries := requestLogEntries(t, logOut)
	require.Len(t, entries, 1, "an answered preflight must be logged exactly once, got: %s", logOut.String())
	require.Equal(t, "debug", entries[0]["level"])
	require.Equal(t, float64(http.StatusNoContent), entries[0]["status"])
}

// TestRouterLogsAllowedCrossOriginRequest covers contract V4: the fix does not
// change what an allowed cross-origin caller sees — the route still runs, the
// CORS header is still set, and the request is logged once.
func TestRouterLogsAllowedCrossOriginRequest(t *testing.T) {
	srv, logOut := corsRestrictedServer(t, zerolog.DebugLevel)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/health", nil)
	req.Header.Set("Origin", "http://allowed.example")
	recorder := httptest.NewRecorder()
	srv.Router.ServeHTTP(recorder, req)

	require.Equal(t, http.StatusOK, recorder.Code)
	require.Equal(t, "ok", recorder.Body.String())
	require.Equal(t, "http://allowed.example", recorder.Header().Get("Access-Control-Allow-Origin"))

	entries := requestLogEntries(t, logOut)
	require.Len(t, entries, 1, "an allowed request must be logged exactly once, got: %s", logOut.String())
	require.Equal(t, "debug", entries[0]["level"])
	require.Equal(t, float64(http.StatusOK), entries[0]["status"])
}
