package middleware

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
)

// captureLogger points the package logger at an in-memory buffer and returns
// a function that restores the previous logger.
func captureLogger(t *testing.T, level zerolog.Level) (*bytes.Buffer, func()) {
	t.Helper()
	prev := logger.Logger
	var buf bytes.Buffer
	logger.Logger = zerolog.New(&buf).With().Timestamp().Logger().Level(level)
	return &buf, func() { logger.Logger = prev }
}

func setupGinLoggerRouter() *gin.Engine {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(GinLogger())
	router.GET("/hello", func(c *gin.Context) {
		c.String(http.StatusOK, "ok")
	})
	return router
}

// TestGinLoggerEmitsStructuredDebugLine verifies that a request through the
// middleware produces exactly one structured zerolog line at DEBUG level with
// the expected fields, and that the response still succeeds.
func TestGinLoggerEmitsStructuredDebugLine(t *testing.T) {
	buf, restore := captureLogger(t, zerolog.DebugLevel)
	defer restore()

	router := setupGinLoggerRouter()
	req := httptest.NewRequest(http.MethodGet, "/hello?x=1", nil)
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)

	require.Equal(t, http.StatusOK, recorder.Code)

	lines := bytes.Split(bytes.TrimSpace(buf.Bytes()), []byte("\n"))
	require.Len(t, lines, 1, "expected exactly one structured log line")

	var entry map[string]interface{}
	require.NoError(t, json.Unmarshal(lines[0], &entry))

	require.Equal(t, "debug", entry["level"])
	require.Equal(t, "http_request", entry["message"])
	require.Equal(t, "GET", entry["method"])
	require.Equal(t, "/hello?x=1", entry["path"])
	require.Equal(t, float64(http.StatusOK), entry["status"])
	require.NotEmpty(t, entry["client_ip"])
	require.NotEmpty(t, entry["latency"])
}

// TestGinLoggerSilentAtInfoLevel verifies that at the default INFO level a
// successful request emits nothing, keeping default output free of per-request
// noise (contract C6).
func TestGinLoggerSilentAtInfoLevel(t *testing.T) {
	buf, restore := captureLogger(t, zerolog.InfoLevel)
	defer restore()

	router := setupGinLoggerRouter()
	req := httptest.NewRequest(http.MethodGet, "/hello", nil)
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)

	require.Equal(t, http.StatusOK, recorder.Code)
	require.Empty(t, buf.Bytes(), "no request log should be emitted at info level")
}

// TestGinLoggerLogsNonOKStatus verifies that when a handler aborts with an
// error status the status is passed through to the response and still appears
// in the structured line.
func TestGinLoggerLogsNonOKStatus(t *testing.T) {
	buf, restore := captureLogger(t, zerolog.DebugLevel)
	defer restore()

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(GinLogger())
	router.GET("/boom", func(c *gin.Context) {
		_ = c.Error(errors.New("handler exploded"))
		c.AbortWithStatus(http.StatusInternalServerError)
	})

	req := httptest.NewRequest(http.MethodGet, "/boom", nil)
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)

	require.Equal(t, http.StatusInternalServerError, recorder.Code)

	var entry map[string]interface{}
	require.NoError(t, json.Unmarshal(bytes.TrimSpace(buf.Bytes()), &entry))
	require.Equal(t, float64(http.StatusInternalServerError), entry["status"])
	require.Equal(t, "http_request", entry["message"])
	// the error field is only populated via c.Error(), not AbortWithStatus alone
	require.Contains(t, entry["error"], "handler exploded")
}

// routerReturning builds a router whose single route responds with the given
// status, so a test can drive the middleware's severity mapping.
func routerReturning(status int) *gin.Engine {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(GinLogger())
	router.GET("/probe", func(c *gin.Context) {
		c.String(status, "body")
	})
	return router
}

// requestLogLevel drives one request through a router that answers with the
// given status and returns the level of the single log line it produced, or ""
// when nothing was logged at the supplied logger level.
func requestLogLevel(t *testing.T, loggerLevel zerolog.Level, status int) string {
	t.Helper()

	buf, restore := captureLogger(t, loggerLevel)
	defer restore()

	recorder := httptest.NewRecorder()
	routerReturning(status).ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/probe", nil))
	require.Equal(t, status, recorder.Code)

	raw := bytes.TrimSpace(buf.Bytes())
	if len(raw) == 0 {
		return ""
	}
	lines := bytes.Split(raw, []byte("\n"))
	require.Len(t, lines, 1, "expected exactly one structured log line")

	var entry map[string]interface{}
	require.NoError(t, json.Unmarshal(lines[0], &entry))
	require.Equal(t, "http_request", entry["message"])
	level, _ := entry["level"].(string)
	return level
}

// TestGinLoggerSeverityFollowsStatus covers contract C6: successful requests
// are debug-level, client errors are warnings and server errors are errors, so
// failures remain visible at the default info level. A 404 is the exception —
// it is logged at info, because a request for a route that does not exist is
// routine internet noise rather than an operator signal.
func TestGinLoggerSeverityFollowsStatus(t *testing.T) {
	cases := []struct {
		name   string
		status int
		want   string
	}{
		{"success is debug", http.StatusOK, "debug"},
		{"redirect is debug", http.StatusMovedPermanently, "debug"},
		{"not found is info", http.StatusNotFound, "info"},
		{"bad request is warn", http.StatusBadRequest, "warn"},
		{"unauthorized is warn", http.StatusUnauthorized, "warn"},
		{"forbidden is warn", http.StatusForbidden, "warn"},
		{"unprocessable is warn", http.StatusUnprocessableEntity, "warn"},
		{"server error is error", http.StatusInternalServerError, "error"},
		{"bad gateway is error", http.StatusBadGateway, "error"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, requestLogLevel(t, zerolog.DebugLevel, tc.status))
		})
	}
}

// TestGinLoggerFailuresVisibleAtInfoLevel covers the operator-facing half of
// contract C6: with the logger at its default info level, successful traffic
// stays silent while 4xx and 5xx responses are still reported.
func TestGinLoggerFailuresVisibleAtInfoLevel(t *testing.T) {
	require.Equal(t, "", requestLogLevel(t, zerolog.InfoLevel, http.StatusOK),
		"successful requests must not be logged at info level")
	require.Equal(t, "warn", requestLogLevel(t, zerolog.InfoLevel, http.StatusUnauthorized),
		"client errors must be visible at info level")
	require.Equal(t, "error", requestLogLevel(t, zerolog.InfoLevel, http.StatusServiceUnavailable),
		"server errors must be visible at info level")
}

// TestGinLoggerMissingRoutesDoNotWarn covers the operator-facing half of the
// 404 carve-out: an operator running the control plane at warn level (the
// hosted default asked for in #559) sees nothing at all from favicon probes and
// scanners, and nothing they see is at a level that trips alerting.
func TestGinLoggerMissingRoutesDoNotWarn(t *testing.T) {
	require.Equal(t, "", requestLogLevel(t, zerolog.WarnLevel, http.StatusNotFound),
		"missing routes must be silent for an operator running at warn")
	require.Equal(t, "info", requestLogLevel(t, zerolog.InfoLevel, http.StatusNotFound),
		"missing routes stay visible at the default level, just not as warnings")
}
