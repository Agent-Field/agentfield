package middleware

import (
	"bytes"
	"encoding/json"
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

// TestGinLoggerSilentAtInfoLevel verifies that at the default INFO level the
// middleware emits nothing, keeping default output free of per-request noise.
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