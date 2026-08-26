package server

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
	useStructuredRequestLogging(router)
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
	useStructuredRequestLogging(router)
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
	useStructuredRequestLogging(router)
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
