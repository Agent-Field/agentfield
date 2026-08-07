package middleware

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// captureLogger swaps the global logger for one writing to buf and returns a
// restore func. It sets the level to debug so the middleware's DEBUG-level
// request line is emitted.
func captureLogger(buf *bytes.Buffer) func() {
	prev := logger.Logger
	logger.Logger = zerolog.New(buf).With().Timestamp().Logger().Level(zerolog.DebugLevel)
	return func() { logger.Logger = prev }
}

func TestGinLoggerEmitsStructuredRequestLine(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var buf bytes.Buffer
	restore := captureLogger(&buf)
	defer restore()

	r := gin.New()
	r.Use(GinLogger())
	r.GET("/health", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodGet, "/health?probe=1", nil)
	req.RemoteAddr = "203.0.113.7:1234"
	w := httptest.NewRecorder()

	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	out := buf.String()
	require.NotEmpty(t, out, "expected a structured log line to be emitted")
	assert.Contains(t, out, `"level":"debug"`)
	assert.Contains(t, out, `"method":"GET"`)
	assert.Contains(t, out, `"path":"/health?probe=1"`)
	assert.Contains(t, out, `"status":200`)
	assert.Contains(t, out, `"http_request"`)
	// The verbose gin [GIN] stdout line must not appear.
	assert.NotContains(t, out, "[GIN]")
}

func TestGinLoggerPassesThroughHandlerError(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var buf bytes.Buffer
	restore := captureLogger(&buf)
	defer restore()

	r := gin.New()
	r.Use(GinLogger())
	r.GET("/boom", func(c *gin.Context) {
		c.AbortWithStatus(http.StatusInternalServerError)
	})

	req := httptest.NewRequest(http.MethodGet, "/boom", nil)
	w := httptest.NewRecorder()

	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusInternalServerError, w.Code)
	out := buf.String()
	assert.Contains(t, out, `"status":500`)
	assert.True(t, strings.Contains(out, `"http_request"`), "expected request line even on error")
}
