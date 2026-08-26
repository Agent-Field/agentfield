package middleware

import (
	"net/http"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"
)

// requestLogEvent picks the severity for a request log line from its response
// status: server errors are logged at ERROR, client errors at WARN and
// everything else at DEBUG. That keeps the default (info) output free of
// per-request noise while still surfacing failures without having to lower the
// level to debug.
func requestLogEvent(statusCode int) *zerolog.Event {
	switch {
	case statusCode >= http.StatusInternalServerError:
		return logger.Logger.Error()
	case statusCode >= http.StatusBadRequest:
		return logger.Logger.Warn()
	default:
		return logger.Logger.Debug()
	}
}

// GinLogger replaces gin's default stdout logger. It emits one structured
// log line per request through the control-plane's zerolog logger, at a
// severity derived from the response status (see requestLogEvent). This keeps
// the default output free of the duplicated, overly verbose [GIN] lines that
// gin.Default() writes to stdout, without hiding failing requests.
func GinLogger() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		path := c.Request.URL.Path
		raw := c.Request.URL.RawQuery

		c.Next()

		latency := time.Since(start)
		clientIP := c.ClientIP()
		method := c.Request.Method
		statusCode := c.Writer.Status()
		errorMessage := c.Errors.ByType(gin.ErrorTypePrivate).String()

		if raw != "" {
			path = path + "?" + raw
		}

		requestLogEvent(statusCode).
			Str("client_ip", clientIP).
			Str("method", method).
			Str("path", path).
			Int("status", statusCode).
			Dur("latency", latency).
			Str("error", errorMessage).
			Msg("http_request")
	}
}
