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
//
// 404 is the exception among the 4xx: a request for a route that does not
// exist is the caller's problem, not an operator signal. Favicon probes,
// requests for /ui/ before the UI is built and internet background noise
// against a hosted control plane would otherwise raise a warning each, which
// is precisely the alert-level noise operators asked to be rid of. It is
// logged at INFO so it is still visible by default but never trips alerting
// keyed on level >= warn. All other 4xx stay at WARN.
func requestLogEvent(statusCode int) *zerolog.Event {
	switch {
	case statusCode >= http.StatusInternalServerError:
		return logger.Logger.Error()
	case statusCode == http.StatusNotFound:
		return logger.Logger.Info()
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
