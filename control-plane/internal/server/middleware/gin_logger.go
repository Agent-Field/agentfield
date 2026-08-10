package middleware

import (
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/gin-gonic/gin"
)

// GinLogger replaces gin's default stdout logger. It emits one structured
// log line per request through the control-plane's zerolog logger at DEBUG
// level, so request-level detail only appears when the log level is lowered
// to debug. This keeps the default (info) output free of the duplicated,
// overly verbose [GIN] lines that gin.Default() writes to stdout.
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

		logger.Logger.Debug().
			Str("client_ip", clientIP).
			Str("method", method).
			Str("path", path).
			Int("status", statusCode).
			Dur("latency", latency).
			Str("error", errorMessage).
			Msg("http_request")
	}
}