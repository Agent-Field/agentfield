package handlers

import (
	"context"

	"github.com/gin-gonic/gin"
)

// WithStreamContext makes a streaming handler stop when either its client
// disconnects or the server begins shutting down.
func WithStreamContext(streamCtx context.Context, handler gin.HandlerFunc) gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx, cancel := context.WithCancel(c.Request.Context())
		defer cancel()
		go func() {
			select {
			case <-streamCtx.Done():
				cancel()
			case <-ctx.Done():
			}
		}()
		c.Request = c.Request.WithContext(ctx)
		handler(c)
	}
}
