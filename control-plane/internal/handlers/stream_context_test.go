package handlers

import (
	"context"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestWithStreamContextEndsHandlerWhenServerContextIsCancelled(t *testing.T) {
	gin.SetMode(gin.TestMode)
	streamCtx, cancelStream := context.WithCancel(context.Background())
	started := make(chan struct{})
	done := make(chan struct{})
	router := gin.New()
	router.GET("/events", WithStreamContext(streamCtx, func(c *gin.Context) {
		close(started)
		<-c.Request.Context().Done()
		close(done)
	}))

	go router.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/events", nil))
	<-started
	cancelStream()

	require.Eventually(t, func() bool {
		select {
		case <-done:
			return true
		default:
			return false
		}
	}, time.Second, time.Millisecond)
}
