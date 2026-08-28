package server

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestHTTPServerIngressTimeoutsDoNotSetStreamWriteTimeout(t *testing.T) {
	s := &AgentFieldServer{Router: gin.New()}
	httpServer := s.newHTTPServer(":0")
	require.Equal(t, 10*time.Second, httpServer.ReadHeaderTimeout)
	require.Equal(t, 120*time.Second, httpServer.IdleTimeout)
	require.Equal(t, 1<<20, httpServer.MaxHeaderBytes)
	require.Zero(t, httpServer.ReadTimeout)
	require.Zero(t, httpServer.WriteTimeout)
}

func TestMaxExecuteBodyHandlerRejectsOversizeBeforeHandler(t *testing.T) {
	called := false
	handler := maxExecuteBodyHandler(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		called = true
	}), 4)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/execute/async/node.reasoner", strings.NewReader("12345"))
	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, req)
	require.Equal(t, http.StatusRequestEntityTooLarge, resp.Code)
	require.JSONEq(t, `{"error":"request body too large"}`, resp.Body.String())
	require.False(t, called)
}

func TestMaxExecuteBodyHandlerDoesNotCapOtherRoutes(t *testing.T) {
	handler := maxExecuteBodyHandler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		_, _ = w.Write(body)
	}), 4)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/logs", strings.NewReader("12345"))
	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, req)
	require.Equal(t, http.StatusOK, resp.Code)
	require.Equal(t, "12345", resp.Body.String())

	req = httptest.NewRequest(http.MethodPost, "/api/v1/executions/abc/status", strings.NewReader("12345"))
	resp = httptest.NewRecorder()
	handler.ServeHTTP(resp, req)
	require.Equal(t, http.StatusOK, resp.Code)
	require.Equal(t, "12345", resp.Body.String())
}
