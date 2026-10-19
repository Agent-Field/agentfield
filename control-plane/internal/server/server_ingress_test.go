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

// Session tool calls run the execute pipeline in-process, so the execute body
// cap applies to them directly, including path variants gin routes to the tool
// handler (an empty session id).
func TestMaxExecuteBodyHandlerCapsSessionToolCalls(t *testing.T) {
	for _, path := range []string{
		"/api/v1/session-instances/sess-1/tools/node.reasoner",
		"/api/v1/session-instances//tools/node.reasoner",
		"/api/v1/session-instances/sess-1/realtime-offer",
	} {
		called := false
		handler := maxExecuteBodyHandler(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
			called = true
		}), 4)
		req := httptest.NewRequest(http.MethodPost, path, strings.NewReader("12345"))
		resp := httptest.NewRecorder()
		handler.ServeHTTP(resp, req)
		require.Equal(t, http.StatusRequestEntityTooLarge, resp.Code, path)
		require.JSONEq(t, `{"error":"request body too large"}`, resp.Body.String())
		require.False(t, called, path)
	}

	passthrough := maxExecuteBodyHandler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		_, _ = w.Write(body)
	}), 4)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/session-targets/node.session/start", strings.NewReader("12345"))
	resp := httptest.NewRecorder()
	passthrough.ServeHTTP(resp, req)
	require.Equal(t, http.StatusOK, resp.Code)
	require.Equal(t, "12345", resp.Body.String())
}

// The cap must hold for every path the production router sends to the session
// tool chain, not only the canonical one.
func TestNewHTTPServerCapsSessionToolBodiesThroughRouter(t *testing.T) {
	t.Setenv("AGENTFIELD_MAX_EXECUTE_BODY_BYTES", "16")
	router := gin.New()
	reached := 0
	router.POST("/api/v1/session-instances/:session_id/tools/:tool", func(c *gin.Context) {
		reached++
		c.Status(http.StatusAccepted)
	})
	s := &AgentFieldServer{Router: router}
	httpServer := s.newHTTPServer(":0")
	large := `{"input":{"padding":"` + strings.Repeat("x", 64) + `"}}`
	for _, path := range []string{
		"/api/v1/session-instances/sess-1/tools/node.reasoner",
		"/api/v1/session-instances//tools/node.reasoner",
	} {
		resp := httptest.NewRecorder()
		httpServer.Handler.ServeHTTP(resp, httptest.NewRequest(http.MethodPost, path, strings.NewReader(large)))
		require.Equal(t, http.StatusRequestEntityTooLarge, resp.Code, path)
	}
	require.Zero(t, reached)

	resp := httptest.NewRecorder()
	httpServer.Handler.ServeHTTP(resp, httptest.NewRequest(http.MethodPost, "/api/v1/session-instances//tools/node.reasoner", strings.NewReader(`{}`)))
	require.Equal(t, http.StatusAccepted, resp.Code, "gin routes an empty session id to the tool chain")
	require.Equal(t, 1, reached)
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

func TestMaxRequestBodyHandlerRejectsOversizeRegistrationBeforeHandler(t *testing.T) {
	for _, tc := range []struct {
		name    string
		chunked bool
	}{
		{name: "content length"},
		{name: "chunked", chunked: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			called := false
			handler := maxRequestBodyHandler(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
				called = true
			}), 32, 4)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/nodes/register", strings.NewReader("12345"))
			if tc.chunked {
				req.ContentLength = -1
				req.TransferEncoding = []string{"chunked"}
			}
			resp := httptest.NewRecorder()
			handler.ServeHTTP(resp, req)
			require.Equal(t, http.StatusRequestEntityTooLarge, resp.Code)
			require.JSONEq(t, `{"error":"request body too large"}`, resp.Body.String())
			require.False(t, called)
		})
	}
}

func TestMaxRequestBodyHandlerAllowsNormalRegistration(t *testing.T) {
	handler := maxRequestBodyHandler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		_, _ = w.Write(body)
	}), 4, 8)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/nodes/register-serverless", strings.NewReader("normal"))
	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, req)
	require.Equal(t, http.StatusOK, resp.Code)
	require.Equal(t, "normal", resp.Body.String())
}

func TestNewHTTPServerHonoursRegisterBodyCapEnv(t *testing.T) {
	t.Setenv("AGENTFIELD_MAX_REGISTER_BODY_BYTES", "4")
	router := gin.New()
	called := false
	router.POST("/api/v1/nodes/register", func(c *gin.Context) { called = true; c.Status(http.StatusOK) })
	s := &AgentFieldServer{Router: router}
	httpServer := s.newHTTPServer(":0")

	req := httptest.NewRequest(http.MethodPost, "/api/v1/nodes/register", strings.NewReader("12345"))
	resp := httptest.NewRecorder()
	httpServer.Handler.ServeHTTP(resp, req)
	require.Equal(t, http.StatusRequestEntityTooLarge, resp.Code)
	require.JSONEq(t, `{"error":"request body too large"}`, resp.Body.String())
	require.False(t, called)

	req = httptest.NewRequest(http.MethodPost, "/api/v1/nodes/register", strings.NewReader("ok"))
	resp = httptest.NewRecorder()
	httpServer.Handler.ServeHTTP(resp, req)
	require.Equal(t, http.StatusOK, resp.Code)
	require.True(t, called)
}

func TestNewHTTPServerFallsBackToDefaultRegisterCapOnInvalidEnv(t *testing.T) {
	t.Setenv("AGENTFIELD_MAX_REGISTER_BODY_BYTES", "not-a-number")
	router := gin.New()
	called := false
	router.POST("/api/v1/nodes", func(c *gin.Context) { called = true; c.Status(http.StatusOK) })
	s := &AgentFieldServer{Router: router}
	httpServer := s.newHTTPServer(":0")

	req := httptest.NewRequest(http.MethodPost, "/api/v1/nodes", strings.NewReader(strings.Repeat("x", 1024)))
	resp := httptest.NewRecorder()
	httpServer.Handler.ServeHTTP(resp, req)
	require.Equal(t, http.StatusOK, resp.Code, "an invalid cap must fall back to the 8 MiB default, not reject")
	require.True(t, called)
}

type failingBody struct{}

func (failingBody) Read([]byte) (int, error) { return 0, io.ErrUnexpectedEOF }
func (failingBody) Close() error             { return nil }

func TestMaxRequestBodyHandlerRejectsUnreadableRegistrationBody(t *testing.T) {
	called := false
	handler := maxRequestBodyHandler(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		called = true
	}), 32, 32)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/nodes/register", nil)
	req.Body = failingBody{}
	req.ContentLength = -1
	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, req)
	require.Equal(t, http.StatusBadRequest, resp.Code)
	require.False(t, called)
}
