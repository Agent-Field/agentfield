package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestPprofIndexWithValidAdminToken(t *testing.T) {
	t.Parallel()

	gin.SetMode(gin.TestMode)

	srv := &AgentFieldServer{
		Router: gin.New(),
		config: &config.Config{},
	}
	srv.config.Features.DID.Authorization.AdminToken = "test-admin-token"
	srv.registerPprofRoutes()

	req, _ := http.NewRequest(http.MethodGet, "/debug/pprof/", nil)
	req.Header.Set("X-Admin-Token", "test-admin-token")
	w := httptest.NewRecorder()
	srv.Router.ServeHTTP(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.Contains(t, w.Body.String(), "Types of profiles")
	require.Contains(t, w.Body.String(), "goroutine")
	require.Contains(t, w.Body.String(), "heap")
}

func TestPprofIndexWithoutToken(t *testing.T) {
	t.Parallel()

	gin.SetMode(gin.TestMode)

	srv := &AgentFieldServer{
		Router: gin.New(),
		config: &config.Config{},
	}
	srv.config.Features.DID.Authorization.AdminToken = "test-admin-token"
	srv.registerPprofRoutes()

	req, _ := http.NewRequest(http.MethodGet, "/debug/pprof/", nil)
	w := httptest.NewRecorder()
	srv.Router.ServeHTTP(w, req)

	require.Equal(t, http.StatusUnauthorized, w.Code)
}

func TestPprofIndexWithWrongToken(t *testing.T) {
	t.Parallel()

	gin.SetMode(gin.TestMode)

	srv := &AgentFieldServer{
		Router: gin.New(),
		config: &config.Config{},
	}
	srv.config.Features.DID.Authorization.AdminToken = "test-admin-token"
	srv.registerPprofRoutes()

	req, _ := http.NewRequest(http.MethodGet, "/debug/pprof/", nil)
	req.Header.Set("X-Admin-Token", "wrong-token")
	w := httptest.NewRecorder()
	srv.Router.ServeHTTP(w, req)

	require.Equal(t, http.StatusUnauthorized, w.Code)
}

func TestPprofNamedProfileGoroutine(t *testing.T) {
	t.Parallel()

	gin.SetMode(gin.TestMode)

	srv := &AgentFieldServer{
		Router: gin.New(),
		config: &config.Config{},
	}
	srv.config.Features.DID.Authorization.AdminToken = "test-admin-token"
	srv.registerPprofRoutes()

	req, _ := http.NewRequest(http.MethodGet, "/debug/pprof/goroutine?debug=1", nil)
	req.Header.Set("X-Admin-Token", "test-admin-token")
	w := httptest.NewRecorder()
	srv.Router.ServeHTTP(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.True(t, strings.Contains(w.Body.String(), "goroutine") || w.Body.Len() > 0)
}

func TestPprofNamedProfileHeap(t *testing.T) {
	t.Parallel()

	gin.SetMode(gin.TestMode)

	srv := &AgentFieldServer{
		Router: gin.New(),
		config: &config.Config{},
	}
	srv.config.Features.DID.Authorization.AdminToken = "test-admin-token"
	srv.registerPprofRoutes()

	req, _ := http.NewRequest(http.MethodGet, "/debug/pprof/heap?debug=1", nil)
	req.Header.Set("X-Admin-Token", "test-admin-token")
	w := httptest.NewRecorder()
	srv.Router.ServeHTTP(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.True(t, w.Body.Len() > 0)
}

func TestPprofNamedProfileWithoutToken(t *testing.T) {
	t.Parallel()

	gin.SetMode(gin.TestMode)

	srv := &AgentFieldServer{
		Router: gin.New(),
		config: &config.Config{},
	}
	srv.config.Features.DID.Authorization.AdminToken = "test-admin-token"
	srv.registerPprofRoutes()

	req, _ := http.NewRequest(http.MethodGet, "/debug/pprof/goroutine?debug=1", nil)
	w := httptest.NewRecorder()
	srv.Router.ServeHTTP(w, req)

	require.Equal(t, http.StatusUnauthorized, w.Code)
}

func TestPprofNoAdminTokenConfigured(t *testing.T) {
	t.Parallel()

	gin.SetMode(gin.TestMode)

	srv := &AgentFieldServer{
		Router: gin.New(),
		config: &config.Config{},
	}
	srv.registerPprofRoutes()

	req, _ := http.NewRequest(http.MethodGet, "/debug/pprof/", nil)
	w := httptest.NewRecorder()
	srv.Router.ServeHTTP(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	require.Contains(t, w.Body.String(), "Types of profiles")
}
