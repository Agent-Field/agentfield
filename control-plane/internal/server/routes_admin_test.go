package server

import (
	"bytes"
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

	require.Equal(t, http.StatusForbidden, w.Code)
	require.JSONEq(t, `{"error":"forbidden","message":"admin token required for this operation (use X-Admin-Token header)"}`, w.Body.String())
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

	require.Equal(t, http.StatusForbidden, w.Code)
	require.JSONEq(t, `{"error":"forbidden","message":"admin token required for this operation (use X-Admin-Token header)"}`, w.Body.String())
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

	require.Equal(t, http.StatusForbidden, w.Code)
	require.JSONEq(t, `{"error":"forbidden","message":"admin token required for this operation (use X-Admin-Token header)"}`, w.Body.String())
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

func TestPprofSymbolSupportsPOST(t *testing.T) {
	t.Parallel()

	gin.SetMode(gin.TestMode)
	srv := &AgentFieldServer{
		Router: gin.New(),
		config: &config.Config{},
	}
	srv.config.Features.DID.Authorization.AdminToken = "test-admin-token"
	srv.registerPprofRoutes()

	req := httptest.NewRequest(http.MethodPost, "/debug/pprof/symbol", bytes.NewBufferString("0x0\n"))
	req.Header.Set("X-Admin-Token", "test-admin-token")
	w := httptest.NewRecorder()
	srv.Router.ServeHTTP(w, req)

	require.NotEqual(t, http.StatusNotFound, w.Code)
	require.Equal(t, http.StatusOK, w.Code)
	require.Contains(t, w.Body.String(), "num_symbols:")
}

func TestPprofComposesAPIKeyAndAdminTokenAuth(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name       string
		apiKey     string
		adminToken string
		headers    map[string]string
		wantStatus int
	}{
		{name: "neither credential", apiKey: "api-key", adminToken: "admin-token", wantStatus: http.StatusUnauthorized},
		{name: "API key only", apiKey: "api-key", adminToken: "admin-token", headers: map[string]string{"X-API-Key": "api-key"}, wantStatus: http.StatusForbidden},
		{name: "admin token only", apiKey: "api-key", adminToken: "admin-token", headers: map[string]string{"X-Admin-Token": "admin-token"}, wantStatus: http.StatusUnauthorized},
		{name: "both credentials", apiKey: "api-key", adminToken: "admin-token", headers: map[string]string{"X-API-Key": "api-key", "X-Admin-Token": "admin-token"}, wantStatus: http.StatusOK},
		{name: "empty admin token falls back to API key", apiKey: "api-key", headers: map[string]string{"X-API-Key": "api-key"}, wantStatus: http.StatusOK},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := &AgentFieldServer{
				Router: gin.New(),
				config: &config.Config{},
			}
			srv.config.API.Auth.APIKey = tt.apiKey
			srv.config.Features.DID.Authorization.AdminToken = tt.adminToken
			srv.applyGlobalMiddleware()
			srv.registerPprofRoutes()

			req := httptest.NewRequest(http.MethodGet, "/debug/pprof/", nil)
			for name, value := range tt.headers {
				req.Header.Set(name, value)
			}
			w := httptest.NewRecorder()
			srv.Router.ServeHTTP(w, req)

			require.Equal(t, tt.wantStatus, w.Code)
		})
	}
}
