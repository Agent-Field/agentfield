package middleware

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestValidateAdminTokenAuth_RejectsImplicitDisable(t *testing.T) {
	err := ValidateAdminTokenAuth(AdminAuthConfig{})
	require.Error(t, err)
	require.Contains(t, err.Error(), "AGENTFIELD_INSECURE_ADMIN_NO_TOKEN=true")
}

func TestValidateAdminTokenAuth_AcceptsExplicitInsecure(t *testing.T) {
	err := ValidateAdminTokenAuth(AdminAuthConfig{InsecureDisableAdminAuth: true})
	require.NoError(t, err)
}

func TestValidateAdminTokenAuth_AcceptsConfiguredToken(t *testing.T) {
	err := ValidateAdminTokenAuth(AdminAuthConfig{AdminToken: "admin-secret"})
	require.NoError(t, err)
}

func TestAdminTokenAuth(t *testing.T) {
	tests := []struct {
		name        string
		config      AdminAuthConfig
		headerToken string
		wantStatus  int
		wantBody    string
	}{
		{
			name:       "empty token with explicit insecure allows request through",
			config:     AdminAuthConfig{InsecureDisableAdminAuth: true},
			wantStatus: http.StatusOK,
		},
		{
			name:       "empty token without insecure flag returns 401",
			config:     AdminAuthConfig{},
			wantStatus: http.StatusUnauthorized,
			wantBody:   "admin authentication required but admin token is not configured",
		},
		{
			name:        "valid admin token",
			config:      AdminAuthConfig{AdminToken: "admin-secret"},
			headerToken: "admin-secret",
			wantStatus:  http.StatusOK,
		},
		{
			name:       "missing admin token header",
			config:     AdminAuthConfig{AdminToken: "admin-secret"},
			wantStatus: http.StatusForbidden,
			wantBody:   "admin token required",
		},
		{
			name:        "invalid admin token header",
			config:      AdminAuthConfig{AdminToken: "admin-secret"},
			headerToken: "wrong-secret",
			wantStatus:  http.StatusForbidden,
			wantBody:    "admin token required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gin.SetMode(gin.TestMode)
			router := gin.New()
			router.Use(AdminTokenAuth(tt.config))
			router.GET("/admin", func(c *gin.Context) {
				c.String(http.StatusOK, "ok")
			})

			req := httptest.NewRequest(http.MethodGet, "/admin", nil)
			if tt.headerToken != "" {
				req.Header.Set("X-Admin-Token", tt.headerToken)
			}

			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, req)

			require.Equal(t, tt.wantStatus, recorder.Code)
			if tt.wantBody != "" {
				require.Contains(t, recorder.Body.String(), tt.wantBody)
			}
		})
	}
}

func TestAdminTokenAuth_EmptyTokenFailsClosed(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(AdminTokenAuth(AdminAuthConfig{}))
	router.GET("/admin/tags", func(c *gin.Context) {
		c.String(http.StatusOK, "ok")
	})

	// Request with only a valid API key (simulated via no admin token header)
	// should be rejected because admin token is not configured.
	req := httptest.NewRequest(http.MethodGet, "/admin/tags", nil)
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)

	require.Equal(t, http.StatusUnauthorized, recorder.Code)

	var resp map[string]string
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &resp))
	require.Equal(t, "unauthorized", resp["error"])
	require.Contains(t, resp["message"], "admin token is not configured")
}

func TestAdminTokenAuth_APIKeyOnlyReturns401WhenAdminTokenConfigured(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(AdminTokenAuth(AdminAuthConfig{AdminToken: "admin-secret"}))
	router.GET("/admin/policies", func(c *gin.Context) {
		c.String(http.StatusOK, "ok")
	})

	// Request with API key header but NO admin token — should return 403
	req := httptest.NewRequest(http.MethodGet, "/admin/policies", nil)
	req.Header.Set("X-API-Key", "some-api-key")
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)

	require.Equal(t, http.StatusForbidden, recorder.Code)
	require.Contains(t, recorder.Body.String(), "admin token required")
}
