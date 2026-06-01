package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestAdminTokenAuth(t *testing.T) {
	tests := []struct {
		name        string
		adminToken  string
		headerToken string
		wantStatus  int
	}{
		{
			name:       "disabled allows request through",
			adminToken: "",
			wantStatus: http.StatusOK,
		},
		{
			name:        "valid admin token",
			adminToken:  "admin-secret",
			headerToken: "admin-secret",
			wantStatus:  http.StatusOK,
		},
		{
			name:        "missing admin token header",
			adminToken:  "admin-secret",
			wantStatus:  http.StatusForbidden,
		},
		{
			name:        "invalid admin token header",
			adminToken:  "admin-secret",
			headerToken: "wrong-secret",
			wantStatus:  http.StatusForbidden,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gin.SetMode(gin.TestMode)
			router := gin.New()
			router.Use(AdminTokenAuth(tt.adminToken))
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
			if tt.wantStatus == http.StatusForbidden {
				require.Contains(t, recorder.Body.String(), "admin token required")
			}
		})
	}
}
