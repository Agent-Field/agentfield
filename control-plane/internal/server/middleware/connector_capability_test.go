package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func setupConnectorCapabilityRouter(capName string, capabilities map[string]config.ConnectorCapability) *gin.Engine {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(ConnectorCapabilityCheck(capName, capabilities))

	handler := func(c *gin.Context) {
		c.String(http.StatusOK, "ok")
	}

	for _, method := range []string{
		http.MethodGet,
		http.MethodHead,
		http.MethodOptions,
		http.MethodPost,
		http.MethodPut,
		http.MethodDelete,
		http.MethodPatch,
	} {
		router.Handle(method, "/resource", handler)
	}

	return router
}

func TestConnectorCapabilityDisabledReturnsForbidden(t *testing.T) {
	router := setupConnectorCapabilityRouter("calendar", map[string]config.ConnectorCapability{
		"calendar": {Enabled: false},
	})

	req := httptest.NewRequest(http.MethodGet, "/resource", nil)
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)

	require.Equal(t, http.StatusForbidden, recorder.Code)
	require.Contains(t, recorder.Body.String(), "capability_disabled")
}

func TestConnectorCapabilityReadOnlyBlocksWriteMethodsAndAllowsSafeMethods(t *testing.T) {
	router := setupConnectorCapabilityRouter("calendar", map[string]config.ConnectorCapability{
		"calendar": {Enabled: true, ReadOnly: true},
	})

	tests := []struct {
		method     string
		wantStatus int
		wantBody   string
	}{
		{method: http.MethodGet, wantStatus: http.StatusOK},
		{method: http.MethodHead, wantStatus: http.StatusOK},
		{method: http.MethodOptions, wantStatus: http.StatusOK},
		{method: http.MethodPost, wantStatus: http.StatusForbidden, wantBody: "read_only"},
		{method: http.MethodPut, wantStatus: http.StatusForbidden, wantBody: "read_only"},
		{method: http.MethodDelete, wantStatus: http.StatusForbidden, wantBody: "read_only"},
		{method: http.MethodPatch, wantStatus: http.StatusForbidden, wantBody: "read_only"},
	}

	for _, tt := range tests {
		t.Run(tt.method, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, "/resource", nil)
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, req)

			require.Equal(t, tt.wantStatus, recorder.Code)
			if tt.wantBody != "" {
				require.Contains(t, recorder.Body.String(), tt.wantBody)
			}
		})
	}
}

func TestConnectorCapabilityMissingOrNilCapabilitiesFailClosed(t *testing.T) {
	tests := []struct {
		name         string
		capabilities map[string]config.ConnectorCapability
	}{
		{
			name:         "missing capability key",
			capabilities: map[string]config.ConnectorCapability{},
		},
		{
			name:         "nil capabilities map",
			capabilities: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			router := setupConnectorCapabilityRouter("calendar", tt.capabilities)

			req := httptest.NewRequest(http.MethodGet, "/resource", nil)
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, req)

			require.Equal(t, http.StatusForbidden, recorder.Code)
			require.Contains(t, recorder.Body.String(), "capability_disabled")
		})
	}
}

func TestConnectorCapabilityEnabledWritableAllowsAllMethods(t *testing.T) {
	router := setupConnectorCapabilityRouter("calendar", map[string]config.ConnectorCapability{
		"calendar": {Enabled: true, ReadOnly: false},
	})

	for _, method := range []string{
		http.MethodGet,
		http.MethodHead,
		http.MethodOptions,
		http.MethodPost,
		http.MethodPut,
		http.MethodDelete,
		http.MethodPatch,
	} {
		t.Run(method, func(t *testing.T) {
			req := httptest.NewRequest(method, "/resource", nil)
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, req)

			require.Equal(t, http.StatusOK, recorder.Code)
		})
	}
}
