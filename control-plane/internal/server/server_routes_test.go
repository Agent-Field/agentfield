package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/your-org/agentfield/control-plane/internal/config"
	"github.com/your-org/agentfield/control-plane/internal/services"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestSetupRoutesRegistersMetricsAndUI(t *testing.T) {
	t.Parallel()

	gin.SetMode(gin.TestMode)

	srv := &AgentFieldServer{
		Router: gin.New(),
		config: &config.Config{
			UI:  config.UIConfig{Enabled: true, Mode: "embedded"},
			API: config.APIConfig{},
		},
	}

	srv.setupRoutes()

	t.Run("metrics endpoint", func(t *testing.T) {
		req, _ := http.NewRequest(http.MethodGet, "/metrics", nil)
		req.Header.Set("Origin", "http://localhost:5173")
		w := httptest.NewRecorder()
		srv.Router.ServeHTTP(w, req)
		require.Equal(t, http.StatusOK, w.Code)
		require.Equal(t, "http://localhost:5173", w.Header().Get("Access-Control-Allow-Origin"))
	})

	t.Run("root redirect", func(t *testing.T) {
		req, _ := http.NewRequest(http.MethodGet, "/", nil)
		w := httptest.NewRecorder()
		srv.Router.ServeHTTP(w, req)
		require.Equal(t, http.StatusMovedPermanently, w.Code)
		require.Equal(t, "/ui/", w.Header().Get("Location"))
	})
}

//nolint:unused // Reserved for future test cases
type stubHealthMonitor struct {
	*services.HealthMonitor
}

func TestUnregisterAgentFromMonitoringResponses(t *testing.T) {
	t.Parallel()

	gin.SetMode(gin.TestMode)

	srv := &AgentFieldServer{}

	t.Run("missing node id returns 400", func(t *testing.T) {
		req, _ := http.NewRequest(http.MethodDelete, "/internal/nodes//monitor", nil)
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Params = append(c.Params, gin.Param{Key: "node_id", Value: ""})
		c.Request = req

		srv.unregisterAgentFromMonitoring(c)
		require.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("health monitor unavailable", func(t *testing.T) {
		req, _ := http.NewRequest(http.MethodDelete, "/internal/nodes/node-1/monitor", nil)
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Params = append(c.Params, gin.Param{Key: "node_id", Value: "node-1"})
		c.Request = req

		srv.unregisterAgentFromMonitoring(c)
		require.Equal(t, http.StatusServiceUnavailable, w.Code)
	})

	t.Run("successful unregister", func(t *testing.T) {
		hm := services.NewHealthMonitor(nil, services.HealthMonitorConfig{}, nil, nil, nil, nil)
		req, _ := http.NewRequest(http.MethodDelete, "/internal/nodes/node-42/monitor", nil)
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Params = append(c.Params, gin.Param{Key: "node_id", Value: "node-42"})
		c.Request = req

		srv.healthMonitor = hm
		srv.unregisterAgentFromMonitoring(c)
		require.Equal(t, http.StatusOK, w.Code)
	})
}
