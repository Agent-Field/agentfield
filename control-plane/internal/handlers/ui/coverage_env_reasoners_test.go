package ui

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func newEnvRouter(handler *EnvHandler) *gin.Engine {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	group := router.Group("/api/ui/v1/agents")
	group.GET("/:agentId/env", handler.GetEnvHandler)
	group.PUT("/:agentId/env", handler.PutEnvHandler)
	group.PATCH("/:agentId/env", handler.PatchEnvHandler)
	group.DELETE("/:agentId/env/:key", handler.DeleteEnvVarHandler)
	return router
}

func TestEnvHandlerCoverage(t *testing.T) {
	require.NotNil(t, NewEnvHandler(nil, nil, "/tmp"))

	t.Run("get env validations and masking", func(t *testing.T) {
		storage := &overrideStorage{StorageProvider: setupTestStorage(t)}
		handler := NewEnvHandler(storage, nil, t.TempDir())
		router := newEnvRouter(handler)
		installPath := t.TempDir()

		rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents//env?packageId=pkg", nil)
		require.Equal(t, http.StatusBadRequest, rec.Code)
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/agent-1/env", nil)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		storage.getAgentPackageFn = func(ctx context.Context, packageID string) (*types.AgentPackage, error) {
			if packageID == "pkg-missing" {
				return nil, errors.New("missing")
			}
			if packageID == "pkg" {
				return &types.AgentPackage{
					ID:          "pkg",
					InstallPath: installPath,
					ConfigurationSchema: json.RawMessage(`{
						"user_environment": {
							"required": [{"name":"SECRET_KEY","type":"secret"}],
							"optional": [{"name":"VISIBLE","type":"string"}]
						}
					}`),
				}, nil
			}
			return nil, errors.New("missing")
		}
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/agent-1/env?packageId=pkg-missing", nil)
		require.Equal(t, http.StatusNotFound, rec.Code)

		require.NoError(t, os.WriteFile(filepath.Join(installPath, ".env"), []byte("SECRET_KEY=supersecret\nVISIBLE=value\n"), 0600))
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/agent-1/env?packageId=pkg", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		var body EnvResponse
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
		require.True(t, body.FileExists)
		require.Contains(t, body.MaskedKeys, "SECRET_KEY")
		require.Contains(t, body.Variables["SECRET_KEY"], "...")
	})

	t.Run("put env validation and success", func(t *testing.T) {
		storage := &overrideStorage{StorageProvider: setupTestStorage(t)}
		handler := NewEnvHandler(storage, nil, t.TempDir())
		router := newEnvRouter(handler)

		rec := performJSONRequest(router, http.MethodPut, "/api/ui/v1/agents/agent-1/env?packageId=pkg", map[string]any{"variables": "bad"})
		require.Equal(t, http.StatusBadRequest, rec.Code)

		installPath := t.TempDir()
		storage.getAgentPackageFn = func(ctx context.Context, packageID string) (*types.AgentPackage, error) {
			return &types.AgentPackage{ID: "pkg", InstallPath: installPath}, nil
		}
		storage.validateAgentConfigurationFn = func(ctx context.Context, agentID, packageID string, config map[string]interface{}) (*types.ConfigurationValidationResult, error) {
			if config["KEY"] == "value" {
				return &types.ConfigurationValidationResult{Valid: false, Errors: []string{"bad"}}, nil
			}
			return &types.ConfigurationValidationResult{Valid: true}, nil
		}
		rec = performJSONRequest(router, http.MethodPut, "/api/ui/v1/agents/agent-1/env?packageId=pkg", map[string]any{
			"variables": map[string]string{"KEY": "value"},
		})
		require.Equal(t, http.StatusBadRequest, rec.Code)

		rec = performJSONRequest(router, http.MethodPut, "/api/ui/v1/agents/agent-1/env?packageId=pkg", map[string]any{
			"variables": map[string]string{"KEY": "hello world"},
		})
		require.Equal(t, http.StatusOK, rec.Code)
		content, err := os.ReadFile(filepath.Join(installPath, ".env"))
		require.NoError(t, err)
		require.Contains(t, string(content), `KEY="hello world"`)
	})

	t.Run("patch and delete env branches", func(t *testing.T) {
		storage := &overrideStorage{StorageProvider: setupTestStorage(t)}
		installPath := t.TempDir()
		envPath := filepath.Join(installPath, ".env")
		require.NoError(t, os.WriteFile(envPath, []byte("KEEP=value\nREMOVE=gone\n"), 0600))
		handler := NewEnvHandler(storage, nil, t.TempDir())
		router := newEnvRouter(handler)

		storage.getAgentPackageFn = func(ctx context.Context, packageID string) (*types.AgentPackage, error) {
			return &types.AgentPackage{ID: "pkg", InstallPath: installPath}, nil
		}
		storage.validateAgentConfigurationFn = func(ctx context.Context, agentID, packageID string, config map[string]interface{}) (*types.ConfigurationValidationResult, error) {
			if _, ok := config["REMOVE"]; ok {
				return &types.ConfigurationValidationResult{Valid: true}, nil
			}
			if _, ok := config["ADD"]; ok {
				return &types.ConfigurationValidationResult{Valid: false, Errors: []string{"required"}}, nil
			}
			return &types.ConfigurationValidationResult{Valid: true}, nil
		}
		rec := performJSONRequest(router, http.MethodPatch, "/api/ui/v1/agents/agent-1/env?packageId=pkg", map[string]any{
			"variables": map[string]string{"ADD": "with space"},
		})
		require.Equal(t, http.StatusOK, rec.Code)

		rec = performJSONRequest(router, http.MethodDelete, "/api/ui/v1/agents/agent-1/env/REMOVE?packageId=pkg", nil)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		rec = performJSONRequest(router, http.MethodDelete, "/api/ui/v1/agents/agent-1/env/GHOST?packageId=pkg", nil)
		require.Equal(t, http.StatusNotFound, rec.Code)

		storage.validateAgentConfigurationFn = func(ctx context.Context, agentID, packageID string, config map[string]interface{}) (*types.ConfigurationValidationResult, error) {
			return &types.ConfigurationValidationResult{Valid: true}, nil
		}
		rec = performJSONRequest(router, http.MethodDelete, "/api/ui/v1/agents/agent-1/env/ADD?packageId=pkg", nil)
		require.Equal(t, http.StatusOK, rec.Code)
	})
}

func TestReasonersHandlerCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)
	storage := &overrideStorage{StorageProvider: setupTestStorage(t)}
	handler := NewReasonersHandler(storage)
	require.NotNil(t, handler)

	router := gin.New()
	router.GET("/api/ui/v1/reasoners", handler.GetAllReasonersHandler)
	router.GET("/api/ui/v1/reasoners/:reasonerId", handler.GetReasonerDetailsHandler)
	router.GET("/api/ui/v1/reasoners/:reasonerId/performance", handler.GetPerformanceMetricsHandler)
	router.GET("/api/ui/v1/reasoners/:reasonerId/history", handler.GetExecutionHistoryHandler)
	router.GET("/api/ui/v1/reasoners/:reasonerId/templates", handler.GetExecutionTemplatesHandler)
	router.POST("/api/ui/v1/reasoners/:reasonerId/templates", handler.SaveExecutionTemplateHandler)

	t.Run("all reasoners failure and success", func(t *testing.T) {
		storage.listAgentsFn = func(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
			return nil, errors.New("boom")
		}
		rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/reasoners", nil)
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		now := time.Now().UTC()
		active := types.HealthStatusActive
		storage.listAgentsFn = func(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
			require.NotNil(t, filters.HealthStatus)
			require.Equal(t, active, *filters.HealthStatus)
			return []*types.AgentNode{
			{
				ID:            "agent-1",
				Version:       "1.0.0",
				HealthStatus:  types.HealthStatusActive,
				LastHeartbeat: now,
				Reasoners: []types.ReasonerDefinition{
					{ID: "plan", Tags: []string{"alpha"}},
					{ID: "search", Tags: []string{"beta"}},
				},
			},
		}, nil
		}
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/reasoners?status=online&search=plan&limit=1&offset=0", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		var body ReasonersResponse
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
		require.Len(t, body.Reasoners, 1)
		require.Equal(t, 1, body.OnlineCount)
	})

	t.Run("reasoner details and metrics branches", func(t *testing.T) {
		rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/reasoners//performance", nil)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/reasoners/bad/performance", nil)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		storage.getReasonerPerformanceMetricsFn = func(ctx context.Context, reasonerID string) (*types.ReasonerPerformanceMetrics, error) {
			return nil, errors.New("boom")
		}
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/reasoners/agent-1.plan/performance", nil)
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		storage.getReasonerPerformanceMetricsFn = func(ctx context.Context, reasonerID string) (*types.ReasonerPerformanceMetrics, error) {
			return &types.ReasonerPerformanceMetrics{
				SuccessRate:     0.5,
				TotalExecutions: 10,
			}, nil
		}
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/reasoners/agent-1.plan/performance", nil)
		require.Equal(t, http.StatusOK, rec.Code)

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/reasoners/bad", nil)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		storage.getAgentFn = func(ctx context.Context, id string) (*types.AgentNode, error) {
			if id == "agent-404" {
				return nil, errors.New("missing")
			}
			return &types.AgentNode{
				ID:            "agent-1",
				Version:       "1.0.0",
				HealthStatus:  types.HealthStatusActive,
				LastHeartbeat: time.Now().UTC(),
				Reasoners:     []types.ReasonerDefinition{{ID: "plan"}},
			}, nil
		}
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/reasoners/agent-404.plan", nil)
		require.Equal(t, http.StatusNotFound, rec.Code)

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/reasoners/agent-1.missing", nil)
		require.Equal(t, http.StatusNotFound, rec.Code)
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/reasoners/agent-1.plan", nil)
		require.Equal(t, http.StatusOK, rec.Code)
	})

	t.Run("history and templates branches", func(t *testing.T) {
		storage.getReasonerExecutionHistoryFn = func(ctx context.Context, reasonerID string, page, limit int) (*types.ReasonerExecutionHistory, error) {
			return nil, errors.New("boom")
		}
		rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/reasoners/agent-1.plan/history", nil)
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		storage.getReasonerExecutionHistoryFn = func(ctx context.Context, reasonerID string, page, limit int) (*types.ReasonerExecutionHistory, error) {
			return &types.ReasonerExecutionHistory{Total: 1, Page: page, Limit: limit}, nil
		}
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/reasoners/agent-1.plan/history?page=0&limit=999", nil)
		require.Equal(t, http.StatusOK, rec.Code)

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/reasoners/bad/templates", nil)
		require.Equal(t, http.StatusBadRequest, rec.Code)
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/reasoners/agent-1.plan/templates", nil)
		require.Equal(t, http.StatusOK, rec.Code)

		rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/reasoners/bad/templates", map[string]any{"name": "x"})
		require.Equal(t, http.StatusBadRequest, rec.Code)
		rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/reasoners/agent-1.plan/templates", map[string]any{"name": "x", "description": "y", "input": map[string]any{"ticker": "NVDA"}})
		require.Equal(t, http.StatusCreated, rec.Code)
	})
}
