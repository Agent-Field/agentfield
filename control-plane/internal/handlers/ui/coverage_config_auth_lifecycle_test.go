package ui

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	storagepkg "github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

type mockLifecycleAgentService struct {
	mock.Mock
}

func (m *mockLifecycleAgentService) RunAgent(name string, options domain.RunOptions) (*domain.RunningAgent, error) {
	args := m.Called(name, options)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*domain.RunningAgent), args.Error(1)
}

func (m *mockLifecycleAgentService) StopAgent(name string) error {
	args := m.Called(name)
	return args.Error(0)
}

func (m *mockLifecycleAgentService) GetAgentStatus(name string) (*domain.AgentStatus, error) {
	args := m.Called(name)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*domain.AgentStatus), args.Error(1)
}

func (m *mockLifecycleAgentService) ListRunningAgents() ([]domain.RunningAgent, error) {
	args := m.Called()
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]domain.RunningAgent), args.Error(1)
}

type stubExecutionRecordStore struct {
	records []*types.Execution
	err     error
}

type overrideStorage struct {
	storagepkg.StorageProvider
	getAgentPackageFn              func(context.Context, string) (*types.AgentPackage, error)
	getAgentConfigurationFn        func(context.Context, string, string) (*types.AgentConfiguration, error)
	validateAgentConfigurationFn   func(context.Context, string, string, map[string]interface{}) (*types.ConfigurationValidationResult, error)
	storeAgentConfigurationFn      func(context.Context, *types.AgentConfiguration) error
	updateAgentConfigurationFn     func(context.Context, *types.AgentConfiguration) error
	getAgentFn                     func(context.Context, string) (*types.AgentNode, error)
	listAgentsFn                   func(context.Context, types.AgentFilters) ([]*types.AgentNode, error)
	getReasonerPerformanceMetricsFn func(context.Context, string) (*types.ReasonerPerformanceMetrics, error)
	getReasonerExecutionHistoryFn  func(context.Context, string, int, int) (*types.ReasonerExecutionHistory, error)
	queryAgentPackagesFn           func(context.Context, types.PackageFilters) ([]*types.AgentPackage, error)
	getConfigFn                    func(context.Context, string) (*storagepkg.ConfigEntry, error)
	setConfigFn                    func(context.Context, string, string, string) error
	hasExecutionWebhookFn          func(context.Context, string) (bool, error)
}

func (s *stubExecutionRecordStore) QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
	if s.err != nil {
		return nil, s.err
	}
	return s.records, nil
}

func (s *stubExecutionRecordStore) GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error) {
	if s.err != nil {
		return nil, s.err
	}
	for _, record := range s.records {
		if record.ExecutionID == executionID {
			return record, nil
		}
	}
	return nil, errors.New("not found")
}

func (s *overrideStorage) GetAgentPackage(ctx context.Context, packageID string) (*types.AgentPackage, error) {
	if s.getAgentPackageFn != nil {
		return s.getAgentPackageFn(ctx, packageID)
	}
	return s.StorageProvider.GetAgentPackage(ctx, packageID)
}

func (s *overrideStorage) GetAgentConfiguration(ctx context.Context, agentID, packageID string) (*types.AgentConfiguration, error) {
	if s.getAgentConfigurationFn != nil {
		return s.getAgentConfigurationFn(ctx, agentID, packageID)
	}
	return s.StorageProvider.GetAgentConfiguration(ctx, agentID, packageID)
}

func (s *overrideStorage) ValidateAgentConfiguration(ctx context.Context, agentID, packageID string, config map[string]interface{}) (*types.ConfigurationValidationResult, error) {
	if s.validateAgentConfigurationFn != nil {
		return s.validateAgentConfigurationFn(ctx, agentID, packageID, config)
	}
	return s.StorageProvider.ValidateAgentConfiguration(ctx, agentID, packageID, config)
}

func (s *overrideStorage) StoreAgentConfiguration(ctx context.Context, config *types.AgentConfiguration) error {
	if s.storeAgentConfigurationFn != nil {
		return s.storeAgentConfigurationFn(ctx, config)
	}
	return s.StorageProvider.StoreAgentConfiguration(ctx, config)
}

func (s *overrideStorage) UpdateAgentConfiguration(ctx context.Context, config *types.AgentConfiguration) error {
	if s.updateAgentConfigurationFn != nil {
		return s.updateAgentConfigurationFn(ctx, config)
	}
	return s.StorageProvider.UpdateAgentConfiguration(ctx, config)
}

func (s *overrideStorage) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	if s.getAgentFn != nil {
		return s.getAgentFn(ctx, id)
	}
	return s.StorageProvider.GetAgent(ctx, id)
}

func (s *overrideStorage) ListAgents(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
	if s.listAgentsFn != nil {
		return s.listAgentsFn(ctx, filters)
	}
	return s.StorageProvider.ListAgents(ctx, filters)
}

func (s *overrideStorage) GetReasonerPerformanceMetrics(ctx context.Context, reasonerID string) (*types.ReasonerPerformanceMetrics, error) {
	if s.getReasonerPerformanceMetricsFn != nil {
		return s.getReasonerPerformanceMetricsFn(ctx, reasonerID)
	}
	return s.StorageProvider.GetReasonerPerformanceMetrics(ctx, reasonerID)
}

func (s *overrideStorage) GetReasonerExecutionHistory(ctx context.Context, reasonerID string, page, limit int) (*types.ReasonerExecutionHistory, error) {
	if s.getReasonerExecutionHistoryFn != nil {
		return s.getReasonerExecutionHistoryFn(ctx, reasonerID, page, limit)
	}
	return s.StorageProvider.GetReasonerExecutionHistory(ctx, reasonerID, page, limit)
}

func (s *overrideStorage) QueryAgentPackages(ctx context.Context, filters types.PackageFilters) ([]*types.AgentPackage, error) {
	if s.queryAgentPackagesFn != nil {
		return s.queryAgentPackagesFn(ctx, filters)
	}
	return s.StorageProvider.QueryAgentPackages(ctx, filters)
}

func (s *overrideStorage) GetConfig(ctx context.Context, key string) (*storagepkg.ConfigEntry, error) {
	if s.getConfigFn != nil {
		return s.getConfigFn(ctx, key)
	}
	return s.StorageProvider.GetConfig(ctx, key)
}

func (s *overrideStorage) SetConfig(ctx context.Context, key, value, source string) error {
	if s.setConfigFn != nil {
		return s.setConfigFn(ctx, key, value, source)
	}
	return s.StorageProvider.SetConfig(ctx, key, value, source)
}

func (s *overrideStorage) HasExecutionWebhook(ctx context.Context, executionID string) (bool, error) {
	if s.hasExecutionWebhookFn != nil {
		return s.hasExecutionWebhookFn(ctx, executionID)
	}
	return s.StorageProvider.HasExecutionWebhook(ctx, executionID)
}

func newConfigRouter(storage storagepkg.StorageProvider) *gin.Engine {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	handler := NewConfigHandler(storage)
	group := router.Group("/api/ui/v1/agents")
	group.GET("/:agentId/config/schema", handler.GetConfigSchemaHandler)
	group.GET("/:agentId/config", handler.GetConfigHandler)
	group.POST("/:agentId/config", handler.SetConfigHandler)
	return router
}

func TestConfigHandlerCoverage(t *testing.T) {
	t.Run("schema handler covers success and failures", func(t *testing.T) {
		router := newConfigRouter(setupTestStorage(t))

		rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents//config/schema?packageId=pkg", nil)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/agent-1/config/schema", nil)
		require.Equal(t, http.StatusBadRequest, rec.Code)
	})

	t.Run("schema handler package lookup and parse failures", func(t *testing.T) {
		storage := &overrideStorage{StorageProvider: setupTestStorage(t)}
		router := newConfigRouter(storage)
		storage.getAgentPackageFn = func(ctx context.Context, packageID string) (*types.AgentPackage, error) {
			return nil, errors.New("missing")
		}
		rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/agent-1/config/schema?packageId=missing", nil)
		require.Equal(t, http.StatusNotFound, rec.Code)

		storage = &overrideStorage{StorageProvider: setupTestStorage(t)}
		router = newConfigRouter(storage)
		storage.getAgentPackageFn = func(ctx context.Context, packageID string) (*types.AgentPackage, error) {
			return &types.AgentPackage{
			ID:                  "pkg",
			Name:                "pkg",
			Version:             "1.0.0",
			ConfigurationSchema: json.RawMessage(`{invalid`),
		}, nil
		}
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/agent-1/config/schema?packageId=pkg", nil)
		require.Equal(t, http.StatusInternalServerError, rec.Code)
	})

	t.Run("schema handler success", func(t *testing.T) {
		storage := &overrideStorage{StorageProvider: setupTestStorage(t)}
		router := newConfigRouter(storage)
		desc := "package description"
		storage.getAgentPackageFn = func(ctx context.Context, packageID string) (*types.AgentPackage, error) {
			return &types.AgentPackage{
			ID:                  "pkg",
			Name:                "Package",
			Version:             "1.0.0",
			Description:         &desc,
			ConfigurationSchema: json.RawMessage(`{"required":{"token":{"type":"secret"}}}`),
		}, nil
		}

		rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/agent-1/config/schema?packageId=pkg", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		var body map[string]any
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
		require.Equal(t, "agent-1", body["agent_id"])
		require.Equal(t, "pkg", body["package_id"])
	})

	t.Run("get config and set config cover branches", func(t *testing.T) {
		now := time.Now().UTC()
		storage := &overrideStorage{StorageProvider: setupTestStorage(t)}
		router := newConfigRouter(storage)

		rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/agent-1/config", nil)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		storage.getAgentConfigurationFn = func(ctx context.Context, agentID, packageID string) (*types.AgentConfiguration, error) {
			if agentID == "agent-1" && packageID == "pkg" {
				return nil, errors.New("not found")
			}
			return storage.StorageProvider.GetAgentConfiguration(ctx, agentID, packageID)
		}
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/agent-1/config?packageId=pkg", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		var missing map[string]any
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &missing))
		require.Equal(t, "draft", missing["status"])

		storage.getAgentConfigurationFn = func(ctx context.Context, agentID, packageID string) (*types.AgentConfiguration, error) {
			return &types.AgentConfiguration{
			AgentID:         "agent-1",
			PackageID:       "pkg",
			Configuration:   map[string]interface{}{"token": "abc"},
			EncryptedFields: []string{"token"},
			Status:          types.ConfigurationStatusActive,
			Version:         2,
			CreatedAt:       now,
			UpdatedAt:       now,
		}, nil
		}
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/agent-1/config?packageId=pkg", nil)
		require.Equal(t, http.StatusOK, rec.Code)

		rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/agents/agent-1/config?packageId=pkg", map[string]any{"configuration": "bad"})
		require.Equal(t, http.StatusBadRequest, rec.Code)

		storage.validateAgentConfigurationFn = func(ctx context.Context, agentID, packageID string, config map[string]interface{}) (*types.ConfigurationValidationResult, error) {
			return nil, errors.New("boom")
		}
		rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/agents/agent-1/config?packageId=pkg", map[string]any{
			"configuration": map[string]any{"token": "abc"},
		})
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		storage.validateAgentConfigurationFn = func(ctx context.Context, agentID, packageID string, config map[string]interface{}) (*types.ConfigurationValidationResult, error) {
			return &types.ConfigurationValidationResult{Valid: false, Errors: []string{"token required"}}, nil
		}
		rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/agents/agent-1/config?packageId=pkg", map[string]any{
			"configuration": map[string]any{"token": "abc"},
		})
		require.Equal(t, http.StatusBadRequest, rec.Code)

		status := "active"
		storage.validateAgentConfigurationFn = func(ctx context.Context, agentID, packageID string, config map[string]interface{}) (*types.ConfigurationValidationResult, error) {
			return &types.ConfigurationValidationResult{Valid: true}, nil
		}
		storage.getAgentConfigurationFn = func(ctx context.Context, agentID, packageID string) (*types.AgentConfiguration, error) {
			return nil, errors.New("missing")
		}
		storage.storeAgentConfigurationFn = func(ctx context.Context, config *types.AgentConfiguration) error {
			require.Equal(t, 1, config.Version)
			return nil
		}
		rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/agents/agent-1/config?packageId=pkg", map[string]any{
			"configuration": map[string]any{"token": "abc"},
			"status":        status,
		})
		require.Equal(t, http.StatusCreated, rec.Code)

		existing := &types.AgentConfiguration{
			AgentID:       "agent-1",
			PackageID:     "pkg",
			Configuration: map[string]interface{}{"token": "old"},
			Status:        types.ConfigurationStatusDraft,
			Version:       5,
		}
		storage.validateAgentConfigurationFn = func(ctx context.Context, agentID, packageID string, config map[string]interface{}) (*types.ConfigurationValidationResult, error) {
			return &types.ConfigurationValidationResult{Valid: true}, nil
		}
		storage.getAgentConfigurationFn = func(ctx context.Context, agentID, packageID string) (*types.AgentConfiguration, error) {
			return existing, nil
		}
		storage.updateAgentConfigurationFn = func(ctx context.Context, config *types.AgentConfiguration) error {
			require.Equal(t, 6, config.Version)
			return nil
		}
		rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/agents/agent-1/config?packageId=pkg", map[string]any{
			"configuration": map[string]any{"token": "new"},
		})
		require.Equal(t, http.StatusOK, rec.Code)
	})
}

func TestAuthorizationHandlerCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("list agents failure", func(t *testing.T) {
		storage := &overrideStorage{StorageProvider: setupTestStorage(t)}
		handler := NewAuthorizationHandler(storage)
		router := gin.New()
		router.GET("/api/ui/v1/authorization/agents", handler.GetAgentsWithTagsHandler)
		storage.listAgentsFn = func(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
			return nil, errors.New("boom")
		}

		rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/authorization/agents", nil)
		require.Equal(t, http.StatusInternalServerError, rec.Code)
	})

	t.Run("list agents success normalizes nil tags", func(t *testing.T) {
		storage := &overrideStorage{StorageProvider: setupTestStorage(t)}
		handler := NewAuthorizationHandler(storage)
		router := gin.New()
		router.GET("/api/ui/v1/authorization/agents", handler.GetAgentsWithTagsHandler)

		now := time.Date(2026, 4, 8, 12, 0, 0, 0, time.UTC)
		storage.listAgentsFn = func(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
			return []*types.AgentNode{
			{ID: "agent-a", LifecycleStatus: types.AgentStatusReady, RegisteredAt: now},
			{ID: "agent-b", ProposedTags: []string{"finance"}, ApprovedTags: []string{"approved"}, LifecycleStatus: types.AgentStatusOffline, RegisteredAt: now},
		}, nil
		}

		rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/authorization/agents", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		var body struct {
			Agents []AgentTagSummaryResponse `json:"agents"`
			Total  int                      `json:"total"`
		}
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
		require.Len(t, body.Agents, 2)
		require.Equal(t, 2, body.Total)
		require.Empty(t, body.Agents[0].ProposedTags)
		require.Empty(t, body.Agents[0].ApprovedTags)
		require.Equal(t, "2026-04-08T12:00:00Z", body.Agents[0].RegisteredAt)
	})
}

func TestLifecycleHandlerCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := &overrideStorage{StorageProvider: setupTestStorage(t)}
	agentSvc := &mockLifecycleAgentService{}
	handler := NewLifecycleHandler(storage, agentSvc)
	router := gin.New()
	router.POST("/api/ui/v1/agents/:agentId/start", handler.StartAgentHandler)
	router.POST("/api/ui/v1/agents/:agentId/stop", handler.StopAgentHandler)
	router.GET("/api/ui/v1/agents/:agentId/status", handler.GetAgentStatusHandler)
	router.GET("/api/ui/v1/agents/running", handler.ListRunningAgentsHandler)
	router.POST("/api/ui/v1/agents/:agentId/reconcile", handler.ReconcileAgentHandler)

	t.Run("helpers", func(t *testing.T) {
		storage.getAgentFn = func(ctx context.Context, id string) (*types.AgentNode, error) {
			if id == "agent-base" {
				return &types.AgentNode{ID: "agent-base", BaseURL: "http://remote:9000"}, nil
			}
			return nil, errors.New("missing")
		}
		require.Equal(t, "http://remote:9000", handler.getAgentBaseURL(context.Background(), "agent-base", 7777))
		require.Equal(t, "http://localhost:7777", handler.getAgentBaseURL(context.Background(), "missing-base", 7777))
		require.Equal(t, "configured", getConfigurationStatus(&types.AgentConfiguration{Status: types.ConfigurationStatusActive}))
		require.Equal(t, "partially_configured", getConfigurationStatus(&types.AgentConfiguration{Status: types.ConfigurationStatusDraft}))
		require.Equal(t, "not_configured", getConfigurationStatus(nil))
		require.Equal(t, "running", getAgentLifecycleStatus(&domain.AgentStatus{IsRunning: true}, "configured", true))
		require.Equal(t, "not_configured", getAgentLifecycleStatus(&domain.AgentStatus{IsRunning: false}, "draft", true))
		require.Equal(t, "stopped", getAgentLifecycleStatus(&domain.AgentStatus{IsRunning: false}, "configured", false))
	})

	t.Run("start agent success and not found", func(t *testing.T) {
		startedAt := time.Now().UTC()
		storage.getAgentFn = func(ctx context.Context, id string) (*types.AgentNode, error) {
			if id == "agent-start" {
				return &types.AgentNode{ID: "agent-start", BaseURL: "http://registered:8080"}, nil
			}
			return nil, errors.New("missing")
		}
		agentSvc.On("RunAgent", "agent-start", domain.RunOptions{Port: 1234, Detach: false}).Return(&domain.RunningAgent{
			Name:      "agent-start",
			PID:       99,
			Port:      1234,
			StartedAt: startedAt,
			LogFile:   "/tmp/agent.log",
		}, nil).Once()
		rec := performJSONRequest(router, http.MethodPost, "/api/ui/v1/agents/agent-start/start", map[string]any{
			"port":   1234,
			"detach": false,
		})
		require.Equal(t, http.StatusOK, rec.Code)

		agentSvc.On("RunAgent", "missing-agent", domain.RunOptions{Port: 0, Detach: true}).Return(nil, errors.New("agent not found")).Once()
		req := httptest.NewRequest(http.MethodPost, "/api/ui/v1/agents/missing-agent/start", bytes.NewBufferString("{bad"))
		req.Header.Set("Content-Type", "application/json")
		rec = httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusNotFound, rec.Code)
	})

	t.Run("stop agent branches", func(t *testing.T) {
		agentSvc.On("GetAgentStatus", "ghost").Return(nil, errors.New("missing")).Once()
		rec := performJSONRequest(router, http.MethodPost, "/api/ui/v1/agents/ghost/stop", nil)
		require.Equal(t, http.StatusNotFound, rec.Code)

		agentSvc.On("GetAgentStatus", "idle").Return(&domain.AgentStatus{Name: "idle", IsRunning: false}, nil).Once()
		rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/agents/idle/stop", nil)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		agentSvc.On("GetAgentStatus", "busy").Return(&domain.AgentStatus{Name: "busy", IsRunning: true}, nil).Once()
		agentSvc.On("StopAgent", "busy").Return(errors.New("fail")).Once()
		rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/agents/busy/stop", nil)
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		agentSvc.On("GetAgentStatus", "ok").Return(&domain.AgentStatus{Name: "ok", IsRunning: true}, nil).Once()
		agentSvc.On("StopAgent", "ok").Return(nil).Once()
		rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/agents/ok/stop", nil)
		require.Equal(t, http.StatusOK, rec.Code)
	})

	t.Run("status branches", func(t *testing.T) {
		storage.getAgentPackageFn = func(ctx context.Context, packageID string) (*types.AgentPackage, error) {
			if packageID == "missing" {
				return nil, errors.New("missing")
			}
			if packageID == "pkg-agent" {
				return &types.AgentPackage{
					ID:                  "pkg-agent",
					Name:                "Pkg Agent",
					ConfigurationSchema: json.RawMessage(`{"required":{"token":{"type":"secret"}}}`),
				}, nil
			}
			if packageID == "running-agent" {
				return &types.AgentPackage{
					ID:                  "running-agent",
					Name:                "Running Agent",
					ConfigurationSchema: json.RawMessage(`{"required":{"token":{"type":"secret"}}}`),
				}, nil
			}
			if packageID == "agent-list" {
				desc := "agent"
				author := "author"
				return &types.AgentPackage{
					ID:          "agent-list",
					Name:        "agent-list",
					Version:     "1.0.0",
					Description: &desc,
					Author:      &author,
				}, nil
			}
			return nil, errors.New("missing")
		}
		rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/missing/status", nil)
		require.Equal(t, http.StatusNotFound, rec.Code)

		agentSvc.On("GetAgentStatus", "pkg-agent").Return(nil, errors.New("not installed")).Once()
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/pkg-agent/status", nil)
		require.Equal(t, http.StatusOK, rec.Code)

		agentSvc.On("GetAgentStatus", "running-agent").Return(&domain.AgentStatus{
			Name:      "running-agent",
			IsRunning: true,
			PID:       100,
			Port:      7070,
			Uptime:    "1m",
			LastSeen:  time.Now().UTC(),
		}, nil).Once()
		storage.getAgentConfigurationFn = func(ctx context.Context, agentID, packageID string) (*types.AgentConfiguration, error) {
			return nil, errors.New("missing")
		}
		storage.getAgentFn = func(ctx context.Context, id string) (*types.AgentNode, error) {
			if id == "running-agent" {
				return &types.AgentNode{ID: "running-agent", BaseURL: "http://svc:7070"}, nil
			}
			return nil, errors.New("missing")
		}
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/running-agent/status", nil)
		require.Equal(t, http.StatusOK, rec.Code)
	})

	t.Run("running agents and reconcile", func(t *testing.T) {
		agentSvc.On("ListRunningAgents").Return(nil, errors.New("boom")).Once()
		rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/running", nil)
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		agentSvc.On("ListRunningAgents").Return([]domain.RunningAgent{{
			Name:      "agent-list",
			PID:       55,
			Port:      9090,
			Status:    "running",
			StartedAt: time.Now().UTC(),
			LogFile:   "/tmp/log",
		}}, nil).Once()
		storage.getAgentFn = func(ctx context.Context, id string) (*types.AgentNode, error) {
			return nil, errors.New("missing")
		}
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/running", nil)
		require.Equal(t, http.StatusOK, rec.Code)

		agentSvc.On("GetAgentStatus", "rec-missing").Return(nil, errors.New("not found")).Once()
		rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/agents/rec-missing/reconcile", nil)
		require.Equal(t, http.StatusNotFound, rec.Code)

		agentSvc.On("GetAgentStatus", "rec-ok").Return(&domain.AgentStatus{
			Name:      "rec-ok",
			IsRunning: true,
			PID:       12,
			Port:      8081,
			Uptime:    "2m",
			LastSeen:  time.Now().UTC(),
		}, nil).Once()
		rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/agents/rec-ok/reconcile", nil)
		require.Equal(t, http.StatusOK, rec.Code)
	})
}

func TestRecentActivityCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("cache and relative time helpers", func(t *testing.T) {
		cache := NewRecentActivityCache()
		require.Equal(t, 15*time.Second, cache.ttl)
		_, found := cache.Get()
		require.False(t, found)
		response := &RecentActivityResponse{}
		cache.Set(response)
		got, found := cache.Get()
		require.True(t, found)
		require.Same(t, response, got)
		cache.timestamp = time.Now().Add(-16 * time.Second)
		_, found = cache.Get()
		require.False(t, found)

		require.Equal(t, "just now", formatRelativeTime(time.Now()))
		require.Equal(t, "30s ago", formatRelativeTime(time.Now().Add(-30*time.Second)))
		require.Equal(t, "5m ago", formatRelativeTime(time.Now().Add(-5*time.Minute)))
		require.Equal(t, "2h ago", formatRelativeTime(time.Now().Add(-2*time.Hour)))
		require.Equal(t, "3d ago", formatRelativeTime(time.Now().Add(-72*time.Hour)))
	})

	t.Run("recent activity handler uses cache and store", func(t *testing.T) {
		storage := &overrideStorage{StorageProvider: setupTestStorage(t)}
		handler := NewRecentActivityHandler(storage)
		handler.store = &stubExecutionRecordStore{
			records: []*types.Execution{{
				ExecutionID: "exec-1",
				AgentNodeID: "agent-1",
				ReasonerID:  "agent-1.plan",
				Status:      "completed",
				StartedAt:   time.Now().Add(-2 * time.Minute),
				DurationMS:  func() *int64 { v := int64(1500); return &v }(),
			}},
		}

		storage.getAgentFn = func(ctx context.Context, id string) (*types.AgentNode, error) {
			return nil, errors.New("missing")
		}
		rec := httptest.NewRecorder()
		ctx, router := gin.CreateTestContext(rec)
		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/recent", nil)
		ctx.Request = req
		handler.GetRecentActivityHandler(ctx)
		require.Equal(t, http.StatusOK, rec.Code)

		rec2 := httptest.NewRecorder()
		ctx2, _ := gin.CreateTestContext(rec2)
		ctx2.Request = httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/recent", nil)
		_ = router
		handler.GetRecentActivityHandler(ctx2)
		require.Equal(t, http.StatusOK, rec2.Code)
	})

	t.Run("recent executions error path", func(t *testing.T) {
		handler := NewRecentActivityHandler(setupTestStorage(t))
		handler.store = &stubExecutionRecordStore{err: errors.New("boom")}
		_, err := handler.getRecentExecutions(context.Background())
		require.Error(t, err)
	})
}
