package ui

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type dashboardOverrideStorage struct {
	*overrideStorage
	queryExecutionRecordsFn func(context.Context, types.ExecutionFilter) ([]*types.Execution, error)
}

func (s *dashboardOverrideStorage) QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
	if s.queryExecutionRecordsFn != nil {
		return s.queryExecutionRecordsFn(ctx, filter)
	}
	return s.overrideStorage.StorageProvider.QueryExecutionRecords(ctx, filter)
}

func TestNodesHandlerAdditionalCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := setupTestStorage(t)
	uiService := services.NewUIService(store, &MockAgentClientForUI{}, &MockAgentServiceForUI{}, nil)
	handler := NewNodesHandler(uiService)
	router := gin.New()
	router.GET("/api/ui/v1/nodes/:nodeId/status", handler.GetNodeStatusHandler)
	router.POST("/api/ui/v1/nodes/:nodeId/status/refresh", handler.RefreshNodeStatusHandler)

	rec := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(rec)
	ctx.Request = httptest.NewRequest(http.MethodGet, "/", nil)
	handler.GetNodeStatusHandler(ctx)
	require.Equal(t, http.StatusBadRequest, rec.Code)

	rec = httptest.NewRecorder()
	ctx, _ = gin.CreateTestContext(rec)
	ctx.Request = httptest.NewRequest(http.MethodPost, "/", nil)
	handler.RefreshNodeStatusHandler(ctx)
	require.Equal(t, http.StatusBadRequest, rec.Code)

	rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/nodes/node-1/status", nil)
	require.Equal(t, http.StatusInternalServerError, rec.Code)

	rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/nodes/node-1/status/refresh", nil)
	require.Equal(t, http.StatusInternalServerError, rec.Code)
}

func TestDashboardSummaryHandlerAdditionalCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("success and cache hit", func(t *testing.T) {
		store := &dashboardOverrideStorage{overrideStorage: &overrideStorage{StorageProvider: setupTestStorage(t)}}
		agentService := &mockLifecycleAgentService{}
		handler := NewDashboardHandler(store, agentService)
		router := gin.New()
		router.GET("/api/ui/v1/dashboard", handler.GetDashboardSummaryHandler)

		store.queryExecutionRecordsFn = func(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
			return []*types.Execution{{Status: string(types.ExecutionStatusSucceeded)}}, nil
		}
		store.queryAgentPackagesFn = func(ctx context.Context, filters types.PackageFilters) ([]*types.AgentPackage, error) {
			return []*types.AgentPackage{
				{ID: "pkg-open"},
				{ID: "pkg-configured", ConfigurationSchema: []byte(`{"required":{"token":{"type":"secret"}}}`)},
			}, nil
		}
		store.getAgentConfigurationFn = func(ctx context.Context, agentID, packageID string) (*types.AgentConfiguration, error) {
			if packageID == "pkg-configured" {
				return &types.AgentConfiguration{Status: types.ConfigurationStatusDraft}, nil
			}
			return nil, errors.New("missing")
		}
		store.listAgentsFn = func(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
			return []*types.AgentNode{{ID: "agent-1"}, {ID: "agent-2"}}, nil
		}
		agentService.On("GetAgentStatus", "agent-1").Return(&domain.AgentStatus{IsRunning: true}, nil).Once()
		agentService.On("GetAgentStatus", "agent-2").Return(nil, errors.New("offline")).Once()

		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/api/ui/v1/dashboard", nil))
		require.Equal(t, http.StatusOK, resp.Code)

		store.listAgentsFn = func(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
			return nil, errors.New("should not be called after cache warm")
		}
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/api/ui/v1/dashboard", nil))
		require.Equal(t, http.StatusOK, resp.Code)
	})

	t.Run("error when collector fails", func(t *testing.T) {
		store := &dashboardOverrideStorage{overrideStorage: &overrideStorage{StorageProvider: setupTestStorage(t)}}
		handler := NewDashboardHandler(store, &mockLifecycleAgentService{})
		router := gin.New()
		router.GET("/api/ui/v1/dashboard", handler.GetDashboardSummaryHandler)

		store.listAgentsFn = func(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
			return nil, errors.New("boom")
		}
		store.queryExecutionRecordsFn = func(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
			return []*types.Execution{}, nil
		}
		store.queryAgentPackagesFn = func(ctx context.Context, filters types.PackageFilters) ([]*types.AgentPackage, error) {
			return []*types.AgentPackage{}, nil
		}

		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/api/ui/v1/dashboard", nil))
		require.Equal(t, http.StatusInternalServerError, resp.Code)
	})
}
