package ui

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	storagepkg "github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type identityOverrideStorage struct {
	storagepkg.StorageProvider
	listExecutionVCsFn  func(context.Context, types.VCFilters) ([]*types.ExecutionVCInfo, error)
	countExecutionVCsFn func(context.Context, types.VCFilters) (int, error)
}

func (s *identityOverrideStorage) ListExecutionVCs(ctx context.Context, filters types.VCFilters) ([]*types.ExecutionVCInfo, error) {
	if s.listExecutionVCsFn != nil {
		return s.listExecutionVCsFn(ctx, filters)
	}
	return s.StorageProvider.ListExecutionVCs(ctx, filters)
}

func (s *identityOverrideStorage) CountExecutionVCs(ctx context.Context, filters types.VCFilters) (int, error) {
	if s.countExecutionVCsFn != nil {
		return s.countExecutionVCsFn(ctx, filters)
	}
	return s.StorageProvider.CountExecutionVCs(ctx, filters)
}

func TestIdentityHandlersAdditionalCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("search credentials parses aliases and limits", func(t *testing.T) {
		base := setupTestStorage(t)
		store := &identityOverrideStorage{StorageProvider: base}
		handler := NewIdentityHandlers(store, nil)
		router := gin.New()
		handler.RegisterRoutes(router.Group("/api/ui/v1"))

		start := time.Date(2026, 4, 8, 10, 0, 0, 0, time.UTC)
		store.listExecutionVCsFn = func(ctx context.Context, filters types.VCFilters) ([]*types.ExecutionVCInfo, error) {
			require.Equal(t, 100, filters.Limit)
			require.Equal(t, 3, filters.Offset)
			require.NotNil(t, filters.WorkflowID)
			require.Equal(t, "wf-1", *filters.WorkflowID)
			require.NotNil(t, filters.SessionID)
			require.Equal(t, "session-1", *filters.SessionID)
			require.Nil(t, filters.Status)
			require.NotNil(t, filters.IssuerDID)
			require.Equal(t, "did:issuer", *filters.IssuerDID)
			require.NotNil(t, filters.ExecutionID)
			require.Equal(t, "exec-1", *filters.ExecutionID)
			require.NotNil(t, filters.CallerDID)
			require.Equal(t, "did:caller", *filters.CallerDID)
			require.NotNil(t, filters.TargetDID)
			require.Equal(t, "did:target", *filters.TargetDID)
			require.NotNil(t, filters.AgentNodeID)
			require.Equal(t, "node-1", *filters.AgentNodeID)
			require.NotNil(t, filters.Search)
			require.Equal(t, "needle", *filters.Search)
			require.NotNil(t, filters.CreatedAfter)
			require.True(t, filters.CreatedAfter.Equal(start))
			require.Nil(t, filters.CreatedBefore)
			agentNodeID := "node-1"
			workflowName := "wf-one"
			return []*types.ExecutionVCInfo{{
				VCID:         "vc-1",
				ExecutionID:  "exec-1",
				WorkflowID:   "wf-1",
				WorkflowName: &workflowName,
				SessionID:    "session-1",
				AgentNodeID:  &agentNodeID,
				IssuerDID:    "did:issuer",
				TargetDID:    "did:target",
				CallerDID:    "did:caller",
				Status:       "completed",
				CreatedAt:    start.Add(time.Hour),
			}}, nil
		}
		store.countExecutionVCsFn = func(ctx context.Context, filters types.VCFilters) (int, error) {
			require.Equal(t, 0, filters.Limit)
			require.Equal(t, 0, filters.Offset)
			return 4, nil
		}

		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/identity/credentials/search?limit=250&offset=3&workflow_id=wf-1&session_id=session-1&status=all&issuer_did=did:issuer&execution_id=exec-1&caller_did=did:caller&target_did=did:target&agent_id=node-1&q=needle&start_time=2026-04-08T10:00:00Z&end_time=not-a-time", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		body := decodeJSONResponse[map[string]any](t, resp)
		require.Equal(t, float64(4), body["total"])
		require.Equal(t, float64(100), body["limit"])
		require.Equal(t, float64(3), body["offset"])
		require.Equal(t, false, body["has_more"])
	})

	t.Run("search credentials handles list and count errors", func(t *testing.T) {
		base := setupTestStorage(t)
		store := &identityOverrideStorage{StorageProvider: base}
		handler := NewIdentityHandlers(store, nil)
		router := gin.New()
		handler.RegisterRoutes(router.Group("/api/ui/v1"))

		store.listExecutionVCsFn = func(ctx context.Context, filters types.VCFilters) ([]*types.ExecutionVCInfo, error) {
			return nil, errors.New("boom")
		}
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/api/ui/v1/identity/credentials/search?status=pending&limit=5", nil))
		require.Equal(t, http.StatusInternalServerError, resp.Code)

		store.listExecutionVCsFn = func(ctx context.Context, filters types.VCFilters) ([]*types.ExecutionVCInfo, error) {
			require.NotNil(t, filters.Status)
			require.Equal(t, "pending", *filters.Status)
			return []*types.ExecutionVCInfo{}, nil
		}
		store.countExecutionVCsFn = func(ctx context.Context, filters types.VCFilters) (int, error) {
			return 0, errors.New("count failed")
		}
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/api/ui/v1/identity/credentials/search?status=pending&limit=5", nil))
		require.Equal(t, http.StatusInternalServerError, resp.Code)
	})

	t.Run("resolve agent did web helper", func(t *testing.T) {
		ls, registry, _, vcService, didWebService, ctx := setupDIDHandlerServices(t)
		didWeb := seedDIDHandlerData(t, ls, registry, vcService, didWebService, ctx)
		handler := NewIdentityHandlers(ls, didWebService)

		rec := httptest.NewRecorder()
		ginCtx, _ := gin.CreateTestContext(rec)
		ginCtx.Request = httptest.NewRequest(http.MethodGet, "/", nil)

		require.Equal(t, didWeb, handler.resolveAgentDIDWeb(ginCtx, "node-1"))
		require.Empty(t, handler.resolveAgentDIDWeb(ginCtx, "missing"))
	})
}

func TestDashboardHelperAdditionalCoverage(t *testing.T) {
	t.Run("agents summary counts running and storage errors", func(t *testing.T) {
		store := &overrideStorage{StorageProvider: setupTestStorage(t)}
		agentService := &mockLifecycleAgentService{}
		handler := NewDashboardHandler(store, agentService)

		store.listAgentsFn = func(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
			return []*types.AgentNode{
				{ID: "agent-1"},
				{ID: "agent-2"},
			}, nil
		}
		agentService.On("GetAgentStatus", "agent-1").Return(&domain.AgentStatus{IsRunning: true}, nil).Once()
		agentService.On("GetAgentStatus", "agent-2").Return(nil, errors.New("offline")).Once()

		summary, err := handler.getAgentsSummary(context.Background())
		require.NoError(t, err)
		require.Equal(t, 2, summary.Total)
		require.Equal(t, 1, summary.Running)

		store.listAgentsFn = func(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
			return nil, errors.New("boom")
		}
		_, err = handler.getAgentsSummary(context.Background())
		require.Error(t, err)
	})

	t.Run("packages summary covers installed branches and query errors", func(t *testing.T) {
		store := &overrideStorage{StorageProvider: setupTestStorage(t)}
		handler := NewDashboardHandler(store, &mockLifecycleAgentService{})

		store.queryAgentPackagesFn = func(ctx context.Context, filters types.PackageFilters) ([]*types.AgentPackage, error) {
			return []*types.AgentPackage{
				{ID: "pkg-open"},
				{ID: "pkg-active", ConfigurationSchema: []byte(`{"required":{"token":{"type":"secret"}}}`)},
				{ID: "pkg-missing", ConfigurationSchema: []byte(`{"required":{"token":{"type":"secret"}}}`)},
			}, nil
		}
		store.getAgentConfigurationFn = func(ctx context.Context, agentID, packageID string) (*types.AgentConfiguration, error) {
			if packageID == "pkg-active" {
				return &types.AgentConfiguration{Status: types.ConfigurationStatusActive}, nil
			}
			return nil, errors.New("missing")
		}

		summary, err := handler.getPackagesSummary(context.Background())
		require.NoError(t, err)
		require.Equal(t, 3, summary.Available)
		require.Equal(t, 2, summary.Installed)

		store.queryAgentPackagesFn = func(ctx context.Context, filters types.PackageFilters) ([]*types.AgentPackage, error) {
			return nil, errors.New("boom")
		}
		_, err = handler.getPackagesSummary(context.Background())
		require.Error(t, err)
	})
}
