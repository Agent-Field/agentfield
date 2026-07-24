package ui

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	storagepkg "github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

type timelineStore struct {
	queryFn func(context.Context, types.ExecutionFilter) ([]*types.Execution, error)
	getFn   func(context.Context, string) (*types.Execution, error)
	calls   int
}

func (s *timelineStore) QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
	s.calls++
	if s.queryFn != nil {
		return s.queryFn(ctx, filter)
	}
	return nil, nil
}

func (s *timelineStore) GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error) {
	if s.getFn != nil {
		return s.getFn(ctx, executionID)
	}
	return nil, errors.New("not found")
}

func newPackageRouter(storage storagepkg.StorageProvider) *gin.Engine {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	handler := NewPackageHandler(storage)
	group := router.Group("/api/ui/v1/agents/packages")
	group.GET("", handler.ListPackagesHandler)
	group.GET("/:packageId/details", handler.GetPackageDetailsHandler)
	return router
}

func newNodeLogSettingsRouter(handler *NodeLogSettingsHandler) *gin.Engine {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.GET("/api/ui/v1/settings/node-log-proxy", handler.GetNodeLogProxySettingsHandler)
	router.PUT("/api/ui/v1/settings/node-log-proxy", handler.PutNodeLogProxySettingsHandler)
	return router
}

func TestErrorResponseWrappers(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name    string
		call    func(*gin.Context)
		status  int
		message string
	}{
		{
			name:    "respond error",
			call:    func(c *gin.Context) { RespondError(c, http.StatusTeapot, "teapot") },
			status:  http.StatusTeapot,
			message: "teapot",
		},
		{
			name:    "bad request",
			call:    func(c *gin.Context) { RespondBadRequest(c, "bad request") },
			status:  http.StatusBadRequest,
			message: "bad request",
		},
		{
			name:    "not found",
			call:    func(c *gin.Context) { RespondNotFound(c, "missing") },
			status:  http.StatusNotFound,
			message: "missing",
		},
		{
			name:    "internal error",
			call:    func(c *gin.Context) { RespondInternalError(c, "boom") },
			status:  http.StatusInternalServerError,
			message: "boom",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			ctx, _ := gin.CreateTestContext(rec)
			ctx.Request = httptest.NewRequest(http.MethodGet, "/", nil)
			tc.call(ctx)

			require.Equal(t, tc.status, rec.Code)
			var body ErrorResponse
			require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
			require.Equal(t, tc.message, body.Error)
		})
	}
}

func TestPackageHandlerCoverageNonIntegration(t *testing.T) {
	t.Run("list packages covers filters and configuration states", func(t *testing.T) {
		storage := &overrideStorage{StorageProvider: setupTestStorage(t)}
		router := newPackageRouter(storage)
		description := "Searchable package"
		author := "Author"

		storage.queryAgentPackagesFn = func(ctx context.Context, filters types.PackageFilters) ([]*types.AgentPackage, error) {
			return []*types.AgentPackage{
				{
					ID:                  "pkg-active",
					Name:                "Package Active",
					Version:             "1.0.0",
					Description:         &description,
					Author:              &author,
					InstallPath:         "/tmp/pkg-active",
					ConfigurationSchema: json.RawMessage(`{"required":{"token":{"type":"secret"}}}`),
				},
				{
					ID:          "pkg-open",
					Name:        "Open Package",
					Version:     "2.0.0",
					InstallPath: "/tmp/pkg-open",
				},
				{
					ID:                  "pkg-draft",
					Name:                "Package Draft",
					Version:             "3.0.0",
					InstallPath:         "/tmp/pkg-draft",
					ConfigurationSchema: json.RawMessage(`{"required":{"token":{"type":"secret"}}}`),
				},
			}, nil
		}
		storage.getAgentConfigurationFn = func(ctx context.Context, agentID, packageID string) (*types.AgentConfiguration, error) {
			switch packageID {
			case "pkg-active":
				return &types.AgentConfiguration{
					AgentID:       agentID,
					PackageID:     packageID,
					Status:        types.ConfigurationStatusActive,
					Configuration: map[string]interface{}{"token": "set"},
				}, nil
			case "pkg-draft":
				return &types.AgentConfiguration{
					AgentID:       agentID,
					PackageID:     packageID,
					Status:        types.ConfigurationStatusDraft,
					Configuration: map[string]interface{}{"token": "draft"},
				}, nil
			default:
				return nil, errors.New("missing config")
			}
		}

		rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/packages?status=configured&search=searchable", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		body := decodeJSONResponse[PackageListResponse](t, rec)
		require.Len(t, body.Packages, 1)
		require.Equal(t, "pkg-active", body.Packages[0].ID)
		require.True(t, body.Packages[0].ConfigurationRequired)
		require.True(t, body.Packages[0].ConfigurationComplete)
		require.Equal(t, 1, body.Total)

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/packages?search=open", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		body = decodeJSONResponse[PackageListResponse](t, rec)
		require.Len(t, body.Packages, 1)
		require.Equal(t, "pkg-open", body.Packages[0].ID)
		require.False(t, body.Packages[0].ConfigurationRequired)
		require.True(t, body.Packages[0].ConfigurationComplete)

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/packages?status=not_configured", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		body = decodeJSONResponse[PackageListResponse](t, rec)
		require.Len(t, body.Packages, 0)
	})

	t.Run("list packages handles storage error", func(t *testing.T) {
		storage := &overrideStorage{StorageProvider: setupTestStorage(t)}
		storage.queryAgentPackagesFn = func(ctx context.Context, filters types.PackageFilters) ([]*types.AgentPackage, error) {
			return nil, errors.New("boom")
		}
		rec := performJSONRequest(newPackageRouter(storage), http.MethodGet, "/api/ui/v1/agents/packages", nil)
		require.Equal(t, http.StatusInternalServerError, rec.Code)
	})

	t.Run("get package details covers direct validation and branches", func(t *testing.T) {
		handler := NewPackageHandler(nil)
		rec := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodGet, "/", nil)
		handler.GetPackageDetailsHandler(ctx)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		storage := &overrideStorage{StorageProvider: setupTestStorage(t)}
		router := newPackageRouter(storage)
		storage.getAgentPackageFn = func(ctx context.Context, packageID string) (*types.AgentPackage, error) {
			switch packageID {
			case "missing":
				return nil, errors.New("missing")
			case "bad-schema":
				return &types.AgentPackage{
					ID:                  "bad-schema",
					Name:                "Bad Schema",
					Version:             "1.0.0",
					ConfigurationSchema: json.RawMessage(`{invalid`),
				}, nil
			case "configured":
				description := "Configured package"
				author := "Configured author"
				return &types.AgentPackage{
					ID:                  "configured",
					Name:                "Configured",
					Version:             "2.0.0",
					Description:         &description,
					Author:              &author,
					InstallPath:         "/tmp/configured",
					ConfigurationSchema: json.RawMessage(`{"required":{"token":{"type":"secret"}}}`),
				}, nil
			case "no-config":
				return &types.AgentPackage{
					ID:          "no-config",
					Name:        "No Config",
					Version:     "3.0.0",
					InstallPath: "/tmp/no-config",
				}, nil
			default:
				return nil, errors.New("unexpected")
			}
		}
		storage.getAgentConfigurationFn = func(ctx context.Context, agentID, packageID string) (*types.AgentConfiguration, error) {
			if packageID == "configured" {
				return &types.AgentConfiguration{
					AgentID:       agentID,
					PackageID:     packageID,
					Status:        types.ConfigurationStatusActive,
					Configuration: map[string]interface{}{"token": "abc"},
				}, nil
			}
			return nil, errors.New("missing config")
		}

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/packages/missing/details", nil)
		require.Equal(t, http.StatusNotFound, rec.Code)

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/packages/bad-schema/details", nil)
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/packages/configured/details", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		configured := decodeJSONResponse[PackageDetailsResponse](t, rec)
		require.Equal(t, "configured", configured.ID)
		require.Equal(t, "Configured package", configured.Description)
		require.True(t, configured.Configuration.Required)
		require.True(t, configured.Configuration.Complete)
		require.Equal(t, map[string]interface{}{"token": "abc"}, configured.Configuration.Current)
		require.Equal(t, "configured", configured.Status)

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/packages/no-config/details", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		openPkg := decodeJSONResponse[PackageDetailsResponse](t, rec)
		require.False(t, openPkg.Configuration.Required)
		require.True(t, openPkg.Configuration.Complete)
		require.Empty(t, openPkg.Configuration.Current)
		require.Equal(t, "configured", openPkg.Status)
	})

	t.Run("package helper methods cover remaining branches", func(t *testing.T) {
		handler := &PackageHandler{storage: &overrideStorage{
			StorageProvider: setupTestStorage(t),
			getAgentConfigurationFn: func(ctx context.Context, agentID, packageID string) (*types.AgentConfiguration, error) {
				switch packageID {
				case "draft":
					return &types.AgentConfiguration{Status: types.ConfigurationStatusDraft}, nil
				case "active":
					return &types.AgentConfiguration{Status: types.ConfigurationStatusActive}, nil
				case "other":
					return &types.AgentConfiguration{Status: "invalid"}, nil
				default:
					return nil, errors.New("missing")
				}
			},
		}}

		require.Equal(t, "not_configured", handler.determinePackageStatus(context.Background(), &types.AgentPackage{
			ID:                  "missing",
			ConfigurationSchema: json.RawMessage(`{"required":{"x":{}}}`),
		}))
		require.Equal(t, "configured", handler.determinePackageStatus(context.Background(), &types.AgentPackage{
			ID:                  "draft",
			ConfigurationSchema: json.RawMessage(`{"required":{"x":{}}}`),
		}))
		require.Equal(t, "configured", handler.determinePackageStatus(context.Background(), &types.AgentPackage{
			ID:                  "active",
			ConfigurationSchema: json.RawMessage(`{"required":{"x":{}}}`),
		}))
		require.Equal(t, "not_configured", handler.determinePackageStatus(context.Background(), &types.AgentPackage{
			ID:                  "other",
			ConfigurationSchema: json.RawMessage(`{"required":{"x":{}}}`),
		}))
		require.Equal(t, "configured", handler.determinePackageStatus(context.Background(), &types.AgentPackage{
			ID: "none",
		}))
		require.True(t, handler.matchesSearch(&types.AgentPackage{
			ID:   "pkg-1",
			Name: "Example",
		}, "example"))
		require.Equal(t, "", handler.safeStringValue(nil))
	})
}

func TestExecutionTimelineCoverage(t *testing.T) {
	t.Run("cache lifecycle and constructor", func(t *testing.T) {
		cache := NewExecutionTimelineCache()
		require.Equal(t, 5*time.Minute, cache.ttl)
		_, found := cache.Get()
		require.False(t, found)

		response := &ExecutionTimelineResponse{CacheTimestamp: "now"}
		cache.Set(response)
		got, found := cache.Get()
		require.True(t, found)
		require.Same(t, response, got)

		cache.timestamp = time.Now().Add(-cache.ttl - time.Second)
		_, found = cache.Get()
		require.False(t, found)

		handler := NewExecutionTimelineHandler(nil)
		require.NotNil(t, handler.cache)
	})

	t.Run("generate timeline handles query error", func(t *testing.T) {
		handler := &ExecutionTimelineHandler{
			store: &timelineStore{
				queryFn: func(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
					return nil, errors.New("boom")
				},
			},
			cache: NewExecutionTimelineCache(),
		}

		_, _, err := handler.generateTimelineData(context.Background())
		require.Error(t, err)
		require.Contains(t, err.Error(), "failed to query executions")
	})

	t.Run("generate timeline and handler cache success", func(t *testing.T) {
		now := time.Now().Truncate(time.Hour)
		durationSucceeded := int64(300)
		durationFailed := int64(900)
		durationRunning := int64(600)

		store := &timelineStore{
			queryFn: func(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
				require.NotNil(t, filter.StartTime)
				require.NotNil(t, filter.EndTime)
				require.Equal(t, 50000, filter.Limit)
				require.Equal(t, "started_at", filter.SortBy)
				require.False(t, filter.SortDescending)
				require.True(t, filter.ExcludePayloads)

				return []*types.Execution{
					{
						ExecutionID: "exec-success",
						StartedAt:   now.Add(-2 * time.Hour).Add(15 * time.Minute),
						Status:      string(types.ExecutionStatusSucceeded),
						DurationMS:  &durationSucceeded,
					},
					{
						ExecutionID: "exec-failed",
						StartedAt:   now.Add(-2 * time.Hour).Add(35 * time.Minute),
						Status:      string(types.ExecutionStatusFailed),
						DurationMS:  &durationFailed,
					},
					{
						ExecutionID: "exec-running",
						StartedAt:   now.Add(-1 * time.Hour).Add(10 * time.Minute),
						Status:      string(types.ExecutionStatusRunning),
						DurationMS:  &durationRunning,
					},
				}, nil
			},
		}

		handler := &ExecutionTimelineHandler{
			store: store,
			cache: NewExecutionTimelineCache(),
		}

		timeline, summary, err := handler.generateTimelineData(context.Background())
		require.NoError(t, err)
		require.Len(t, timeline, 24)
		require.Equal(t, 3, summary.TotalExecutions)
		require.Equal(t, 1, summary.TotalErrors)
		require.Equal(t, 2, summary.PeakExecutions)

		var foundPeak bool
		for _, point := range timeline {
			if point.Executions == 2 {
				foundPeak = true
				require.Equal(t, 1, point.Successful)
				require.Equal(t, 1, point.Failed)
				require.InDelta(t, 50.0, point.SuccessRate, 0.001)
				require.Equal(t, int64(600), point.AvgDurationMS)
			}
			if point.Executions == 1 {
				require.Equal(t, 1, point.Running)
				require.Equal(t, int64(600), point.AvgDurationMS)
			}
		}
		require.True(t, foundPeak)
		require.NotEmpty(t, summary.PeakHour)
		require.InDelta(t, 25.0, summary.AvgSuccessRate, 0.001)

		router := gin.New()
		router.GET("/api/ui/v1/executions/timeline", handler.GetExecutionTimelineHandler)

		rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/executions/timeline", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		first := decodeJSONResponse[ExecutionTimelineResponse](t, rec)
		require.Len(t, first.TimelineData, 24)
		require.Equal(t, 2, store.calls)

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/executions/timeline", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		second := decodeJSONResponse[ExecutionTimelineResponse](t, rec)
		require.Equal(t, first.CacheTimestamp, second.CacheTimestamp)
		require.Equal(t, 2, store.calls)
	})
}

func TestNodeLogSettingsCoverage(t *testing.T) {
	t.Run("env locks map reflects overrides", func(t *testing.T) {
		t.Setenv("AGENTFIELD_NODE_LOG_PROXY_CONNECT_TIMEOUT", "5s")
		t.Setenv("AGENTFIELD_NODE_LOG_PROXY_STREAM_IDLE_TIMEOUT", "")
		t.Setenv("AGENTFIELD_NODE_LOG_PROXY_MAX_DURATION", "")
		t.Setenv("AGENTFIELD_NODE_LOG_MAX_TAIL_LINES", "99")

		locks := envLocksNodeLogProxy()
		require.True(t, locks["connect_timeout"])
		require.False(t, locks["stream_idle_timeout"])
		require.False(t, locks["max_stream_duration"])
		require.True(t, locks["max_tail_lines"])
	})

	t.Run("get handler validates configuration and returns effective values", func(t *testing.T) {
		router := newNodeLogSettingsRouter(&NodeLogSettingsHandler{})
		rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/settings/node-log-proxy", nil)
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		t.Setenv("AGENTFIELD_NODE_LOG_PROXY_CONNECT_TIMEOUT", "")
		handler := &NodeLogSettingsHandler{
			ReadConfig: func(fn func(*config.Config)) {
				fn(&config.Config{
					AgentField: config.AgentFieldConfig{
						NodeLogProxy: config.NodeLogProxyConfig{
							ConnectTimeout:    7 * time.Second,
							MaxTailLines:      250,
							MaxStreamDuration: 2 * time.Minute,
						},
					},
				})
			},
		}
		rec = performJSONRequest(newNodeLogSettingsRouter(handler), http.MethodGet, "/api/ui/v1/settings/node-log-proxy", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		var body map[string]map[string]interface{}
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
		require.Equal(t, "7s", body["effective"]["connect_timeout"])
		require.Equal(t, "1m0s", body["effective"]["stream_idle_timeout"])
		require.Equal(t, float64(250), body["effective"]["max_tail_lines"])
	})

	t.Run("put handler covers validation and success branches", func(t *testing.T) {
		rec := performJSONRequest(newNodeLogSettingsRouter(&NodeLogSettingsHandler{}), http.MethodPut, "/api/ui/v1/settings/node-log-proxy", map[string]interface{}{})
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		t.Setenv("AGENTFIELD_NODE_LOG_PROXY_CONNECT_TIMEOUT", "9s")
		handler := &NodeLogSettingsHandler{
			Storage:     &overrideStorage{StorageProvider: setupTestStorage(t)},
			ReadConfig:  func(fn func(*config.Config)) { fn(&config.Config{}) },
			WriteConfig: func(fn func(*config.Config)) {},
		}
		rec = performJSONRequest(newNodeLogSettingsRouter(handler), http.MethodPut, "/api/ui/v1/settings/node-log-proxy", map[string]interface{}{"connect_timeout": "1s"})
		require.Equal(t, http.StatusConflict, rec.Code)

		t.Setenv("AGENTFIELD_NODE_LOG_PROXY_CONNECT_TIMEOUT", "")
		router := newNodeLogSettingsRouter(handler)
		req := httptest.NewRequest(http.MethodPut, "/api/ui/v1/settings/node-log-proxy", httptest.NewRecorder().Body)
		req.Header.Set("Content-Type", "application/json")
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusBadRequest, resp.Code)

		tests := []struct {
			name string
			body map[string]interface{}
		}{
			{name: "invalid connect timeout", body: map[string]interface{}{"connect_timeout": "bad"}},
			{name: "invalid idle timeout", body: map[string]interface{}{"stream_idle_timeout": "bad"}},
			{name: "invalid max stream duration", body: map[string]interface{}{"max_stream_duration": "bad"}},
			{name: "no fields", body: map[string]interface{}{}},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				rec := performJSONRequest(router, http.MethodPut, "/api/ui/v1/settings/node-log-proxy", tc.body)
				require.Equal(t, http.StatusBadRequest, rec.Code)
			})
		}

		persistErrHandler := &NodeLogSettingsHandler{
			Storage: &overrideStorage{
				StorageProvider: setupTestStorage(t),
				getConfigFn: func(ctx context.Context, key string) (*storagepkg.ConfigEntry, error) {
					return nil, errors.New("config read failed")
				},
			},
			ReadConfig:  func(fn func(*config.Config)) { fn(&config.Config{}) },
			WriteConfig: func(fn func(*config.Config)) {},
		}
		rec = performJSONRequest(newNodeLogSettingsRouter(persistErrHandler), http.MethodPut, "/api/ui/v1/settings/node-log-proxy", map[string]interface{}{"connect_timeout": "2s"})
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		var cfg config.Config
		var persisted string
		successHandler := &NodeLogSettingsHandler{
			Storage: &overrideStorage{
				StorageProvider: setupTestStorage(t),
				getConfigFn: func(ctx context.Context, key string) (*storagepkg.ConfigEntry, error) {
					return &storagepkg.ConfigEntry{Key: key, Value: "agentfield:\n  node_log_proxy:\n    connect_timeout: 1s\n"}, nil
				},
				setConfigFn: func(ctx context.Context, key, value, source string) error {
					require.Equal(t, agentfieldYAMLConfigKey, key)
					require.Equal(t, "ui", source)
					persisted = value
					return nil
				},
			},
			ReadConfig: func(fn func(*config.Config)) { fn(&cfg) },
			WriteConfig: func(fn func(*config.Config)) {
				fn(&cfg)
			},
		}
		rec = performJSONRequest(newNodeLogSettingsRouter(successHandler), http.MethodPut, "/api/ui/v1/settings/node-log-proxy", map[string]interface{}{
			"connect_timeout":     "3s",
			"stream_idle_timeout": "45s",
			"max_stream_duration": "5m",
			"max_tail_lines":      321,
		})
		require.Equal(t, http.StatusOK, rec.Code)

		require.Equal(t, 3*time.Second, cfg.AgentField.NodeLogProxy.ConnectTimeout)
		require.Equal(t, 45*time.Second, cfg.AgentField.NodeLogProxy.StreamIdleTimeout)
		require.Equal(t, 5*time.Minute, cfg.AgentField.NodeLogProxy.MaxStreamDuration)
		require.Equal(t, 321, cfg.AgentField.NodeLogProxy.MaxTailLines)

		var persistedRoot map[string]interface{}
		require.NoError(t, yaml.Unmarshal([]byte(persisted), &persistedRoot))
		agentfield := persistedRoot["agentfield"].(map[string]interface{})
		nodeLogProxy := agentfield["node_log_proxy"].(map[string]interface{})
		require.Equal(t, "3s", nodeLogProxy["connect_timeout"])
		require.Equal(t, "45s", nodeLogProxy["stream_idle_timeout"])
		require.Equal(t, "5m0s", nodeLogProxy["max_stream_duration"])
		require.Equal(t, 321, nodeLogProxy["max_tail_lines"])
	})

	t.Run("persist overlay handles malformed and empty config", func(t *testing.T) {
		storage := &overrideStorage{
			StorageProvider: setupTestStorage(t),
			getConfigFn: func(ctx context.Context, key string) (*storagepkg.ConfigEntry, error) {
				return &storagepkg.ConfigEntry{Key: key, Value: "{"}, nil
			},
		}
		err := persistNodeLogProxyOverlay(context.Background(), storage, config.NodeLogProxyConfig{ConnectTimeout: time.Second})
		require.Error(t, err)

		var stored string
		storage = &overrideStorage{
			StorageProvider: setupTestStorage(t),
			getConfigFn: func(ctx context.Context, key string) (*storagepkg.ConfigEntry, error) {
				return nil, nil
			},
			setConfigFn: func(ctx context.Context, key, value, source string) error {
				stored = value
				return nil
			},
		}
		err = persistNodeLogProxyOverlay(context.Background(), storage, config.NodeLogProxyConfig{
			ConnectTimeout:    2 * time.Second,
			MaxTailLines:      10,
			MaxStreamDuration: time.Minute,
		})
		require.NoError(t, err)
		require.Contains(t, stored, "connect_timeout: 2s")
		require.Contains(t, stored, "max_stream_duration: 1m0s")
		require.Contains(t, stored, "max_tail_lines: 10")
	})
}
