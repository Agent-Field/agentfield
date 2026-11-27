package agentic

import (
	"context"
	"encoding/json"
	"errors"
	"net/http/httptest"
	"runtime"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeLoadStore is a minimal LoadStore that records calls and returns canned data.
type fakeLoadStore struct {
	agents       []*types.AgentNode
	agentsErr    error
	summaries    []*storage.RunSummaryAggregation
	summariesErr error

	listAgentsCalls int
	lastFilter      types.ExecutionFilter
}

func (f *fakeLoadStore) ListAgents(_ context.Context, _ types.AgentFilters) ([]*types.AgentNode, error) {
	f.listAgentsCalls++
	return f.agents, f.agentsErr
}

func (f *fakeLoadStore) QueryRunSummaries(_ context.Context, filter types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error) {
	f.lastFilter = filter
	return f.summaries, len(f.summaries), f.summariesErr
}

func TestRecommendedMaxConcurrent(t *testing.T) {
	cases := []struct {
		cores int
		want  int
	}{
		{cores: 0, want: 1},
		{cores: 1, want: 1},
		{cores: 2, want: 1},
		{cores: 3, want: 1},
		{cores: 4, want: 2},
		{cores: 8, want: 4},
		{cores: 16, want: 8},
	}
	for _, tc := range cases {
		got := RecommendedMaxConcurrent(tc.cores)
		assert.Equal(t, tc.want, got, "cores=%d", tc.cores)
		assert.GreaterOrEqual(t, got, 1, "recommendation must never drop below 1")
	}
}

func TestCachedLoadProvider_CachesWithinTTL(t *testing.T) {
	var calls int
	p := &cachedLoadProvider{
		ttl: time.Minute,
		compute: func() (*LoadInfo, error) {
			calls++
			return &LoadInfo{RunningAgents: 1, TotalAgents: 2, CPUCores: 4, RecommendedMaxConcurrent: 2}, nil
		},
	}

	first := p.Load()
	second := p.Load()
	require.NotNil(t, first)
	assert.Equal(t, 1, calls, "second call within TTL must be served from cache")
	assert.Same(t, first, second, "cached call returns the same snapshot")
}

func TestCachedLoadProvider_RecomputesAfterTTL(t *testing.T) {
	var calls int
	p := &cachedLoadProvider{
		ttl: time.Minute,
		compute: func() (*LoadInfo, error) {
			calls++
			return &LoadInfo{RunningAgents: calls}, nil
		},
	}

	p.Load()
	require.Equal(t, 1, calls)

	// Force the cache to look stale without sleeping.
	p.cachedAt = time.Now().Add(-2 * p.ttl)
	third := p.Load()
	assert.Equal(t, 2, calls)
	assert.Equal(t, 2, third.RunningAgents)
}

func TestCachedLoadProvider_ErrorFallsBackToCache(t *testing.T) {
	var fail bool
	p := &cachedLoadProvider{
		ttl: time.Minute,
		compute: func() (*LoadInfo, error) {
			if fail {
				return nil, errors.New("boom")
			}
			return &LoadInfo{RunningAgents: 7}, nil
		},
	}

	first := p.Load()
	require.NotNil(t, first)

	p.cachedAt = time.Now().Add(-2 * p.ttl) // expire
	fail = true
	second := p.Load()
	assert.Same(t, first, second, "compute error should return the last good snapshot")
}

func TestCachedLoadProvider_ErrorWithNoCacheReturnsNil(t *testing.T) {
	p := &cachedLoadProvider{
		ttl:     time.Minute,
		compute: func() (*LoadInfo, error) { return nil, errors.New("boom") },
	}
	assert.Nil(t, p.Load())
}

func TestRespondOK_AttachesLoadWhenProviderSet(t *testing.T) {
	SetLoadProvider(func() *LoadInfo {
		return &LoadInfo{
			RunningAgents:            2,
			TotalAgents:              3,
			ActiveExecutions:         1,
			CPUCores:                 8,
			RecommendedMaxConcurrent: 4,
		}
	})
	defer SetLoadProvider(nil)

	router := gin.New()
	router.GET("/t", func(c *gin.Context) { respondOK(c, gin.H{"k": "v"}) })
	req := httptest.NewRequest("GET", "/t", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	require.NotNil(t, resp.Meta)
	require.NotNil(t, resp.Meta.Load)
	assert.Equal(t, 2, resp.Meta.Load.RunningAgents)
	assert.Equal(t, 3, resp.Meta.Load.TotalAgents)
	assert.Equal(t, 1, resp.Meta.Load.ActiveExecutions)
	assert.GreaterOrEqual(t, resp.Meta.Load.RecommendedMaxConcurrent, 1)
}

func TestRespondOK_OmitsLoadWhenProviderUnset(t *testing.T) {
	SetLoadProvider(nil)

	router := gin.New()
	router.GET("/t", func(c *gin.Context) { respondOK(c, gin.H{"k": "v"}) })
	req := httptest.NewRequest("GET", "/t", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.Nil(t, resp.Meta, "meta.load must be omitted when no provider is registered")
}

func TestRespondOK_OmitsLoadWhenProviderReturnsNil(t *testing.T) {
	SetLoadProvider(func() *LoadInfo { return nil })
	defer SetLoadProvider(nil)

	router := gin.New()
	router.GET("/t", func(c *gin.Context) { respondOK(c, gin.H{"k": "v"}) })
	req := httptest.NewRequest("GET", "/t", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	assert.Nil(t, resp.Meta)
}

// TestMetaLoadSurvivesCLIMetaMerge proves the envelope marshals meta.load as
// plain nested JSON, so the `af agent` CLI meta merge (which decodes the body
// into map[string]interface{} and copies payload["meta"] key by key) preserves
// load without any typed knowledge of it.
func TestMetaLoadSurvivesCLIMetaMerge(t *testing.T) {
	resp := AgenticResponse{
		OK:   true,
		Data: map[string]interface{}{"x": 1},
		Meta: &MetaInfo{Load: &LoadInfo{
			RunningAgents:            1,
			TotalAgents:              2,
			ActiveExecutions:         3,
			CPUCores:                 4,
			RecommendedMaxConcurrent: 2,
		}},
	}
	raw, err := json.Marshal(resp)
	require.NoError(t, err)

	// Mimic proxyToServer's generic decode + meta merge in agent_commands.go.
	var decoded map[string]interface{}
	require.NoError(t, json.Unmarshal(raw, &decoded))

	existing, ok := decoded["meta"].(map[string]interface{})
	require.True(t, ok, "meta must decode as a generic object")

	merged := map[string]interface{}{}
	for k, v := range existing {
		merged[k] = v
	}
	merged["server"] = "http://localhost:8080"

	load, ok := merged["load"].(map[string]interface{})
	require.True(t, ok, "meta.load must survive the merge as a nested object")
	assert.Equal(t, float64(1), load["running_agents"])
	assert.Equal(t, float64(2), load["total_agents"])
	assert.Equal(t, float64(3), load["active_executions"])
	assert.Equal(t, float64(4), load["cpu_cores"])
	assert.Equal(t, float64(2), load["recommended_max_concurrent"])
}

func TestActiveExecutionCount(t *testing.T) {
	cases := []struct {
		name   string
		counts map[string]int
		want   int
	}{
		{name: "nil map", counts: nil, want: 0},
		{name: "empty map", counts: map[string]int{}, want: 0},
		{name: "only terminal statuses", counts: map[string]int{"succeeded": 5, "failed": 2, "cancelled": 1, "timeout": 3}, want: 0},
		{name: "only non-terminal statuses", counts: map[string]int{"running": 2, "queued": 3}, want: 5},
		{name: "mixed statuses count only non-terminal", counts: map[string]int{"running": 4, "succeeded": 10, "failed": 1}, want: 4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, activeExecutionCount(tc.counts))
		})
	}
}

func TestComputeStorageLoad_CountsAgentsAndActiveExecutions(t *testing.T) {
	store := &fakeLoadStore{
		agents: []*types.AgentNode{
			{ID: "a", HealthStatus: types.HealthStatusActive},
			{ID: "b", HealthStatus: types.HealthStatusActive},
			{ID: "c", HealthStatus: types.HealthStatusInactive},
			nil, // a nil entry must not panic and is not counted as running
		},
		summaries: []*storage.RunSummaryAggregation{
			{RunID: "r1", StatusCounts: map[string]int{"running": 2, "succeeded": 7}},
			nil, // a nil summary is skipped
			{RunID: "r2", StatusCounts: map[string]int{"queued": 1, "failed": 4}},
		},
	}

	info, err := computeStorageLoad(store)
	require.NoError(t, err)
	require.NotNil(t, info)

	assert.Equal(t, 2, info.RunningAgents, "only health-active agents are running")
	assert.Equal(t, 4, info.TotalAgents, "total is every registered entry")
	assert.Equal(t, 3, info.ActiveExecutions, "non-terminal executions summed across runs")
	assert.Equal(t, runtime.NumCPU(), info.CPUCores)
	assert.Equal(t, RecommendedMaxConcurrent(runtime.NumCPU()), info.RecommendedMaxConcurrent)
	assert.True(t, store.lastFilter.ActiveOnly, "active executions must be queried with ActiveOnly")
}

func TestComputeStorageLoad_ListAgentsErrorFailsSnapshot(t *testing.T) {
	store := &fakeLoadStore{agentsErr: errors.New("agents unavailable")}

	info, err := computeStorageLoad(store)
	require.Error(t, err)
	assert.Nil(t, info)
}

func TestComputeStorageLoad_RunSummariesErrorLeavesActiveAtZero(t *testing.T) {
	store := &fakeLoadStore{
		agents:       []*types.AgentNode{{ID: "a", HealthStatus: types.HealthStatusActive}},
		summariesErr: errors.New("summaries unavailable"),
	}

	info, err := computeStorageLoad(store)
	require.NoError(t, err, "active_executions is best-effort and must not fail the snapshot")
	require.NotNil(t, info)
	assert.Equal(t, 1, info.RunningAgents)
	assert.Equal(t, 1, info.TotalAgents)
	assert.Equal(t, 0, info.ActiveExecutions)
}

func TestNewStorageLoadProvider_ServesCachedSnapshotWithinTTL(t *testing.T) {
	store := &fakeLoadStore{
		agents: []*types.AgentNode{{ID: "a", HealthStatus: types.HealthStatusActive}},
	}
	provider := NewStorageLoadProvider(store, time.Minute)

	first := provider()
	second := provider()

	require.NotNil(t, first)
	assert.Equal(t, 1, first.RunningAgents)
	assert.Same(t, first, second, "a second call within the TTL returns the cached snapshot")
	assert.Equal(t, 1, store.listAgentsCalls, "storage is queried once per TTL window")
}
