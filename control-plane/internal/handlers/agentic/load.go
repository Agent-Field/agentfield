package agentic

import (
	"context"
	"runtime"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// LoadInfo is ambient machine-load metadata stamped onto every successful
// agentic response (meta.load). It lets the driving agent decide whether to
// launch more heavy work with zero extra round-trips.
type LoadInfo struct {
	// RunningAgents is the count of health-active registered agents.
	RunningAgents int `json:"running_agents"`
	// TotalAgents is every registered agent, active or not.
	TotalAgents int `json:"total_agents"`
	// ActiveExecutions is the number of non-terminal executions in flight now.
	ActiveExecutions int `json:"active_executions"`
	// CPUCores is runtime.NumCPU() on the control-plane host.
	CPUCores int `json:"cpu_cores"`
	// RecommendedMaxConcurrent is a suggested ceiling on concurrent heavy runs.
	RecommendedMaxConcurrent int `json:"recommended_max_concurrent"`
}

var (
	loadProviderMu sync.RWMutex
	loadProvider   func() *LoadInfo
)

// SetLoadProvider registers the package-level provider respondOK uses to stamp
// meta.load onto every successful response. Pass nil to disable. Wiring happens
// once at route registration (see registerAgenticRoutes). A nil provider — or a
// provider that returns nil — simply omits meta.load; load metadata never fails
// a response.
func SetLoadProvider(fn func() *LoadInfo) {
	loadProviderMu.Lock()
	loadProvider = fn
	loadProviderMu.Unlock()
}

// currentLoad invokes the registered provider, returning nil when unset. It is
// as cheap as the provider itself (the storage-backed provider caches).
func currentLoad() *LoadInfo {
	loadProviderMu.RLock()
	fn := loadProvider
	loadProviderMu.RUnlock()
	if fn == nil {
		return nil
	}
	return fn()
}

// RecommendedMaxConcurrent derives a safe ceiling on concurrent heavy runs from
// the CPU core count: half the cores, floored at 1. This is deliberately
// cores-based — the Go stdlib cannot portably read total system RAM, so a
// memory-aware refinement (accounting for each run's memory footprint) is a
// follow-up.
func RecommendedMaxConcurrent(cpuCores int) int {
	rec := cpuCores / 2
	if rec < 1 {
		rec = 1
	}
	return rec
}

// LoadStore is the minimal storage surface the load provider reads.
type LoadStore interface {
	ListAgents(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error)
	QueryRunSummaries(ctx context.Context, filter types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error)
}

// cachedLoadProvider memoizes a compute function for a short TTL so a burst of
// agentic calls doesn't hammer storage — the LoadInfo is recomputed at most
// once per ttl window.
type cachedLoadProvider struct {
	ttl     time.Duration
	compute func() (*LoadInfo, error)

	mu       sync.Mutex
	cached   *LoadInfo
	cachedAt time.Time
}

// Load returns the cached LoadInfo when still fresh, otherwise recomputes. On a
// compute error it falls back to the last good value (possibly nil), so a
// transient storage hiccup degrades to omitting meta.load rather than failing
// the response.
func (p *cachedLoadProvider) Load() *LoadInfo {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.cached != nil && time.Since(p.cachedAt) < p.ttl {
		return p.cached
	}
	info, err := p.compute()
	if err != nil || info == nil {
		return p.cached
	}
	p.cached = info
	p.cachedAt = time.Now()
	return info
}

// NewStorageLoadProvider builds a cached provider backed by storage, suitable
// for SetLoadProvider. running_agents counts health-active registered agents;
// active_executions sums non-terminal executions across in-flight runs (the
// same definition the executions/active handler uses).
func NewStorageLoadProvider(store LoadStore, ttl time.Duration) func() *LoadInfo {
	p := &cachedLoadProvider{
		ttl:     ttl,
		compute: func() (*LoadInfo, error) { return computeStorageLoad(store) },
	}
	return p.Load
}

func computeStorageLoad(store LoadStore) (*LoadInfo, error) {
	ctx := context.Background()

	agents, err := store.ListAgents(ctx, types.AgentFilters{})
	if err != nil {
		return nil, err
	}
	running := 0
	for _, a := range agents {
		if a != nil && a.HealthStatus == types.HealthStatusActive {
			running++
		}
	}

	// active_executions is best-effort: a failure here should not sink the rest
	// of the load snapshot, so a query error just leaves the count at zero.
	active := 0
	if summaries, _, err := store.QueryRunSummaries(ctx, types.ExecutionFilter{ActiveOnly: true}); err == nil {
		for _, agg := range summaries {
			if agg == nil {
				continue
			}
			active += activeExecutionCount(agg.StatusCounts)
		}
	}

	cores := runtime.NumCPU()
	return &LoadInfo{
		RunningAgents:            running,
		TotalAgents:              len(agents),
		ActiveExecutions:         active,
		CPUCores:                 cores,
		RecommendedMaxConcurrent: RecommendedMaxConcurrent(cores),
	}, nil
}

// activeExecutionCount sums every non-terminal execution in a run's status
// counts — mirrors the definition the executions/active handler uses.
func activeExecutionCount(statusCounts map[string]int) int {
	active := 0
	for status, count := range statusCounts {
		if !types.IsTerminalExecutionStatus(status) {
			active += count
		}
	}
	return active
}
