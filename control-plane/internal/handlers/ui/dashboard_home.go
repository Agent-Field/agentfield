package ui

import (
	"net/http"
	"sort"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// homeWindow is one of the windows the Home page offers. The window always
// ends now; the baseline it is compared with is the seven days before it.
type homeWindow struct {
	Preset   string
	Duration time.Duration
	Bucket   time.Duration
}

var homeWindows = map[string]homeWindow{
	"1h":  {Preset: "1h", Duration: time.Hour, Bucket: 5 * time.Minute},
	"6h":  {Preset: "6h", Duration: 6 * time.Hour, Bucket: 30 * time.Minute},
	"24h": {Preset: "24h", Duration: 24 * time.Hour, Bucket: time.Hour},
	"7d":  {Preset: "7d", Duration: 7 * 24 * time.Hour, Bucket: 12 * time.Hour},
}

// homeBaselinePeriod is how far back the "usual" level looks.
const homeBaselinePeriod = 7 * 24 * time.Hour

// homeCacheTTL absorbs bursts of refreshes (several tabs, coalesced live
// updates) without making the page noticeably stale.
const homeCacheTTL = 5 * time.Second

// homeBaselineTTL is how long a computed baseline is reused. It covers seven
// days, so a few minutes of drift does not change what "usual" means, and
// reusing it keeps the scan of a week of executions off every refresh.
const homeBaselineTTL = 5 * time.Minute

// homeMaxStoppedAgents bounds how many silent agents get a blast-radius query.
const homeMaxStoppedAgents = 10

// DeveloperHomeResponse is the payload of GET /api/ui/v1/dashboard/home.
type DeveloperHomeResponse struct {
	GeneratedAt time.Time          `json:"generated_at"`
	Window      HomeWindowInfo     `json:"window"`
	Agents      []HomeAgent        `json:"agents"`
	AgentCounts HomeAgentCounts    `json:"agent_counts"`
	Stopped     []HomeStoppedAgent `json:"stopped_agents"`
	*types.DeveloperHomeData
	// Forwarder is present only when an observability webhook is configured.
	Forwarder *HomeForwarder `json:"forwarder,omitempty"`
}

// HomeWindowInfo describes the window the response covers.
type HomeWindowInfo struct {
	Preset        string    `json:"preset"`
	Start         time.Time `json:"start"`
	End           time.Time `json:"end"`
	BucketSeconds int64     `json:"bucket_seconds"`
}

// HomeAgent is a registered agent and the status derived from its recorded
// heartbeat and lifecycle.
type HomeAgent struct {
	ID            string    `json:"id"`
	Version       string    `json:"version,omitempty"`
	Status        string    `json:"status"`
	Health        string    `json:"health"`
	Lifecycle     string    `json:"lifecycle"`
	LastHeartbeat time.Time `json:"last_heartbeat"`
	Reasoners     int       `json:"reasoners"`
	Skills        int       `json:"skills"`
}

// HomeAgentCounts groups registered agents by derived status.
type HomeAgentCounts struct {
	Registered      int `json:"registered"`
	Running         int `json:"running"`
	Offline         int `json:"offline"`
	Degraded        int `json:"degraded"`
	Starting        int `json:"starting"`
	PendingApproval int `json:"pending_approval"`
	Unknown         int `json:"unknown"`
}

// HomeStoppedAgent is a registered agent whose heartbeats stopped, with the
// entry reasoners whose runs in the window went through it.
type HomeStoppedAgent struct {
	ID            string                          `json:"id"`
	LastHeartbeat time.Time                       `json:"last_heartbeat"`
	EntryPoints   []types.DeveloperHomeEntryPoint `json:"entry_points"`
}

// HomeForwarder reports the observability forwarder's dead-letter depth.
type HomeForwarder struct {
	Enabled     bool  `json:"enabled"`
	DeadLetters int64 `json:"dead_letters"`
}

type homeCacheEntry struct {
	at   time.Time
	data *DeveloperHomeResponse
}

type homeBaselineEntry struct {
	at        time.Time
	baseline  *types.DeveloperHomeBaseline
	spendRows int64
	spendCost *float64
}

type homeCache struct {
	mu        sync.Mutex
	entries   map[string]homeCacheEntry
	baselines map[string]homeBaselineEntry
}

func (c *homeCache) getBaseline(key string, now time.Time) (homeBaselineEntry, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.baselines[key]
	if !ok || now.Sub(e.at) >= homeBaselineTTL {
		return homeBaselineEntry{}, false
	}
	return e, true
}

func (c *homeCache) setBaseline(key string, e homeBaselineEntry) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.baselines == nil {
		c.baselines = map[string]homeBaselineEntry{}
	}
	c.baselines[key] = e
}

func (c *homeCache) get(key string, now time.Time) *DeveloperHomeResponse {
	c.mu.Lock()
	defer c.mu.Unlock()
	if e, ok := c.entries[key]; ok && now.Sub(e.at) < homeCacheTTL {
		return e.data
	}
	return nil
}

func (c *homeCache) set(key string, now time.Time, data *DeveloperHomeResponse) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.entries == nil {
		c.entries = map[string]homeCacheEntry{}
	}
	c.entries[key] = homeCacheEntry{at: now, data: data}
}

// GetDeveloperHomeHandler serves the Home page.
// GET /api/ui/v1/dashboard/home?window=1h|6h|24h|7d (default 1h)
func (h *DashboardHandler) GetDeveloperHomeHandler(c *gin.Context) {
	ctx := c.Request.Context()
	now := time.Now().UTC()
	if h.now != nil {
		now = h.now().UTC()
	}

	w, ok := homeWindows[c.DefaultQuery("window", "1h")]
	if !ok {
		RespondBadRequest(c, "window must be one of 1h, 6h, 24h, 7d")
		return
	}
	if cached := h.homeCache.get(w.Preset, now); cached != nil {
		c.JSON(http.StatusOK, cached)
		return
	}

	start := now.Add(-w.Duration)
	query := types.DeveloperHomeQuery{
		Start:         start,
		End:           now,
		BucketSeconds: int64(w.Bucket.Seconds()),
		Limit:         12,
	}
	cachedBaseline, haveBaseline := h.homeCache.getBaseline(w.Preset, now)
	if !haveBaseline {
		query.BaselineStart = start.Add(-homeBaselinePeriod)
	}
	data, err := h.storage.QueryDeveloperHome(ctx, query)
	if err != nil {
		logger.Logger.Error().Err(err).Msg("failed to query developer home aggregates")
		RespondInternalError(c, "failed to load home data")
		return
	}
	if haveBaseline {
		data.Baseline = cachedBaseline.baseline
		if cachedBaseline.spendRows > 0 {
			if data.Spend == nil {
				data.Spend = &types.DeveloperHomeSpend{}
			}
			data.Spend.BaselineRows = cachedBaseline.spendRows
			data.Spend.BaselineCostUSD = cachedBaseline.spendCost
		}
	} else {
		entry := homeBaselineEntry{at: now, baseline: data.Baseline}
		if data.Spend != nil {
			entry.spendRows = data.Spend.BaselineRows
			entry.spendCost = data.Spend.BaselineCostUSD
		}
		h.homeCache.setBaseline(w.Preset, entry)
	}

	agents, err := h.storage.ListAgents(ctx, types.AgentFilters{})
	if err != nil {
		logger.Logger.Error().Err(err).Msg("failed to list agents for developer home")
		RespondInternalError(c, "failed to load agent data")
		return
	}

	resp := &DeveloperHomeResponse{
		GeneratedAt: now,
		Window: HomeWindowInfo{
			Preset: w.Preset, Start: start, End: now, BucketSeconds: int64(w.Bucket.Seconds()),
		},
		Agents:            make([]HomeAgent, 0, len(agents)),
		Stopped:           []HomeStoppedAgent{},
		DeveloperHomeData: data,
	}

	for _, agent := range agents {
		status := recordedAgentStatus(agent)
		resp.Agents = append(resp.Agents, HomeAgent{
			ID:            agent.ID,
			Version:       agent.Version,
			Status:        status,
			Health:        string(agent.HealthStatus),
			Lifecycle:     string(agent.LifecycleStatus),
			LastHeartbeat: agent.LastHeartbeat,
			Reasoners:     len(agent.Reasoners),
			Skills:        len(agent.Skills),
		})
		resp.AgentCounts.Registered++
		switch status {
		case agentStatusRunning:
			resp.AgentCounts.Running++
		case agentStatusOffline:
			resp.AgentCounts.Offline++
		case agentStatusDegraded:
			resp.AgentCounts.Degraded++
		case agentStatusStarting:
			resp.AgentCounts.Starting++
		case agentStatusPendingApproval:
			resp.AgentCounts.PendingApproval++
		default:
			resp.AgentCounts.Unknown++
		}
	}
	sort.Slice(resp.Agents, func(i, j int) bool { return resp.Agents[i].ID < resp.Agents[j].ID })

	// Agents whose heartbeats stopped, most recently silent first, each with
	// the entry points whose runs went through it in the window.
	for _, agent := range resp.Agents {
		if agent.Status == agentStatusOffline {
			resp.Stopped = append(resp.Stopped, HomeStoppedAgent{ID: agent.ID, LastHeartbeat: agent.LastHeartbeat})
		}
	}
	sort.Slice(resp.Stopped, func(i, j int) bool {
		return resp.Stopped[i].LastHeartbeat.After(resp.Stopped[j].LastHeartbeat)
	})
	if len(resp.Stopped) > homeMaxStoppedAgents {
		resp.Stopped = resp.Stopped[:homeMaxStoppedAgents]
	}
	for i := range resp.Stopped {
		entries, err := h.storage.QueryBlastRadius(ctx, resp.Stopped[i].ID, start, now)
		if err != nil {
			logger.Logger.Error().Err(err).Msg("failed to query blast radius for developer home")
			RespondInternalError(c, "failed to load home data")
			return
		}
		resp.Stopped[i].EntryPoints = entries
	}

	if hook, err := h.storage.GetObservabilityWebhook(ctx); err == nil && hook != nil {
		dead, err := h.storage.CountForwarderDeadLetters(ctx)
		if err == nil {
			resp.Forwarder = &HomeForwarder{Enabled: hook.Enabled, DeadLetters: dead}
		}
	}

	h.homeCache.set(w.Preset, now, resp)
	c.JSON(http.StatusOK, resp)
}
