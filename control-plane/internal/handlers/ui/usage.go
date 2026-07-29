package ui

import (
	"context"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// maxUsageBuckets caps the number of timeseries buckets a client may request.
const maxUsageBuckets = 200

// usageStatsStore is the narrow storage capability the usage endpoint needs.
type usageStatsStore interface {
	GetUsageStats(ctx context.Context, since *time.Time) (*types.UsageStatsAggregation, error)
	GetUsageTimeseries(ctx context.Context, since *time.Time, now time.Time, buckets int) (*types.UsageTimeseries, error)
	GetUsageTimeseriesByModel(ctx context.Context, since *time.Time, now time.Time, buckets int) ([]types.UsageModelSeries, error)
}

// UsageHandler serves aggregated token/cost usage statistics for the UI/tray.
type UsageHandler struct {
	store usageStatsStore
}

// NewUsageHandler creates a new UsageHandler.
func NewUsageHandler(store usageStatsStore) *UsageHandler {
	return &UsageHandler{store: store}
}

// UsageStatsResponse is the response body for GET /api/ui/v1/usage/stats.
// Series is only present when a valid buckets query param was supplied.
type UsageStatsResponse struct {
	Window        string             `json:"window"`
	Totals        UsageStatsTotals   `json:"totals"`
	ByModel       []UsageStatsGroup  `json:"by_model"`
	ByProvider    []UsageStatsGroup  `json:"by_provider"`
	ByAgent       []UsageStatsGroup  `json:"by_agent"`
	ByHarness     []UsageStatsGroup  `json:"by_harness"`
	LastUpdated   *string            `json:"last_updated"`
	Series        *UsageSeries       `json:"series,omitempty"`
	SeriesByModel []UsageModelSeries `json:"series_by_model,omitempty"`
}

// UsageModelSeries is one per-model token series (tokens only, no cost). Key is
// the model name, or "other" for the rolled-up remainder.
type UsageModelSeries struct {
	Key    string                  `json:"key"`
	Points []UsageModelSeriesPoint `json:"points"`
}

// UsageModelSeriesPoint is one zero-filled per-model bucket. T is the bucket
// start (RFC3339 UTC); tokens only.
type UsageModelSeriesPoint struct {
	T           string `json:"t"`
	TotalTokens int64  `json:"total_tokens"`
}

// UsageSeries is the optional bucketed token/cost timeseries.
type UsageSeries struct {
	BucketSeconds int64              `json:"bucket_seconds"`
	Points        []UsageSeriesPoint `json:"points"`
}

// UsageSeriesPoint is one zero-filled bucket. T is the bucket start time
// (RFC3339 UTC); CostUSD is null when the bucket has no cost.
type UsageSeriesPoint struct {
	T           string   `json:"t"`
	TotalTokens int64    `json:"total_tokens"`
	CostUSD     *float64 `json:"cost_usd"`
}

// UsageStatsTotals holds the window's aggregate totals.
type UsageStatsTotals struct {
	CostUSD             *float64 `json:"cost_usd"`
	InputTokens         int64    `json:"input_tokens"`
	OutputTokens        int64    `json:"output_tokens"`
	CacheReadTokens     int64    `json:"cache_read_tokens"`
	CacheCreationTokens int64    `json:"cache_creation_tokens"`
	TotalTokens         int64    `json:"total_tokens"`
	ExecutionsWithUsage int64    `json:"executions_with_usage"`
}

// UsageStatsGroup is one grouped bucket. Provider is only emitted for by_model.
type UsageStatsGroup struct {
	Key          string   `json:"key"`
	Provider     string   `json:"provider,omitempty"`
	CostUSD      *float64 `json:"cost_usd"`
	InputTokens  int64    `json:"input_tokens"`
	OutputTokens int64    `json:"output_tokens"`
	TotalTokens  int64    `json:"total_tokens"`
	Entries      int64    `json:"entries"`
}

// GetUsageStatsHandler handles token/cost usage aggregation requests.
// GET /api/ui/v1/usage/stats?window=<1h|24h|7d|30d|all>
func (h *UsageHandler) GetUsageStatsHandler(c *gin.Context) {
	now := time.Now().UTC()
	window := normalizeUsageWindow(c.Query("window"))
	since := usageWindowSince(window, now)

	agg, err := h.store.GetUsageStats(c.Request.Context(), since)
	if err != nil {
		RespondInternalError(c, "failed to aggregate usage: "+err.Error())
		return
	}
	if agg == nil {
		agg = &types.UsageStatsAggregation{}
	}

	resp := UsageStatsResponse{
		Window: window,
		Totals: UsageStatsTotals{
			CostUSD:             agg.Totals.CostUSD,
			InputTokens:         agg.Totals.InputTokens,
			OutputTokens:        agg.Totals.OutputTokens,
			CacheReadTokens:     agg.Totals.CacheReadTokens,
			CacheCreationTokens: agg.Totals.CacheCreationTokens,
			TotalTokens:         agg.Totals.TotalTokens,
			ExecutionsWithUsage: agg.Totals.ExecutionsWithUsage,
		},
		ByModel:    toUsageGroups(agg.ByModel),
		ByProvider: toUsageGroups(agg.ByProvider),
		ByAgent:    toUsageGroups(agg.ByAgent),
		ByHarness:  toUsageGroups(agg.ByHarness),
	}
	if agg.LastUpdated != nil {
		formatted := agg.LastUpdated.UTC().Format(time.RFC3339)
		resp.LastUpdated = &formatted
	}

	// Optional bucketed timeseries. Absent/invalid buckets → no series field.
	if buckets, ok := parseUsageBuckets(c.Query("buckets")); ok {
		series, err := h.store.GetUsageTimeseries(c.Request.Context(), since, now, buckets)
		if err != nil {
			RespondInternalError(c, "failed to compute usage timeseries: "+err.Error())
			return
		}
		resp.Series = toUsageSeries(series)

		// Optional per-model series, gated on series_by=model in addition to a valid
		// buckets param. Any other series_by value leaves the key absent.
		if strings.EqualFold(strings.TrimSpace(c.Query("series_by")), "model") {
			byModel, err := h.store.GetUsageTimeseriesByModel(c.Request.Context(), since, now, buckets)
			if err != nil {
				RespondInternalError(c, "failed to compute usage timeseries by model: "+err.Error())
				return
			}
			resp.SeriesByModel = toUsageModelSeries(byModel)
		}
	}

	c.JSON(http.StatusOK, resp)
}

// parseUsageBuckets validates the buckets query param. It returns (n, true) only
// for an integer in [1, maxUsageBuckets]; anything absent, non-numeric, or out
// of range yields (0, false) so the response carries no series field.
func parseUsageBuckets(raw string) (int, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return 0, false
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 1 || n > maxUsageBuckets {
		return 0, false
	}
	return n, true
}

// toUsageSeries converts a storage timeseries into the response shape, formatting
// each bucket start as RFC3339 UTC.
func toUsageSeries(series *types.UsageTimeseries) *UsageSeries {
	if series == nil {
		return nil
	}
	points := make([]UsageSeriesPoint, 0, len(series.Points))
	for _, p := range series.Points {
		points = append(points, UsageSeriesPoint{
			T:           p.Start.UTC().Format(time.RFC3339),
			TotalTokens: p.TotalTokens,
			CostUSD:     p.CostUSD,
		})
	}
	return &UsageSeries{
		BucketSeconds: series.BucketSeconds,
		Points:        points,
	}
}

// toUsageModelSeries converts storage per-model series into the response shape,
// formatting each bucket start as RFC3339 UTC. A nil/empty input yields nil so
// the series_by_model key is omitted.
func toUsageModelSeries(series []types.UsageModelSeries) []UsageModelSeries {
	if len(series) == 0 {
		return nil
	}
	out := make([]UsageModelSeries, 0, len(series))
	for _, s := range series {
		points := make([]UsageModelSeriesPoint, 0, len(s.Points))
		for _, p := range s.Points {
			points = append(points, UsageModelSeriesPoint{
				T:           p.Start.UTC().Format(time.RFC3339),
				TotalTokens: p.TotalTokens,
			})
		}
		out = append(out, UsageModelSeries{Key: s.Key, Points: points})
	}
	return out
}

func toUsageGroups(groups []types.UsageStatsGroup) []UsageStatsGroup {
	out := make([]UsageStatsGroup, 0, len(groups))
	for _, g := range groups {
		out = append(out, UsageStatsGroup{
			Key:          g.Key,
			Provider:     g.Provider,
			CostUSD:      g.CostUSD,
			InputTokens:  g.InputTokens,
			OutputTokens: g.OutputTokens,
			TotalTokens:  g.TotalTokens,
			Entries:      g.Entries,
		})
	}
	return out
}

// normalizeUsageWindow validates the window query param, defaulting to "24h".
func normalizeUsageWindow(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "1h":
		return "1h"
	case "24h":
		return "24h"
	case "7d":
		return "7d"
	case "30d":
		return "30d"
	case "all":
		return "all"
	default:
		return "24h"
	}
}

// usageWindowSince returns the inclusive lower time bound for a window, or nil
// for the "all" window (no lower bound).
func usageWindowSince(window string, now time.Time) *time.Time {
	var d time.Duration
	switch window {
	case "1h":
		d = time.Hour
	case "24h":
		d = 24 * time.Hour
	case "7d":
		d = 7 * 24 * time.Hour
	case "30d":
		d = 30 * 24 * time.Hour
	case "all":
		return nil
	default:
		d = 24 * time.Hour
	}
	since := now.Add(-d)
	return &since
}
