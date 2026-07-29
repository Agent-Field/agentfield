package ui

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

type fakeUsageStatsStore struct {
	gotSince          *time.Time
	agg               *types.UsageStatsAggregation
	series            *types.UsageTimeseries
	byModel           []types.UsageModelSeries
	gotBuckets        int
	gotByModelBuckets int
	timeseriesHit     bool
	byModelHit        bool
}

func (f *fakeUsageStatsStore) GetUsageStats(ctx context.Context, since *time.Time) (*types.UsageStatsAggregation, error) {
	f.gotSince = since
	return f.agg, nil
}

func (f *fakeUsageStatsStore) GetUsageTimeseries(ctx context.Context, since *time.Time, now time.Time, buckets int) (*types.UsageTimeseries, error) {
	f.timeseriesHit = true
	f.gotBuckets = buckets
	return f.series, nil
}

func (f *fakeUsageStatsStore) GetUsageTimeseriesByModel(ctx context.Context, since *time.Time, now time.Time, buckets int) ([]types.UsageModelSeries, error) {
	f.byModelHit = true
	f.gotByModelBuckets = buckets
	return f.byModel, nil
}

func TestNormalizeUsageWindow(t *testing.T) {
	cases := map[string]string{
		"":        "24h",
		"garbage": "24h",
		"1H":      "1h",
		"24h":     "24h",
		"7d":      "7d",
		"30D":     "30d",
		"all":     "all",
	}
	for in, want := range cases {
		if got := normalizeUsageWindow(in); got != want {
			t.Errorf("normalizeUsageWindow(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestUsageWindowSince(t *testing.T) {
	now := time.Date(2026, 7, 17, 12, 0, 0, 0, time.UTC)
	if s := usageWindowSince("all", now); s != nil {
		t.Errorf("all window should have nil since, got %v", s)
	}
	if s := usageWindowSince("1h", now); s == nil || !s.Equal(now.Add(-time.Hour)) {
		t.Errorf("1h since = %v, want %v", s, now.Add(-time.Hour))
	}
	if s := usageWindowSince("7d", now); s == nil || !s.Equal(now.Add(-7*24*time.Hour)) {
		t.Errorf("7d since = %v", s)
	}
}

func TestGetUsageStatsHandlerResponseShape(t *testing.T) {
	gin.SetMode(gin.TestMode)
	cost := 1.23
	store := &fakeUsageStatsStore{
		agg: &types.UsageStatsAggregation{
			Totals: types.UsageStatsTotals{
				CostUSD: &cost, InputTokens: 1000, OutputTokens: 2000,
				TotalTokens: 3000, ExecutionsWithUsage: 42,
			},
			ByModel: []types.UsageStatsGroup{
				{Key: "claude-opus-4-8", Provider: "anthropic", CostUSD: &cost, TotalTokens: 3000, Entries: 10},
			},
		},
	}
	h := NewUsageHandler(store)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/usage/stats", nil)
	h.GetUsageStatsHandler(c)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", w.Code, w.Body.String())
	}
	// Default window applied.
	if store.gotSince == nil {
		t.Errorf("expected non-nil since for default 24h window")
	}

	var resp map[string]json.RawMessage
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	for _, key := range []string{"window", "totals", "by_model", "by_provider", "by_agent", "by_harness", "last_updated"} {
		if _, ok := resp[key]; !ok {
			t.Errorf("response missing key %q", key)
		}
	}
	var window string
	_ = json.Unmarshal(resp["window"], &window)
	if window != "24h" {
		t.Errorf("window = %q, want 24h", window)
	}
	// last_updated should be JSON null (no rows had a timestamp).
	if string(resp["last_updated"]) != "null" {
		t.Errorf("last_updated = %s, want null", resp["last_updated"])
	}
	// by_provider/agent/harness must be arrays (not null) for the tray.
	if string(resp["by_provider"]) != "[]" {
		t.Errorf("by_provider = %s, want []", resp["by_provider"])
	}
	// No buckets param → no series key at all.
	if _, ok := resp["series"]; ok {
		t.Errorf("series key present without buckets param: %s", resp["series"])
	}
}

func TestParseUsageBuckets(t *testing.T) {
	cases := map[string]struct {
		wantN  int
		wantOK bool
	}{
		"":        {0, false},
		"0":       {0, false},
		"-1":      {0, false},
		"201":     {0, false},
		"1000":    {0, false},
		"garbage": {0, false},
		"1.5":     {0, false},
		"1":       {1, true},
		" 24 ":    {24, true},
		"200":     {200, true},
	}
	for in, want := range cases {
		gotN, gotOK := parseUsageBuckets(in)
		if gotN != want.wantN || gotOK != want.wantOK {
			t.Errorf("parseUsageBuckets(%q) = (%d,%v), want (%d,%v)", in, gotN, gotOK, want.wantN, want.wantOK)
		}
	}
}

func newUsageHandlerRequest(t *testing.T, store usageStatsStore, target string) map[string]json.RawMessage {
	t.Helper()
	gin.SetMode(gin.TestMode)
	h := NewUsageHandler(store)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, target, nil)
	h.GetUsageStatsHandler(c)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", w.Code, w.Body.String())
	}
	var resp map[string]json.RawMessage
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	return resp
}

func TestGetUsageStatsHandlerBucketsValidation(t *testing.T) {
	// Invalid/absent buckets values must yield NO series key at all in the JSON,
	// and must not touch the timeseries store method.
	for _, raw := range []string{"", "0", "-5", "201", "999", "abc", "1.2"} {
		store := &fakeUsageStatsStore{agg: &types.UsageStatsAggregation{}}
		target := "/usage/stats"
		if raw != "" {
			target += "?buckets=" + raw
		}
		resp := newUsageHandlerRequest(t, store, target)
		if _, ok := resp["series"]; ok {
			t.Errorf("buckets=%q: series key present, want absent (%s)", raw, resp["series"])
		}
		if store.timeseriesHit {
			t.Errorf("buckets=%q: GetUsageTimeseries was called, want skipped", raw)
		}
	}
}

func TestGetUsageStatsHandlerSeriesShape(t *testing.T) {
	cost := 0.12
	start := time.Date(2026, 7, 17, 0, 0, 0, 0, time.UTC)
	store := &fakeUsageStatsStore{
		agg: &types.UsageStatsAggregation{},
		series: &types.UsageTimeseries{
			BucketSeconds: 1800,
			Points: []types.UsageTimeseriesPoint{
				{Start: start, TotalTokens: 1234, CostUSD: &cost},
				{Start: start.Add(30 * time.Minute), TotalTokens: 0, CostUSD: nil},
			},
		},
	}
	resp := newUsageHandlerRequest(t, store, "/usage/stats?window=1h&buckets=2")

	if !store.timeseriesHit {
		t.Fatalf("GetUsageTimeseries not called")
	}
	if store.gotBuckets != 2 {
		t.Errorf("gotBuckets = %d, want 2", store.gotBuckets)
	}
	raw, ok := resp["series"]
	if !ok {
		t.Fatalf("series key missing")
	}
	var series struct {
		BucketSeconds int64 `json:"bucket_seconds"`
		Points        []struct {
			T           string   `json:"t"`
			TotalTokens int64    `json:"total_tokens"`
			CostUSD     *float64 `json:"cost_usd"`
		} `json:"points"`
	}
	if err := json.Unmarshal(raw, &series); err != nil {
		t.Fatalf("invalid series JSON: %v", err)
	}
	if series.BucketSeconds != 1800 {
		t.Errorf("bucket_seconds = %d, want 1800", series.BucketSeconds)
	}
	if len(series.Points) != 2 {
		t.Fatalf("points has %d entries, want 2", len(series.Points))
	}
	if series.Points[0].T != "2026-07-17T00:00:00Z" {
		t.Errorf("points[0].t = %q, want 2026-07-17T00:00:00Z", series.Points[0].T)
	}
	if series.Points[0].TotalTokens != 1234 {
		t.Errorf("points[0].total_tokens = %d, want 1234", series.Points[0].TotalTokens)
	}
	if series.Points[0].CostUSD == nil || *series.Points[0].CostUSD != 0.12 {
		t.Errorf("points[0].cost_usd = %v, want 0.12", series.Points[0].CostUSD)
	}
	// Empty bucket: cost_usd must be JSON null.
	pt1 := struct {
		CostUSD json.RawMessage `json:"cost_usd"`
	}{}
	var rawPoints struct {
		Points []json.RawMessage `json:"points"`
	}
	_ = json.Unmarshal(raw, &rawPoints)
	_ = json.Unmarshal(rawPoints.Points[1], &pt1)
	if string(pt1.CostUSD) != "null" {
		t.Errorf("points[1].cost_usd = %s, want null", pt1.CostUSD)
	}
	// Without series_by=model, the series_by_model key must be absent.
	if _, ok := resp["series_by_model"]; ok {
		t.Errorf("series_by_model present without series_by=model: %s", resp["series_by_model"])
	}
}

func TestGetUsageStatsHandlerSeriesByModelGating(t *testing.T) {
	start := time.Date(2026, 7, 17, 0, 0, 0, 0, time.UTC)
	newStore := func() *fakeUsageStatsStore {
		return &fakeUsageStatsStore{
			agg:    &types.UsageStatsAggregation{},
			series: &types.UsageTimeseries{BucketSeconds: 1800, Points: []types.UsageTimeseriesPoint{{Start: start}}},
			byModel: []types.UsageModelSeries{
				{Key: "m1", Points: []types.UsageModelTimeseriesPoint{{Start: start, TotalTokens: 5}}},
			},
		}
	}

	// series_by absent, or any value other than "model", or missing buckets →
	// series_by_model key absent and the by-model store method is never called.
	absentCases := []string{
		"/usage/stats?buckets=2",                   // valid buckets, no series_by
		"/usage/stats?buckets=2&series_by=",        // empty series_by
		"/usage/stats?buckets=2&series_by=x",       // bogus series_by
		"/usage/stats?series_by=model",             // series_by=model but no buckets
		"/usage/stats?series_by=model&buckets=0",   // invalid buckets
		"/usage/stats?series_by=model&buckets=abc", // invalid buckets
	}
	for _, target := range absentCases {
		store := newStore()
		resp := newUsageHandlerRequest(t, store, target)
		if _, ok := resp["series_by_model"]; ok {
			t.Errorf("%s: series_by_model present, want absent (%s)", target, resp["series_by_model"])
		}
		if store.byModelHit {
			t.Errorf("%s: GetUsageTimeseriesByModel called, want skipped", target)
		}
	}
}

func TestGetUsageStatsHandlerSeriesByModelShape(t *testing.T) {
	start := time.Date(2026, 7, 17, 0, 0, 0, 0, time.UTC)
	store := &fakeUsageStatsStore{
		agg:    &types.UsageStatsAggregation{},
		series: &types.UsageTimeseries{BucketSeconds: 1800, Points: []types.UsageTimeseriesPoint{{Start: start}, {Start: start.Add(30 * time.Minute)}}},
		byModel: []types.UsageModelSeries{
			{Key: "claude-opus-4-8", Points: []types.UsageModelTimeseriesPoint{
				{Start: start, TotalTokens: 1234},
				{Start: start.Add(30 * time.Minute), TotalTokens: 0},
			}},
			{Key: "other", Points: []types.UsageModelTimeseriesPoint{
				{Start: start, TotalTokens: 7},
				{Start: start.Add(30 * time.Minute), TotalTokens: 8},
			}},
		},
	}
	resp := newUsageHandlerRequest(t, store, "/usage/stats?window=1h&buckets=2&series_by=Model")

	if !store.byModelHit {
		t.Fatalf("GetUsageTimeseriesByModel not called")
	}
	if store.gotByModelBuckets != 2 {
		t.Errorf("gotByModelBuckets = %d, want 2", store.gotByModelBuckets)
	}
	raw, ok := resp["series_by_model"]
	if !ok {
		t.Fatalf("series_by_model key missing")
	}
	var got []struct {
		Key    string `json:"key"`
		Points []struct {
			T           string `json:"t"`
			TotalTokens int64  `json:"total_tokens"`
		} `json:"points"`
	}
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("invalid series_by_model JSON: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("series_by_model has %d entries, want 2", len(got))
	}
	if got[0].Key != "claude-opus-4-8" || got[1].Key != "other" {
		t.Errorf("keys = %q/%q, want claude-opus-4-8/other", got[0].Key, got[1].Key)
	}
	if len(got[0].Points) != 2 {
		t.Fatalf("entry 0 points = %d, want 2", len(got[0].Points))
	}
	if got[0].Points[0].T != "2026-07-17T00:00:00Z" {
		t.Errorf("points[0].t = %q, want 2026-07-17T00:00:00Z", got[0].Points[0].T)
	}
	if got[0].Points[0].TotalTokens != 1234 {
		t.Errorf("points[0].total_tokens = %d, want 1234", got[0].Points[0].TotalTokens)
	}
	// Grid alignment: series_by_model timestamps match the plain series exactly.
	var series struct {
		Points []struct {
			T string `json:"t"`
		} `json:"points"`
	}
	_ = json.Unmarshal(resp["series"], &series)
	if len(series.Points) != len(got[0].Points) {
		t.Fatalf("series has %d points, series_by_model entry has %d", len(series.Points), len(got[0].Points))
	}
	for i := range series.Points {
		if series.Points[i].T != got[0].Points[i].T {
			t.Errorf("grid mismatch at %d: series=%q by_model=%q", i, series.Points[i].T, got[0].Points[i].T)
		}
	}
	// Points must not carry a cost field (tokens only).
	var rawEntries []struct {
		Points []map[string]json.RawMessage `json:"points"`
	}
	_ = json.Unmarshal(raw, &rawEntries)
	if _, hasCost := rawEntries[0].Points[0]["cost_usd"]; hasCost {
		t.Errorf("series_by_model point unexpectedly carries cost_usd")
	}
}
