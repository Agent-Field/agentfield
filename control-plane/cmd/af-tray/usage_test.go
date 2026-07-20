package main

import (
	"bytes"
	"fmt"
	"image/png"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"
)

// These tests pin the usage-card behaviours the tray depends on. Like the other
// af-tray tests they avoid any GUI/CGO dependency so they run on the Linux CI
// even though the menu itself is macOS-only.

// Contract: the usage URL targets the resolved port and carries the window.
func TestUsageStatsURL(t *testing.T) {
	t.Setenv("AGENTFIELD_PORT", "9091")
	if got, want := usageStatsURL("7d"), "http://localhost:9091/api/ui/v1/usage/stats?window=7d"; got != want {
		t.Errorf("usageStatsURL() = %q, want %q", got, want)
	}
}

// Contract: parseUsageStats reads totals + groups, tolerating null cost fields
// and preserving the server's group order.
func TestParseUsageStats(t *testing.T) {
	body := []byte(`{
		"window":"24h",
		"totals":{"cost_usd":1.23,"input_tokens":1000,"output_tokens":2000,"cache_read_tokens":0,"cache_creation_tokens":0,"total_tokens":3000,"executions_with_usage":42},
		"by_model":[
			{"key":"claude-opus-4-8","provider":"anthropic","cost_usd":0.9,"input_tokens":1,"output_tokens":2,"total_tokens":2000,"entries":10},
			{"key":"gpt-4o","provider":"openai","cost_usd":null,"input_tokens":1,"output_tokens":2,"total_tokens":1000,"entries":5}
		],
		"by_provider":[{"key":"anthropic","cost_usd":1.1,"total_tokens":2100,"entries":15}],
		"by_agent":[],
		"by_harness":[{"key":"claude-code","cost_usd":1.0,"total_tokens":1500,"entries":8}],
		"last_updated":"2026-07-17T12:00:00Z"
	}`)
	u, err := parseUsageStats(body)
	if err != nil {
		t.Fatalf("parseUsageStats: %v", err)
	}
	if u.Status != usageOK {
		t.Errorf("Status = %v, want usageOK", u.Status)
	}
	if u.Window != "24h" || u.TotalTokens != 3000 || u.Executions != 42 {
		t.Errorf("totals = %+v, want window=24h total=3000 exec=42", u)
	}
	if u.CostUSD == nil || *u.CostUSD != 1.23 {
		t.Errorf("CostUSD = %v, want 1.23", u.CostUSD)
	}
	if len(u.ByModel) != 2 || u.ByModel[0].Key != "claude-opus-4-8" {
		t.Fatalf("ByModel = %+v, want 2 groups, opus first", u.ByModel)
	}
	if u.ByModel[1].CostUSD != nil {
		t.Errorf("second model cost = %v, want nil (null tolerated)", u.ByModel[1].CostUSD)
	}
	if len(u.ByProvider) != 1 || len(u.ByHarness) != 1 {
		t.Errorf("providers/harnesses = %d/%d, want 1/1", len(u.ByProvider), len(u.ByHarness))
	}
	if u.LastUpdated == nil {
		t.Error("LastUpdated = nil, want parsed time")
	}
	if !u.hasData() {
		t.Error("hasData() = false, want true for a window with tokens")
	}
}

// Contract: an empty window (no usage) parses cleanly with a null last_updated
// and reports hasData()==false.
func TestParseUsageStatsEmpty(t *testing.T) {
	body := []byte(`{"window":"24h","totals":{"cost_usd":null,"total_tokens":0,"executions_with_usage":0},"by_model":[],"by_provider":[],"by_agent":[],"by_harness":[],"last_updated":null}`)
	u, err := parseUsageStats(body)
	if err != nil {
		t.Fatalf("parseUsageStats: %v", err)
	}
	if u.hasData() {
		t.Error("hasData() = true, want false for an empty window")
	}
	if u.LastUpdated != nil {
		t.Errorf("LastUpdated = %v, want nil", u.LastUpdated)
	}
	if u.CostUSD != nil {
		t.Errorf("CostUSD = %v, want nil", u.CostUSD)
	}
}

func TestParseUsageStatsBadJSON(t *testing.T) {
	if _, err := parseUsageStats([]byte("not json")); err == nil {
		t.Error("parseUsageStats on garbage = nil error, want error")
	}
}

// Contract: fetchUsageStats sends the key on 200, maps 401/403 to
// usageAuthRequired, 404 to usageAbsent (older server → hide section), and
// everything else to usageUnavailable.
func TestFetchUsageStats(t *testing.T) {
	var mode, gotKey, gotWindow string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get("X-API-Key")
		gotWindow = r.URL.Query().Get("window")
		switch mode {
		case "ok":
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, `{"window":"24h","totals":{"cost_usd":0.5,"total_tokens":3000,"executions_with_usage":2},"by_model":[],"by_provider":[],"by_harness":[],"last_updated":null}`)
		case "401":
			w.WriteHeader(http.StatusUnauthorized)
		case "403":
			w.WriteHeader(http.StatusForbidden)
		case "404":
			w.WriteHeader(http.StatusNotFound)
		default:
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	defer ts.Close()
	u, _ := url.Parse(ts.URL)
	t.Setenv("AGENTFIELD_PORT", u.Port())

	t.Run("ok sends key + window", func(t *testing.T) {
		mode = "ok"
		s := fetchUsageStats("24h", "secret")
		if s.Status != usageOK || s.TotalTokens != 3000 {
			t.Fatalf("got %+v, want usageOK total=3000", s)
		}
		if gotKey != "secret" {
			t.Errorf("X-API-Key = %q, want secret", gotKey)
		}
		if gotWindow != "24h" {
			t.Errorf("window = %q, want 24h", gotWindow)
		}
	})
	t.Run("401 -> auth required", func(t *testing.T) {
		mode = "401"
		if s := fetchUsageStats("24h", "bad"); s.Status != usageAuthRequired {
			t.Errorf("Status = %v, want usageAuthRequired", s.Status)
		}
	})
	t.Run("403 -> auth required", func(t *testing.T) {
		mode = "403"
		if s := fetchUsageStats("24h", "bad"); s.Status != usageAuthRequired {
			t.Errorf("Status = %v, want usageAuthRequired", s.Status)
		}
	})
	t.Run("404 -> absent", func(t *testing.T) {
		mode = "404"
		if s := fetchUsageStats("24h", "k"); s.Status != usageAbsent {
			t.Errorf("Status = %v, want usageAbsent", s.Status)
		}
	})
	t.Run("500 -> unavailable", func(t *testing.T) {
		mode = "500"
		if s := fetchUsageStats("24h", "k"); s.Status != usageUnavailable {
			t.Errorf("Status = %v, want usageUnavailable", s.Status)
		}
	})
}

func TestFetchUsageStatsUnreachable(t *testing.T) {
	down := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	u, _ := url.Parse(down.URL)
	down.Close()
	t.Setenv("AGENTFIELD_PORT", u.Port())
	if s := fetchUsageStats("24h", ""); s.Status != usageUnavailable {
		t.Errorf("Status = %v, want usageUnavailable", s.Status)
	}
}

// Contract: token counts render compactly with M/k suffixes.
func TestFormatTokens(t *testing.T) {
	cases := map[int64]string{
		0:         "0",
		999:       "999",
		1000:      "1.0k",
		45300:     "45.3k",
		3_000_000: "3.0M",
		1_200_000: "1.2M",
	}
	for in, want := range cases {
		if got := formatTokens(in); got != want {
			t.Errorf("formatTokens(%d) = %q, want %q", in, got, want)
		}
	}
}

// Contract: cost renders with two decimals normally, four for sub-cent amounts,
// and an em dash when unknown (null).
func TestFormatCost(t *testing.T) {
	f := func(v float64) *float64 { return &v }
	cases := []struct {
		in   *float64
		want string
	}{
		{nil, "—"},
		{f(0), "$0.00"},
		{f(1.23), "$1.23"},
		{f(0.0042), "$0.0042"},
		{f(12.4), "$12.40"},
	}
	for _, tc := range cases {
		if got := formatCost(tc.in); got != tc.want {
			t.Errorf("formatCost(%v) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// Contract: model ids are shortened by dropping provider path prefixes and the
// "claude-" vendor prefix, then truncated.
func TestShortModelName(t *testing.T) {
	cases := map[string]string{
		"claude-opus-4-8":                    "opus-4-8",
		"openrouter/anthropic/claude-sonnet": "sonnet",
		"gpt-4o":                             "gpt-4o",
		"a-very-long-model-identifier-xyz":   "a-very-long-model-ide…",
	}
	for in, want := range cases {
		if got := shortModelName(in); got != want {
			t.Errorf("shortModelName(%q) = %q, want %q", in, got, want)
		}
	}
}

// Contract: the parent headline includes cost when known and omits it (tokens
// only) when the cost is null.
func TestUsageHeadline(t *testing.T) {
	c := 1.23
	withCost := usageStats{Status: usageOK, Window: "24h", CostUSD: &c, TotalTokens: 3_000_000}
	if got, want := usageHeadline(withCost), "Usage (24h) — $1.23 · 3.0M tokens"; got != want {
		t.Errorf("usageHeadline(cost) = %q, want %q", got, want)
	}
	noCost := usageStats{Status: usageOK, Window: "24h", TotalTokens: 3_000_000}
	if got, want := usageHeadline(noCost), "Usage (24h) — 3.0M tokens"; got != want {
		t.Errorf("usageHeadline(no cost) = %q, want %q", got, want)
	}
}

// Contract: relative time reads friendly across buckets.
func TestRelativeTime(t *testing.T) {
	now := time.Date(2026, 7, 17, 12, 0, 0, 0, time.UTC)
	cases := []struct {
		ago  time.Duration
		want string
	}{
		{5 * time.Second, "just now"},
		{60 * time.Second, "1 minute ago"},
		{8 * time.Minute, "8 minutes ago"},
		{90 * time.Minute, "1 hour ago"},
		{3 * time.Hour, "3 hours ago"},
		{25 * time.Hour, "1 day ago"},
		{50 * time.Hour, "2 days ago"},
	}
	for _, tc := range cases {
		if got := relativeTime(now.Add(-tc.ago), now); got != tc.want {
			t.Errorf("relativeTime(-%s) = %q, want %q", tc.ago, got, tc.want)
		}
	}
}

// Contract: the footer prefixes "Last updated: " and is empty (hidden) when the
// timestamp is nil.
func TestUsageFooter(t *testing.T) {
	now := time.Date(2026, 7, 17, 12, 0, 0, 0, time.UTC)
	ts := now.Add(-8 * time.Minute)
	if got, want := usageFooter(&ts, now), "Last updated: 8 minutes ago"; got != want {
		t.Errorf("usageFooter() = %q, want %q", got, want)
	}
	if got := usageFooter(nil, now); got != "" {
		t.Errorf("usageFooter(nil) = %q, want empty", got)
	}
}

// ---- Claude subscription quota ---------------------------------------------

// Contract: the OAuth access token is pulled from the keychain JSON blob.
func TestParseClaudeCodeToken(t *testing.T) {
	blob := []byte(`{"claudeAiOauth":{"accessToken":"sk-ant-oat01-xyz","refreshToken":"r","expiresAt":1,"scopes":["user:inference"]}}`)
	tok, err := parseClaudeCodeToken(blob)
	if err != nil {
		t.Fatalf("parseClaudeCodeToken: %v", err)
	}
	if tok != "sk-ant-oat01-xyz" {
		t.Errorf("token = %q, want sk-ant-oat01-xyz", tok)
	}
	if _, err := parseClaudeCodeToken([]byte(`{"claudeAiOauth":{"accessToken":""}}`)); err == nil {
		t.Error("empty token = nil error, want error")
	}
	if _, err := parseClaudeCodeToken([]byte("nope")); err == nil {
		t.Error("garbage = nil error, want error")
	}
}

// Contract: the OAuth usage response parses into 5h/7d utilization + reset
// times, defending against missing/null fields.
func TestParseClaudeQuota(t *testing.T) {
	body := []byte(`{
		"five_hour":{"utilization":17.0,"resets_at":"2026-07-17T16:50:00+00:00"},
		"seven_day":{"utilization":8.0,"resets_at":"2026-07-21T18:00:00+00:00"},
		"limits":[{"kind":"session","percent":17}]
	}`)
	q, err := parseClaudeQuota(body)
	if err != nil {
		t.Fatalf("parseClaudeQuota: %v", err)
	}
	if !q.OK || q.FiveHourPct != 17.0 || q.SevenDayPct != 8.0 {
		t.Errorf("quota = %+v, want 17/8", q)
	}
	if q.FiveHourReset == nil || q.SevenDayReset == nil {
		t.Error("reset times = nil, want parsed")
	}

	// Missing/null fields degrade to zero/nil, not an error.
	partial, err := parseClaudeQuota([]byte(`{"five_hour":{"utilization":null}}`))
	if err != nil {
		t.Fatalf("parseClaudeQuota(partial): %v", err)
	}
	if !partial.OK || partial.FiveHourPct != 0 || partial.FiveHourReset != nil || partial.SevenDayReset != nil {
		t.Errorf("partial = %+v, want zeroed + OK", partial)
	}

	if _, err := parseClaudeQuota([]byte("nope")); err == nil {
		t.Error("garbage = nil error, want error")
	}
}

// Contract: fetchClaudeQuota sends the bearer token + oauth beta header on 200,
// and returns OK=false for an empty token or any non-200 / transport failure.
func TestFetchClaudeQuota(t *testing.T) {
	var gotAuth, gotBeta, mode string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotBeta = r.Header.Get("anthropic-beta")
		switch mode {
		case "ok":
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, `{"five_hour":{"utilization":17.0,"resets_at":"2026-07-17T16:50:00+00:00"},"seven_day":{"utilization":8.0,"resets_at":"2026-07-21T18:00:00+00:00"}}`)
		default:
			w.WriteHeader(http.StatusUnauthorized)
		}
	}))
	defer ts.Close()

	mode = "ok"
	q := fetchClaudeQuota(ts.URL, "tok123")
	if !q.OK || q.FiveHourPct != 17.0 {
		t.Fatalf("quota = %+v, want OK 17%%", q)
	}
	if gotAuth != "Bearer tok123" {
		t.Errorf("Authorization = %q, want Bearer tok123", gotAuth)
	}
	if gotBeta != claudeOAuthBetaHeader {
		t.Errorf("anthropic-beta = %q, want %q", gotBeta, claudeOAuthBetaHeader)
	}

	if q := fetchClaudeQuota(ts.URL, ""); q.OK {
		t.Error("empty token = OK, want OK=false (no request made)")
	}
	mode = "401"
	if q := fetchClaudeQuota(ts.URL, "tok"); q.OK {
		t.Error("401 = OK, want OK=false")
	}
}

func TestPushSparkSampleBounded(t *testing.T) {
	var ring []float64
	for i := 0; i < sparkMaxSamples+10; i++ {
		ring = pushSparkSample(ring, float64(i))
	}
	if len(ring) != sparkMaxSamples {
		t.Fatalf("ring len = %d, want %d", len(ring), sparkMaxSamples)
	}
	if ring[len(ring)-1] != float64(sparkMaxSamples+9) {
		t.Errorf("newest sample lost: last = %v", ring[len(ring)-1])
	}
}

func TestSparklineIconPNG(t *testing.T) {
	cases := map[string][]float64{
		"empty":    nil,
		"single":   {50},
		"flat":     {80, 80, 80},
		"varied":   {10, 90, 40, 70, 20, 100},
		"twoOnly":  {0, 100},
		"negative": {-5, 3},
	}
	for name, samples := range cases {
		b := sparklineIconPNG(samples, levelSparkColor(levelGood))
		if b == nil {
			t.Fatalf("%s: nil PNG", name)
		}
		img, err := png.Decode(bytes.NewReader(b))
		if err != nil {
			t.Fatalf("%s: invalid PNG: %v", name, err)
		}
		if got := img.Bounds().Dx(); got != sparkIconSize {
			t.Errorf("%s: width = %d, want %d", name, got, sparkIconSize)
		}
	}
}
