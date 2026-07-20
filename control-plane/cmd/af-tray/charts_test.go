package main

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// These tests pin the image renderers and the series parsing they consume. Like
// the rest of the af-tray suite they are GUI/CGO-free so they run on Linux CI
// even though the menu itself is macOS-only.

// ---- series parsing --------------------------------------------------------

// Contract: a present "series" key parses into ascending points with tolerant
// null cost, and hasSeries()/showUsageChart() report true.
func TestParseUsageStatsSeries(t *testing.T) {
	body := []byte(`{
		"window":"24h",
		"totals":{"cost_usd":1.0,"total_tokens":3000,"executions_with_usage":2},
		"by_model":[],"by_provider":[],"by_harness":[],
		"series":{"bucket_seconds":1800,"points":[
			{"t":"2026-07-17T00:00:00Z","total_tokens":100,"cost_usd":0.01},
			{"t":"2026-07-17T00:30:00Z","total_tokens":0,"cost_usd":null},
			{"t":"2026-07-17T01:00:00Z","total_tokens":2900,"cost_usd":0.99}
		]}
	}`)
	u, err := parseUsageStats(body)
	if err != nil {
		t.Fatalf("parseUsageStats: %v", err)
	}
	if u.BucketSeconds != 1800 {
		t.Errorf("BucketSeconds = %d, want 1800", u.BucketSeconds)
	}
	if len(u.Series) != 3 {
		t.Fatalf("Series len = %d, want 3", len(u.Series))
	}
	if u.Series[0].TotalTokens != 100 || u.Series[2].TotalTokens != 2900 {
		t.Errorf("series tokens = %d..%d, want 100..2900", u.Series[0].TotalTokens, u.Series[2].TotalTokens)
	}
	if u.Series[1].CostUSD != nil {
		t.Errorf("middle cost = %v, want nil (null tolerated)", u.Series[1].CostUSD)
	}
	if u.Series[0].CostUSD == nil || *u.Series[0].CostUSD != 0.01 {
		t.Errorf("first cost = %v, want 0.01", u.Series[0].CostUSD)
	}
	if u.Series[0].T.IsZero() {
		t.Error("first timestamp not parsed")
	}
	if !u.hasSeries() {
		t.Error("hasSeries() = false, want true")
	}
	if !showUsageChart(u) {
		t.Error("showUsageChart() = false, want true (has data + series)")
	}
}

// Contract: an absent "series" key (older server) leaves Series nil, hasSeries()
// false, and showUsageChart() false — the chart row is hidden and everything
// else works exactly as before.
func TestParseUsageStatsSeriesAbsent(t *testing.T) {
	body := []byte(`{
		"window":"24h",
		"totals":{"cost_usd":1.0,"total_tokens":3000,"executions_with_usage":2},
		"by_model":[],"by_provider":[],"by_harness":[]
	}`)
	u, err := parseUsageStats(body)
	if err != nil {
		t.Fatalf("parseUsageStats: %v", err)
	}
	if u.Series != nil {
		t.Errorf("Series = %v, want nil when key absent", u.Series)
	}
	if u.hasSeries() {
		t.Error("hasSeries() = true, want false when series absent")
	}
	if showUsageChart(u) {
		t.Error("showUsageChart() = true, want false when series absent")
	}
	if !u.hasData() {
		t.Error("hasData() = false, want true (unchanged behavior)")
	}
}

// Contract: garbage series does not crash the whole parse (the outer object is
// still garbage → error), and a present-but-empty series yields no chart.
func TestParseUsageStatsSeriesEmptyAndGarbage(t *testing.T) {
	empty := []byte(`{"window":"24h","totals":{"total_tokens":5,"executions_with_usage":1},"series":{"bucket_seconds":1800,"points":[]}}`)
	u, err := parseUsageStats(empty)
	if err != nil {
		t.Fatalf("parseUsageStats(empty series): %v", err)
	}
	if u.hasSeries() || showUsageChart(u) {
		t.Error("empty series should not show a chart")
	}
	if _, err := parseUsageStats([]byte("not json")); err == nil {
		t.Error("garbage = nil error, want error")
	}
}

// Contract: showUsageChart requires both data and a series; a series with data
// present but all-zero points still charts (flat baseline), which the renderer
// handles.
func TestShowUsageChart(t *testing.T) {
	c := 1.0
	withData := usageStats{Status: usageOK, CostUSD: &c, TotalTokens: 10, Series: []seriesPoint{{TotalTokens: 10}}}
	if !showUsageChart(withData) {
		t.Error("data + series should show chart")
	}
	noSeries := usageStats{Status: usageOK, TotalTokens: 10}
	if showUsageChart(noSeries) {
		t.Error("no series should not show chart")
	}
	noData := usageStats{Status: usageOK, TotalTokens: 0, Series: []seriesPoint{{TotalTokens: 0}}}
	if showUsageChart(noData) {
		t.Error("no data should not show chart")
	}
}

// ---- series_by_model parsing ----------------------------------------------

// Contract: a present "series_by_model" parses into per-model layers (top models
// + a final "other"), hasSeriesByModel()/showStackedChart() report true, and the
// tray prefers the stacked chart.
func TestParseUsageStatsSeriesByModel(t *testing.T) {
	body := []byte(`{
		"window":"24h",
		"totals":{"cost_usd":1.0,"total_tokens":3000,"executions_with_usage":2},
		"by_model":[
			{"key":"claude-opus-4-8","total_tokens":2000},
			{"key":"gpt-4o","total_tokens":1000}
		],
		"by_provider":[],"by_harness":[],
		"series":{"bucket_seconds":1800,"points":[
			{"t":"2026-07-17T00:00:00Z","total_tokens":100},
			{"t":"2026-07-17T00:30:00Z","total_tokens":2900}
		]},
		"series_by_model":[
			{"key":"claude-opus-4-8","points":[{"total_tokens":60},{"total_tokens":1900}]},
			{"key":"gpt-4o","points":[{"total_tokens":40},{"total_tokens":800}]},
			{"key":"other","points":[{"total_tokens":0},{"total_tokens":200}]}
		]
	}`)
	u, err := parseUsageStats(body)
	if err != nil {
		t.Fatalf("parseUsageStats: %v", err)
	}
	if len(u.SeriesByModel) != 3 {
		t.Fatalf("SeriesByModel len = %d, want 3", len(u.SeriesByModel))
	}
	if u.SeriesByModel[0].Key != "claude-opus-4-8" || u.SeriesByModel[2].Key != "other" {
		t.Errorf("keys = %q..%q, want opus..other", u.SeriesByModel[0].Key, u.SeriesByModel[2].Key)
	}
	if got := u.SeriesByModel[0].Tokens; len(got) != 2 || got[1] != 1900 {
		t.Errorf("first model tokens = %v, want [60 1900]", got)
	}
	if !u.hasSeriesByModel() {
		t.Error("hasSeriesByModel() = false, want true")
	}
	if !showStackedChart(u) {
		t.Error("showStackedChart() = false, want true")
	}
}

// Contract: absent "series_by_model" (older server) leaves SeriesByModel nil,
// hasSeriesByModel() false, showStackedChart() false — the tray falls back to the
// single-series chart, which stays available.
func TestParseUsageStatsSeriesByModelAbsent(t *testing.T) {
	body := []byte(`{
		"window":"24h",
		"totals":{"total_tokens":3000,"executions_with_usage":2},
		"by_model":[],"by_provider":[],"by_harness":[],
		"series":{"bucket_seconds":1800,"points":[{"total_tokens":100},{"total_tokens":200}]}
	}`)
	u, err := parseUsageStats(body)
	if err != nil {
		t.Fatalf("parseUsageStats: %v", err)
	}
	if u.SeriesByModel != nil {
		t.Errorf("SeriesByModel = %v, want nil when absent", u.SeriesByModel)
	}
	if u.hasSeriesByModel() || showStackedChart(u) {
		t.Error("stacked chart should be off when series_by_model absent")
	}
	if !showUsageChart(u) {
		t.Error("single-series fallback should still be available")
	}
}

// Contract: an empty series_by_model, and an all-empty-points model, degrade to
// no stacked chart rather than crashing.
func TestParseUsageStatsSeriesByModelEmpty(t *testing.T) {
	empty := []byte(`{"window":"24h","totals":{"total_tokens":5,"executions_with_usage":1},"series_by_model":[]}`)
	u, err := parseUsageStats(empty)
	if err != nil {
		t.Fatalf("parseUsageStats(empty): %v", err)
	}
	if u.hasSeriesByModel() || showStackedChart(u) {
		t.Error("empty series_by_model should not show a stacked chart")
	}
	blank := []byte(`{"window":"24h","totals":{"total_tokens":5,"executions_with_usage":1},"series_by_model":[{"key":"other","points":[]}]}`)
	u2, err := parseUsageStats(blank)
	if err != nil {
		t.Fatalf("parseUsageStats(blank): %v", err)
	}
	if u2.hasSeriesByModel() {
		t.Error("a model with no points should not count as a stacked series")
	}
}

// Contract: stackedChartData assembles bottom-to-top layers with the accent at
// each rank's intensity and neutral gray for the "other" bucket.
func TestStackedChartData(t *testing.T) {
	u := usageStats{
		Status:      usageOK,
		TotalTokens: 10,
		Series: []seriesPoint{
			{T: time.Date(2026, 7, 17, 8, 0, 0, 0, time.Local)},
			{T: time.Date(2026, 7, 17, 20, 0, 0, 0, time.Local)},
		},
		SeriesByModel: []modelSeries{
			{Key: "claude-opus-4-8", Tokens: []float64{100, 41000}},
			{Key: "gpt-4o", Tokens: []float64{50, 60000}},
			{Key: "other", Tokens: []float64{0, 40000}},
		},
	}
	layers, colors := stackedChartData(u)
	if len(layers) != 3 || len(colors) != 3 {
		t.Fatalf("layers/colors = %d/%d, want 3/3", len(layers), len(colors))
	}
	if colors[0] != modelBarColor(0) || colors[1] != modelBarColor(1) {
		t.Error("ranked layers should take rank hues")
	}
	if colors[2] != grayOther {
		t.Errorf("other layer color = %v, want grayOther", colors[2])
	}
	// Layers are zero-filled to the common bucket count and carry the raw tokens.
	if len(layers[0]) != 2 || layers[0][1] != 41000 {
		t.Errorf("layer[0] = %v, want length 2 ending 41000", layers[0])
	}
}

func TestSeriesTokenValues(t *testing.T) {
	pts := []seriesPoint{{TotalTokens: 1}, {TotalTokens: 0}, {TotalTokens: 42}}
	got := seriesTokenValues(pts)
	want := []float64{1, 0, 42}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("value[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// ---- no-bar titles + share -------------------------------------------------

func TestUsageModelTitle(t *testing.T) {
	c := 0.9
	g := usageGroup{Key: "claude-opus-4-8", CostUSD: &c, TotalTokens: 1_200_000}
	if got, want := usageModelTitle(g), "opus-4-8 — 1.2M · $0.90"; got != want {
		t.Errorf("usageModelTitle() = %q, want %q", got, want)
	}
}

func TestModelShare(t *testing.T) {
	g := usageGroup{TotalTokens: 1_200_000}
	if got := modelShare(g, 2_400_000); math.Abs(got-0.5) > 1e-9 {
		t.Errorf("modelShare() = %v, want 0.5", got)
	}
	if got := modelShare(g, 0); got != 0 {
		t.Errorf("modelShare(total=0) = %v, want 0", got)
	}
}

func TestClaudeQuotaTitle(t *testing.T) {
	now := time.Date(2026, 7, 17, 12, 0, 0, 0, time.Local)
	reset := time.Date(2026, 7, 17, 15, 0, 0, 0, time.Local)
	if got, want := claudeQuotaTitle("Claude 5h", 24, &reset, now), "Claude 5h — 24% (resets 3:00 PM)"; got != want {
		t.Errorf("claudeQuotaTitle() = %q, want %q", got, want)
	}
	if got, want := claudeQuotaTitle("Claude 7d", 8, nil, now), "Claude 7d — 8%"; got != want {
		t.Errorf("claudeQuotaTitle(no reset) = %q, want %q", got, want)
	}
	if got, want := claudeQuotaTitle("Claude 5h", 150, nil, now), "Claude 5h — 100%"; got != want {
		t.Errorf("claudeQuotaTitle(over) = %q, want %q", got, want)
	}
}

// ---- palette ---------------------------------------------------------------

// Contract: one accent, ranked by intensity. Every rank is the same accent hue;
// rank 0 is full strength and intensity never rises with rank; the top three
// ranks are distinguishable; and the ramp is never fully transparent.
func TestModelBarColor(t *testing.T) {
	// Same accent hue at every rank.
	for rank := 0; rank < 8; rank++ {
		c := modelBarColor(rank)
		if c.R != accentColor.R || c.G != accentColor.G || c.B != accentColor.B {
			t.Errorf("rank %d hue = (%d,%d,%d), want accent (%d,%d,%d)",
				rank, c.R, c.G, c.B, accentColor.R, accentColor.G, accentColor.B)
		}
	}
	// Rank 0 is full strength.
	if got := modelBarColor(0).A; got != 0xff {
		t.Errorf("rank 0 alpha = %d, want 255 (full strength)", got)
	}
	// Intensity never rises with rank.
	prev := modelBarColor(0).A
	for rank := 1; rank < 8; rank++ {
		a := modelBarColor(rank).A
		if a > prev {
			t.Errorf("rank %d alpha %d > rank %d alpha %d (intensity should not rise)", rank, a, rank-1, prev)
		}
		prev = a
	}
	// Top-3 intensities are distinct.
	a0, a1, a2 := modelBarColor(0).A, modelBarColor(1).A, modelBarColor(2).A
	if a0 == a1 || a1 == a2 || a0 == a2 {
		t.Errorf("top-3 intensities not distinct: %d/%d/%d", a0, a1, a2)
	}
	// Never colorless, at any rank.
	for r := 0; r < 20; r++ {
		if modelBarColor(r).A == 0 {
			t.Errorf("rank %d is fully transparent", r)
		}
	}
}

func TestQuotaBarColor(t *testing.T) {
	green := color.NRGBA{0x30, 0xd1, 0x58, 0xff}
	orange := color.NRGBA{0xe0, 0x8a, 0x3c, 0xff}
	red := color.NRGBA{0xff, 0x45, 0x3a, 0xff}
	cases := []struct {
		pct  float64
		want color.NRGBA
	}{
		{0, green}, {49.9, green}, {50, orange}, {79.9, orange}, {80, red}, {100, red},
	}
	for _, tc := range cases {
		if got := quotaBarColor(tc.pct); got != tc.want {
			t.Errorf("quotaBarColor(%v) = %v, want %v", tc.pct, got, tc.want)
		}
	}
}

// ---- chart renderer --------------------------------------------------------

func decodePNG(t *testing.T, b []byte) *image.NRGBA {
	t.Helper()
	if b == nil {
		t.Fatal("nil PNG")
	}
	img, err := png.Decode(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("invalid PNG: %v", err)
	}
	out := image.NewNRGBA(img.Bounds())
	for y := img.Bounds().Min.Y; y < img.Bounds().Max.Y; y++ {
		for x := img.Bounds().Min.X; x < img.Bounds().Max.X; x++ {
			out.Set(x, y, img.At(x, y))
		}
	}
	return out
}

func nonTransparentCount(img *image.NRGBA) int {
	n := 0
	b := img.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			if img.NRGBAAt(x, y).A > 8 {
				n++
			}
		}
	}
	return n
}

// ---- uniform slot: spacer + compact bar ------------------------------------

// barFillFraction scans the bar's mid-height row within the given bar width and
// returns the fraction of that width covered by opaque fill (alpha>128; the
// neutral track is faint).
func barFillFraction(img *image.NRGBA, barWPx int) float64 {
	b := img.Bounds()
	y := b.Min.Y + b.Dy()/2
	end := 0
	for x := 0; x < barWPx && x < b.Dx(); x++ {
		if img.NRGBAAt(b.Min.X+x, y).A > 128 {
			end = x + 1
		}
	}
	return float64(end) / float64(barWPx)
}

// absU8 is the absolute difference of two bytes.
func absU8(a, b uint8) int {
	if a > b {
		return int(a - b)
	}
	return int(b - a)
}

// inkPixels counts non-transparent (drawn) pixels in a sub-rectangle.
func inkPixels(img *image.NRGBA, x0, y0, x1, y1 int) int {
	n := 0
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			if img.NRGBAAt(x, y).A > 40 {
				n++
			}
		}
	}
	return n
}

// Contract: the spacer is the uniform slot size and fully transparent, so it
// reserves the leading slot without drawing anything.
func TestSpacerImagePNG(t *testing.T) {
	const w, h = usageSlotWidthPt * 2, usageSlotHeightPt * 2
	img := decodePNG(t, spacerImagePNG(w, h))
	if img.Bounds().Dx() != w || img.Bounds().Dy() != h {
		t.Fatalf("dims = %dx%d, want %dx%d", img.Bounds().Dx(), img.Bounds().Dy(), w, h)
	}
	if n := nonTransparentCount(img); n != 0 {
		t.Errorf("spacer has %d non-transparent pixels, want 0", n)
	}
}

// Contract: every row's leading image is EXACTLY the uniform slot size, so every
// native title starts at the same x. The histogram is the single wider graphic
// row; its left edge still aligns with the slot but it spans the row content
// width. This is the geometry the menu mock renders and eyeballs.
func TestUniformSlotWidths(t *testing.T) {
	const sW, sH = usageSlotWidthPt * 2, usageSlotHeightPt * 2
	const bW, bH = usageBarWidthPt * 2, usageBarHeightPt * 2
	slots := map[string][]byte{
		"spacer":      spacerImagePNG(sW, sH),
		"model_bar":   slotBarPNG(0.7, modelBarColor(0), sW, sH, bW, bH),
		"quota_gauge": slotBarPNG(0.4, quotaBarColor(40), sW, sH, bW, bH),
	}
	for name, b := range slots {
		img := decodePNG(t, b)
		if img.Bounds().Dx() != sW || img.Bounds().Dy() != sH {
			t.Errorf("%s: leading image = %dx%d, want %dx%d (every row's slot must be identical)",
				name, img.Bounds().Dx(), img.Bounds().Dy(), sW, sH)
		}
	}
	hist := decodePNG(t, histogramChartPNG(
		[][]float64{makeBumpySeries(48)}, []color.NRGBA{modelBarColor(0)},
		usageChartWidthPt*2, usageChartHeightPt*2))
	if hist.Bounds().Dx() != usageChartWidthPt*2 || hist.Bounds().Dy() != usageChartHeightPt*2 {
		t.Errorf("histogram = %dx%d, want %dx%d", hist.Bounds().Dx(), hist.Bounds().Dy(),
			usageChartWidthPt*2, usageChartHeightPt*2)
	}
}

func TestSlotBarPNG(t *testing.T) {
	const sW, sH = usageSlotWidthPt * 2, usageSlotHeightPt * 2 // 128x24
	const bW, bH = usageBarWidthPt * 2, usageBarHeightPt * 2   // 112x16
	fill := modelBarColor(0)                                   // full-strength accent for geometry

	// Dimensions match the SLOT, not the bar.
	for _, frac := range []float64{0, 0.25, 0.5, 1} {
		img := decodePNG(t, slotBarPNG(frac, fill, sW, sH, bW, bH))
		if img.Bounds().Dx() != sW || img.Bounds().Dy() != sH {
			t.Fatalf("frac %v: dims = %dx%d, want %dx%d", frac, img.Bounds().Dx(), img.Bounds().Dy(), sW, sH)
		}
	}

	img := decodePNG(t, slotBarPNG(1.0, fill, sW, sH, bW, bH))
	// The slot is transparent to the right of the bar (uniform leading padding),
	// so the title always starts at the same x.
	for y := 0; y < sH; y++ {
		for x := bW + 2; x < sW; x++ {
			if img.NRGBAAt(x, y).A > 8 {
				t.Fatalf("slot painted at x=%d past bar width %d — not left-aligned in slot", x, bW)
			}
		}
	}
	// The bar is vertically centered: the very top and bottom slot rows are clear.
	for x := 0; x < bW; x++ {
		if img.NRGBAAt(x, 0).A > 8 || img.NRGBAAt(x, sH-1).A > 8 {
			t.Fatalf("bar touches slot edge at x=%d — not vertically centered", x)
		}
	}

	// Fill proportion within the BAR region ≈ fraction (sampled at mid-height).
	for _, frac := range []float64{0.25, 0.5, 1.0} {
		im := decodePNG(t, slotBarPNG(frac, fill, sW, sH, bW, bH))
		got := barFillFraction(im, bW)
		if math.Abs(got-frac) > 0.06 {
			t.Errorf("frac %v: measured fill = %.3f, want ≈ %v (±0.06)", frac, got, frac)
		}
	}
	if got := barFillFraction(decodePNG(t, slotBarPNG(0, fill, sW, sH, bW, bH)), bW); got > 0.03 {
		t.Errorf("frac 0: measured fill = %.3f, want ~0", got)
	}

	// Rank hue + intensity: a mid-fill pixel matches the rank's accent tint,
	// including its rank-dependent opacity.
	for rank := 0; rank < 3; rank++ {
		want := modelBarColor(rank)
		im := decodePNG(t, slotBarPNG(0.9, want, sW, sH, bW, bH))
		px := im.NRGBAAt(12, sH/2) // well inside the fill
		if !nearColor(px, want, 8) || absU8(px.A, want.A) > 24 {
			t.Errorf("rank %d: mid fill = %v, want ≈ %v", rank, px, want)
		}
	}
}

// ---- histogram renderer ----------------------------------------------------

func TestHistogramChartPNG(t *testing.T) {
	const w, h = usageChartWidthPt * 2, usageChartHeightPt * 2 // 400x56
	bucketW := float64(w) / 48

	// Dimensions (single-model / single accent hue).
	img := decodePNG(t, histogramChartPNG(
		[][]float64{makeBumpySeries(48)}, []color.NRGBA{modelBarColor(0)}, w, h))
	if img.Bounds().Dx() != w || img.Bounds().Dy() != h {
		t.Fatalf("dims = %dx%d, want %dx%d", img.Bounds().Dx(), img.Bounds().Dy(), w, h)
	}

	// Sparse: one nonzero bucket among 48. The spike rises high; the other buckets
	// keep a baseline stub, so the timeline reads as continuous rather than broken.
	sparse := make([]float64, 48)
	sparse[24] = 1000
	sp := decodePNG(t, histogramChartPNG([][]float64{sparse}, []color.NRGBA{modelBarColor(0)}, w, h))
	if inkPixels(sp, 0, 0, w, h/4) == 0 {
		t.Error("sparse: the lone spike did not rise into the top quarter")
	}
	stubCols := 0
	for x := 0; x < w; x++ {
		if sp.NRGBAAt(x, h-1).A > 8 {
			stubCols++
		}
	}
	if stubCols < w/2 {
		t.Errorf("sparse: baseline stubs cover %d/%d columns, want ≥ half (continuous timeline)", stubCols, w)
	}
	// A bucket far from the spike is empty above its baseline stub.
	for y := 0; y < h-4; y++ {
		if sp.NRGBAAt(int(bucketW*2)+2, y).A > 40 {
			t.Fatalf("sparse: an empty bucket painted above the baseline at y=%d", y)
		}
	}

	// All-zero: only baseline stubs — nothing painted in the top half.
	zero := decodePNG(t, histogramChartPNG([][]float64{repeatFloat(0, 48)}, []color.NRGBA{modelBarColor(0)}, w, h))
	for y := 0; y < h/2; y++ {
		for x := 0; x < w; x++ {
			if zero.NRGBAAt(x, y).A > 40 {
				t.Fatalf("all-zero histogram painted at y=%d (should be baseline stubs only)", y)
			}
		}
	}

	// Multi-model stacking order: layers[0] (full accent) at the BOTTOM, grayOther
	// on top. In a full-height bucket the bottom reads accent, the top reads gray.
	colors := []color.NRGBA{modelBarColor(0), modelBarColor(1), modelBarColor(2), grayOther}
	layers := [][]float64{
		repeatFloat(1000, 48), // rank 0 accent (bottom)
		repeatFloat(600, 48),  // rank 1
		repeatFloat(300, 48),  // rank 2
		repeatFloat(400, 48),  // other, gray (top)
	}
	ms := decodePNG(t, histogramChartPNG(layers, colors, w, h))
	col := int(bucketW*24) + 2 // safely inside a bar, not a gap
	bottomPx := ms.NRGBAAt(col, h-3)
	if !nearColor(bottomPx, accentColor, 20) || bottomPx.A < 200 {
		t.Errorf("stack bottom = %v, want full-strength accent (rank 0 at the bottom)", bottomPx)
	}
	topPx := ms.NRGBAAt(col, 5)
	if !nearColor(topPx, grayOther, 30) {
		t.Errorf("stack top = %v, want grayOther (other bucket on top)", topPx)
	}
}

// ---- menu-bar widget + trend ----------------------------------------------

// synthBadge builds a tiny opaque square PNG to stand in for the brand badge in
// GUI-free tests (the real asset is darwin-embedded).
func synthBadge(t *testing.T, n int, c color.NRGBA) []byte {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, n, n))
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			img.SetNRGBA(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func nearColor(a, b color.NRGBA, tol int) bool {
	d := func(x, y uint8) int {
		v := int(x) - int(y)
		if v < 0 {
			return -v
		}
		return v
	}
	return d(a.R, b.R) <= tol && d(a.G, b.G) <= tol && d(a.B, b.B) <= tol
}

// ---- helpers ---------------------------------------------------------------

func repeatFloat(v float64, n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = v
	}
	return out
}

// scaleSeries returns a copy of s with every value multiplied by k.
func scaleSeries(s []float64, k float64) []float64 {
	out := make([]float64, len(s))
	for i, v := range s {
		out[i] = v * k
	}
	return out
}

// shiftSeries returns a copy of s with its humps shifted by `by` buckets, so a
// stacked set of layers has visibly different silhouettes.
func shiftSeries(s []float64, by int) []float64 {
	out := make([]float64, len(s))
	for i := range s {
		j := i - by
		if j < 0 {
			j += len(s)
		}
		out[i] = s[j%len(s)]
	}
	return out
}

// makeBumpySeries builds a realistic-looking token series: a couple of humps of
// activity with quiet gaps, so the dumped chart looks like real usage.
func makeBumpySeries(n int) []float64 {
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		x := float64(i)
		v := 1800*math.Exp(-math.Pow((x-12)/5, 2)) + // morning hump
			3200*math.Exp(-math.Pow((x-32)/6, 2)) + // afternoon hump
			300*math.Sin(x/2.2) + 400 // gentle ripple + floor
		if v < 0 {
			v = 0
		}
		out[i] = v
	}
	return out
}

// ---- env-gated PNG dump ----------------------------------------------------
//
// Kept in-tree but skipped by default. Run it to eyeball the renders:
//
//	AF_TRAY_DUMP=1 go test ./cmd/af-tray/ -run TestDumpCharts
//
// Output goes to $AF_TRAY_DUMP_DIR (default: ./_chart_dump). The submenu graphics
// are composited on BOTH a dark and a light mock menu background (macOS menus
// follow system appearance and systray can't detect it). The menu-bar widget is
// composited onto BOTH a dark and light mock menu-bar strip, alongside a couple
// of fake neighbor icons, so its legibility and crowding can be judged the way it
// will actually read in the system menu bar.
func TestDumpCharts(t *testing.T) {
	if os.Getenv("AF_TRAY_DUMP") == "" {
		t.Skip("set AF_TRAY_DUMP=1 to dump chart PNGs")
	}
	dir := os.Getenv("AF_TRAY_DUMP_DIR")
	if dir == "" {
		dir = "_chart_dump"
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}

	menuDark := color.NRGBA{0x2b, 0x34, 0x40, 0xff}  // ~ macOS dark menu
	menuLight := color.NRGBA{0xec, 0xec, 0xec, 0xff} // ~ macOS light menu

	writePNG := func(name string, img image.Image) {
		var buf bytes.Buffer
		if err := png.Encode(&buf, img); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), buf.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// dumpBoth composites the PNG on both menu backgrounds and writes _dark/_light.
	dumpBoth := func(name string, b []byte) {
		if b == nil {
			t.Fatalf("%s: nil PNG", name)
		}
		writePNG(name+"_dark.png", compositeOn(t, b, menuDark))
		writePNG(name+"_light.png", compositeOn(t, b, menuLight))
	}

	// --- Renders at 2x (retina) pixel dimensions, matching the tray. ---
	const slotW, slotH = usageSlotWidthPt * 2, usageSlotHeightPt * 2     // uniform leading slot (128x24)
	const barW, barH = usageBarWidthPt * 2, usageBarHeightPt * 2         // compact bar inside slot (112x16)
	const chartW, chartH = usageChartWidthPt * 2, usageChartHeightPt * 2 // histogram (400x56)

	// Realistic bumpy multi-model data with differently-shifted humps.
	base := makeBumpySeries(48)
	layers := [][]float64{
		base,
		scaleSeries(shiftSeries(base, 8), 0.7),
		scaleSeries(shiftSeries(base, 18), 0.4),
		repeatFloat(350, 48),
	}
	colors := []color.NRGBA{modelBarColor(0), modelBarColor(1), modelBarColor(2), grayOther}

	// Histogram: dense (multi-model), single-hue, sparse (one lone bucket), flat.
	dense := histogramChartPNG(layers, colors, chartW, chartH)
	dumpBoth("histogram_dense", dense)
	dumpBoth("histogram_single", histogramChartPNG(
		[][]float64{base}, []color.NRGBA{modelBarColor(0)}, chartW, chartH))
	sparse := make([]float64, 48)
	sparse[31] = 5200
	dumpBoth("histogram_sparse", histogramChartPNG(
		[][]float64{sparse}, []color.NRGBA{modelBarColor(0)}, chartW, chartH))
	dumpBoth("histogram_flat", histogramChartPNG(
		[][]float64{repeatFloat(0, 48)}, []color.NRGBA{modelBarColor(0)}, chartW, chartH))

	// Uniform slot: transparent spacer + compact model bars (accent intensity ramp)
	// + Claude quota gauges (semantic green/amber/red). All identical slot size.
	dumpBoth("slot_spacer", spacerImagePNG(slotW, slotH))
	dumpBoth("model_bar0", slotBarPNG(0.78, modelBarColor(0), slotW, slotH, barW, barH))
	dumpBoth("model_bar1", slotBarPNG(0.46, modelBarColor(1), slotW, slotH, barW, barH))
	dumpBoth("model_bar2", slotBarPNG(0.19, modelBarColor(2), slotW, slotH, barW, barH))
	dumpBoth("quota_bar_24", slotBarPNG(0.24, quotaBarColor(24), slotW, slotH, barW, barH))
	dumpBoth("quota_bar_67", slotBarPNG(0.67, quotaBarColor(67), slotW, slotH, barW, barH))
	dumpBoth("quota_bar_91", slotBarPNG(0.91, quotaBarColor(91), slotW, slotH, barW, barH))

	// --- Full menu mock: every row is [uniform slot][native-title placeholder],
	// stacked in submenu order, so the regularity of the left edge can be judged
	// visually. The placeholder's left edge is identical on every row by
	// construction (it is asserted pixel-exact in TestUniformSlotWidths). ---
	spacer := spacerImagePNG(slotW, slotH)
	rows := []mockRow{
		{slot: dense, isChart: true},
		{slot: spacer, placeholderW: 150},              // 7d summary line
		{slot: spacer, placeholderW: 92, header: true}, // "Top models"
		{slot: slotBarPNG(0.78, modelBarColor(0), slotW, slotH, barW, barH), placeholderW: 150},
		{slot: slotBarPNG(0.46, modelBarColor(1), slotW, slotH, barW, barH), placeholderW: 140},
		{slot: slotBarPNG(0.19, modelBarColor(2), slotW, slotH, barW, barH), placeholderW: 132},
		{slot: spacer, placeholderW: 84, header: true},  // "Providers"
		{slot: spacer, placeholderW: 156},               // provider rollup
		{slot: spacer, placeholderW: 150, header: true}, // "Claude subscription"
		{slot: slotBarPNG(0.67, quotaBarColor(67), slotW, slotH, barW, barH), placeholderW: 176},
		{slot: slotBarPNG(0.24, quotaBarColor(24), slotW, slotH, barW, barH), placeholderW: 176},
		{slot: spacer, placeholderW: 120}, // footer
	}
	// text/header placeholder tints per background (mimic native menu text).
	writePNG("menu_mock_dark.png", buildMenuMock(t, rows, menuDark, chartW,
		color.NRGBA{0xe6, 0xe8, 0xec, 0xff}, color.NRGBA{0x9a, 0x9d, 0xa4, 0xff}))
	writePNG("menu_mock_light.png", buildMenuMock(t, rows, menuLight, chartW,
		color.NRGBA{0x2a, 0x2c, 0x30, 0xff}, color.NRGBA{0x6c, 0x70, 0x78, 0xff}))

	// --- Menu-bar status badge variants on both strips. ---
	badge, err := os.ReadFile("assets/icon_active.png")
	if err != nil || len(badge) == 0 {
		badge = synthBadge(t, 44, color.NRGBA{0xe0, 0x8a, 0x3c, 0xff})
	}
	states := []struct {
		name  string
		state serverState
		phase int
	}{
		{"status_running", serverRunning, 0},
		{"status_starting_p0", serverStarting, 0},
		{"status_starting_p3", serverStarting, 3},
		{"status_stopped", serverStopped, 0},
	}
	stripDark := color.NRGBA{0x1c, 0x1f, 0x24, 0xff}
	stripLight := color.NRGBA{0xe8, 0xe8, 0xea, 0xff}
	neighborOnDark := color.NRGBA{0xd0, 0xd2, 0xd6, 0xff}
	neighborOnLight := color.NRGBA{0x3a, 0x3a, 0x3c, 0xff}
	for _, sv := range states {
		img := statusBadgePNG(badge, sv.state, sv.phase)
		if img == nil {
			t.Fatalf("%s: nil status badge PNG", sv.name)
		}
		writePNG(sv.name+"_dark.png", buildMenuBarStrip(t, img, stripDark, neighborOnDark))
		writePNG(sv.name+"_light.png", buildMenuBarStrip(t, img, stripLight, neighborOnLight))
	}

	t.Logf("dumped composited chart + menu-bar PNGs to %s", dir)
}

// buildMenuBarStrip composites the menu-bar widget onto a mock menu-bar strip
// with a couple of fake neighbor icons (gray squares) so crowding and legibility
// can be judged as they would read in the real menu bar. A blank gap is left to
// the widget's right where the native cost/token title (SetTitle) would appear.
func buildMenuBarStrip(t *testing.T, widget []byte, bg, neighbor color.NRGBA) image.Image {
	t.Helper()
	w := decodePNG(t, widget)
	const (
		h          = 44 // 22pt menu bar @2x
		pad        = 14
		icon       = 30
		gap        = 18
		titleSpace = 72 // blank space where the native "$0.99" title renders
	)
	total := pad + 2*(icon+gap) + w.Bounds().Dx() + titleSpace + gap + icon + pad
	out := image.NewNRGBA(image.Rect(0, 0, total, h))
	for y := 0; y < h; y++ {
		for x := 0; x < total; x++ {
			out.SetNRGBA(x, y, bg)
		}
	}
	drawSquare := func(x0 int) {
		y0 := (h - icon) / 2
		for y := 0; y < icon; y++ {
			for x := 0; x < icon; x++ {
				// Soften the corners so the placeholder reads as an icon, not a block.
				if (x < 4 || x >= icon-4) && (y < 4 || y >= icon-4) {
					continue
				}
				out.SetNRGBA(x0+x, y0+y, neighbor)
			}
		}
	}
	x := pad
	drawSquare(x)
	x += icon + gap
	drawSquare(x)
	x += icon + gap
	overDraw(out, w, x, (h-w.Bounds().Dy())/2)
	x += w.Bounds().Dx() + titleSpace + gap
	drawSquare(x)
	return out
}

// compositeOn draws a PNG over a solid background with a small margin, returning
// the flattened image (so transparent renders can be judged on a real menu hue).
func compositeOn(t *testing.T, pngBytes []byte, bg color.NRGBA) image.Image {
	t.Helper()
	fg := decodePNG(t, pngBytes)
	const m = 10
	out := image.NewNRGBA(image.Rect(0, 0, fg.Bounds().Dx()+2*m, fg.Bounds().Dy()+2*m))
	for y := 0; y < out.Bounds().Dy(); y++ {
		for x := 0; x < out.Bounds().Dx(); x++ {
			out.SetNRGBA(x, y, bg)
		}
	}
	overDraw(out, fg, m, m)
	return out
}

// mockRow is one submenu row in the menu mock: its leading slot image plus a
// gray rounded placeholder standing in for the native title (systray renders the
// title text itself; the mock only proves the geometry). A chart row spans the
// full width and has no placeholder.
type mockRow struct {
	slot         []byte
	placeholderW int
	header       bool
	isChart      bool
}

// buildMenuMock stacks the rows as [leading slot image][gray rounded title
// placeholder] on a mock menu background. Every non-chart row places its slot
// image at the SAME leading x and its placeholder at leftInset+slotWidth+gap, so
// the left edge of every title is pixel-identical — exactly what the redesign
// promises. `text` tints value placeholders, `head` tints section-header ones.
func buildMenuMock(t *testing.T, rows []mockRow, bg color.NRGBA, chartWPx int, text, head color.NRGBA) image.Image {
	t.Helper()
	const (
		leftInset = 24 // menu content inset (mirrors the systray image gutter)
		nativeGap = 14 // gap between the leading image and the native title
		rowGap    = 8
		phH       = 12 // placeholder height (stands in for a line of title text)
		rightPad  = 28
	)
	slotWPx := usageSlotWidthPt * 2
	placeholderLeft := leftInset + slotWPx + nativeGap

	// Overall dimensions.
	maxPh := 0
	totalH := rowGap
	for _, r := range rows {
		sh := decodePNG(t, r.slot).Bounds().Dy()
		rh := sh
		if !r.isChart && phH > rh {
			rh = phH
		}
		totalH += rh + rowGap
		if r.placeholderW > maxPh {
			maxPh = r.placeholderW
		}
	}
	width := placeholderLeft + maxPh + rightPad
	if w := leftInset + chartWPx + rightPad; w > width {
		width = w
	}
	out := image.NewNRGBA(image.Rect(0, 0, width, totalH))
	for y := 0; y < totalH; y++ {
		for x := 0; x < width; x++ {
			out.SetNRGBA(x, y, bg)
		}
	}

	y := rowGap
	for _, r := range rows {
		slot := decodePNG(t, r.slot)
		sh := slot.Bounds().Dy()
		rh := sh
		if !r.isChart && phH > rh {
			rh = phH
		}
		// Leading image at the uniform inset, vertically centered in the row.
		overDraw(out, slot, leftInset, y+(rh-sh)/2)
		// Title placeholder (skip for the full-width chart row).
		if !r.isChart && r.placeholderW > 0 {
			col := text
			if r.header {
				col = head
			}
			drawRoundedBlock(out, placeholderLeft, y+(rh-phH)/2, r.placeholderW, phH, col)
		}
		y += rh + rowGap
	}
	return out
}

// drawRoundedBlock fills a rounded rectangle of size w×h at (x0,y0) in col.
func drawRoundedBlock(dst *image.NRGBA, x0, y0, w, h int, col color.NRGBA) {
	radius := float64(h) / 2
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if !insideRoundedRect(float64(x)+0.5, float64(y)+0.5, 0.5, 0.5, float64(w)-0.5, float64(h)-0.5, radius) {
				continue
			}
			dx, dy := x0+x, y0+y
			if dx < 0 || dy < 0 || dx >= dst.Bounds().Dx() || dy >= dst.Bounds().Dy() {
				continue
			}
			dst.SetNRGBA(dx, dy, col)
		}
	}
}

// overDraw alpha-composites src onto dst at (ox,oy) using straight NRGBA math.
func overDraw(dst, src *image.NRGBA, ox, oy int) {
	b := src.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			s := src.NRGBAAt(x, y)
			if s.A == 0 {
				continue
			}
			dx, dy := ox+x, oy+y
			if dx < 0 || dy < 0 || dx >= dst.Bounds().Dx() || dy >= dst.Bounds().Dy() {
				continue
			}
			d := dst.NRGBAAt(dx, dy)
			sa := float64(s.A) / 255
			da := float64(d.A) / 255
			oa := sa + da*(1-sa)
			blend := func(sc, dc uint8) uint8 {
				v := (float64(sc)*sa + float64(dc)*da*(1-sa)) / oa
				if v > 255 {
					v = 255
				}
				return uint8(v + 0.5)
			}
			dst.SetNRGBA(dx, dy, color.NRGBA{blend(s.R, d.R), blend(s.G, d.G), blend(s.B, d.B), uint8(oa*255 + 0.5)})
		}
	}
}

func TestDeriveServerState(t *testing.T) {
	cases := []struct {
		healthy, proc bool
		want          serverState
	}{
		{true, true, serverRunning},
		{true, false, serverRunning}, // healthy wins even if pgrep misses
		{false, true, serverStarting},
		{false, false, serverStopped},
	}
	for _, c := range cases {
		if got := deriveServerState(c.healthy, c.proc); got != c.want {
			t.Errorf("deriveServerState(%v, %v) = %v, want %v", c.healthy, c.proc, got, c.want)
		}
	}
}

func TestStatusBadgePNG(t *testing.T) {
	badge := synthBadge(t, 44, color.NRGBA{0x22, 0x22, 0x22, 0xff})
	for _, state := range []serverState{serverRunning, serverStarting, serverStopped} {
		b := statusBadgePNG(badge, state, 0)
		if b == nil {
			t.Fatalf("state %v: nil PNG", state)
		}
		img := decodePNG(t, b)
		if img.Bounds().Dx() != statusBadgeWidthPt*2 || img.Bounds().Dy() != statusBadgeHeightPt*2 {
			t.Errorf("state %v: dims %v, want %dx%d", state, img.Bounds(), statusBadgeWidthPt*2, statusBadgeHeightPt*2)
		}
	}
	// The glyph region must differ between running (solid green) and stopped
	// (gray ring), and the starting arc must change with phase (animation).
	if bytesEqual(statusBadgePNG(badge, serverRunning, 0), statusBadgePNG(badge, serverStopped, 0)) {
		t.Error("running and stopped badges render identically")
	}
	if bytesEqual(statusBadgePNG(badge, serverStarting, 0), statusBadgePNG(badge, serverStarting, 3)) {
		t.Error("starting arc does not rotate with phase")
	}
	if bytesEqual(statusBadgePNG(badge, serverRunning, 0), statusBadgePNG(badge, serverRunning, 3)) == false {
		t.Error("running badge should ignore phase")
	}
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
