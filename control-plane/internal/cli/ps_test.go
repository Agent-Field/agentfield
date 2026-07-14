package cli

import (
	"strings"
	"testing"
	"time"
)

func TestRenderActiveRunsEmpty(t *testing.T) {
	var out strings.Builder
	renderActiveRuns(&out, activeRunsResponse{Count: 0})
	if got := out.String(); !strings.Contains(got, "No runs in flight.") {
		t.Fatalf("empty render = %q, want no-runs message", got)
	}
}

func TestRenderActiveRunsTable(t *testing.T) {
	started := time.Now().Add(-10 * time.Minute)
	var out strings.Builder
	renderActiveRuns(&out, activeRunsResponse{
		Count: 3,
		Runs: []activeRunItem{
			{
				RunID:            "run_20260714_165410_wtcyrr5q",
				Target:           "pr-af-go.review",
				RootStatus:       "running",
				ActiveExecutions: 4,
				TotalExecutions:  25,
				StartedAt:        started,
				LatestActivity:   started.Add(9 * time.Minute),
			},
		},
	})
	got := out.String()
	for _, want := range []string{"RUN", "TARGET", "LAST ACTIVITY", "pr-af-go.review", "running", "run_20260714_165410_wtcyrr5q"} {
		if !strings.Contains(got, want) {
			t.Fatalf("render output missing %q:\n%s", want, got)
		}
	}
	// More active runs exist than were returned — the truncation must be said.
	if !strings.Contains(got, "3 active runs total (showing 1)") {
		t.Fatalf("render output missing truncation note:\n%s", got)
	}
}
