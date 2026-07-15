package packages

import (
	"strings"
	"testing"
)

func TestInstallSourceLabel(t *testing.T) {
	if got := installSourceLabel("https://x/y", ""); got != "https://x/y" {
		t.Errorf("no ref: got %q", got)
	}
	if got := installSourceLabel("https://x/y", "v1.2.3"); got != "https://x/y @ v1.2.3" {
		t.Errorf("with ref: got %q", got)
	}
}

func TestInstallSummaryPanel(t *testing.T) {
	out := installSummaryPanel("swe-planner", "0.1.0", "https://github.com/x/y", "", "/home/u/.agentfield/packages/swe-planner")
	for _, want := range []string{"Installed swe-planner v0.1.0", "Source", "https://github.com/x/y", "Location", "swe-planner"} {
		if !strings.Contains(out, want) {
			t.Errorf("summary missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "Reference") {
		t.Errorf("no ref given, but Reference shown:\n%s", out)
	}
	withRef := installSummaryPanel("n", "1.0", "url", "main", "/loc")
	if !strings.Contains(withRef, "Reference") || !strings.Contains(withRef, "main") {
		t.Errorf("ref not shown when provided:\n%s", withRef)
	}
}
