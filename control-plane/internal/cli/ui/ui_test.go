package ui

import (
	"strings"
	"testing"
)

func TestTableContainsTitleHeadersAndCells(t *testing.T) {
	out := Table("My Nodes",
		[]string{"NODE", "STATUS"},
		[][]string{{"swe-planner", "up"}, {"pr-af", "down"}})
	for _, want := range []string{"My Nodes", "NODE", "STATUS", "swe-planner", "pr-af"} {
		if !strings.Contains(out, want) {
			t.Errorf("Table output missing %q:\n%s", want, out)
		}
	}
	if !strings.Contains(out, "╭") {
		t.Errorf("expected a rounded border in table output:\n%s", out)
	}
}

func TestTitleAndMutedKeepText(t *testing.T) {
	if !strings.Contains(Title("Hello"), "Hello") {
		t.Error("Title dropped its text")
	}
	if !strings.Contains(Muted("secondary"), "secondary") {
		t.Error("Muted dropped its text")
	}
}

func TestStatusBadge(t *testing.T) {
	cases := map[string]string{
		"running": "running",
		"error":   "error",
		"stopped": "stopped",
		"":        "—",
	}
	for status, want := range cases {
		if got := StatusBadge(status); !strings.Contains(got, want) {
			t.Errorf("StatusBadge(%q) = %q; want substring %q", status, got, want)
		}
	}
}

func TestPanelHasTitleBodyAndBorder(t *testing.T) {
	p := Panel("Title", "body text")
	for _, want := range []string{"Title", "body text", "╭"} {
		if !strings.Contains(p, want) {
			t.Errorf("Panel missing %q:\n%s", want, p)
		}
	}
}

func TestSuccessPanelHasContent(t *testing.T) {
	sp := SuccessPanel("Done", "installed successfully")
	if !strings.Contains(sp, "Done") || !strings.Contains(sp, "installed successfully") {
		t.Errorf("SuccessPanel missing content:\n%s", sp)
	}
}

func TestKVAlignsAndKeepsValues(t *testing.T) {
	out := KV([][2]string{{"Name", "swe-planner"}, {"Port", "8003"}})
	for _, want := range []string{"Name", "swe-planner", "Port", "8003"} {
		if !strings.Contains(out, want) {
			t.Errorf("KV missing %q:\n%s", want, out)
		}
	}
	if got := strings.Count(out, "\n"); got != 1 {
		t.Errorf("expected 2 KV lines (one newline), got %d:\n%q", got, out)
	}
}
