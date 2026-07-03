package packages

import "testing"

func TestNodeDepName(t *testing.T) {
	cases := map[string]string{
		"af://registry/swe-planner":                "swe-planner",
		"af://registry/swe-planner@^1.0":           "swe-planner",
		"af://registry/pr-af/":                     "pr-af",
		"https://github.com/Agent-Field/pr-af":     "pr-af",
		"https://github.com/Agent-Field/pr-af.git": "pr-af",
		"git@github.com:Agent-Field/sec-af.git":    "sec-af",
	}
	for ref, want := range cases {
		if got := NodeDepName(ref); got != want {
			t.Errorf("NodeDepName(%q) = %q, want %q", ref, got, want)
		}
	}
}
