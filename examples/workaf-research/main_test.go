package main

import (
	"strings"
	"testing"
)

// A WorkAF job's ticket is the question; the default is only for a call that
// says nothing (the JSON-library ticket once planned SQLite drivers).
func TestTheQuestionComesFromTheTicket(t *testing.T) {
	cases := []struct {
		name string
		in   map[string]any
		want string
	}{
		{"direct", map[string]any{"question": "Which YAML library?"}, "Which YAML library?"},
		{"workaf job", map[string]any{"job": 7, "title": "Pick a JSON library", "body": "encoding/json, goccy/go-json or json-iterator?"}, "Pick a JSON library\n\nencoding/json, goccy/go-json or json-iterator?"},
		{"event only", map[string]any{"event": map[string]any{"title": "Pick a JSON library"}}, "Pick a JSON library"},
		{"nothing", map[string]any{}, defaultQuestion},
	}
	for _, c := range cases {
		if got := question(c.in); got != c.want {
			t.Errorf("%s: %q, want %q", c.name, got, c.want)
		}
	}
	if strings.Contains(question(cases[1].in), "SQLite") {
		t.Fatal("a JSON ticket was asked about SQLite")
	}
}

func TestTheInvestigationsAreThePlansOptions(t *testing.T) {
	planned := map[string]any{"options": []any{
		map[string]any{"name": "encoding/json"}, map[string]any{"name": "github.com/goccy/go-json"},
		map[string]any{"name": "encoding/json"}, map[string]any{"name": ""},
		map[string]any{"name": "a"}, map[string]any{"name": "b"}, map[string]any{"name": "c"},
	}}
	got := optionsOf(planned)
	if len(got) != mostOptions || str(got[0], "name") != "encoding/json" || str(got[1], "name") != "github.com/goccy/go-json" || str(got[2], "name") != "a" {
		t.Fatalf("options %v", got)
	}
	if len(optionsOf(map[string]any{})) != 0 {
		t.Fatal("a plan with no options named some")
	}
}
