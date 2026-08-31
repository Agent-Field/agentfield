package types

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestValidateRunLinkURL(t *testing.T) {
	tests := []struct {
		url string
		ok  bool
	}{
		{"http://x.test/a?b=c#d", true},
		{"https://github.com/o/r/pull/1", true},
		{"javascript:alert(1)", false}, {"data:text/html,x", false},
		{"file:///etc/passwd", false}, {"https://user:pass@example.com", false},
		{"mailto:a@b.c", false}, {"example.com", false}, {"http://", false}, {"", false},
		{"https://x.test/" + strings.Repeat("a", MaxRunLinkURLBytes), false},
	}
	for _, test := range tests {
		t.Run(test.url, func(t *testing.T) {
			err := ValidateRunLinkURL(test.url)
			if (err == nil) != test.ok {
				t.Fatalf("ValidateRunLinkURL() error = %v, want ok=%v", err, test.ok)
			}
		})
	}
}

func TestParseRunMetadata(t *testing.T) {
	tests := []struct {
		raw  string
		want *RunMetadata
	}{
		{"", nil}, {"{}", nil}, {"{", nil}, {`{"run":"x"}`, nil}, {`{"run":3}`, nil}, {`{"run":null}`, nil},
		{`{"run":{"display_name":"Release"}}`, &RunMetadata{DisplayName: "Release"}},
	}
	for _, test := range tests {
		got := ParseRunMetadata(json.RawMessage(test.raw))
		if test.want == nil && got != nil || test.want != nil && (got == nil || got.DisplayName != test.want.DisplayName) {
			t.Fatalf("ParseRunMetadata(%q) = %#v, want %#v", test.raw, got, test.want)
		}
	}
}
