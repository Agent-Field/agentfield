package utils

import (
	"regexp"
	"strings"
	"testing"
)

func TestGeneratedIDsMatchExpectedFormat(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		prefix string
		fn     func() string
	}{
		{name: "workflow", prefix: "wf", fn: GenerateWorkflowID},
		{name: "execution", prefix: "exec", fn: GenerateExecutionID},
		{name: "run", prefix: "run", fn: GenerateRunID},
		{name: "request", prefix: "req", fn: GenerateAgentFieldRequestID},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			id := tc.fn()
			pattern := "^" + tc.prefix + "_\\d{8}_\\d{6}_[a-z0-9]{8}$"
			if matched, err := regexp.MatchString(pattern, id); err != nil || !matched {
				t.Fatalf("id %q did not match pattern %q, err=%v", id, pattern, err)
			}
		})
	}
}

func TestValidateWorkflowID(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		value string
		want  bool
	}{
		{name: "empty", value: "", want: false},
		{name: "valid", value: "wf_20260408_120000_deadbeef", want: true},
		{name: "maxLength", value: strings.Repeat("a", 255), want: true},
		{name: "tooLong", value: strings.Repeat("a", 256), want: false},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			if got := ValidateWorkflowID(tc.value); got != tc.want {
				t.Fatalf("ValidateWorkflowID(%q) = %v, want %v", tc.value, got, tc.want)
			}
		})
	}
}

func TestGenerateRandomString(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		length int
	}{
		{name: "zero", length: 0},
		{name: "short", length: 8},
		{name: "long", length: 64},
	}

	validChars := regexp.MustCompile(`^[a-z0-9]*$`)

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got := generateRandomString(tc.length)
			if len(got) != tc.length {
				t.Fatalf("len(generateRandomString(%d)) = %d, want %d", tc.length, len(got), tc.length)
			}
			if !validChars.MatchString(got) {
				t.Fatalf("generateRandomString(%d) returned invalid characters: %q", tc.length, got)
			}
		})
	}
}
