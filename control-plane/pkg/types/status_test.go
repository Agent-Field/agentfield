package types

import "testing"

func TestNormalizeExecutionStatus(t *testing.T) {
	cases := map[string]string{
		"":                  string(ExecutionStatusUnknown),
		"  ":                string(ExecutionStatusUnknown),
		"Completed":         string(ExecutionStatusSucceeded),
		"success":           string(ExecutionStatusSucceeded),
		"FAILED":            string(ExecutionStatusFailed),
		"canceled":          string(ExecutionStatusCancelled),
		"TIMED_OUT":         string(ExecutionStatusTimeout),
		"waiting":           string(ExecutionStatusWaiting),
		"awaiting_approval": string(ExecutionStatusWaiting),
		"processing":        string(ExecutionStatusRunning),
		"custom-status":     string(ExecutionStatusUnknown),
	}

	for input, expected := range cases {
		if got := NormalizeExecutionStatus(input); got != expected {
			t.Fatalf("NormalizeExecutionStatus(%q) = %q, want %q", input, got, expected)
		}
	}
}

func TestIsTerminalExecutionStatus(t *testing.T) {
	terminals := []string{"succeeded", "failed", "cancelled", "timeout", "completed"}
	nonTerminals := []string{"pending", "queued", "waiting", "running", "processing"}

	for _, status := range terminals {
		if !IsTerminalExecutionStatus(status) {
			t.Fatalf("expected %q to be terminal", status)
		}
	}

	for _, status := range nonTerminals {
		if IsTerminalExecutionStatus(status) {
			t.Fatalf("expected %q to be non-terminal", status)
		}
	}
}
