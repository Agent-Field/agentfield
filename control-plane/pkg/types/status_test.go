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

func TestAgentNodeHoldsExecutionsOfflineReadsRegistrationMetadata(t *testing.T) {
	if (*AgentNode)(nil).HoldsExecutionsOffline() {
		t.Fatal("nil AgentNode must not opt in to offline hold")
	}

	cases := []struct {
		name   string
		custom map[string]interface{}
		want   bool
	}{
		{name: "nil custom metadata", custom: nil, want: false},
		{name: "empty custom metadata", custom: map[string]interface{}{}, want: false},
		{name: "explicit boolean false", custom: map[string]interface{}{AgentMetadataHoldsExecutionsOffline: false}, want: false},
		{name: "explicit boolean true", custom: map[string]interface{}{AgentMetadataHoldsExecutionsOffline: true}, want: true},
		{name: "string true", custom: map[string]interface{}{AgentMetadataHoldsExecutionsOffline: "true"}, want: true},
		{name: "string false", custom: map[string]interface{}{AgentMetadataHoldsExecutionsOffline: "false"}, want: false},
		{name: "other string", custom: map[string]interface{}{AgentMetadataHoldsExecutionsOffline: "yes"}, want: false},
		{name: "non-boolean non-string value", custom: map[string]interface{}{AgentMetadataHoldsExecutionsOffline: 1}, want: false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			node := &AgentNode{Metadata: AgentMetadata{Custom: tc.custom}}
			if got := node.HoldsExecutionsOffline(); got != tc.want {
				t.Fatalf("HoldsExecutionsOffline(%v) = %v, want %v", tc.custom, got, tc.want)
			}
		})
	}
}

