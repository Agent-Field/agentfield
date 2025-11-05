package storage

import (
	"strings"
	"testing"
)

var workflowExecutionLifecycleColumns = []string{
	"workflow_id",
	"execution_id",
	"haxen_request_id",
	"run_id",
	"session_id",
	"actor_id",
	"agent_node_id",
	"parent_workflow_id",
	"parent_execution_id",
	"root_workflow_id",
	"workflow_depth",
	"reasoner_id",
	"input_data",
	"output_data",
	"input_size",
	"output_size",
	"status",
	"started_at",
	"completed_at",
	"duration_ms",
	"state_version",
	"last_event_sequence",
	"active_children",
	"pending_children",
	"pending_terminal_status",
	"status_reason",
	"lease_owner",
	"lease_expires_at",
	"error_message",
	"retry_count",
	"workflow_name",
	"workflow_tags",
	"notes",
	"created_at",
	"updated_at",
}

func TestWorkflowExecutionInsertQueriesCoverLifecycleColumns(t *testing.T) {
	tests := []struct {
		name        string
		query       string
		placeholder string
		requireTail string
	}{
		{name: "sqlite", query: sqliteWorkflowExecutionInsertQuery, placeholder: "?"},
	}

	const expectedPlaceholders = 35

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			assertLifecycleColumnsPresent(t, tc.name, tc.query)

			count := strings.Count(tc.query, tc.placeholder)
			if count != expectedPlaceholders {
				t.Fatalf("expected %s query to have %d '%s' placeholders, got %d", tc.name, expectedPlaceholders, tc.placeholder, count)
			}

			if tc.requireTail != "" && !strings.Contains(tc.query, tc.requireTail) {
				t.Fatalf("expected %s query to include placeholder %s", tc.name, tc.requireTail)
			}
		})
	}
}

func assertLifecycleColumnsPresent(t *testing.T, backend, query string) {
	t.Helper()
	for _, col := range workflowExecutionLifecycleColumns {
		if !strings.Contains(query, col) {
			t.Fatalf("expected %s insert query to include column %q", backend, col)
		}
	}
}
