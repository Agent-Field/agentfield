package ui

import (
	"bytes"
	"context"
	"errors"
	"io"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

type stubPayloadStore struct {
	payloads map[string][]byte
	err      error
	opens    []string
}

func (s *stubPayloadStore) SaveFromReader(ctx context.Context, r io.Reader) (*services.PayloadRecord, error) {
	return nil, nil
}

func (s *stubPayloadStore) SaveBytes(ctx context.Context, data []byte) (*services.PayloadRecord, error) {
	return nil, nil
}

func (s *stubPayloadStore) Open(ctx context.Context, uri string) (io.ReadCloser, error) {
	s.opens = append(s.opens, uri)
	if s.err != nil {
		return nil, s.err
	}
	return io.NopCloser(bytes.NewReader(s.payloads[uri])), nil
}

func (s *stubPayloadStore) Remove(ctx context.Context, uri string) error {
	return nil
}

func TestExecutionHelpersSummaryAndDetails(t *testing.T) {
	payloadStore := &stubPayloadStore{payloads: map[string][]byte{
		"payload://input":  []byte(`{"prompt":"hello"}`),
		"payload://output": []byte(`{"answer":42}`),
	}}
	handler := &ExecutionHandler{payloads: payloadStore}
	now := time.Date(2026, 4, 8, 16, 0, 0, 0, time.UTC)
	sessionID := "session-1"
	actorID := "actor-1"
	statusReason := "ok"
	errorMessage := "boom"
	duration := int64(1500)
	completedAt := now.Add(2 * time.Second)
	notes := []types.ExecutionNote{
		{Message: "started", Timestamp: now},
		{Message: "completed", Timestamp: completedAt},
	}

	exec := &types.Execution{
		ExecutionID:   "exec-1",
		RunID:         "run-1",
		SessionID:     &sessionID,
		ActorID:       &actorID,
		AgentNodeID:   "agent-1",
		ReasonerID:    "reasoner-1",
		Status:        "SUCCESS",
		StatusReason:  &statusReason,
		StartedAt:     now,
		CompletedAt:   &completedAt,
		DurationMS:    &duration,
		InputPayload:  []byte(`{}`),
		InputURI:      dashboardStringPtr("payload://input"),
		ResultPayload: []byte(corruptedJSONSentinel),
		ResultURI:     dashboardStringPtr("payload://output"),
		ErrorMessage:  &errorMessage,
		Notes:         notes,
		UpdatedAt:     now.Add(5 * time.Second),
	}

	summary := handler.toExecutionSummary(exec)
	require.Equal(t, "succeeded", summary.Status)
	require.Equal(t, len(exec.InputPayload), summary.InputSize)
	require.Equal(t, len(exec.ResultPayload), summary.OutputSize)
	require.Equal(t, 1500, summary.DurationMS)
	require.Equal(t, now, summary.CreatedAt)

	details := handler.toExecutionDetails(context.Background(), exec)
	require.Equal(t, "exec-1", details.ExecutionID)
	require.Equal(t, "run-1", details.WorkflowID)
	require.Equal(t, "succeeded", details.Status)
	require.Equal(t, map[string]any{"prompt": "hello"}, details.InputData)
	require.Equal(t, map[string]any{"answer": float64(42)}, details.OutputData)
	require.Equal(t, len(`{"prompt":"hello"}`), details.InputSize)
	require.Equal(t, len(`{"answer":42}`), details.OutputSize)
	require.Equal(t, 2, details.NotesCount)
	require.NotNil(t, details.LatestNote)
	require.Equal(t, "completed", details.LatestNote.Message)
	require.Equal(t, completedAt.Format(time.RFC3339), *details.CompletedAt)
	require.Equal(t, 1500, *details.DurationMS)
	require.Equal(t, []string{"payload://input", "payload://output"}, payloadStore.opens)

	exec.Notes = nil
	payloadStore.opens = nil
	details = handler.toExecutionDetails(context.Background(), exec)
	require.Empty(t, details.Notes)
	require.Nil(t, details.LatestNote)
	require.Equal(t, 0, details.NotesCount)
	require.Equal(t, []string{"payload://input", "payload://output"}, payloadStore.opens)
}

func TestExecutionHelpersResolveFallbacks(t *testing.T) {
	payloadStore := &stubPayloadStore{payloads: map[string][]byte{
		"payload://fallback": []byte(`{"loaded":true}`),
	}}
	handler := &ExecutionHandler{payloads: payloadStore}

	data, size := handler.resolveExecutionData(context.Background(), []byte(`{"inline":true}`), dashboardStringPtr("payload://fallback"))
	require.Equal(t, map[string]any{"inline": true}, data)
	require.Equal(t, len(`{"inline":true}`), size)
	require.Empty(t, payloadStore.opens)

	data, size = handler.resolveExecutionData(context.Background(), []byte(` { } `), dashboardStringPtr("payload://fallback"))
	require.Equal(t, map[string]any{"loaded": true}, data)
	require.Equal(t, len(`{"loaded":true}`), size)
	require.Equal(t, []string{"payload://fallback"}, payloadStore.opens)

	payloadStore.err = errors.New("missing payload")
	data, size = handler.resolveExecutionData(context.Background(), []byte(corruptedJSONSentinel), dashboardStringPtr("payload://fallback"))
	require.Equal(t, corruptedJSONSentinel, data)
	require.Equal(t, len(corruptedJSONSentinel), size)

	payloadStore.err = nil
	loaded, loadedSize, err := handler.loadPayloadData(context.Background(), "payload://fallback")
	require.NoError(t, err)
	require.Equal(t, map[string]any{"loaded": true}, loaded)
	require.Equal(t, len(`{"loaded":true}`), loadedSize)
}

func TestExecutionHelpersPayloadParsingAndFormatting(t *testing.T) {
	require.Nil(t, decodePayload(nil))
	require.Nil(t, decodePayload([]byte("   ")))
	require.Equal(t, map[string]any{"hello": "world"}, decodePayload([]byte(`{"hello":"world"}`)))
	require.Equal(t, "plain-text", decodePayload([]byte("plain-text")))

	require.False(t, hasMeaningfulData(nil))
	require.False(t, hasMeaningfulData("   "))
	require.False(t, hasMeaningfulData(corruptedJSONSentinel))
	require.False(t, hasMeaningfulData([]interface{}{}))
	require.False(t, hasMeaningfulData(map[string]any{}))
	require.False(t, hasMeaningfulData(map[string]any{"error": corruptedJSONSentinel}))
	require.True(t, hasMeaningfulData([]interface{}{1}))
	require.True(t, hasMeaningfulData([]byte("x")))
	require.True(t, hasMeaningfulData(map[string]any{"ok": true}))

	require.Equal(t, 4, parsePositiveIntOrDefault("4", 9))
	require.Equal(t, 9, parsePositiveIntOrDefault("0", 9))
	require.Equal(t, 9, parsePositiveIntOrDefault("bad", 9))
	require.Equal(t, 10, parseBoundedIntOrDefault("20", 5, 1, 10))
	require.Equal(t, 5, parseBoundedIntOrDefault("-1", 5, 1, 10))
	require.Equal(t, 6, parseBoundedIntOrDefault("6", 5, 1, 10))

	parsedTime, err := parseTimePtrValue("2026-04-08T16:00:00Z")
	require.NoError(t, err)
	require.Equal(t, time.Date(2026, 4, 8, 16, 0, 0, 0, time.UTC), *parsedTime)
	nilTime, err := parseTimePtrValue("   ")
	require.NoError(t, err)
	require.Nil(t, nilTime)
	_, err = parseTimePtrValue("not-a-time")
	require.Error(t, err)

	require.Equal(t, "status", sanitizeExecutionSortField("STATUS"))
	require.Equal(t, "reasoner_id", sanitizeExecutionSortField("task_name"))
	require.Equal(t, "duration_ms", sanitizeExecutionSortField("duration"))
	require.Equal(t, "agent_node_id", sanitizeExecutionSortField("agent_node_id"))
	require.Equal(t, "execution_id", sanitizeExecutionSortField("execution_id"))
	require.Equal(t, "run_id", sanitizeExecutionSortField("workflow_id"))
	require.Equal(t, "started_at", sanitizeExecutionSortField("started"))
	require.Equal(t, "started_at", sanitizeExecutionSortField("unknown"))

	require.Equal(t, 1, computeTotalPages(0, 10))
	require.Equal(t, 1, computeTotalPages(10, 0))
	require.Equal(t, 3, computeTotalPages(21, 10))

	grouped := (&ExecutionHandler{}).groupExecutionSummaries([]ExecutionSummary{
		{ExecutionID: "exec-1", Status: "running", AgentNodeID: "agent-a", ReasonerID: "reasoner-a"},
		{ExecutionID: "exec-2", Status: "failed", AgentNodeID: "agent-a", ReasonerID: "reasoner-b"},
		{ExecutionID: "exec-3", Status: "failed", AgentNodeID: "agent-b", ReasonerID: "reasoner-b"},
	}, "reasoner")
	require.Len(t, grouped["reasoner-a"], 1)
	require.Len(t, grouped["reasoner-b"], 2)
	require.Len(t, (&ExecutionHandler{}).groupExecutionSummaries(grouped["reasoner-b"], "status")["failed"], 2)
	require.Len(t, (&ExecutionHandler{}).groupExecutionSummaries(grouped["reasoner-b"], "unknown")["ungrouped"], 2)

	now := time.Date(2026, 4, 8, 16, 0, 0, 0, time.UTC)
	require.Equal(t, "", formatRelativeTimeString(now, time.Time{}))
	require.Equal(t, "just now", formatRelativeTimeString(now, now.Add(-20*time.Second)))
	require.Equal(t, "5m ago", formatRelativeTimeString(now, now.Add(-5*time.Minute)))
	require.Equal(t, "3h ago", formatRelativeTimeString(now, now.Add(-3*time.Hour)))
	require.Equal(t, "2d ago", formatRelativeTimeString(now, now.Add(-50*time.Hour)))

	require.Equal(t, "—", formatDurationDisplay(nil))
	require.Equal(t, "—", formatDurationDisplay(dashboardInt64Ptr(0)))
	require.Equal(t, "500ms", formatDurationDisplay(dashboardInt64Ptr(500)))
	require.Equal(t, "1.5s", formatDurationDisplay(dashboardInt64Ptr(1500)))
	require.Equal(t, "2m 5s", formatDurationDisplay(dashboardInt64Ptr(125000)))
	require.Equal(t, "3h", formatDurationDisplay(dashboardInt64Ptr(3*60*60*1000)))
	require.Equal(t, "3h 2m", formatDurationDisplay(dashboardInt64Ptr((3*60*60+2*60)*1000)))
}
