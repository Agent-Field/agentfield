package types

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestReasonerDefinitionJSONRoundTrip(t *testing.T) {
	original := ReasonerDefinition{
		ID:           "my-reasoner",
		InputSchema:  json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}}}`),
		OutputSchema: json.RawMessage(`{"type":"object","properties":{"y":{"type":"integer"}}}`),
		Tags:         []string{"tag1", "tag2"},
		ProposedTags: []string{"proposed1"},
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded ReasonerDefinition
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, original.ID, decoded.ID)
	assert.JSONEq(t, string(original.InputSchema), string(decoded.InputSchema))
	assert.JSONEq(t, string(original.OutputSchema), string(decoded.OutputSchema))
	assert.Equal(t, original.Tags, decoded.Tags)
	assert.Equal(t, original.ProposedTags, decoded.ProposedTags)
}

func TestReasonerDefinitionOmitEmptyFields(t *testing.T) {
	// Tags and ProposedTags are omitempty — they should not appear in JSON when nil
	r := ReasonerDefinition{
		ID:          "r1",
		InputSchema: json.RawMessage(`{}`),
	}

	data, err := json.Marshal(r)
	require.NoError(t, err)

	var m map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(data, &m))

	_, hasTags := m["tags"]
	_, hasProposed := m["proposed_tags"]
	assert.False(t, hasTags, "tags should be omitted when nil")
	assert.False(t, hasProposed, "proposed_tags should be omitted when nil")
}

func TestSkillDefinitionJSONRoundTrip(t *testing.T) {
	original := SkillDefinition{
		ID:           "my-skill",
		InputSchema:  json.RawMessage(`{"type":"object"}`),
		Tags:         []string{"skill-tag"},
		ProposedTags: []string{"pt"},
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded SkillDefinition
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, original.ID, decoded.ID)
	assert.JSONEq(t, string(original.InputSchema), string(decoded.InputSchema))
	assert.Equal(t, original.Tags, decoded.Tags)
	assert.Equal(t, original.ProposedTags, decoded.ProposedTags)
}

func TestCommunicationConfigJSONRoundTrip(t *testing.T) {
	original := CommunicationConfig{
		Protocols:         []string{"http", "websocket"},
		WebSocketEndpoint: "ws://localhost:8001/ws",
		HeartbeatInterval: "30s",
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded CommunicationConfig
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, original, decoded)
}

func TestNodeRegistrationRequestJSONRoundTrip(t *testing.T) {
	now := time.Now().UTC().Truncate(time.Second)
	invURL := "http://localhost:8001/invoke"

	original := NodeRegistrationRequest{
		ID:      "node-1",
		TeamID:  "team-1",
		BaseURL: "http://localhost:8001",
		Version: "1.2.3",
		Reasoners: []ReasonerDefinition{
			{
				ID:          "r1",
				InputSchema: json.RawMessage(`{"type":"object"}`),
				Tags:        []string{"foo"},
			},
		},
		Skills: []SkillDefinition{
			{ID: "s1"},
		},
		CommunicationConfig: CommunicationConfig{
			Protocols: []string{"http"},
		},
		HealthStatus:  "healthy",
		LastHeartbeat: now,
		RegisteredAt:  now,
		Metadata:      map[string]any{"env": "test"},
		DeploymentType: "serverless",
		InvocationURL:  &invURL,
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded NodeRegistrationRequest
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, original.ID, decoded.ID)
	assert.Equal(t, original.TeamID, decoded.TeamID)
	assert.Equal(t, original.BaseURL, decoded.BaseURL)
	assert.Equal(t, original.Version, decoded.Version)
	assert.Equal(t, original.HealthStatus, decoded.HealthStatus)
	assert.Equal(t, original.DeploymentType, decoded.DeploymentType)
	require.NotNil(t, decoded.InvocationURL)
	assert.Equal(t, *original.InvocationURL, *decoded.InvocationURL)
	assert.Equal(t, len(original.Reasoners), len(decoded.Reasoners))
	assert.Equal(t, original.Reasoners[0].ID, decoded.Reasoners[0].ID)
	assert.Equal(t, len(original.Skills), len(decoded.Skills))
	assert.Equal(t, original.Skills[0].ID, decoded.Skills[0].ID)
}

func TestNodeRegistrationResponseJSONRoundTrip(t *testing.T) {
	original := NodeRegistrationResponse{
		ID:              "node-1",
		ResolvedBaseURL: "http://10.0.0.1:8001",
		Message:         "registered",
		Success:         true,
		Status:          "active",
		ProposedTags:    []string{"ptag"},
		PendingTags:     []string{"pending-tag"},
		AutoApprovedTags: []string{"auto-tag"},
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded NodeRegistrationResponse
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, original.ID, decoded.ID)
	assert.Equal(t, original.ResolvedBaseURL, decoded.ResolvedBaseURL)
	assert.Equal(t, original.Message, decoded.Message)
	assert.Equal(t, original.Success, decoded.Success)
	assert.Equal(t, original.Status, decoded.Status)
	assert.Equal(t, original.ProposedTags, decoded.ProposedTags)
	assert.Equal(t, original.PendingTags, decoded.PendingTags)
	assert.Equal(t, original.AutoApprovedTags, decoded.AutoApprovedTags)
	// RegisteredAt has json:"-" so it should not be serialized/deserialized
	assert.True(t, decoded.RegisteredAt.IsZero(), "RegisteredAt should not be serialized (json:\"-\")")
}

func TestNodeStatusUpdateJSONRoundTrip(t *testing.T) {
	score := 95
	original := NodeStatusUpdate{
		Phase:       "running",
		Version:     "1.0.0",
		HealthScore: &score,
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded NodeStatusUpdate
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, original.Phase, decoded.Phase)
	assert.Equal(t, original.Version, decoded.Version)
	require.NotNil(t, decoded.HealthScore)
	assert.Equal(t, *original.HealthScore, *decoded.HealthScore)
}

func TestNodeStatusUpdateNilHealthScore(t *testing.T) {
	original := NodeStatusUpdate{
		Phase: "draining",
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var m map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(data, &m))

	// health_score is omitempty (nil pointer) — should not appear
	_, has := m["health_score"]
	assert.False(t, has, "health_score should be omitted when nil")
}

func TestLeaseResponseJSONRoundTrip(t *testing.T) {
	original := LeaseResponse{
		LeaseSeconds:     120,
		NextLeaseRenewal: "2026-04-05T10:00:00Z",
		Message:          "lease granted",
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded LeaseResponse
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, original, decoded)
}

func TestActionAckRequestJSONRoundTrip(t *testing.T) {
	durationMS := 500
	original := ActionAckRequest{
		ActionID:   "action-1",
		Status:     "succeeded",
		DurationMS: &durationMS,
		ResultRef:  "ref-1",
		Result:     json.RawMessage(`{"output":"done"}`),
		Error:      "",
		Notes:      []string{"note1"},
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded ActionAckRequest
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, original.ActionID, decoded.ActionID)
	assert.Equal(t, original.Status, decoded.Status)
	require.NotNil(t, decoded.DurationMS)
	assert.Equal(t, *original.DurationMS, *decoded.DurationMS)
	assert.Equal(t, original.ResultRef, decoded.ResultRef)
	assert.JSONEq(t, string(original.Result), string(decoded.Result))
	assert.Equal(t, original.Notes, decoded.Notes)
}

func TestShutdownRequestJSONRoundTrip(t *testing.T) {
	original := ShutdownRequest{
		Reason:          "maintenance",
		Version:         "1.0.0",
		ExpectedRestart: "2026-04-05T12:00:00Z",
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded ShutdownRequest
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, original, decoded)
}

func TestWorkflowExecutionEventJSONRoundTrip(t *testing.T) {
	parentExecID := "parent-exec-1"
	parentWfID := "parent-wf-1"
	statusReason := "all good"
	durationMS := int64(1234)

	original := WorkflowExecutionEvent{
		ExecutionID:       "exec-1",
		WorkflowID:        "wf-1",
		RunID:             "run-1",
		ReasonerID:        "my-reasoner",
		Type:              "execution",
		AgentNodeID:       "node-1",
		Status:            "succeeded",
		StatusReason:      &statusReason,
		ParentExecutionID: &parentExecID,
		ParentWorkflowID:  &parentWfID,
		InputData:         map[string]interface{}{"key": "value"},
		Result:            map[string]interface{}{"output": "ok"},
		Error:             "",
		DurationMS:        &durationMS,
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded WorkflowExecutionEvent
	err = json.Unmarshal(data, &decoded)
	require.NoError(t, err)

	assert.Equal(t, original.ExecutionID, decoded.ExecutionID)
	assert.Equal(t, original.WorkflowID, decoded.WorkflowID)
	assert.Equal(t, original.RunID, decoded.RunID)
	assert.Equal(t, original.ReasonerID, decoded.ReasonerID)
	assert.Equal(t, original.Type, decoded.Type)
	assert.Equal(t, original.AgentNodeID, decoded.AgentNodeID)
	assert.Equal(t, original.Status, decoded.Status)
	require.NotNil(t, decoded.StatusReason)
	assert.Equal(t, *original.StatusReason, *decoded.StatusReason)
	require.NotNil(t, decoded.ParentExecutionID)
	assert.Equal(t, *original.ParentExecutionID, *decoded.ParentExecutionID)
	require.NotNil(t, decoded.ParentWorkflowID)
	assert.Equal(t, *original.ParentWorkflowID, *decoded.ParentWorkflowID)
	require.NotNil(t, decoded.DurationMS)
	assert.Equal(t, *original.DurationMS, *decoded.DurationMS)
}

func TestWorkflowExecutionEventOmitEmptyOptionals(t *testing.T) {
	event := WorkflowExecutionEvent{
		ExecutionID: "exec-1",
		Status:      "running",
	}

	data, err := json.Marshal(event)
	require.NoError(t, err)

	var m map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(data, &m))

	for _, field := range []string{"parent_execution_id", "parent_workflow_id", "status_reason", "duration_ms"} {
		_, has := m[field]
		assert.False(t, has, "field %q should be omitted when nil", field)
	}
}

func TestExecutionStatusConstants(t *testing.T) {
	// Verify the constant values are as expected by the rest of the codebase
	assert.Equal(t, "pending", ExecutionStatusPending)
	assert.Equal(t, "queued", ExecutionStatusQueued)
	assert.Equal(t, "waiting", ExecutionStatusWaiting)
	assert.Equal(t, "running", ExecutionStatusRunning)
	assert.Equal(t, "succeeded", ExecutionStatusSucceeded)
	assert.Equal(t, "failed", ExecutionStatusFailed)
	assert.Equal(t, "cancelled", ExecutionStatusCancelled)
	assert.Equal(t, "timeout", ExecutionStatusTimeout)
}
