package cli

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

func TestVerifyProvenanceJSON_LegacyChainEmptyComponents(t *testing.T) {
	legacy := types.WorkflowVCChainResponse{
		WorkflowID:   "wf-empty",
		ComponentVCs: []types.ExecutionVC{},
		WorkflowVC:   types.WorkflowVC{WorkflowID: "wf-empty"},
		Status:       "completed",
	}
	raw, err := json.Marshal(legacy)
	if err != nil {
		t.Fatal(err)
	}
	res := VerifyProvenanceJSON(raw, VerifyOptions{})
	if res.FormatValid != true || res.Type != "workflow" {
		t.Fatalf("expected workflow parse, got type=%q formatValid=%v", res.Type, res.FormatValid)
	}
}

func TestVerifyProvenanceJSON_InvalidJSON(t *testing.T) {
	res := VerifyProvenanceJSON([]byte(`not json`), VerifyOptions{})
	if res.FormatValid || res.Valid {
		t.Fatalf("expected invalid")
	}
}

func TestResolveWebDID_DecodesEncodedPortInHost(t *testing.T) {
	_, err := resolveWebDID("did:web:example.com%3A8443:agents:test-agent")
	if err == nil {
		t.Fatalf("expected network fetch to fail in test environment")
	}
	if strings.Contains(err.Error(), "invalid did:web domain") || strings.Contains(err.Error(), "invalid URL") {
		t.Fatalf("expected encoded port to be accepted before fetch, got %v", err)
	}
}
