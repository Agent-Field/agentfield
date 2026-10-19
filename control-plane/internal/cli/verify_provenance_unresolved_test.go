package cli

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

func TestVerifyProvenanceJSON_UnresolvedDIDsFailVerification(t *testing.T) {
	chain := types.WorkflowVCChainResponse{
		WorkflowID:     "wf-unresolved",
		ComponentVCs:   []types.ExecutionVC{},
		WorkflowVC:     types.WorkflowVC{WorkflowID: "wf-unresolved"},
		Status:         "completed",
		UnresolvedDIDs: []string{"did:web:example.com:agents:gone"},
	}
	raw, err := json.Marshal(chain)
	if err != nil {
		t.Fatal(err)
	}

	res := VerifyProvenanceJSON(raw, VerifyOptions{})
	if res.Valid {
		t.Fatal("expected a chain with unresolved DIDs to fail verification")
	}
	if len(res.UnresolvedDIDs) != 1 || res.UnresolvedDIDs[0] != "did:web:example.com:agents:gone" {
		t.Fatalf("unexpected unresolved DIDs: %v", res.UnresolvedDIDs)
	}
	found := false
	for _, issue := range res.Comprehensive.CriticalIssues {
		if issue.Type == "unresolved_dids" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected unresolved_dids critical issue, got %+v", res.Comprehensive.CriticalIssues)
	}
	warning := unresolvedDIDsWarning(res)
	if !strings.Contains(warning, "cannot be fully verified") || !strings.Contains(warning, "did:web:example.com:agents:gone") {
		t.Fatalf("unexpected warning: %q", warning)
	}

	// Without unresolved DIDs the same chain verifies and no warning is shown.
	chain.UnresolvedDIDs = nil
	chain.Verifiable = true
	raw, err = json.Marshal(chain)
	if err != nil {
		t.Fatal(err)
	}
	res = VerifyProvenanceJSON(raw, VerifyOptions{})
	if len(res.UnresolvedDIDs) != 0 || unresolvedDIDsWarning(res) != "" {
		t.Fatalf("expected no unresolved DID warning, got %v", res.UnresolvedDIDs)
	}
	if !res.Valid {
		t.Fatalf("expected empty verifiable chain to be valid, got %+v", res.Comprehensive.CriticalIssues)
	}
}

func TestC8F8VerifiabilityCompatibility(t *testing.T) {
	for _, tc := range []struct {
		name, fields string
		valid        bool
	}{
		{"old export", "", true},
		{"explicit false", `,"verifiable":false`, false},
		{"explicit true", `,"verifiable":true`, true},
		{"empty required issuer", `,"unresolved_dids":[""]`, false},
		{"whitespace required issuer", `,"unresolved_dids":[" "]`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := `{"workflow_id":"wf","component_vcs":[],"workflow_vc":{"workflow_id":"wf"},"status":"completed"` + tc.fields + `}`
			result := VerifyProvenanceJSON([]byte(raw), VerifyOptions{})
			if result.Valid != tc.valid {
				t.Fatalf("valid=%v want=%v: %+v", result.Valid, tc.valid, result.Comprehensive)
			}
		})
	}
}
