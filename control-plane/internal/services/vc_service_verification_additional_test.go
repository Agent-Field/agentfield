package services

import (
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestVCServiceVerificationHelpers(t *testing.T) {
	service := NewVCService(&config.DIDConfig{Enabled: true}, nil, nil)

	execVC := &types.ExecutionVC{
		VCID:        "vc-1",
		ExecutionID: "exec-1",
		WorkflowID:  "workflow-1",
		SessionID:   "session-1",
		IssuerDID:   "did:key:issuer-a",
		CallerDID:   "did:key:caller-a",
		TargetDID:   "did:key:target-a",
		InputHash:   "input-a",
		OutputHash:  "output-a",
		Status:      "succeeded",
		Signature:   "signature-a",
	}

	vcDoc := &types.VCDocument{
		Context:      []string{"https://example.com/context"},
		Type:         []string{"VerifiableCredential"},
		ID:           "doc-1",
		Issuer:       "did:key:issuer-b",
		IssuanceDate: "not-a-timestamp",
		CredentialSubject: types.VCCredentialSubject{
			ExecutionID: "exec-2",
			WorkflowID:  "workflow-2",
			SessionID:   "session-2",
			Caller:      types.VCCaller{DID: "did:key:caller-b"},
			Target:      types.VCTarget{DID: "did:key:target-b"},
			Execution: types.VCExecution{
				InputHash:  "input-b",
				OutputHash: "output-b",
				Status:     "failed",
			},
		},
		Proof: types.VCProof{ProofValue: "signature-b"},
	}

	integrity := service.performIntegrityChecks(execVC, vcDoc)
	require.False(t, integrity.MetadataConsistency)
	require.False(t, integrity.FieldConsistency)
	require.False(t, integrity.TimestampValidation)
	require.False(t, integrity.HashValidation)
	require.False(t, integrity.StructuralIntegrity)
	require.GreaterOrEqual(t, len(integrity.Issues), 10)

	compliance := service.performComplianceChecks(vcDoc)
	require.False(t, compliance.W3CCompliance)
	require.False(t, compliance.AgentFieldStandardCompliance)
	require.Len(t, compliance.Issues, 2)

	require.ErrorContains(t, service.validateVCStructure(&types.VCDocument{}), "missing @context")
	require.ErrorContains(t, service.validateTimestamp("bad-timestamp"), "cannot parse")
	require.Equal(t, "null", string(marshalDataOrNull(nil)))
	require.Equal(t, "succeeded", service.determineWorkflowStatus([]types.ExecutionVC{{Status: "succeeded"}}))
}

func TestVCServiceGetWorkflowVCStatusSummaries(t *testing.T) {
	vcService, _, _, _ := setupVCTestEnvironment(t)

	summaries, err := vcService.GetWorkflowVCStatusSummaries([]string{"workflow-summary", "", "workflow-summary", "missing"})
	require.NoError(t, err)
	require.Contains(t, summaries, "workflow-summary")
	require.Contains(t, summaries, "missing")
	require.False(t, summaries["workflow-summary"].HasVCs)
	require.Equal(t, 0, summaries["workflow-summary"].VCCount)
	require.Equal(t, "none", summaries["workflow-summary"].VerificationStatus)
}

func TestVCServiceVerifyWorkflowVCComprehensiveSuccess(t *testing.T) {
	vcService, didService, _, _ := setupVCTestEnvironment(t)

	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-verify-workflow",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner-1"}},
	}
	regResp, err := didService.RegisterAgent(req)
	require.NoError(t, err)

	for i := 1; i <= 2; i++ {
		execCtx := &types.ExecutionContext{
			ExecutionID:  "exec-verify-" + string(rune('0'+i)),
			WorkflowID:   "workflow-verify-success",
			SessionID:    "session-verify",
			CallerDID:    regResp.IdentityPackage.ReasonerDIDs["reasoner-1"].DID,
			TargetDID:    "",
			AgentNodeDID: regResp.IdentityPackage.AgentDID.DID,
			Timestamp:    time.Now(),
		}

		_, err = vcService.GenerateExecutionVC(execCtx, []byte(`{"step":"input"}`), []byte(`{"step":"output"}`), "succeeded", nil, 25)
		require.NoError(t, err)
	}

	result, err := vcService.VerifyWorkflowVCComprehensive("workflow-verify-success")
	require.NoError(t, err)
	require.True(t, result.Valid)
	require.Empty(t, result.CriticalIssues)
	require.Greater(t, result.OverallScore, 0.0)
	require.True(t, result.IntegrityChecks.MetadataConsistency)
	require.True(t, result.ComplianceChecks.W3CCompliance)
}

func TestVCServiceDetermineWorkflowStatusVariants(t *testing.T) {
	service := NewVCService(&config.DIDConfig{Enabled: true}, nil, nil)

	tests := []struct {
		name     string
		statuses []string
		want     string
	}{
		{name: "no vcs", statuses: nil, want: string(types.ExecutionStatusPending)},
		{name: "timeout wins over cancelled", statuses: []string{"cancelled", "timeout"}, want: string(types.ExecutionStatusTimeout)},
		{name: "cancelled wins over running", statuses: []string{"running", "cancelled"}, want: string(types.ExecutionStatusCancelled)},
		{name: "running wins over queued", statuses: []string{"queued", "running"}, want: string(types.ExecutionStatusRunning)},
		{name: "queued wins over pending", statuses: []string{"pending", "queued"}, want: string(types.ExecutionStatusQueued)},
		{name: "pending wins over unknown", statuses: []string{"unknown", "pending"}, want: string(types.ExecutionStatusPending)},
		{name: "unknown when only unknown", statuses: []string{"unknown"}, want: string(types.ExecutionStatusUnknown)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			executionVCs := make([]types.ExecutionVC, 0, len(tt.statuses))
			for _, status := range tt.statuses {
				executionVCs = append(executionVCs, types.ExecutionVC{Status: status})
			}
			require.Equal(t, tt.want, service.determineWorkflowStatus(executionVCs))
		})
	}

	require.Equal(t, 2, service.countCompletedSteps([]types.ExecutionVC{
		{Status: "succeeded"},
		{Status: "failed"},
		{Status: "succeeded"},
	}))
}
