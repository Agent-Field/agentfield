package services

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

type workflowSummaryStorageStub struct {
	storage.StorageProvider
	aggregations []*types.WorkflowVCStatusAggregation
}

func (s *workflowSummaryStorageStub) ListWorkflowVCStatusSummaries(ctx context.Context, workflowIDs []string) ([]*types.WorkflowVCStatusAggregation, error) {
	return s.aggregations, nil
}

func TestVCServiceGetWorkflowVCStatusSummariesAggregationBranches(t *testing.T) {
	baseTime := time.Date(2026, 4, 9, 12, 0, 0, 0, time.UTC)
	stub := &workflowSummaryStorageStub{
		aggregations: []*types.WorkflowVCStatusAggregation{
			nil,
			{
				WorkflowID:    "workflow-verified",
				VCCount:       2,
				VerifiedCount: 2,
				FailedCount:   0,
				LastCreatedAt: services95TimePtr(baseTime.Add(-time.Minute)),
			},
			{
				WorkflowID:    "workflow-failed",
				VCCount:       2,
				VerifiedCount: 1,
				FailedCount:   1,
			},
			{
				WorkflowID:    "workflow-pending",
				VCCount:       2,
				VerifiedCount: 1,
				FailedCount:   0,
				LastCreatedAt: services95TimePtr(baseTime),
			},
		},
	}
	vcService := NewVCService(&config.DIDConfig{Enabled: true}, nil, nil)
	vcService.vcStorage = NewVCStorageWithStorage(stub)

	summaries, err := vcService.GetWorkflowVCStatusSummaries([]string{
		"workflow-verified",
		"workflow-failed",
		"workflow-pending",
		"",
		"workflow-verified",
		"workflow-missing",
	})
	require.NoError(t, err)

	require.Equal(t, "verified", summaries["workflow-verified"].VerificationStatus)
	require.Equal(t, 2, summaries["workflow-verified"].VCCount)
	require.Equal(t, 2, summaries["workflow-verified"].VerifiedCount)
	require.Equal(t, 0, summaries["workflow-verified"].FailedCount)
	require.Equal(t, baseTime.Add(-time.Minute).Format(time.RFC3339), summaries["workflow-verified"].LastVCCreated)

	require.Equal(t, "failed", summaries["workflow-failed"].VerificationStatus)
	require.Equal(t, 2, summaries["workflow-failed"].VCCount)
	require.Equal(t, 1, summaries["workflow-failed"].VerifiedCount)
	require.Equal(t, 1, summaries["workflow-failed"].FailedCount)

	require.Equal(t, "pending", summaries["workflow-pending"].VerificationStatus)
	require.Equal(t, 2, summaries["workflow-pending"].VCCount)
	require.Equal(t, 1, summaries["workflow-pending"].VerifiedCount)
	require.Equal(t, 0, summaries["workflow-pending"].FailedCount)
	require.Equal(t, baseTime.Format(time.RFC3339), summaries["workflow-pending"].LastVCCreated)

	require.Equal(t, "none", summaries["workflow-missing"].VerificationStatus)
	require.False(t, summaries["workflow-missing"].HasVCs)
}

func services95TimePtr(t time.Time) *time.Time {
	return &t
}

func TestVCServiceSignatureHelperErrorBranches(t *testing.T) {
	vcService, didService, _, _ := setupVCTestEnvironment(t)

	regResp, err := didService.RegisterAgent(&types.DIDRegistrationRequest{
		AgentNodeID: "agent-signature-branches",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner-1"}},
	})
	require.NoError(t, err)

	identity, err := didService.ResolveDID(regResp.IdentityPackage.ReasonerDIDs["reasoner-1"].DID)
	require.NoError(t, err)

	vcDoc := &types.VCDocument{
		Context:      []string{"https://www.w3.org/2018/credentials/v1"},
		Type:         []string{"VerifiableCredential", "AgentFieldExecutionCredential"},
		ID:           "urn:agentfield:vc:test-signature",
		Issuer:       identity.DID,
		IssuanceDate: time.Now().UTC().Format(time.RFC3339),
		CredentialSubject: types.VCCredentialSubject{
			ExecutionID: "exec-signature",
			WorkflowID:  "workflow-signature",
			SessionID:   "session-signature",
			Caller:      types.VCCaller{DID: identity.DID},
			Execution:   types.VCExecution{Status: string(types.ExecutionStatusSucceeded)},
		},
	}

	workflowDoc := &types.WorkflowVCDocument{
		Context:      []string{"https://www.w3.org/2018/credentials/v1"},
		Type:         []string{"VerifiableCredential", "AgentFieldWorkflowCredential"},
		ID:           "urn:agentfield:workflow-vc:test-signature",
		Issuer:       identity.DID,
		IssuanceDate: time.Now().UTC().Format(time.RFC3339),
		CredentialSubject: types.WorkflowVCCredentialSubject{
			WorkflowID:     "workflow-signature",
			SessionID:      "session-signature",
			ComponentVCIDs: []string{"vc-1"},
			TotalSteps:     1,
			CompletedSteps: 1,
			Status:         string(types.ExecutionStatusSucceeded),
			StartTime:      time.Now().UTC().Format(time.RFC3339),
			SnapshotTime:   time.Now().UTC().Format(time.RFC3339),
			Orchestrator:   types.VCCaller{DID: identity.DID},
		},
	}

	signature, err := vcService.signVC(vcDoc, identity)
	require.NoError(t, err)
	vcDoc.Proof = types.VCProof{ProofValue: signature}

	valid, err := vcService.verifyVCSignature(vcDoc, identity)
	require.NoError(t, err)
	require.True(t, valid)

	shortSeed := base64.RawURLEncoding.EncodeToString([]byte("short-seed"))
	shortPublicKey := base64.RawURLEncoding.EncodeToString([]byte("short-public-key"))

	signTests := []struct {
		name      string
		identity   *types.DIDIdentity
		wantErr   string
	}{
		{name: "invalid private jwk json", identity: &types.DIDIdentity{PrivateKeyJWK: "{not-json}"}, wantErr: "failed to parse private key JWK"},
		{name: "missing private key d", identity: &types.DIDIdentity{PrivateKeyJWK: `{"kty":"OKP"}`}, wantErr: "missing 'd' parameter"},
		{name: "invalid private key encoding", identity: &types.DIDIdentity{PrivateKeyJWK: `{"d":"%%%invalid%%%"}`}, wantErr: "failed to decode private key seed"},
		{name: "invalid private key length", identity: &types.DIDIdentity{PrivateKeyJWK: `{"d":"` + shortSeed + `"}`}, wantErr: "invalid private key seed length"},
	}

	for _, tt := range signTests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := vcService.signVC(vcDoc, tt.identity)
			require.ErrorContains(t, err, tt.wantErr)

			_, err = vcService.signWorkflowVC(workflowDoc, tt.identity)
			require.ErrorContains(t, err, tt.wantErr)
		})
	}

	verifyTests := []struct {
		name     string
		identity *types.DIDIdentity
		proof    string
		wantErr  string
	}{
		{name: "invalid public jwk json", identity: &types.DIDIdentity{PublicKeyJWK: "{not-json}"}, proof: signature, wantErr: "failed to parse public key JWK"},
		{name: "missing public key x", identity: &types.DIDIdentity{PublicKeyJWK: `{"kty":"OKP"}`}, proof: signature, wantErr: "missing 'x' parameter"},
		{name: "invalid public key encoding", identity: &types.DIDIdentity{PublicKeyJWK: `{"x":"%%%invalid%%%"}`}, proof: signature, wantErr: "failed to decode public key"},
		{name: "invalid public key length", identity: &types.DIDIdentity{PublicKeyJWK: `{"x":"` + shortPublicKey + `"}`}, proof: signature, wantErr: "invalid public key length"},
		{name: "invalid signature encoding", identity: identity, proof: "%%%invalid%%%", wantErr: "failed to decode signature"},
	}

	for _, tt := range verifyTests {
		t.Run(tt.name, func(t *testing.T) {
			docCopy := *vcDoc
			docCopy.Proof = types.VCProof{ProofValue: tt.proof}

			valid, err := vcService.verifyVCSignature(&docCopy, tt.identity)
			require.ErrorContains(t, err, tt.wantErr)
			require.False(t, valid)
		})
	}

	hashDisabledCfg := *didService.config
	hashDisabled := &VCService{config: &hashDisabledCfg}
	hashDisabled.config.VCRequirements.HashSensitiveData = false
	require.Empty(t, hashDisabled.hashData([]byte("secret")))

	hashEnabledCfg := *didService.config
	hashEnabled := &VCService{config: &hashEnabledCfg}
	hashEnabled.config.VCRequirements.HashSensitiveData = true
	require.NotEmpty(t, hashEnabled.hashData([]byte("secret")))

	unsignedAgentTagVC := &types.AgentTagVCDocument{
		Issuer: identity.DID,
		Proof:  &types.VCProof{Type: "UnsignedAuditRecord"},
	}
	valid, err = vcService.VerifyAgentTagVCSignature(unsignedAgentTagVC)
	require.ErrorContains(t, err, "VC has no valid signature")
	require.False(t, valid)
}

func TestVCServiceVerifyWorkflowVCComprehensiveAggregatesNegativeFindings(t *testing.T) {
	vcService, didService, _, _ := setupVCTestEnvironment(t)

	regResp, err := didService.RegisterAgent(&types.DIDRegistrationRequest{
		AgentNodeID: "agent-workflow-negative",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner-1"}},
	})
	require.NoError(t, err)

	callerDID := regResp.IdentityPackage.ReasonerDIDs["reasoner-1"].DID
	now := time.Now().UTC()

	badDoc := json.RawMessage(`{
		"@context":["https://example.com/custom"],
		"type":["VerifiableCredential"],
		"id":"urn:agentfield:vc:negative-1",
		"issuer":"did:key:missing",
		"issuanceDate":"not-a-timestamp",
		"credentialSubject":{
			"executionId":"exec-negative-doc",
			"workflowId":"workflow-negative-findings",
			"sessionId":"session-negative",
			"caller":{"did":"did:key:other-caller"},
			"target":{"did":"did:key:other-target"},
			"execution":{"inputHash":"other-input","outputHash":"other-output","status":"failed"}
		},
		"proof":{"proofValue":"not-the-same-signature"}
	}`)

	require.NoError(t, vcService.vcStorage.StoreExecutionVC(context.Background(), &types.ExecutionVC{
		VCID:         "vc-negative-1",
		ExecutionID:  "exec-negative-meta",
		WorkflowID:   "workflow-negative-findings",
		SessionID:    "session-negative",
		IssuerDID:    callerDID,
		CallerDID:    callerDID,
		TargetDID:    callerDID,
		VCDocument:   badDoc,
		Signature:    "meta-signature",
		DocumentSize: int64(len(badDoc)),
		InputHash:    "meta-input",
		OutputHash:   "meta-output",
		Status:       string(types.ExecutionStatusSucceeded),
		CreatedAt:    now.Add(-time.Minute),
	}))

	require.NoError(t, vcService.vcStorage.StoreExecutionVC(context.Background(), &types.ExecutionVC{
		VCID:         "vc-negative-2",
		ExecutionID:  "exec-negative-parse",
		WorkflowID:   "workflow-negative-findings",
		SessionID:    "session-negative",
		IssuerDID:    callerDID,
		CallerDID:    callerDID,
		VCDocument:   json.RawMessage(`{"broken"`),
		Signature:    "sig",
		DocumentSize: 9,
		Status:       string(types.ExecutionStatusFailed),
		CreatedAt:    now,
	}))

	result, err := vcService.VerifyWorkflowVCComprehensive("workflow-negative-findings")
	require.NoError(t, err)
	require.False(t, result.Valid)
	require.False(t, result.IntegrityChecks.MetadataConsistency)
	require.False(t, result.IntegrityChecks.FieldConsistency)
	require.False(t, result.IntegrityChecks.TimestampValidation)
	require.False(t, result.IntegrityChecks.HashValidation)
	require.False(t, result.IntegrityChecks.StructuralIntegrity)
	require.False(t, result.SecurityAnalysis.DIDAuthenticity)
	require.NotEmpty(t, result.SecurityAnalysis.TamperEvidence)
	require.False(t, result.ComplianceChecks.W3CCompliance)
	require.False(t, result.ComplianceChecks.AgentFieldStandardCompliance)
	require.NotEmpty(t, result.CriticalIssues)
	require.NotEmpty(t, result.Warnings)

	var criticalTypes []string
	for _, issue := range result.CriticalIssues {
		criticalTypes = append(criticalTypes, issue.Type)
	}
	require.Contains(t, criticalTypes, "parse_error")
	require.Contains(t, criticalTypes, "did_resolution_failed")

	var warningTypes []string
	for _, issue := range result.Warnings {
		warningTypes = append(warningTypes, issue.Type)
	}
	require.Contains(t, warningTypes, "tamper_evidence")
	require.Contains(t, warningTypes, "w3c_compliance_failure")
	require.Contains(t, warningTypes, "agentfield_compliance_failure")
}

func TestExecutionsUIServiceGroupExecutionsAdditionalBranches(t *testing.T) {
	service := &ExecutionsUIService{}
	now := time.Date(2026, 4, 9, 12, 0, 0, 0, time.UTC)
	sessionID := "session-1"
	actorID := "actor-1"

	executions := []*types.WorkflowExecution{
		{WorkflowID: "wf-1", ExecutionID: "exec-1", SessionID: &sessionID, ActorID: &actorID, AgentNodeID: "node-1", Status: "running", StartedAt: now},
		{WorkflowID: "wf-2", ExecutionID: "exec-2", AgentNodeID: "node-2", Status: "failed", StartedAt: now.Add(time.Minute)},
	}

	sessionGroups := service.groupExecutions(executions, "session")
	require.Len(t, sessionGroups, 2)
	require.Equal(t, "session-1", groupedExecutionByKey(t, sessionGroups, "session-1").GroupLabel)
	require.Equal(t, "No Session", groupedExecutionByKey(t, sessionGroups, "no-session").GroupLabel)

	actorGroups := service.groupExecutions(executions, "actor")
	require.Len(t, actorGroups, 2)
	require.Equal(t, "actor-1", groupedExecutionByKey(t, actorGroups, "actor-1").GroupLabel)
	require.Equal(t, "No Actor", groupedExecutionByKey(t, actorGroups, "no-actor").GroupLabel)

	statusGroups := service.groupExecutions(executions, "status")
	require.Len(t, statusGroups, 2)
	require.Equal(t, 1, groupedExecutionByKey(t, statusGroups, "running").Count)
	require.Equal(t, 1, groupedExecutionByKey(t, statusGroups, "failed").Count)

	require.Empty(t, service.groupExecutions(executions, "invalid"))
	require.Empty(t, service.groupExecutions(executions, "none"))
}

func TestHealthMonitorUnifiedAndFallbackBranches(t *testing.T) {
	t.Run("unified status manager path updates active and inactive", func(t *testing.T) {
		storageStub := &uiStorageStub{
			agentsByID: map[string]*types.AgentNode{
				"node-1": {ID: "node-1", LifecycleStatus: types.AgentStatusOffline},
			},
		}
		statusManager := NewStatusManager(storageStub, StatusManagerConfig{}, nil, nil)
		presence := NewPresenceManager(statusManager, PresenceManagerConfig{HeartbeatTTL: time.Minute})
		hm := NewHealthMonitor(storageStub, HealthMonitorConfig{}, nil, nil, statusManager, presence)

		hm.markAgentActive("node-1")
		require.Equal(t, types.HealthStatusActive, storageStub.updatedHealth["node-1"])
		require.Equal(t, types.AgentStatusReady, storageStub.updatedLifecycle["node-1"])
		require.True(t, presence.HasLease("node-1"))

		hm.markAgentInactive("node-1", 2)
		require.Equal(t, types.HealthStatusInactive, storageStub.updatedHealth["node-1"])
		require.Equal(t, types.AgentStatusOffline, storageStub.updatedLifecycle["node-1"])
	})

	t.Run("status manager failure falls back to direct health update", func(t *testing.T) {
		storageStub := &uiStorageStub{
			getAgentErr: errors.New("lookup failed"),
			agentsByID: map[string]*types.AgentNode{
				"node-2": {ID: "node-2"},
			},
		}
		statusManager := NewStatusManager(storageStub, StatusManagerConfig{}, nil, nil)
		hm := NewHealthMonitor(storageStub, HealthMonitorConfig{}, nil, nil, statusManager, nil)

		hm.markAgentActive("node-2")
		require.Equal(t, types.HealthStatusActive, storageStub.updatedHealth["node-2"])

		hm.markAgentInactive("node-2", 3)
		require.Equal(t, types.HealthStatusInactive, storageStub.updatedHealth["node-2"])
	})
}
