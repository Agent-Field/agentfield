package services

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

func setupVCTestEnvironment(t *testing.T) (*VCService, *DIDService, storage.StorageProvider, context.Context) {
	t.Helper()

	provider, ctx := setupTestStorage(t)
	registry := NewDIDRegistryWithStorage(provider)
	require.NoError(t, registry.Initialize())

	keystoreDir := filepath.Join(t.TempDir(), "keys")
	ks, err := NewKeystoreService(&config.KeystoreConfig{Path: keystoreDir, Type: "local"})
	require.NoError(t, err)

	didCfg := &config.DIDConfig{
		Enabled:  true,
		Keystore: config.KeystoreConfig{Path: keystoreDir, Type: "local"},
		VCRequirements: config.VCRequirements{
			RequireVCForExecution: true,
			PersistExecutionVC:    true,
			HashSensitiveData:     true,
		},
	}

	didService := NewDIDService(didCfg, ks, registry)
	agentfieldID := "agentfield-vc-test"
	require.NoError(t, didService.Initialize(agentfieldID))

	vcService := NewVCService(didCfg, didService, provider)
	require.NoError(t, vcService.Initialize())

	return vcService, didService, provider, ctx
}

func setupExecutionVCForVerificationTest(t *testing.T) (*VCService, *DIDService, context.Context, *types.ExecutionVC, *types.VCDocument) {
	t.Helper()

	vcService, didService, _, ctx := setupVCTestEnvironment(t)

	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-verify",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
	}

	regResp, err := didService.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, regResp.Success)

	callerDID := regResp.IdentityPackage.ReasonerDIDs["reasoner1"].DID

	execCtx := &types.ExecutionContext{
		ExecutionID:  "exec-verify",
		WorkflowID:   "workflow-1",
		SessionID:    "session-1",
		CallerDID:    callerDID,
		TargetDID:    "",
		AgentNodeDID: regResp.IdentityPackage.AgentDID.DID,
		Timestamp:    time.Now(),
	}

	vc, err := vcService.GenerateExecutionVC(execCtx, []byte(`{"input": "test"}`), []byte(`{"output": "result"}`), "succeeded", nil, 100)
	require.NoError(t, err)

	var vcDoc types.VCDocument
	require.NoError(t, json.Unmarshal(vc.VCDocument, &vcDoc))

	return vcService, didService, ctx, vc, &vcDoc
}

func marshalSignedVCDocument(t *testing.T, vcService *VCService, didService *DIDService, vcDoc *types.VCDocument) json.RawMessage {
	t.Helper()

	issuerIdentity, err := didService.ResolveDID(vcDoc.Issuer)
	require.NoError(t, err)

	signature, err := vcService.signVC(vcDoc, issuerIdentity)
	require.NoError(t, err)

	vcDoc.Proof.ProofValue = signature

	vcDocument, err := json.Marshal(vcDoc)
	require.NoError(t, err)

	return vcDocument
}

type executionVCListStub struct {
	storage.StorageProvider
	records     []*types.ExecutionVCInfo
	err         error
	lastFilters types.VCFilters
}

func (s *executionVCListStub) ListExecutionVCs(_ context.Context, filters types.VCFilters) ([]*types.ExecutionVCInfo, error) {
	s.lastFilters = filters
	if s.err != nil {
		return nil, s.err
	}
	return s.records, nil
}

func TestVCService_IsExecutionVCEnabled(t *testing.T) {
	tests := []struct {
		name     string
		config   *config.DIDConfig
		expected bool
	}{
		{
			name:     "nil service",
			config:   nil,
			expected: false,
		},
		{
			name: "disabled DID system",
			config: &config.DIDConfig{
				Enabled: false,
				VCRequirements: config.VCRequirements{
					RequireVCForExecution: true,
				},
			},
			expected: false,
		},
		{
			name: "enabled but VC not required",
			config: &config.DIDConfig{
				Enabled: true,
				VCRequirements: config.VCRequirements{
					RequireVCForExecution: false,
				},
			},
			expected: false,
		},
		{
			name: "enabled and VC required",
			config: &config.DIDConfig{
				Enabled: true,
				VCRequirements: config.VCRequirements{
					RequireVCForExecution: true,
				},
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var svc *VCService
			if tt.config != nil {
				svc = &VCService{config: tt.config}
			}
			result := svc.IsExecutionVCEnabled()
			require.Equal(t, tt.expected, result)
		})
	}
}

func TestVCService_ShouldPersistExecutionVC(t *testing.T) {
	tests := []struct {
		name     string
		config   *config.DIDConfig
		expected bool
	}{
		{
			name:     "nil service",
			config:   nil,
			expected: false,
		},
		{
			name: "disabled DID system",
			config: &config.DIDConfig{
				Enabled: false,
				VCRequirements: config.VCRequirements{
					PersistExecutionVC: true,
				},
			},
			expected: false,
		},
		{
			name: "enabled but persistence disabled",
			config: &config.DIDConfig{
				Enabled: true,
				VCRequirements: config.VCRequirements{
					PersistExecutionVC: false,
				},
			},
			expected: false,
		},
		{
			name: "enabled and persistence enabled",
			config: &config.DIDConfig{
				Enabled: true,
				VCRequirements: config.VCRequirements{
					PersistExecutionVC: true,
				},
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var svc *VCService
			if tt.config != nil {
				svc = &VCService{config: tt.config}
			}
			result := svc.ShouldPersistExecutionVC()
			require.Equal(t, tt.expected, result)
		})
	}
}

func TestVCService_GenerateExecutionVC_Success(t *testing.T) {
	vcService, didService, provider, ctx := setupVCTestEnvironment(t)

	// Register an agent first
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-vc-test",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
		Skills:      []types.SkillDefinition{{ID: "skill1"}},
	}

	regResp, err := didService.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, regResp.Success)

	callerDID := regResp.IdentityPackage.ReasonerDIDs["reasoner1"].DID
	targetDID := regResp.IdentityPackage.SkillDIDs["skill1"].DID

	execCtx := &types.ExecutionContext{
		ExecutionID:  "exec-1",
		WorkflowID:   "workflow-1",
		SessionID:    "session-1",
		CallerDID:    callerDID,
		TargetDID:    targetDID,
		AgentNodeDID: regResp.IdentityPackage.AgentDID.DID,
		Timestamp:    time.Now(),
	}

	inputData := []byte(`{"input": "test"}`)
	outputData := []byte(`{"output": "result"}`)
	status := "succeeded"

	vc, err := vcService.GenerateExecutionVC(execCtx, inputData, outputData, status, nil, 100)
	require.NoError(t, err)
	require.NotNil(t, vc)
	require.NotEmpty(t, vc.VCID)
	require.Equal(t, execCtx.ExecutionID, vc.ExecutionID)
	require.Equal(t, execCtx.WorkflowID, vc.WorkflowID)
	require.Equal(t, execCtx.SessionID, vc.SessionID)
	require.Equal(t, callerDID, vc.CallerDID)
	require.Equal(t, targetDID, vc.TargetDID)
	require.NotEmpty(t, vc.Signature)
	require.NotEmpty(t, vc.VCDocument)
	require.NotEmpty(t, vc.InputHash)
	require.NotEmpty(t, vc.OutputHash)
	require.Equal(t, "succeeded", vc.Status)
	require.Greater(t, vc.DocumentSize, int64(0))

	// Verify VC document structure
	var vcDoc types.VCDocument
	err = json.Unmarshal(vc.VCDocument, &vcDoc)
	require.NoError(t, err)
	require.NotEmpty(t, vcDoc.Context)
	require.NotEmpty(t, vcDoc.Type)
	require.NotEmpty(t, vcDoc.ID)
	require.Equal(t, callerDID, vcDoc.Issuer)
	require.NotEmpty(t, vcDoc.IssuanceDate)
	require.NotEmpty(t, vcDoc.Proof.ProofValue)
	require.Equal(t, "Ed25519Signature2020", vcDoc.Proof.Type)

	// Verify VC was stored
	storedVC, err := provider.GetExecutionVC(ctx, vc.VCID)
	require.NoError(t, err)
	require.NotNil(t, storedVC)
	require.Equal(t, vc.VCID, storedVC.VCID)
}

func TestVCService_GenerateExecutionVC_WithError(t *testing.T) {
	vcService, didService, _, _ := setupVCTestEnvironment(t)

	// Register an agent
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-vc-error",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
	}

	regResp, err := didService.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, regResp.Success)

	callerDID := regResp.IdentityPackage.ReasonerDIDs["reasoner1"].DID

	execCtx := &types.ExecutionContext{
		ExecutionID:  "exec-error",
		WorkflowID:   "workflow-1",
		SessionID:    "session-1",
		CallerDID:    callerDID,
		TargetDID:    "",
		AgentNodeDID: regResp.IdentityPackage.AgentDID.DID,
		Timestamp:    time.Now(),
	}

	errorMsg := "test error message"
	vc, err := vcService.GenerateExecutionVC(execCtx, []byte(`{}`), nil, "failed", &errorMsg, 50)
	require.NoError(t, err)
	require.NotNil(t, vc)
	require.Equal(t, "failed", vc.Status)

	// Verify error message is in VC document
	var vcDoc types.VCDocument
	err = json.Unmarshal(vc.VCDocument, &vcDoc)
	require.NoError(t, err)
	require.Equal(t, errorMsg, vcDoc.CredentialSubject.Execution.ErrorMessage)
}

func TestVCService_GenerateExecutionVC_DisabledSystem(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	cfg := &config.DIDConfig{
		Enabled: false,
		VCRequirements: config.VCRequirements{
			RequireVCForExecution: true,
		},
	}

	vcService := NewVCService(cfg, nil, provider)

	execCtx := &types.ExecutionContext{
		ExecutionID: "exec-1",
		WorkflowID:  "workflow-1",
		SessionID:   "session-1",
		CallerDID:   "did:key:test",
		Timestamp:   time.Now(),
	}

	_, err := vcService.GenerateExecutionVC(execCtx, []byte(`{}`), []byte(`{}`), "succeeded", nil, 100)
	require.Error(t, err)
	require.Contains(t, err.Error(), "DID system is disabled")
	_ = ctx
}

func TestVCService_GenerateExecutionVC_VCNotRequired(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	cfg := &config.DIDConfig{
		Enabled: true,
		VCRequirements: config.VCRequirements{
			RequireVCForExecution: false,
		},
	}

	vcService := NewVCService(cfg, nil, provider)

	execCtx := &types.ExecutionContext{
		ExecutionID: "exec-1",
		WorkflowID:  "workflow-1",
		SessionID:   "session-1",
		CallerDID:   "did:key:test",
		Timestamp:   time.Now(),
	}

	vc, err := vcService.GenerateExecutionVC(execCtx, []byte(`{}`), []byte(`{}`), "succeeded", nil, 100)
	require.NoError(t, err)
	require.Nil(t, vc) // Should return nil when VC generation is disabled by config
	_ = ctx
}

func TestVCService_GenerateExecutionVC_EmptyData(t *testing.T) {
	vcService, didService, _, _ := setupVCTestEnvironment(t)

	// Register an agent
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-empty",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
	}

	regResp, err := didService.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, regResp.Success)

	callerDID := regResp.IdentityPackage.ReasonerDIDs["reasoner1"].DID

	execCtx := &types.ExecutionContext{
		ExecutionID:  "exec-empty",
		WorkflowID:   "workflow-1",
		SessionID:    "session-1",
		CallerDID:    callerDID,
		TargetDID:    "",
		AgentNodeDID: regResp.IdentityPackage.AgentDID.DID,
		Timestamp:    time.Now(),
	}

	// Test with nil input/output
	vc, err := vcService.GenerateExecutionVC(execCtx, nil, nil, "succeeded", nil, 0)
	require.NoError(t, err)
	require.NotNil(t, vc)

	// Test with empty slices
	vc2, err := vcService.GenerateExecutionVC(execCtx, []byte{}, []byte{}, "succeeded", nil, 0)
	require.NoError(t, err)
	require.NotNil(t, vc2)
}

func TestVCService_GenerateExecutionVC_LongErrorMessage(t *testing.T) {
	vcService, didService, _, _ := setupVCTestEnvironment(t)

	// Register an agent
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-long-error",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
	}

	regResp, err := didService.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, regResp.Success)

	callerDID := regResp.IdentityPackage.ReasonerDIDs["reasoner1"].DID

	execCtx := &types.ExecutionContext{
		ExecutionID:  "exec-long-error",
		WorkflowID:   "workflow-1",
		SessionID:    "session-1",
		CallerDID:    callerDID,
		TargetDID:    "",
		AgentNodeDID: regResp.IdentityPackage.AgentDID.DID,
		Timestamp:    time.Now(),
	}

	// Create error message longer than 500 characters
	longError := make([]byte, 600)
	for i := range longError {
		longError[i] = 'a'
	}
	errorMsg := string(longError)

	vc, err := vcService.GenerateExecutionVC(execCtx, []byte(`{}`), nil, "failed", &errorMsg, 50)
	require.NoError(t, err)
	require.NotNil(t, vc)

	// Verify error message was truncated
	var vcDoc types.VCDocument
	err = json.Unmarshal(vc.VCDocument, &vcDoc)
	require.NoError(t, err)
	require.LessOrEqual(t, len(vcDoc.CredentialSubject.Execution.ErrorMessage), 500+len("...[truncated]"))
	require.Contains(t, vcDoc.CredentialSubject.Execution.ErrorMessage, "...[truncated]")
}

func TestVCService_VerifyVC_Success(t *testing.T) {
	vcService, _, _, vc, _ := setupExecutionVCForVerificationTest(t)

	// Verify the VC
	verifyResp, err := vcService.VerifyVC(vc.VCDocument)
	require.NoError(t, err)
	require.NotNil(t, verifyResp)
	require.True(t, verifyResp.Valid)
	require.Empty(t, verifyResp.Reason)
	require.Equal(t, vc.IssuerDID, verifyResp.IssuerDID)
	require.NotEmpty(t, verifyResp.IssuedAt)
	require.Contains(t, verifyResp.Message, "verified successfully")
}

func TestVCService_VerifyVC_InvalidDocument(t *testing.T) {
	vcService, _, _, _ := setupVCTestEnvironment(t)

	invalidDoc := json.RawMessage(`{"invalid":`)
	verifyResp, err := vcService.VerifyVC(invalidDoc)
	require.NoError(t, err)
	require.NotNil(t, verifyResp)
	require.False(t, verifyResp.Valid)
	require.Equal(t, types.VCVerificationReasonInvalidDocument, verifyResp.Reason)
	require.Contains(t, verifyResp.Error, "failed to parse VC document")
}

func TestVCService_VerifyVC_DisabledSystem(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	cfg := &config.DIDConfig{
		Enabled: false,
	}

	vcService := NewVCService(cfg, nil, provider)

	doc := json.RawMessage(`{"@context": ["https://www.w3.org/2018/credentials/v1"]}`)
	verifyResp, err := vcService.VerifyVC(doc)
	require.NoError(t, err)
	require.NotNil(t, verifyResp)
	require.False(t, verifyResp.Valid)
	require.Equal(t, types.VCVerificationReasonSystemDisabled, verifyResp.Reason)
	require.Contains(t, verifyResp.Error, "DID system is disabled")
	_ = ctx
}

func TestVCService_VerifyVC_TamperedSignature(t *testing.T) {
	vcService, _, _, vc, vcDoc := setupExecutionVCForVerificationTest(t)

	// Tamper with the signature
	vcDoc.Proof.ProofValue = "tampered_signature"

	tamperedDoc, err := json.Marshal(vcDoc)
	require.NoError(t, err)

	verifyResp, err := vcService.VerifyVC(tamperedDoc)
	require.NoError(t, err)
	require.NotNil(t, verifyResp)
	require.False(t, verifyResp.Valid)
	require.Equal(t, types.VCVerificationReasonInvalidSignature, verifyResp.Reason)
	require.Contains(t, verifyResp.Message, "Invalid signature")
	require.Equal(t, vc.IssuerDID, vcDoc.Issuer)
}

func TestVCService_VerifyVC_InvalidSignatureEncoding(t *testing.T) {
	vcService, _, _, _, vcDoc := setupExecutionVCForVerificationTest(t)

	vcDoc.Proof.ProofValue = "not base64!"

	invalidDoc, err := json.Marshal(vcDoc)
	require.NoError(t, err)

	verifyResp, err := vcService.VerifyVC(invalidDoc)
	require.NoError(t, err)
	require.NotNil(t, verifyResp)
	require.False(t, verifyResp.Valid)
	require.Equal(t, types.VCVerificationReasonInvalidSignature, verifyResp.Reason)
	require.Contains(t, verifyResp.Error, "failed to verify signature")
}

func TestVCService_VerifyVC_InvalidIssuerDID(t *testing.T) {
	vcService, _, _, _, vcDoc := setupExecutionVCForVerificationTest(t)

	// Change the issuer DID to an invalid one
	vcDoc.Issuer = "did:key:invalid"

	invalidDoc, err := json.Marshal(vcDoc)
	require.NoError(t, err)

	verifyResp, err := vcService.VerifyVC(invalidDoc)
	require.NoError(t, err)
	require.NotNil(t, verifyResp)
	require.False(t, verifyResp.Valid)
	require.Equal(t, types.VCVerificationReasonUnknownIssuer, verifyResp.Reason)
	require.Contains(t, verifyResp.Error, "failed to resolve issuer DID")
}

func TestVCService_VerifyVC_EnforcesLifecycleAndProofValidation(t *testing.T) {
	tests := []struct {
		name              string
		mutate            func(t *testing.T, vcService *VCService, didService *DIDService, ctx context.Context, execVC *types.ExecutionVC, vcDoc *types.VCDocument) json.RawMessage
		wantValid         bool
		wantReason        types.VCVerificationReason
		wantMessageSubstr string
		wantErrorSubstr   string
	}{
		{
			name:      "valid",
			wantValid: true,
		},
		{
			name: "expired",
			mutate: func(t *testing.T, vcService *VCService, didService *DIDService, ctx context.Context, execVC *types.ExecutionVC, vcDoc *types.VCDocument) json.RawMessage {
				vcDoc.ExpirationDate = formatVCDateTime(time.Now().Add(-1 * time.Hour))
				return marshalSignedVCDocument(t, vcService, didService, vcDoc)
			},
			wantReason:        types.VCVerificationReasonExpired,
			wantMessageSubstr: "expired",
		},
		{
			name: "not before in future",
			mutate: func(t *testing.T, vcService *VCService, didService *DIDService, ctx context.Context, execVC *types.ExecutionVC, vcDoc *types.VCDocument) json.RawMessage {
				vcDoc.NotBefore = formatVCDateTime(time.Now().Add(1 * time.Hour))
				return marshalSignedVCDocument(t, vcService, didService, vcDoc)
			},
			wantReason:        types.VCVerificationReasonNotYetValid,
			wantMessageSubstr: "not valid before",
		},
		{
			name: "revoked",
			mutate: func(t *testing.T, vcService *VCService, didService *DIDService, ctx context.Context, execVC *types.ExecutionVC, vcDoc *types.VCDocument) json.RawMessage {
				execVC.Status = "revoked"
				require.NoError(t, vcService.vcStorage.StoreExecutionVC(ctx, execVC))
				return execVC.VCDocument
			},
			wantReason:        types.VCVerificationReasonRevoked,
			wantMessageSubstr: "revoked",
		},
		{
			name: "proof purpose mismatch",
			mutate: func(t *testing.T, vcService *VCService, didService *DIDService, ctx context.Context, execVC *types.ExecutionVC, vcDoc *types.VCDocument) json.RawMessage {
				vcDoc.Proof.ProofPurpose = "authentication"
				vcDocument, err := json.Marshal(vcDoc)
				require.NoError(t, err)
				return vcDocument
			},
			wantReason:        types.VCVerificationReasonProofPurposeMismatch,
			wantMessageSubstr: "proofPurpose",
		},
		{
			name: "unknown issuer",
			mutate: func(t *testing.T, vcService *VCService, didService *DIDService, ctx context.Context, execVC *types.ExecutionVC, vcDoc *types.VCDocument) json.RawMessage {
				vcDoc.Issuer = "did:key:unknown"
				vcDocument, err := json.Marshal(vcDoc)
				require.NoError(t, err)
				return vcDocument
			},
			wantReason:      types.VCVerificationReasonUnknownIssuer,
			wantErrorSubstr: "failed to resolve issuer DID",
		},
		{
			name: "invalid signature",
			mutate: func(t *testing.T, vcService *VCService, didService *DIDService, ctx context.Context, execVC *types.ExecutionVC, vcDoc *types.VCDocument) json.RawMessage {
				vcDoc.Proof.ProofValue = "tampered_signature"
				vcDocument, err := json.Marshal(vcDoc)
				require.NoError(t, err)
				return vcDocument
			},
			wantReason:        types.VCVerificationReasonInvalidSignature,
			wantMessageSubstr: "Invalid signature",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vcService, didService, ctx, execVC, vcDoc := setupExecutionVCForVerificationTest(t)

			vcDocument := execVC.VCDocument
			if tt.mutate != nil {
				vcDocument = tt.mutate(t, vcService, didService, ctx, execVC, vcDoc)
			}

			verifyResp, err := vcService.VerifyVC(vcDocument)
			require.NoError(t, err)
			require.NotNil(t, verifyResp)
			require.Equal(t, tt.wantValid, verifyResp.Valid)
			require.Equal(t, tt.wantReason, verifyResp.Reason)

			if tt.wantValid {
				require.Equal(t, execVC.IssuerDID, verifyResp.IssuerDID)
				require.NotEmpty(t, verifyResp.IssuedAt)
				require.Contains(t, verifyResp.Message, "verified successfully")
				return
			}

			if tt.wantMessageSubstr != "" {
				require.Contains(t, verifyResp.Message, tt.wantMessageSubstr)
			}
			if tt.wantErrorSubstr != "" {
				require.Contains(t, verifyResp.Error, tt.wantErrorSubstr)
			}
		})
	}
}

func TestParseVCDateTime(t *testing.T) {
	ts := time.Date(2026, time.May, 10, 14, 30, 0, 0, time.FixedZone("UTC-7", -7*60*60))

	parsed, err := parseVCDateTime("issuanceDate", formatVCDateTime(ts))
	require.NoError(t, err)
	require.Equal(t, ts.UTC(), parsed)

	_, err = parseVCDateTime("issuanceDate", "not-a-time")
	require.Error(t, err)
	require.Contains(t, err.Error(), "invalid issuanceDate")
}

func TestVCService_VerifyVCProofPurpose(t *testing.T) {
	vcService := &VCService{}

	tests := []struct {
		name       string
		vcDoc      *types.VCDocument
		wantReason types.VCVerificationReason
		wantNil    bool
	}{
		{
			name:       "nil document",
			vcDoc:      nil,
			wantReason: types.VCVerificationReasonInvalidDocument,
		},
		{
			name:    "expected proof purpose",
			vcDoc:   &types.VCDocument{Proof: types.VCProof{ProofPurpose: expectedVCProofPurpose}},
			wantNil: true,
		},
		{
			name:       "mismatched proof purpose",
			vcDoc:      &types.VCDocument{Proof: types.VCProof{ProofPurpose: "authentication"}},
			wantReason: types.VCVerificationReasonProofPurposeMismatch,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := vcService.verifyVCProofPurpose(tt.vcDoc)
			if tt.wantNil {
				require.Nil(t, resp)
				return
			}

			require.NotNil(t, resp)
			require.Equal(t, tt.wantReason, resp.Reason)
		})
	}
}

func TestVCService_VerifyVCValidityWindow(t *testing.T) {
	vcService := &VCService{}
	now := time.Date(2026, time.May, 10, 14, 0, 0, 0, time.UTC)
	baseDoc := &types.VCDocument{
		IssuanceDate: formatVCDateTime(now.Add(-1 * time.Hour)),
	}

	cloneDoc := func() *types.VCDocument {
		doc := *baseDoc
		return &doc
	}

	tests := []struct {
		name              string
		mutate            func(*types.VCDocument)
		wantReason        types.VCVerificationReason
		wantMessageSubstr string
		wantErrorSubstr   string
		wantNil           bool
	}{
		{
			name:            "nil document",
			wantReason:      types.VCVerificationReasonInvalidDocument,
			wantErrorSubstr: "VC document is nil",
			mutate:          nil,
		},
		{
			name: "invalid issuance date",
			mutate: func(doc *types.VCDocument) {
				doc.IssuanceDate = "yesterday-ish"
			},
			wantReason:      types.VCVerificationReasonInvalidDocument,
			wantErrorSubstr: "failed to validate VC issuance date",
		},
		{
			name: "issuance date in future",
			mutate: func(doc *types.VCDocument) {
				doc.IssuanceDate = formatVCDateTime(now.Add(1 * time.Hour))
			},
			wantReason:        types.VCVerificationReasonNotYetValid,
			wantMessageSubstr: "issuanceDate",
		},
		{
			name: "invalid not before",
			mutate: func(doc *types.VCDocument) {
				doc.NotBefore = "not-yet"
			},
			wantReason:      types.VCVerificationReasonInvalidDocument,
			wantErrorSubstr: "failed to validate VC notBefore",
		},
		{
			name: "not before in future",
			mutate: func(doc *types.VCDocument) {
				doc.NotBefore = formatVCDateTime(now.Add(2 * time.Hour))
			},
			wantReason:        types.VCVerificationReasonNotYetValid,
			wantMessageSubstr: "not valid before",
		},
		{
			name:    "no expiration date",
			wantNil: true,
		},
		{
			name: "invalid expiration date",
			mutate: func(doc *types.VCDocument) {
				doc.ExpirationDate = "someday"
			},
			wantReason:      types.VCVerificationReasonInvalidDocument,
			wantErrorSubstr: "failed to validate VC expiration date",
		},
		{
			name: "expired",
			mutate: func(doc *types.VCDocument) {
				doc.ExpirationDate = formatVCDateTime(now.Add(-1 * time.Minute))
			},
			wantReason:        types.VCVerificationReasonExpired,
			wantMessageSubstr: "expired",
		},
		{
			name: "expiration in future",
			mutate: func(doc *types.VCDocument) {
				doc.ExpirationDate = formatVCDateTime(now.Add(1 * time.Hour))
			},
			wantNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var doc *types.VCDocument
			if tt.name == "nil document" {
				doc = nil
			} else {
				doc = cloneDoc()
				if tt.mutate != nil {
					tt.mutate(doc)
				}
			}

			resp := vcService.verifyVCValidityWindow(doc, now)
			if tt.wantNil {
				require.Nil(t, resp)
				return
			}

			require.NotNil(t, resp)
			require.Equal(t, tt.wantReason, resp.Reason)
			if tt.wantMessageSubstr != "" {
				require.Contains(t, resp.Message, tt.wantMessageSubstr)
			}
			if tt.wantErrorSubstr != "" {
				require.Contains(t, resp.Error, tt.wantErrorSubstr)
			}
		})
	}
}

func TestVCService_VerifyVCRevocation(t *testing.T) {
	baseDoc := &types.VCDocument{
		Issuer: "did:key:issuer",
		CredentialSubject: types.VCCredentialSubject{
			ExecutionID: "exec-verify",
			Target: types.VCTarget{
				DID: "did:key:target",
			},
			Execution: types.VCExecution{
				Status: "succeeded",
			},
		},
	}

	cloneDoc := func() *types.VCDocument {
		doc := *baseDoc
		return &doc
	}

	t.Run("nil document", func(t *testing.T) {
		resp := (&VCService{}).verifyVCRevocation(nil)
		require.NotNil(t, resp)
		require.Equal(t, types.VCVerificationReasonInvalidDocument, resp.Reason)
		require.Contains(t, resp.Error, "VC document is nil")
	})

	t.Run("credential subject revoked", func(t *testing.T) {
		doc := cloneDoc()
		doc.CredentialSubject.Execution.Status = " Revoked "

		resp := (&VCService{}).verifyVCRevocation(doc)
		require.NotNil(t, resp)
		require.Equal(t, types.VCVerificationReasonRevoked, resp.Reason)
		require.Contains(t, resp.Message, "revoked")
	})

	t.Run("missing storage provider returns nil", func(t *testing.T) {
		resp := (&VCService{}).verifyVCRevocation(cloneDoc())
		require.Nil(t, resp)
	})

	t.Run("storage lookup error is ignored", func(t *testing.T) {
		provider := &executionVCListStub{err: errors.New("lookup failed")}
		vcService := &VCService{vcStorage: NewVCStorageWithStorage(provider)}

		resp := vcService.verifyVCRevocation(cloneDoc())
		require.Nil(t, resp)
	})

	t.Run("nil records are skipped and filters are populated", func(t *testing.T) {
		provider := &executionVCListStub{
			records: []*types.ExecutionVCInfo{
				nil,
				{Status: "succeeded"},
			},
		}
		vcService := &VCService{vcStorage: NewVCStorageWithStorage(provider)}

		resp := vcService.verifyVCRevocation(cloneDoc())
		require.Nil(t, resp)
		require.NotNil(t, provider.lastFilters.ExecutionID)
		require.NotNil(t, provider.lastFilters.IssuerDID)
		require.NotNil(t, provider.lastFilters.TargetDID)
		require.Equal(t, baseDoc.CredentialSubject.ExecutionID, *provider.lastFilters.ExecutionID)
		require.Equal(t, baseDoc.Issuer, *provider.lastFilters.IssuerDID)
		require.Equal(t, baseDoc.CredentialSubject.Target.DID, *provider.lastFilters.TargetDID)
	})

	t.Run("revoked record from storage", func(t *testing.T) {
		provider := &executionVCListStub{
			records: []*types.ExecutionVCInfo{
				{Status: " revoked "},
			},
		}
		vcService := &VCService{vcStorage: NewVCStorageWithStorage(provider)}

		resp := vcService.verifyVCRevocation(cloneDoc())
		require.NotNil(t, resp)
		require.Equal(t, types.VCVerificationReasonRevoked, resp.Reason)
		require.Contains(t, resp.Message, "revoked")
	})
}

func TestVCService_GetWorkflowVCStatusSummaries(t *testing.T) {
	vcService, _, _, _ := setupVCTestEnvironment(t)

	// Test with empty workflow IDs
	summaries, err := vcService.GetWorkflowVCStatusSummaries([]string{})
	require.NoError(t, err)
	require.Empty(t, summaries)

	// Test with nil workflow IDs
	summaries, err = vcService.GetWorkflowVCStatusSummaries(nil)
	require.NoError(t, err)
	require.Empty(t, summaries)

	// Test with workflow IDs (no VCs stored yet)
	summaries, err = vcService.GetWorkflowVCStatusSummaries([]string{"workflow-1", "workflow-2"})
	require.NoError(t, err)
	require.Len(t, summaries, 2)
	require.Equal(t, "none", summaries["workflow-1"].VerificationStatus)
	require.Equal(t, "none", summaries["workflow-2"].VerificationStatus)
	require.False(t, summaries["workflow-1"].HasVCs)
	require.False(t, summaries["workflow-2"].HasVCs)
}

func TestVCService_GetWorkflowVCStatusSummaries_WithEmptyIDs(t *testing.T) {
	vcService, _, _, _ := setupVCTestEnvironment(t)

	// Test with empty string IDs (should be skipped)
	summaries, err := vcService.GetWorkflowVCStatusSummaries([]string{"workflow-1", "", "workflow-2", ""})
	require.NoError(t, err)
	require.Len(t, summaries, 2)
	require.Contains(t, summaries, "workflow-1")
	require.Contains(t, summaries, "workflow-2")
}

func TestVCService_GetWorkflowVCStatusSummaries_DisabledSystem(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	cfg := &config.DIDConfig{
		Enabled: false,
	}

	vcService := NewVCService(cfg, nil, provider)

	summaries, err := vcService.GetWorkflowVCStatusSummaries([]string{"workflow-1"})
	require.NoError(t, err)
	require.Len(t, summaries, 1)
	require.Equal(t, "none", summaries["workflow-1"].VerificationStatus)
	_ = ctx
}

func TestVCService_GetWorkflowVCChain_DisabledSystem(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	cfg := &config.DIDConfig{
		Enabled: false,
	}

	vcService := NewVCService(cfg, nil, provider)

	chain, err := vcService.GetWorkflowVCChain("workflow-1")
	require.Error(t, err)
	require.Nil(t, chain)
	require.Contains(t, err.Error(), "DID system is disabled")
	_ = ctx
}

func TestVCService_GetWorkflowVCChain_NoVCs(t *testing.T) {
	vcService, _, _, _ := setupVCTestEnvironment(t)

	chain, err := vcService.GetWorkflowVCChain("workflow-nonexistent")
	require.NoError(t, err)
	require.NotNil(t, chain)
	require.Equal(t, "workflow-nonexistent", chain.WorkflowID)
	require.Empty(t, chain.ComponentVCs)
	require.Equal(t, 0, chain.TotalSteps)
	require.Equal(t, "pending", chain.Status)
}

func TestVCService_CreateWorkflowVC_DisabledSystem(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	cfg := &config.DIDConfig{
		Enabled: false,
	}

	vcService := NewVCService(cfg, nil, provider)

	workflowVC, err := vcService.CreateWorkflowVC("workflow-1", "session-1", []string{"vc-1", "vc-2"})
	require.Error(t, err)
	require.Nil(t, workflowVC)
	require.Contains(t, err.Error(), "DID system is disabled")
	_ = ctx
}

func TestVCService_CreateWorkflowVC_Success(t *testing.T) {
	vcService, didService, provider, ctx := setupVCTestEnvironment(t)

	// Register an agent first
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-workflow-vc",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
	}

	regResp, err := didService.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, regResp.Success)

	callerDID := regResp.IdentityPackage.ReasonerDIDs["reasoner1"].DID

	// Generate some execution VCs
	execCtx1 := &types.ExecutionContext{
		ExecutionID:  "exec-1",
		WorkflowID:   "workflow-1",
		SessionID:    "session-1",
		CallerDID:    callerDID,
		TargetDID:    "",
		AgentNodeDID: regResp.IdentityPackage.AgentDID.DID,
		Timestamp:    time.Now(),
	}

	vc1, err := vcService.GenerateExecutionVC(execCtx1, []byte(`{"input": "1"}`), []byte(`{"output": "1"}`), "succeeded", nil, 100)
	require.NoError(t, err)

	execCtx2 := &types.ExecutionContext{
		ExecutionID:  "exec-2",
		WorkflowID:   "workflow-1",
		SessionID:    "session-1",
		CallerDID:    callerDID,
		TargetDID:    "",
		AgentNodeDID: regResp.IdentityPackage.AgentDID.DID,
		Timestamp:    time.Now(),
	}

	vc2, err := vcService.GenerateExecutionVC(execCtx2, []byte(`{"input": "2"}`), []byte(`{"output": "2"}`), "succeeded", nil, 200)
	require.NoError(t, err)

	// Create workflow VC
	workflowVC, err := vcService.CreateWorkflowVC("workflow-1", "session-1", []string{vc1.VCID, vc2.VCID})
	require.NoError(t, err)
	require.NotNil(t, workflowVC)
	require.Equal(t, "workflow-1", workflowVC.WorkflowID)
	require.Equal(t, "session-1", workflowVC.SessionID)
	require.Len(t, workflowVC.ComponentVCs, 2)
	require.Equal(t, 2, workflowVC.TotalSteps)
	require.Equal(t, 2, workflowVC.CompletedSteps)
	require.NotEmpty(t, workflowVC.WorkflowVCID)

	// Verify workflow VC was stored - GetWorkflowVC looks up by workflow_vc_id, not workflow_id
	storedVC, err := provider.GetWorkflowVC(ctx, workflowVC.WorkflowVCID)
	require.NoError(t, err)
	require.NotNil(t, storedVC)
	require.Equal(t, workflowVC.WorkflowVCID, storedVC.WorkflowVCID)
}

func TestVCService_QueryExecutionVCs_DisabledSystem(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	cfg := &config.DIDConfig{
		Enabled: false,
	}

	vcService := NewVCService(cfg, nil, provider)

	filters := &types.VCFilters{
		WorkflowID: stringPtr("workflow-1"),
	}

	vcs, err := vcService.QueryExecutionVCs(filters)
	require.Error(t, err)
	require.Nil(t, vcs)
	require.Contains(t, err.Error(), "DID system is disabled")
	_ = ctx
}

func TestVCService_QueryExecutionVCs_Success(t *testing.T) {
	vcService, didService, _, _ := setupVCTestEnvironment(t)

	// Register an agent
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-query",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
	}

	regResp, err := didService.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, regResp.Success)

	callerDID := regResp.IdentityPackage.ReasonerDIDs["reasoner1"].DID

	// Generate execution VCs
	execCtx1 := &types.ExecutionContext{
		ExecutionID:  "exec-query-1",
		WorkflowID:   "workflow-query",
		SessionID:    "session-query",
		CallerDID:    callerDID,
		TargetDID:    "",
		AgentNodeDID: regResp.IdentityPackage.AgentDID.DID,
		Timestamp:    time.Now(),
	}

	vc1, err := vcService.GenerateExecutionVC(execCtx1, []byte(`{"input": "1"}`), []byte(`{"output": "1"}`), "succeeded", nil, 100)
	require.NoError(t, err)

	execCtx2 := &types.ExecutionContext{
		ExecutionID:  "exec-query-2",
		WorkflowID:   "workflow-query",
		SessionID:    "session-query",
		CallerDID:    callerDID,
		TargetDID:    "",
		AgentNodeDID: regResp.IdentityPackage.AgentDID.DID,
		Timestamp:    time.Now(),
	}

	vc2, err := vcService.GenerateExecutionVC(execCtx2, []byte(`{"input": "2"}`), []byte(`{"output": "2"}`), "succeeded", nil, 200)
	require.NoError(t, err)

	// Query by workflow ID
	filters := &types.VCFilters{
		WorkflowID: stringPtr("workflow-query"),
	}

	vcs, err := vcService.QueryExecutionVCs(filters)
	require.NoError(t, err)
	require.Len(t, vcs, 2)

	vcIDs := make(map[string]bool)
	for _, vc := range vcs {
		vcIDs[vc.VCID] = true
	}
	require.True(t, vcIDs[vc1.VCID])
	require.True(t, vcIDs[vc2.VCID])
}

func TestVCService_GetExecutionVCByExecutionID(t *testing.T) {
	vcService, didService, _, _ := setupVCTestEnvironment(t)

	// Register an agent
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-get-exec",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
	}

	regResp, err := didService.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, regResp.Success)

	callerDID := regResp.IdentityPackage.ReasonerDIDs["reasoner1"].DID

	execCtx := &types.ExecutionContext{
		ExecutionID:  "exec-get-by-id",
		WorkflowID:   "workflow-1",
		SessionID:    "session-1",
		CallerDID:    callerDID,
		TargetDID:    "",
		AgentNodeDID: regResp.IdentityPackage.AgentDID.DID,
		Timestamp:    time.Now(),
	}

	vc, err := vcService.GenerateExecutionVC(execCtx, []byte(`{"input": "test"}`), []byte(`{"output": "result"}`), "succeeded", nil, 100)
	require.NoError(t, err)

	// Get VC by execution ID
	retrievedVC, err := vcService.GetExecutionVCByExecutionID("exec-get-by-id")
	require.NoError(t, err)
	require.NotNil(t, retrievedVC)
	require.Equal(t, vc.VCID, retrievedVC.VCID)
	require.Equal(t, "exec-get-by-id", retrievedVC.ExecutionID)
}

func TestVCService_GetExecutionVCByExecutionID_NotFound(t *testing.T) {
	vcService, _, _, _ := setupVCTestEnvironment(t)

	vc, err := vcService.GetExecutionVCByExecutionID("nonexistent-exec-id")
	require.Error(t, err)
	require.Nil(t, vc)
	require.Contains(t, err.Error(), "execution VC not found")
}

func TestVCService_ListWorkflowVCs_DisabledSystem(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	cfg := &config.DIDConfig{
		Enabled: false,
	}

	vcService := NewVCService(cfg, nil, provider)

	workflowVCs, err := vcService.ListWorkflowVCs()
	require.Error(t, err)
	require.Nil(t, workflowVCs)
	require.Contains(t, err.Error(), "DID system is disabled")
	_ = ctx
}

func TestVCService_ListWorkflowVCs_Success(t *testing.T) {
	vcService, didService, _, _ := setupVCTestEnvironment(t)

	// Register an agent
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-list-workflow",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
	}

	regResp, err := didService.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, regResp.Success)

	callerDID := regResp.IdentityPackage.ReasonerDIDs["reasoner1"].DID

	// Generate execution VC
	execCtx := &types.ExecutionContext{
		ExecutionID:  "exec-list-1",
		WorkflowID:   "workflow-list",
		SessionID:    "session-list",
		CallerDID:    callerDID,
		TargetDID:    "",
		AgentNodeDID: regResp.IdentityPackage.AgentDID.DID,
		Timestamp:    time.Now(),
	}

	vc, err := vcService.GenerateExecutionVC(execCtx, []byte(`{"input": "test"}`), []byte(`{"output": "result"}`), "succeeded", nil, 100)
	require.NoError(t, err)

	// Create workflow VC
	workflowVC, err := vcService.CreateWorkflowVC("workflow-list", "session-list", []string{vc.VCID})
	require.NoError(t, err)

	// List workflow VCs
	workflowVCs, err := vcService.ListWorkflowVCs()
	require.NoError(t, err)
	require.NotNil(t, workflowVCs)
	require.GreaterOrEqual(t, len(workflowVCs), 1)

	found := false
	for _, wvc := range workflowVCs {
		if wvc.WorkflowVCID == workflowVC.WorkflowVCID {
			found = true
			break
		}
	}
	require.True(t, found)
}

func TestVCService_VerifyExecutionVCComprehensive_Success(t *testing.T) {
	vcService, didService, _, _ := setupVCTestEnvironment(t)

	// Register an agent
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-comprehensive",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
		Skills:      []types.SkillDefinition{{ID: "skill1"}},
	}

	regResp, err := didService.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, regResp.Success)

	callerDID := regResp.IdentityPackage.ReasonerDIDs["reasoner1"].DID
	targetDID := regResp.IdentityPackage.SkillDIDs["skill1"].DID

	execCtx := &types.ExecutionContext{
		ExecutionID:  "exec-comprehensive",
		WorkflowID:   "workflow-comprehensive",
		SessionID:    "session-comprehensive",
		CallerDID:    callerDID,
		TargetDID:    targetDID,
		AgentNodeDID: regResp.IdentityPackage.AgentDID.DID,
		Timestamp:    time.Now(),
	}

	_, err = vcService.GenerateExecutionVC(execCtx, []byte(`{"input": "test"}`), []byte(`{"output": "result"}`), "succeeded", nil, 100)
	require.NoError(t, err)

	// Perform comprehensive verification
	result, err := vcService.VerifyExecutionVCComprehensive("exec-comprehensive")
	require.NoError(t, err)
	require.NotNil(t, result)
	require.True(t, result.Valid)
	require.Greater(t, result.OverallScore, 0.0)
	require.Empty(t, result.CriticalIssues)
	require.True(t, result.IntegrityChecks.MetadataConsistency)
	require.True(t, result.IntegrityChecks.FieldConsistency)
	require.True(t, result.IntegrityChecks.HashValidation)
	require.True(t, result.IntegrityChecks.StructuralIntegrity)
	require.True(t, result.SecurityAnalysis.KeyValidation)
	require.True(t, result.SecurityAnalysis.DIDAuthenticity)
	require.True(t, result.ComplianceChecks.W3CCompliance)
	require.True(t, result.ComplianceChecks.AgentFieldStandardCompliance)
}

func TestVCService_VerifyExecutionVCComprehensive_NotFound(t *testing.T) {
	vcService, _, _, _ := setupVCTestEnvironment(t)

	result, err := vcService.VerifyExecutionVCComprehensive("nonexistent-exec-id")
	require.NoError(t, err)
	require.NotNil(t, result)
	require.False(t, result.Valid)
	require.Equal(t, 0.0, result.OverallScore)
	require.Len(t, result.CriticalIssues, 1)
	require.Equal(t, "vc_not_found", result.CriticalIssues[0].Type)
}

func TestVCService_VerifyExecutionVCComprehensive_DisabledSystem(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	cfg := &config.DIDConfig{
		Enabled: false,
	}

	vcService := NewVCService(cfg, nil, provider)

	result, err := vcService.VerifyExecutionVCComprehensive("exec-1")
	require.NoError(t, err)
	require.NotNil(t, result)
	require.False(t, result.Valid)
	require.Equal(t, 0.0, result.OverallScore)
	require.Len(t, result.CriticalIssues, 1)
	require.Equal(t, "system_disabled", result.CriticalIssues[0].Type)
	_ = ctx
}

func TestVCService_VerifyExecutionVCComprehensive_IssuerMismatch(t *testing.T) {
	vcService, didService, _, ctx := setupVCTestEnvironment(t)

	// Register an agent
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-mismatch",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
	}

	regResp, err := didService.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, regResp.Success)

	callerDID := regResp.IdentityPackage.ReasonerDIDs["reasoner1"].DID

	execCtx := &types.ExecutionContext{
		ExecutionID:  "exec-mismatch",
		WorkflowID:   "workflow-1",
		SessionID:    "session-1",
		CallerDID:    callerDID,
		TargetDID:    "",
		AgentNodeDID: regResp.IdentityPackage.AgentDID.DID,
		Timestamp:    time.Now(),
	}

	_, err = vcService.GenerateExecutionVC(execCtx, []byte(`{"input": "test"}`), []byte(`{"output": "result"}`), "succeeded", nil, 100)
	require.NoError(t, err)

	// Get the VC from storage to tamper with it
	storedVC, err := vcService.vcStorage.GetExecutionVCByExecutionID("exec-mismatch")
	require.NoError(t, err)
	require.NotNil(t, storedVC)

	// Tamper with the VCDocument JSON to change the issuer field inside it.
	// The SQL upsert only updates vc_document (not issuer_did metadata), so we
	// must modify the JSON document to create a mismatch between the stored
	// metadata issuer_did and the issuer field inside the VC document.
	var vcDoc types.VCDocument
	require.NoError(t, json.Unmarshal(storedVC.VCDocument, &vcDoc))
	vcDoc.Issuer = "did:key:tampered"
	tamperedDocBytes, err := json.Marshal(vcDoc)
	require.NoError(t, err)

	tamperedVC := *storedVC
	tamperedVC.VCDocument = tamperedDocBytes

	// Store tampered VC using vcStorage - the upsert updates vc_document but
	// preserves the original issuer_did in metadata, creating the mismatch
	err = vcService.vcStorage.StoreExecutionVC(ctx, &tamperedVC)
	require.NoError(t, err)

	// Verify - should detect issuer mismatch between metadata and VC document
	result, err := vcService.VerifyExecutionVCComprehensive("exec-mismatch")
	require.NoError(t, err)
	require.NotNil(t, result)
	require.False(t, result.Valid)
	require.Greater(t, len(result.CriticalIssues), 0)

	// Find issuer mismatch issue
	found := false
	for _, issue := range result.CriticalIssues {
		if issue.Type == "issuer_mismatch" {
			found = true
			break
		}
	}
	require.True(t, found)
}

func TestVCService_VerifyWorkflowVCComprehensive_DisabledSystem(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	cfg := &config.DIDConfig{
		Enabled: false,
	}

	vcService := NewVCService(cfg, nil, provider)

	result, err := vcService.VerifyWorkflowVCComprehensive("workflow-1")
	require.NoError(t, err)
	require.NotNil(t, result)
	require.False(t, result.Valid)
	require.Equal(t, 0.0, result.OverallScore)
	require.Len(t, result.CriticalIssues, 1)
	require.Equal(t, "system_disabled", result.CriticalIssues[0].Type)
	_ = ctx
}

func TestVCService_VerifyWorkflowVCComprehensive_NoVCs(t *testing.T) {
	vcService, _, _, _ := setupVCTestEnvironment(t)

	result, err := vcService.VerifyWorkflowVCComprehensive("workflow-nonexistent")
	require.NoError(t, err)
	require.NotNil(t, result)
	// Should handle gracefully when workflow has no VCs
}

func TestVCService_DetermineWorkflowStatus(t *testing.T) {
	vcService, didService, _, _ := setupVCTestEnvironment(t)

	// Register an agent
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-status",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
	}

	regResp, err := didService.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, regResp.Success)

	callerDID := regResp.IdentityPackage.ReasonerDIDs["reasoner1"].DID

	// Create VCs with different statuses
	execCtx1 := &types.ExecutionContext{
		ExecutionID:  "exec-status-1",
		WorkflowID:   "workflow-status",
		SessionID:    "session-status",
		CallerDID:    callerDID,
		TargetDID:    "",
		AgentNodeDID: regResp.IdentityPackage.AgentDID.DID,
		Timestamp:    time.Now(),
	}

	vc1, err := vcService.GenerateExecutionVC(execCtx1, []byte(`{"input": "1"}`), []byte(`{"output": "1"}`), "succeeded", nil, 100)
	require.NoError(t, err)

	execCtx2 := &types.ExecutionContext{
		ExecutionID:  "exec-status-2",
		WorkflowID:   "workflow-status",
		SessionID:    "session-status",
		CallerDID:    callerDID,
		TargetDID:    "",
		AgentNodeDID: regResp.IdentityPackage.AgentDID.DID,
		Timestamp:    time.Now(),
	}

	vc2, err := vcService.GenerateExecutionVC(execCtx2, []byte(`{"input": "2"}`), nil, "failed", stringPtr("error"), 50)
	require.NoError(t, err)

	// Get workflow VC chain to test status determination
	chain, err := vcService.GetWorkflowVCChain("workflow-status")
	require.NoError(t, err)
	require.NotNil(t, chain)
	require.Equal(t, "failed", chain.Status) // Should be failed because one VC failed
	require.Len(t, chain.ComponentVCs, 2)

	// Verify both VCs are in the chain
	vcIDs := make(map[string]bool)
	for _, vc := range chain.ComponentVCs {
		vcIDs[vc.VCID] = true
	}
	require.True(t, vcIDs[vc1.VCID])
	require.True(t, vcIDs[vc2.VCID])
}

func TestVCService_DetermineWorkflowStatus_AllSucceeded(t *testing.T) {
	vcService, didService, _, _ := setupVCTestEnvironment(t)

	// Register an agent
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-status-all-success",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
	}

	regResp, err := didService.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, regResp.Success)

	callerDID := regResp.IdentityPackage.ReasonerDIDs["reasoner1"].DID

	// Create multiple succeeded VCs
	for i := 1; i <= 3; i++ {
		execCtx := &types.ExecutionContext{
			ExecutionID:  "exec-success-" + string(rune('0'+i)),
			WorkflowID:   "workflow-all-success",
			SessionID:    "session-success",
			CallerDID:    callerDID,
			TargetDID:    "",
			AgentNodeDID: regResp.IdentityPackage.AgentDID.DID,
			Timestamp:    time.Now(),
		}

		_, err := vcService.GenerateExecutionVC(execCtx, []byte(`{"input": "test"}`), []byte(`{"output": "result"}`), "succeeded", nil, 100)
		require.NoError(t, err)
	}

	// Get workflow VC chain
	chain, err := vcService.GetWorkflowVCChain("workflow-all-success")
	require.NoError(t, err)
	require.NotNil(t, chain)
	require.Equal(t, "succeeded", chain.Status)
	require.Len(t, chain.ComponentVCs, 3)
}

func TestVCService_GenerateExecutionVC_EmptyCallerDID_FallsBackToAgentDID(t *testing.T) {
	vcService, didService, _, _ := setupVCTestEnvironment(t)

	// Register an agent
	req := &types.DIDRegistrationRequest{
		AgentNodeID: "agent-empty-caller",
		Reasoners:   []types.ReasonerDefinition{{ID: "reasoner1"}},
	}

	regResp, err := didService.RegisterAgent(req)
	require.NoError(t, err)
	require.True(t, regResp.Success)

	agentDID := regResp.IdentityPackage.AgentDID.DID

	// Empty CallerDID — should fall back to AgentNodeDID
	execCtx := &types.ExecutionContext{
		ExecutionID:  "exec-empty-caller",
		WorkflowID:   "workflow-1",
		SessionID:    "session-1",
		CallerDID:    "",
		TargetDID:    "",
		AgentNodeDID: agentDID,
		Timestamp:    time.Now(),
	}

	vc, err := vcService.GenerateExecutionVC(execCtx, []byte(`{"input": "test"}`), []byte(`{"output": "result"}`), "succeeded", nil, 100)
	require.NoError(t, err)
	require.NotNil(t, vc, "VC should be generated using agent's own DID as fallback")
	require.Equal(t, agentDID, vc.CallerDID)
	require.Equal(t, agentDID, vc.IssuerDID)
}

func TestVCService_GenerateExecutionVC_BothDIDsEmpty_ReturnsNil(t *testing.T) {
	vcService, _, _, _ := setupVCTestEnvironment(t)

	// Both CallerDID and AgentNodeDID are empty — should return nil gracefully
	execCtx := &types.ExecutionContext{
		ExecutionID:  "exec-no-did",
		WorkflowID:   "workflow-1",
		SessionID:    "session-1",
		CallerDID:    "",
		TargetDID:    "",
		AgentNodeDID: "",
		Timestamp:    time.Now(),
	}

	vc, err := vcService.GenerateExecutionVC(execCtx, []byte(`{"input": "test"}`), []byte(`{"output": "result"}`), "succeeded", nil, 100)
	require.NoError(t, err)
	require.Nil(t, vc, "VC generation should be skipped when no DID is available")
}

// Helper function
func stringPtr(s string) *string {
	return &s
}
