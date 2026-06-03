package cli

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

func TestVCHelpersAndFormatting(t *testing.T) {
	require.Equal(t, "key", getDIDMethod("did:key:abc"))
	require.Equal(t, "unknown", getDIDMethod("invalid"))

	legacy := types.WorkflowVCChainResponse{
		WorkflowID: "wf-1",
		ComponentVCs: []types.ExecutionVC{
			{VCID: "vc-1"},
		},
		WorkflowVC: types.WorkflowVC{WorkflowID: "wf-1", Status: "completed"},
		Status:     "completed",
		DIDResolutionBundle: map[string]types.DIDResolutionEntry{
			"did:key:issuer": {
				Method:       "key",
				PublicKeyJWK: json.RawMessage(`{"x":"abc"}`),
				ResolvedFrom: "bundle",
				ResolvedAt:   "2026-01-01T00:00:00Z",
			},
		},
	}
	chain := convertLegacyChain(legacy)
	require.Equal(t, "wf-1", chain.WorkflowID)
	require.Len(t, chain.ExecutionVCs, 1)
	require.Len(t, chain.DIDResolutionBundle, 1)

	normalizeEnhancedChain(&chain)
	require.Equal(t, 1, chain.TotalExecutions)
	require.Equal(t, 1, chain.CompletedExecutions)
	require.Equal(t, "completed", chain.WorkflowStatus)

	execVC, raw, _ := signedExecutionVC(t, "did:key:issuer")
	parsed, ok := tryParseBareExecutionVC(raw)
	require.True(t, ok)
	require.Equal(t, execVC.WorkflowID, parsed.WorkflowID)
	require.Len(t, parsed.ExecutionVCs, 1)

	_, ok = tryParseBareExecutionVC([]byte(`{"type":["Other"]}`))
	require.False(t, ok)

	_, err := executionVCFromBareDocument([]byte(`{"type":["Other"]}`))
	require.ErrorContains(t, err, "missing VerifiableCredential type")

	result := VCVerificationResult{
		Valid:          true,
		Type:           "workflow",
		WorkflowID:     "wf-1",
		FormatValid:    true,
		SignatureValid: true,
		Message:        "verified",
		Summary:        VerificationSummary{TotalComponents: 1, ValidComponents: 1, TotalDIDs: 1, ResolvedDIDs: 1, TotalSignatures: 1, ValidSignatures: 1},
		VerificationSteps: []VerificationStep{
			{Step: 1, Description: "read", Success: true},
		},
		DIDResolutions: []DIDResolutionResult{
			{DID: "did:key:issuer", Success: true},
		},
	}
	output := captureOutput(t, func() {
		require.NoError(t, outputResult(result, VerifyOptions{OutputFormat: "pretty", Verbose: true}))
		require.NoError(t, outputResult(result, VerifyOptions{OutputFormat: "json"}))
	})
	require.Contains(t, output, "AgentField VC Verification: ✅ VALID")
	require.Contains(t, output, `"workflow_id": "wf-1"`)
}

func TestEnhancedVCVerifierCoverage(t *testing.T) {
	execVC, raw, resolution := signedExecutionVC(t, "did:key:issuer")
	verifier := NewEnhancedVCVerifier(map[string]DIDResolutionInfo{
		"did:key:issuer": resolution,
	}, false)

	t.Run("verify execution vc comprehensive valid", func(t *testing.T) {
		result := verifier.verifyExecutionVCComprehensive(execVC)
		require.True(t, result.Valid)
		require.True(t, result.SignatureValid)
		require.True(t, result.FormatValid)
	})

	t.Run("verify execution vc comprehensive failures", func(t *testing.T) {
		badDocVC := execVC
		badDocVC.VCDocument = []byte(`not-json`)
		result := verifier.verifyExecutionVCComprehensive(badDocVC)
		require.False(t, result.Valid)
		require.False(t, result.FormatValid)

		mismatch := execVC
		mismatch.IssuerDID = "did:key:other"
		result = verifier.verifyExecutionVCComprehensive(mismatch)
		require.False(t, result.Valid)
		require.Contains(t, result.Error, "Issuer DID mismatch")

		missingResolutionVerifier := NewEnhancedVCVerifier(nil, false)
		result = missingResolutionVerifier.verifyExecutionVCComprehensive(execVC)
		require.False(t, result.Valid)
		require.Contains(t, result.Error, "DID resolution failed")

		var doc types.VCDocument
		require.NoError(t, json.Unmarshal(execVC.VCDocument, &doc))
		cases := []struct {
			name   string
			mutate func(*types.ExecutionVC, *types.VCDocument)
			want   string
		}{
			{name: "execution id mismatch", mutate: func(vc *types.ExecutionVC, _ *types.VCDocument) { vc.ExecutionID = "other" }, want: "Execution ID mismatch"},
			{name: "workflow id mismatch", mutate: func(vc *types.ExecutionVC, _ *types.VCDocument) { vc.WorkflowID = "other" }, want: "Workflow ID mismatch"},
			{name: "session id mismatch", mutate: func(vc *types.ExecutionVC, _ *types.VCDocument) { vc.SessionID = "other" }, want: "Session ID mismatch"},
			{name: "caller did mismatch", mutate: func(vc *types.ExecutionVC, _ *types.VCDocument) { vc.CallerDID = "did:key:other" }, want: "Caller DID mismatch"},
			{name: "target did mismatch", mutate: func(vc *types.ExecutionVC, _ *types.VCDocument) { vc.TargetDID = "did:key:other" }, want: "Target DID mismatch"},
			{name: "status mismatch", mutate: func(vc *types.ExecutionVC, _ *types.VCDocument) { vc.Status = "failed" }, want: "Status mismatch"},
			{name: "input hash mismatch", mutate: func(vc *types.ExecutionVC, _ *types.VCDocument) { vc.InputHash = "other" }, want: "Input hash mismatch"},
			{name: "output hash mismatch", mutate: func(vc *types.ExecutionVC, _ *types.VCDocument) { vc.OutputHash = "other" }, want: "Output hash mismatch"},
			{name: "signature mismatch", mutate: func(vc *types.ExecutionVC, _ *types.VCDocument) { vc.Signature = "other" }, want: "Signature mismatch"},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				mutatedVC := execVC
				mutatedDoc := doc
				tc.mutate(&mutatedVC, &mutatedDoc)
				result := verifier.verifyExecutionVCComprehensive(mutatedVC)
				require.False(t, result.Valid)
				require.Contains(t, result.Error, tc.want)
			})
		}
	})

	t.Run("signature timestamp structure and compliance helpers", func(t *testing.T) {
		var vcDoc types.VCDocument
		require.NoError(t, json.Unmarshal(raw, &vcDoc))

		valid, err := verifier.verifyVCSignature(vcDoc, resolution)
		require.NoError(t, err)
		require.True(t, valid)

		invalidResolution := DIDResolutionInfo{PublicKeyJWK: map[string]interface{}{}}
		valid, err = verifier.verifyVCSignature(vcDoc, invalidResolution)
		require.ErrorContains(t, err, "public key JWK is empty")
		require.False(t, valid)

		require.NoError(t, verifier.validateTimestamp(vcDoc.IssuanceDate))
		require.Error(t, verifier.validateTimestamp("not-a-time"))

		require.NoError(t, verifier.validateVCStructure(vcDoc))
		require.ErrorContains(t, verifier.validateVCStructure(types.VCDocument{}), "missing @context")
		require.True(t, verifier.checkExecutionVCMetadataConsistency(execVC))
		require.False(t, verifier.checkExecutionVCMetadataConsistency(types.ExecutionVC{VCDocument: []byte(`bad`)}))
		require.True(t, verifier.checkW3CCompliance(vcDoc))
		vcDoc.Context = []string{"https://example.com"}
		require.False(t, verifier.checkW3CCompliance(vcDoc))
	})

	t.Run("workflow verification and aggregate checks", func(t *testing.T) {
		workflow := types.WorkflowVC{
			WorkflowID:   "wf-1",
			ComponentVCs: []string{"vc-1", "vc-2"},
			VCDocument:   []byte(`not-json`),
		}
		result := verifier.verifyWorkflowVC(workflow, []types.ExecutionVC{execVC})
		require.False(t, result.Valid)
		require.NotEmpty(t, result.Issues)

		workflow.VCDocument = []byte(`{"issuer":"did:key:missing"}`)
		result = verifier.verifyWorkflowVC(workflow, []types.ExecutionVC{execVC})
		require.False(t, result.Valid)
		require.False(t, result.ComponentConsistency)
		require.False(t, result.SignatureValid)
		require.False(t, verifier.checkComponentVCConsistency(workflow, []types.ExecutionVC{execVC}))
		require.True(t, verifier.checkComponentVCConsistency(types.WorkflowVC{ComponentVCs: []string{"vc-1"}}, []types.ExecutionVC{execVC}))

		chain := EnhancedVCChain{
			WorkflowID:   "wf-1",
			ExecutionVCs: []types.ExecutionVC{execVC},
			WorkflowVC:   types.WorkflowVC{},
		}
		integrity := verifier.performIntegrityChecks(chain)
		require.True(t, integrity.MetadataConsistency)
		require.Empty(t, integrity.Issues)

		tampered := execVC
		tampered.Status = "failed"
		security := verifier.performSecurityAnalysis(EnhancedVCChain{ExecutionVCs: []types.ExecutionVC{tampered}})
		require.Contains(t, security.TamperEvidence, "metadata_inconsistency")
		require.Less(t, security.SecurityScore, 100.0)

		compliance := verifier.performComplianceChecks(chain)
		require.True(t, compliance.W3CCompliance)

		invalidCompliance := verifier.performComplianceChecks(EnhancedVCChain{ExecutionVCs: []types.ExecutionVC{{VCID: "x", VCDocument: []byte(`{"@context":["https://example.com"]}`)}}})
		require.False(t, invalidCompliance.W3CCompliance)
		require.NotEmpty(t, invalidCompliance.Issues)

		full := verifier.VerifyEnhancedVCChain(chain)
		require.True(t, full.Valid)
		require.NotEmpty(t, full.ComponentResults)
		require.Greater(t, full.OverallScore, 0.0)
	})

	t.Run("score and status consistency", func(t *testing.T) {
		score := verifier.calculateOverallScore(&ComprehensiveVerificationResult{
			CriticalIssues:   []VerificationIssue{{}, {}},
			Warnings:         []VerificationIssue{{}},
			SecurityAnalysis: SecurityAnalysis{SecurityScore: 80},
		})
		require.Equal(t, 62.5, score)
		require.True(t, verifier.isStatusConsistent("completed", "success"))
		require.False(t, verifier.isStatusConsistent("failed", "success"))
	})
}

func TestVerifyVCAndResolutionHelpers(t *testing.T) {
	execVC, raw, resolution := signedExecutionVC(t, "did:key:issuer")
	chain := EnhancedVCChain{
		WorkflowID:          "wf-1",
		GeneratedAt:         "2026-01-02T03:04:05Z",
		TotalExecutions:     1,
		CompletedExecutions: 1,
		WorkflowStatus:      "completed",
		ExecutionVCs:        []types.ExecutionVC{execVC},
		ComponentVCs:        []types.ExecutionVC{execVC},
		DIDResolutionBundle: map[string]DIDResolutionInfo{"did:key:issuer": resolution},
	}
	path := filepath.Join(t.TempDir(), "chain.json")
	payload, err := json.Marshal(chain)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(path, payload, 0o644))

	output := captureOutput(t, func() {
		require.NoError(t, verifyVC(path, VerifyOptions{OutputFormat: "pretty"}))
	})
	require.Contains(t, output, "AgentField VC Verification:")

	dids := collectUniqueDIDs(chain)
	require.Equal(t, []string{"did:key:issuer"}, dids)

	resolved, err := resolveDID("did:key:issuer", chain.DIDResolutionBundle, VerifyOptions{})
	require.NoError(t, err)
	require.Equal(t, "bundled", resolved.ResolvedFrom)

	_, err = resolveDID("did:key:issuer", nil, VerifyOptions{ResolveWeb: true})
	require.ErrorContains(t, err, "disabled")
	_, err = resolveDID("did:key:issuer", nil, VerifyOptions{Resolver: "https://resolver.example"})
	require.ErrorContains(t, err, "disabled")
	_, err = resolveDID("did:key:missing", nil, VerifyOptions{})
	require.ErrorContains(t, err, "DID resolution failed")

	_, err = resolveKeyDID("bad")
	require.ErrorContains(t, err, "invalid did:key format")
	_, err = resolveKeyDID("did:key:abc")
	require.ErrorContains(t, err, "not fully implemented")

	var vcDoc types.VCDocument
	require.NoError(t, json.Unmarshal(raw, &vcDoc))
	valid, err := verifyVCSignature(vcDoc, resolution)
	require.NoError(t, err)
	require.True(t, valid)

	workflowDoc := types.WorkflowVCDocument{
		Context:      []string{"https://www.w3.org/2018/credentials/v1"},
		Type:         []string{"VerifiableCredential"},
		ID:           "workflow-vc-1",
		Issuer:       "did:key:issuer",
		IssuanceDate: "2026-01-02T03:04:05Z",
	}
	publicKey, privateKey, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)
	workflowCanonical, err := json.Marshal(workflowDoc)
	require.NoError(t, err)
	workflowDoc.Proof = types.VCProof{ProofValue: base64.RawURLEncoding.EncodeToString(ed25519.Sign(privateKey, workflowCanonical))}
	valid, err = verifyWorkflowVCSignature(workflowDoc, DIDResolutionInfo{
		PublicKeyJWK: map[string]interface{}{"x": base64.RawURLEncoding.EncodeToString(publicKey)},
	})
	require.NoError(t, err)
	require.True(t, valid)
}

func TestInitModelInit(t *testing.T) {
	require.Nil(t, (initModel{}).Init())
}

func signedExecutionVC(t *testing.T, issuer string) (types.ExecutionVC, []byte, DIDResolutionInfo) {
	t.Helper()
	publicKey, priv, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)

	doc := types.VCDocument{
		Context:      []string{"https://www.w3.org/2018/credentials/v1"},
		Type:         []string{"VerifiableCredential"},
		ID:           "vc-1",
		Issuer:       issuer,
		IssuanceDate: "2026-01-02T03:04:05Z",
		CredentialSubject: types.VCCredentialSubject{
			ExecutionID: "exec-1",
			WorkflowID:  "wf-1",
			SessionID:   "session-1",
			Caller:      types.VCCaller{DID: "did:key:caller"},
			Target:      types.VCTarget{DID: "did:key:target"},
			Execution: types.VCExecution{
				InputHash:  "input-hash",
				OutputHash: "output-hash",
				Status:     "completed",
			},
		},
	}

	canonical, err := json.Marshal(doc)
	require.NoError(t, err)
	signature := ed25519.Sign(priv, canonical)
	doc.Proof = types.VCProof{ProofValue: base64.RawURLEncoding.EncodeToString(signature)}
	raw, err := json.Marshal(doc)
	require.NoError(t, err)

	return types.ExecutionVC{
		VCID:        "vc-1",
		ExecutionID: "exec-1",
		WorkflowID:  "wf-1",
		SessionID:   "session-1",
		IssuerDID:   issuer,
		TargetDID:   "did:key:target",
		CallerDID:   "did:key:caller",
		VCDocument:  raw,
		Signature:   doc.Proof.ProofValue,
		InputHash:   "input-hash",
		OutputHash:  "output-hash",
		Status:      "completed",
		CreatedAt:   mustParseRFC3339(t, doc.IssuanceDate),
	}, raw, DIDResolutionInfo{
		DID:    issuer,
		Method: "key",
		PublicKeyJWK: map[string]interface{}{
			"x": base64.RawURLEncoding.EncodeToString(publicKey),
		},
	}
}

func mustParseRFC3339(t *testing.T, value string) time.Time {
	t.Helper()
	ts, err := time.Parse(time.RFC3339, value)
	require.NoError(t, err)
	return ts
}
