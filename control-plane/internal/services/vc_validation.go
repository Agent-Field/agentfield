package services

import (
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// VerifyVC verifies a verifiable credential.
func (s *VCService) VerifyVC(vcDocument json.RawMessage) (*types.VCVerificationResponse, error) {
	if !s.config.Enabled {
		return &types.VCVerificationResponse{
			Valid: false,
			Error: "DID system is disabled",
		}, nil
	}

	var vcDoc types.VCDocument
	if err := json.Unmarshal(vcDocument, &vcDoc); err != nil {
		return &types.VCVerificationResponse{
			Valid: false,
			Error: fmt.Sprintf("failed to parse VC document: %v", err),
		}, nil
	}

	// Resolve issuer DID
	issuerIdentity, err := s.didService.ResolveDID(vcDoc.Issuer)
	if err != nil {
		return &types.VCVerificationResponse{
			Valid: false,
			Error: fmt.Sprintf("failed to resolve issuer DID: %v", err),
		}, nil
	}

	// Verify signature
	valid, err := s.verifyVCSignature(&vcDoc, issuerIdentity)
	if err != nil {
		return &types.VCVerificationResponse{
			Valid: false,
			Error: fmt.Sprintf("failed to verify signature: %v", err),
		}, nil
	}

	if !valid {
		return &types.VCVerificationResponse{
			Valid:   false,
			Message: "Invalid signature",
		}, nil
	}

	return &types.VCVerificationResponse{
		Valid:     true,
		IssuerDID: vcDoc.Issuer,
		IssuedAt:  vcDoc.IssuanceDate,
		Message:   "VC verified successfully",
	}, nil
}

// VerifyAgentTagVCSignature verifies the Ed25519 signature on an AgentTagVCDocument.
func (s *VCService) VerifyAgentTagVCSignature(vc *types.AgentTagVCDocument) (bool, error) {
	if vc.Proof == nil || vc.Proof.ProofValue == "" || vc.Proof.Type == "UnsignedAuditRecord" {
		return false, fmt.Errorf("VC has no valid signature")
	}

	// Resolve issuer identity
	issuerIdentity, err := s.didService.ResolveDID(vc.Issuer)
	if err != nil {
		return false, fmt.Errorf("cannot resolve issuer DID %s: %w", vc.Issuer, err)
	}

	// Create canonical representation (without proof)
	vcCopy := *vc
	vcCopy.Proof = nil
	canonicalBytes, err := json.Marshal(vcCopy)
	if err != nil {
		return false, fmt.Errorf("failed to marshal agent tag VC for verification: %w", err)
	}

	// Decode signature
	signatureBytes, err := base64.RawURLEncoding.DecodeString(vc.Proof.ProofValue)
	if err != nil {
		return false, fmt.Errorf("failed to decode signature: %w", err)
	}

	// Parse public key from JWK
	var jwk map[string]interface{}
	if err := json.Unmarshal([]byte(issuerIdentity.PublicKeyJWK), &jwk); err != nil {
		return false, fmt.Errorf("failed to parse issuer public key JWK: %w", err)
	}

	xValue, ok := jwk["x"].(string)
	if !ok {
		return false, fmt.Errorf("invalid issuer public key JWK: missing 'x' parameter")
	}

	publicKeyBytes, err := base64.RawURLEncoding.DecodeString(xValue)
	if err != nil {
		return false, fmt.Errorf("failed to decode public key: %w", err)
	}

	if len(publicKeyBytes) != ed25519.PublicKeySize {
		return false, fmt.Errorf("invalid public key length: got %d, want %d", len(publicKeyBytes), ed25519.PublicKeySize)
	}

	publicKey := ed25519.PublicKey(publicKeyBytes)
	return ed25519.Verify(publicKey, canonicalBytes, signatureBytes), nil
}

// verifyVCSignature verifies the signature of a VC document.
func (s *VCService) verifyVCSignature(vcDoc *types.VCDocument, issuerIdentity *types.DIDIdentity) (bool, error) {
	// Create canonical representation for verification
	vcCopy := *vcDoc
	vcCopy.Proof = types.VCProof{} // Remove proof for verification

	canonicalBytes, err := json.Marshal(vcCopy)
	if err != nil {
		return false, fmt.Errorf("failed to marshal VC for verification: %w", err)
	}

	// Parse public key from JWK
	var jwk map[string]interface{}
	if err := json.Unmarshal([]byte(issuerIdentity.PublicKeyJWK), &jwk); err != nil {
		return false, fmt.Errorf("failed to parse public key JWK: %w", err)
	}

	xValue, ok := jwk["x"].(string)
	if !ok {
		return false, fmt.Errorf("invalid public key JWK: missing 'x' parameter")
	}

	publicKeyBytes, err := base64.RawURLEncoding.DecodeString(xValue)
	if err != nil {
		return false, fmt.Errorf("failed to decode public key: %w", err)
	}

	if len(publicKeyBytes) != ed25519.PublicKeySize {
		return false, fmt.Errorf("invalid public key length: got %d, want %d", len(publicKeyBytes), ed25519.PublicKeySize)
	}

	publicKey := ed25519.PublicKey(publicKeyBytes)

	// Decode signature
	signatureBytes, err := base64.RawURLEncoding.DecodeString(vcDoc.Proof.ProofValue)
	if err != nil {
		return false, fmt.Errorf("failed to decode signature: %w", err)
	}

	// Verify signature
	return ed25519.Verify(publicKey, canonicalBytes, signatureBytes), nil
}

// VerificationIssue represents a specific verification problem
type VerificationIssue struct {
	Type        string `json:"type"`
	Severity    string `json:"severity"` // "critical", "warning", "info"
	Component   string `json:"component"`
	Field       string `json:"field"`
	Expected    string `json:"expected"`
	Actual      string `json:"actual"`
	Description string `json:"description"`
}

// ComprehensiveVCVerificationResult provides detailed verification results
type ComprehensiveVCVerificationResult struct {
	Valid                 bool                  `json:"valid"`
	OverallScore          float64               `json:"overall_score"` // 0-100
	CriticalIssues        []VerificationIssue   `json:"critical_issues"`
	Warnings              []VerificationIssue   `json:"warnings"`
	IntegrityChecks       IntegrityCheckResults `json:"integrity_checks"`
	SecurityAnalysis      SecurityAnalysis      `json:"security_analysis"`
	ComplianceChecks      ComplianceChecks      `json:"compliance_checks"`
	VerificationTimestamp string                `json:"verification_timestamp"`
}

// IntegrityCheckResults represents various integrity verification results
type IntegrityCheckResults struct {
	MetadataConsistency bool                `json:"metadata_consistency"`
	FieldConsistency    bool                `json:"field_consistency"`
	TimestampValidation bool                `json:"timestamp_validation"`
	HashValidation      bool                `json:"hash_validation"`
	StructuralIntegrity bool                `json:"structural_integrity"`
	Issues              []VerificationIssue `json:"issues"`
}

// SecurityAnalysis represents security-focused verification results
type SecurityAnalysis struct {
	SignatureStrength string              `json:"signature_strength"`
	KeyValidation     bool                `json:"key_validation"`
	DIDAuthenticity   bool                `json:"did_authenticity"`
	ReplayProtection  bool                `json:"replay_protection"`
	TamperEvidence    []string            `json:"tamper_evidence"`
	SecurityScore     float64             `json:"security_score"`
	Issues            []VerificationIssue `json:"issues"`
}

// ComplianceChecks represents compliance and audit verification results
type ComplianceChecks struct {
	W3CCompliance                bool                `json:"w3c_compliance"`
	AgentFieldStandardCompliance bool                `json:"agentfield_standard_compliance"`
	AuditTrailIntegrity          bool                `json:"audit_trail_integrity"`
	DataIntegrityChecks          bool                `json:"data_integrity_checks"`
	Issues                       []VerificationIssue `json:"issues"`
}

// VerifyExecutionVCComprehensive performs comprehensive verification of an execution VC
func (s *VCService) VerifyExecutionVCComprehensive(executionID string) (*ComprehensiveVCVerificationResult, error) {
	if !s.config.Enabled {
		return &ComprehensiveVCVerificationResult{
			Valid:                 false,
			OverallScore:          0,
			CriticalIssues:        []VerificationIssue{{Type: "system_disabled", Severity: "critical", Description: "DID system is disabled"}},
			VerificationTimestamp: time.Now().UTC().Format(time.RFC3339),
		}, nil
	}

	// Get the execution VC
	filters := &types.VCFilters{Limit: 1000}
	executionVCs, err := s.vcStorage.QueryExecutionVCs(filters)
	if err != nil {
		return nil, fmt.Errorf("failed to query execution VCs: %w", err)
	}

	var executionVC *types.ExecutionVC
	for _, vc := range executionVCs {
		if vc.ExecutionID == executionID {
			executionVC = &vc
			break
		}
	}

	if executionVC == nil {
		return &ComprehensiveVCVerificationResult{
			Valid:                 false,
			OverallScore:          0,
			CriticalIssues:        []VerificationIssue{{Type: "vc_not_found", Severity: "critical", Description: "VC not found for execution"}},
			VerificationTimestamp: time.Now().UTC().Format(time.RFC3339),
		}, nil
	}

	result := &ComprehensiveVCVerificationResult{
		VerificationTimestamp: time.Now().UTC().Format(time.RFC3339),
		CriticalIssues:        []VerificationIssue{},
		Warnings:              []VerificationIssue{},
	}

	// Parse VC document
	var vcDoc types.VCDocument
	if err := json.Unmarshal(executionVC.VCDocument, &vcDoc); err != nil {
		result.CriticalIssues = append(result.CriticalIssues, VerificationIssue{
			Type:        "parse_error",
			Severity:    "critical",
			Component:   executionVC.VCID,
			Description: fmt.Sprintf("Failed to parse VC document: %v", err),
		})
		result.Valid = false
		result.OverallScore = 0
		return result, nil
	}

	// Perform comprehensive verification checks
	result.IntegrityChecks = s.performIntegrityChecks(executionVC, &vcDoc)
	result.SecurityAnalysis = s.performSecurityAnalysis(executionVC, &vcDoc)
	result.ComplianceChecks = s.performComplianceChecks(&vcDoc)

	// Collect all issues
	allIssues := []VerificationIssue{}
	allIssues = append(allIssues, result.IntegrityChecks.Issues...)
	allIssues = append(allIssues, result.SecurityAnalysis.Issues...)
	allIssues = append(allIssues, result.ComplianceChecks.Issues...)

	// Separate critical issues and warnings
	for _, issue := range allIssues {
		if issue.Severity == "critical" {
			result.CriticalIssues = append(result.CriticalIssues, issue)
		} else if issue.Severity == "warning" {
			result.Warnings = append(result.Warnings, issue)
		}
	}

	// Calculate overall validity and score
	result.Valid = len(result.CriticalIssues) == 0
	result.OverallScore = s.calculateOverallScore(result)

	return result, nil
}

// performIntegrityChecks performs various integrity checks on the VC
func (s *VCService) performIntegrityChecks(execVC *types.ExecutionVC, vcDoc *types.VCDocument) IntegrityCheckResults {
	result := IntegrityCheckResults{
		MetadataConsistency: true,
		FieldConsistency:    true,
		TimestampValidation: true,
		HashValidation:      true,
		StructuralIntegrity: true,
		Issues:              []VerificationIssue{},
	}

	// CRITICAL CHECK 1: Metadata consistency between top-level and VC document
	if execVC.IssuerDID != vcDoc.Issuer {
		result.MetadataConsistency = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "issuer_mismatch",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "issuer_did",
			Expected:    execVC.IssuerDID,
			Actual:      vcDoc.Issuer,
			Description: "Issuer DID mismatch between metadata and VC document",
		})
	}

	// CRITICAL CHECK 2: Execution ID consistency
	if execVC.ExecutionID != vcDoc.CredentialSubject.ExecutionID {
		result.FieldConsistency = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "execution_id_mismatch",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "execution_id",
			Expected:    execVC.ExecutionID,
			Actual:      vcDoc.CredentialSubject.ExecutionID,
			Description: "Execution ID mismatch between metadata and VC document",
		})
	}

	// CRITICAL CHECK 3: Workflow ID consistency
	if execVC.WorkflowID != vcDoc.CredentialSubject.WorkflowID {
		result.FieldConsistency = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "workflow_id_mismatch",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "workflow_id",
			Expected:    execVC.WorkflowID,
			Actual:      vcDoc.CredentialSubject.WorkflowID,
			Description: "Workflow ID mismatch between metadata and VC document",
		})
	}

	// CRITICAL CHECK 4: Session ID consistency
	if execVC.SessionID != vcDoc.CredentialSubject.SessionID {
		result.FieldConsistency = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "session_id_mismatch",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "session_id",
			Expected:    execVC.SessionID,
			Actual:      vcDoc.CredentialSubject.SessionID,
			Description: "Session ID mismatch between metadata and VC document",
		})
	}

	// CRITICAL CHECK 5: Caller DID consistency
	if execVC.CallerDID != vcDoc.CredentialSubject.Caller.DID {
		result.FieldConsistency = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "caller_did_mismatch",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "caller_did",
			Expected:    execVC.CallerDID,
			Actual:      vcDoc.CredentialSubject.Caller.DID,
			Description: "Caller DID mismatch between metadata and VC document",
		})
	}

	// CRITICAL CHECK 6: Target DID consistency
	if execVC.TargetDID != vcDoc.CredentialSubject.Target.DID {
		result.FieldConsistency = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "target_did_mismatch",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "target_did",
			Expected:    execVC.TargetDID,
			Actual:      vcDoc.CredentialSubject.Target.DID,
			Description: "Target DID mismatch between metadata and VC document",
		})
	}

	// CRITICAL CHECK 7: Status consistency (with AgentField system status mapping)
	if !s.isStatusConsistent(execVC.Status, vcDoc.CredentialSubject.Execution.Status) {
		result.FieldConsistency = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "status_mismatch",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "status",
			Expected:    execVC.Status,
			Actual:      vcDoc.CredentialSubject.Execution.Status,
			Description: "Status mismatch between metadata and VC document",
		})
	}

	// CRITICAL CHECK 8: Hash consistency
	if execVC.InputHash != vcDoc.CredentialSubject.Execution.InputHash {
		result.HashValidation = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "input_hash_mismatch",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "input_hash",
			Expected:    execVC.InputHash,
			Actual:      vcDoc.CredentialSubject.Execution.InputHash,
			Description: "Input hash mismatch between metadata and VC document",
		})
	}

	if execVC.OutputHash != vcDoc.CredentialSubject.Execution.OutputHash {
		result.HashValidation = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "output_hash_mismatch",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "output_hash",
			Expected:    execVC.OutputHash,
			Actual:      vcDoc.CredentialSubject.Execution.OutputHash,
			Description: "Output hash mismatch between metadata and VC document",
		})
	}

	// CRITICAL CHECK 9: Signature consistency
	if execVC.Signature != vcDoc.Proof.ProofValue {
		result.StructuralIntegrity = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "signature_mismatch",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "signature",
			Expected:    execVC.Signature,
			Actual:      vcDoc.Proof.ProofValue,
			Description: "Signature mismatch between metadata and VC document",
		})
	}

	// CRITICAL CHECK 10: Timestamp validation
	if err := s.validateTimestamp(vcDoc.IssuanceDate); err != nil {
		result.TimestampValidation = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "invalid_timestamp",
			Severity:    "critical",
			Component:   execVC.VCID,
			Field:       "issuance_date",
			Description: fmt.Sprintf("Invalid timestamp: %v", err),
		})
	}

	// CRITICAL CHECK 11: VC structure validation
	if err := s.validateVCStructure(vcDoc); err != nil {
		result.StructuralIntegrity = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "invalid_structure",
			Severity:    "critical",
			Component:   execVC.VCID,
			Description: fmt.Sprintf("Invalid VC structure: %v", err),
		})
	}

	return result
}

// performSecurityAnalysis performs security-focused analysis
func (s *VCService) performSecurityAnalysis(execVC *types.ExecutionVC, vcDoc *types.VCDocument) SecurityAnalysis {
	result := SecurityAnalysis{
		SignatureStrength: "Ed25519",
		KeyValidation:     true,
		DIDAuthenticity:   true,
		ReplayProtection:  true,
		TamperEvidence:    []string{},
		SecurityScore:     100.0,
		Issues:            []VerificationIssue{},
	}

	// CRITICAL CHECK: Cryptographic signature verification
	issuerIdentity, err := s.didService.ResolveDID(vcDoc.Issuer)
	if err != nil {
		result.DIDAuthenticity = false
		result.SecurityScore -= 50.0
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "did_resolution_failed",
			Severity:    "critical",
			Component:   execVC.VCID,
			Description: fmt.Sprintf("Failed to resolve issuer DID: %v", err),
		})
	} else {
		valid, err := s.verifyVCSignature(vcDoc, issuerIdentity)
		if err != nil || !valid {
			result.KeyValidation = false
			result.SecurityScore -= 40.0
			result.Issues = append(result.Issues, VerificationIssue{
				Type:        "signature_verification_failed",
				Severity:    "critical",
				Component:   execVC.VCID,
				Description: fmt.Sprintf("Signature verification failed: %v", err),
			})
		}
	}

	// Check for tamper evidence
	if evidence := s.detectTamperEvidence(execVC, vcDoc); len(evidence) > 0 {
		result.TamperEvidence = evidence
		result.SecurityScore -= 20.0
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "tamper_evidence",
			Severity:    "warning",
			Component:   execVC.VCID,
			Description: fmt.Sprintf("Tamper evidence detected: %v", evidence),
		})
	}

	return result
}

// performComplianceChecks performs compliance verification
func (s *VCService) performComplianceChecks(vcDoc *types.VCDocument) ComplianceChecks {
	result := ComplianceChecks{
		W3CCompliance:                true,
		AgentFieldStandardCompliance: true,
		AuditTrailIntegrity:          true,
		DataIntegrityChecks:          true,
		Issues:                       []VerificationIssue{},
	}

	// Check W3C compliance
	if !s.checkW3CCompliance(vcDoc) {
		result.W3CCompliance = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "w3c_compliance_failure",
			Severity:    "warning",
			Component:   vcDoc.ID,
			Description: "VC does not meet W3C standards",
		})
	}

	// Check AgentField standard compliance
	if !s.checkAgentFieldStandardCompliance(vcDoc) {
		result.AgentFieldStandardCompliance = false
		result.Issues = append(result.Issues, VerificationIssue{
			Type:        "agentfield_compliance_failure",
			Severity:    "warning",
			Component:   vcDoc.ID,
			Description: "VC does not meet AgentField standard requirements",
		})
	}

	return result
}

// Helper methods for verification checks

func (s *VCService) validateTimestamp(timestamp string) error {
	_, err := time.Parse(time.RFC3339, timestamp)
	return err
}

func (s *VCService) validateVCStructure(vcDoc *types.VCDocument) error {
	// Check required fields
	if len(vcDoc.Context) == 0 {
		return fmt.Errorf("missing @context")
	}
	if len(vcDoc.Type) == 0 {
		return fmt.Errorf("missing type")
	}
	if vcDoc.ID == "" {
		return fmt.Errorf("missing id")
	}
	if vcDoc.Issuer == "" {
		return fmt.Errorf("missing issuer")
	}
	if vcDoc.IssuanceDate == "" {
		return fmt.Errorf("missing issuanceDate")
	}
	return nil
}

func (s *VCService) detectTamperEvidence(execVC *types.ExecutionVC, vcDoc *types.VCDocument) []string {
	evidence := []string{}

	// Check for inconsistencies that indicate tampering
	if execVC.IssuerDID != vcDoc.Issuer {
		evidence = append(evidence, "issuer_did_inconsistency")
	}
	if execVC.ExecutionID != vcDoc.CredentialSubject.ExecutionID {
		evidence = append(evidence, "execution_id_inconsistency")
	}
	if execVC.Signature != vcDoc.Proof.ProofValue {
		evidence = append(evidence, "signature_inconsistency")
	}

	return evidence
}

func (s *VCService) checkW3CCompliance(vcDoc *types.VCDocument) bool {
	// Check W3C VC standard compliance
	requiredContexts := []string{"https://www.w3.org/2018/credentials/v1"}
	for _, required := range requiredContexts {
		found := false
		for _, context := range vcDoc.Context {
			if context == required {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func (s *VCService) checkAgentFieldStandardCompliance(vcDoc *types.VCDocument) bool {
	// Check AgentField-specific compliance requirements
	requiredTypes := []string{"VerifiableCredential", "AgentFieldExecutionCredential"}
	for _, required := range requiredTypes {
		found := false
		for _, vcType := range vcDoc.Type {
			if vcType == required {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func (s *VCService) calculateOverallScore(result *ComprehensiveVCVerificationResult) float64 {
	score := 100.0

	// Deduct points for critical issues
	score -= float64(len(result.CriticalIssues)) * 25.0

	// Deduct points for warnings
	score -= float64(len(result.Warnings)) * 5.0

	// Factor in security score
	score = (score + result.SecurityAnalysis.SecurityScore) / 2.0

	if score < 0 {
		score = 0
	}

	return score
}

// isStatusConsistent checks if status values are consistent, accounting for AgentField system status mapping
func (s *VCService) isStatusConsistent(metadataStatus, vcDocStatus string) bool {
	return types.NormalizeExecutionStatus(metadataStatus) == types.NormalizeExecutionStatus(vcDocStatus)
}

// VerifyWorkflowVCComprehensive performs comprehensive verification of a workflow VC chain
func (s *VCService) VerifyWorkflowVCComprehensive(workflowID string) (*ComprehensiveVCVerificationResult, error) {
	if !s.config.Enabled {
		return &ComprehensiveVCVerificationResult{
			Valid:                 false,
			OverallScore:          0,
			CriticalIssues:        []VerificationIssue{{Type: "system_disabled", Severity: "critical", Description: "DID system is disabled"}},
			VerificationTimestamp: time.Now().UTC().Format(time.RFC3339),
		}, nil
	}

	// Get the workflow VC chain
	vcChain, err := s.GetWorkflowVCChain(workflowID)
	if err != nil {
		return &ComprehensiveVCVerificationResult{
			Valid:                 false,
			OverallScore:          0,
			CriticalIssues:        []VerificationIssue{{Type: "workflow_chain_error", Severity: "critical", Description: fmt.Sprintf("Failed to get workflow VC chain: %v", err)}},
			VerificationTimestamp: time.Now().UTC().Format(time.RFC3339),
		}, nil
	}

	result := &ComprehensiveVCVerificationResult{
		VerificationTimestamp: time.Now().UTC().Format(time.RFC3339),
		CriticalIssues:        []VerificationIssue{},
		Warnings:              []VerificationIssue{},
	}

	// Verify each execution VC in the workflow
	allIntegrityChecks := IntegrityCheckResults{
		MetadataConsistency: true,
		FieldConsistency:    true,
		TimestampValidation: true,
		HashValidation:      true,
		StructuralIntegrity: true,
		Issues:              []VerificationIssue{},
	}

	allSecurityAnalysis := SecurityAnalysis{
		SignatureStrength: "Ed25519",
		KeyValidation:     true,
		DIDAuthenticity:   true,
		ReplayProtection:  true,
		TamperEvidence:    []string{},
		SecurityScore:     100.0,
		Issues:            []VerificationIssue{},
	}

	allComplianceChecks := ComplianceChecks{
		W3CCompliance:                true,
		AgentFieldStandardCompliance: true,
		AuditTrailIntegrity:          true,
		DataIntegrityChecks:          true,
		Issues:                       []VerificationIssue{},
	}

	// Verify each execution VC in the workflow
	for _, execVC := range vcChain.ComponentVCs {
		// Parse VC document
		var vcDoc types.VCDocument
		if err := json.Unmarshal(execVC.VCDocument, &vcDoc); err != nil {
			result.CriticalIssues = append(result.CriticalIssues, VerificationIssue{
				Type:        "parse_error",
				Severity:    "critical",
				Component:   execVC.VCID,
				Description: fmt.Sprintf("Failed to parse VC document: %v", err),
			})
			continue
		}

		// Perform verification checks for this execution VC
		integrityChecks := s.performIntegrityChecks(&execVC, &vcDoc)
		securityAnalysis := s.performSecurityAnalysis(&execVC, &vcDoc)
		complianceChecks := s.performComplianceChecks(&vcDoc)

		// Aggregate results
		if !integrityChecks.MetadataConsistency {
			allIntegrityChecks.MetadataConsistency = false
		}
		if !integrityChecks.FieldConsistency {
			allIntegrityChecks.FieldConsistency = false
		}
		if !integrityChecks.TimestampValidation {
			allIntegrityChecks.TimestampValidation = false
		}
		if !integrityChecks.HashValidation {
			allIntegrityChecks.HashValidation = false
		}
		if !integrityChecks.StructuralIntegrity {
			allIntegrityChecks.StructuralIntegrity = false
		}

		if !securityAnalysis.KeyValidation {
			allSecurityAnalysis.KeyValidation = false
		}
		if !securityAnalysis.DIDAuthenticity {
			allSecurityAnalysis.DIDAuthenticity = false
		}
		if !securityAnalysis.ReplayProtection {
			allSecurityAnalysis.ReplayProtection = false
		}

		if !complianceChecks.W3CCompliance {
			allComplianceChecks.W3CCompliance = false
		}
		if !complianceChecks.AgentFieldStandardCompliance {
			allComplianceChecks.AgentFieldStandardCompliance = false
		}
		if !complianceChecks.AuditTrailIntegrity {
			allComplianceChecks.AuditTrailIntegrity = false
		}
		if !complianceChecks.DataIntegrityChecks {
			allComplianceChecks.DataIntegrityChecks = false
		}

		// Collect all issues
		allIntegrityChecks.Issues = append(allIntegrityChecks.Issues, integrityChecks.Issues...)
		allSecurityAnalysis.Issues = append(allSecurityAnalysis.Issues, securityAnalysis.Issues...)
		allComplianceChecks.Issues = append(allComplianceChecks.Issues, complianceChecks.Issues...)

		// Collect tamper evidence
		allSecurityAnalysis.TamperEvidence = append(allSecurityAnalysis.TamperEvidence, securityAnalysis.TamperEvidence...)

		// Update security score (take minimum)
		if securityAnalysis.SecurityScore < allSecurityAnalysis.SecurityScore {
			allSecurityAnalysis.SecurityScore = securityAnalysis.SecurityScore
		}
	}

	// Verify workflow VC itself if it exists
	if vcChain.WorkflowVC.VCDocument != nil {
		var workflowVCDoc types.WorkflowVCDocument
		if err := json.Unmarshal(vcChain.WorkflowVC.VCDocument, &workflowVCDoc); err != nil {
			result.CriticalIssues = append(result.CriticalIssues, VerificationIssue{
				Type:        "workflow_vc_parse_error",
				Severity:    "critical",
				Component:   vcChain.WorkflowVC.WorkflowVCID,
				Description: fmt.Sprintf("Failed to parse workflow VC document: %v", err),
			})
		} else {
			// Verify workflow VC signature
			issuerIdentity, err := s.didService.ResolveDID(workflowVCDoc.Issuer)
			if err != nil {
				allSecurityAnalysis.DIDAuthenticity = false
				allSecurityAnalysis.Issues = append(allSecurityAnalysis.Issues, VerificationIssue{
					Type:        "workflow_did_resolution_failed",
					Severity:    "critical",
					Component:   vcChain.WorkflowVC.WorkflowVCID,
					Description: fmt.Sprintf("Failed to resolve workflow VC issuer DID: %v", err),
				})
			} else {
				valid, err := s.verifyWorkflowVCSignature(&workflowVCDoc, issuerIdentity)
				if err != nil || !valid {
					allSecurityAnalysis.KeyValidation = false
					allSecurityAnalysis.Issues = append(allSecurityAnalysis.Issues, VerificationIssue{
						Type:        "workflow_signature_verification_failed",
						Severity:    "critical",
						Component:   vcChain.WorkflowVC.WorkflowVCID,
						Description: fmt.Sprintf("Workflow VC signature verification failed: %v", err),
					})
				}
			}

			// Check workflow VC compliance
			if !s.checkWorkflowVCCompliance(&workflowVCDoc) {
				allComplianceChecks.AgentFieldStandardCompliance = false
				allComplianceChecks.Issues = append(allComplianceChecks.Issues, VerificationIssue{
					Type:        "workflow_compliance_failure",
					Severity:    "warning",
					Component:   vcChain.WorkflowVC.WorkflowVCID,
					Description: "Workflow VC does not meet AgentField standard requirements",
				})
			}
		}
	}

	// Set aggregated results
	result.IntegrityChecks = allIntegrityChecks
	result.SecurityAnalysis = allSecurityAnalysis
	result.ComplianceChecks = allComplianceChecks

	// Collect all issues
	allIssues := []VerificationIssue{}
	allIssues = append(allIssues, result.IntegrityChecks.Issues...)
	allIssues = append(allIssues, result.SecurityAnalysis.Issues...)
	allIssues = append(allIssues, result.ComplianceChecks.Issues...)

	// Separate critical issues and warnings
	for _, issue := range allIssues {
		if issue.Severity == "critical" {
			result.CriticalIssues = append(result.CriticalIssues, issue)
		} else if issue.Severity == "warning" {
			result.Warnings = append(result.Warnings, issue)
		}
	}

	// Calculate overall validity and score
	result.Valid = len(result.CriticalIssues) == 0
	result.OverallScore = s.calculateOverallScore(result)

	return result, nil
}

// verifyWorkflowVCSignature verifies the signature of a WorkflowVC document
func (s *VCService) verifyWorkflowVCSignature(vcDoc *types.WorkflowVCDocument, issuerIdentity *types.DIDIdentity) (bool, error) {
	// Create canonical representation for verification
	vcCopy := *vcDoc
	vcCopy.Proof = types.VCProof{} // Remove proof for verification

	canonicalBytes, err := json.Marshal(vcCopy)
	if err != nil {
		return false, fmt.Errorf("failed to marshal workflow VC for verification: %w", err)
	}

	// Parse public key from JWK
	var jwk map[string]interface{}
	if err := json.Unmarshal([]byte(issuerIdentity.PublicKeyJWK), &jwk); err != nil {
		return false, fmt.Errorf("failed to parse public key JWK: %w", err)
	}

	xValue, ok := jwk["x"].(string)
	if !ok {
		return false, fmt.Errorf("invalid public key JWK: missing 'x' parameter")
	}

	publicKeyBytes, err := base64.RawURLEncoding.DecodeString(xValue)
	if err != nil {
		return false, fmt.Errorf("failed to decode public key: %w", err)
	}

	if len(publicKeyBytes) != ed25519.PublicKeySize {
		return false, fmt.Errorf("invalid public key length: got %d, want %d", len(publicKeyBytes), ed25519.PublicKeySize)
	}

	publicKey := ed25519.PublicKey(publicKeyBytes)

	// Decode signature
	signatureBytes, err := base64.RawURLEncoding.DecodeString(vcDoc.Proof.ProofValue)
	if err != nil {
		return false, fmt.Errorf("failed to decode signature: %w", err)
	}

	// Verify signature
	return ed25519.Verify(publicKey, canonicalBytes, signatureBytes), nil
}

// checkWorkflowVCCompliance checks if a workflow VC meets AgentField standard compliance
func (s *VCService) checkWorkflowVCCompliance(vcDoc *types.WorkflowVCDocument) bool {
	// Check AgentField-specific compliance requirements for workflow VCs
	requiredTypes := []string{"VerifiableCredential", "AgentFieldWorkflowCredential"}
	for _, required := range requiredTypes {
		found := false
		for _, vcType := range vcDoc.Type {
			if vcType == required {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}
