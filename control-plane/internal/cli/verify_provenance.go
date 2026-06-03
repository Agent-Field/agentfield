package cli

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// MaxVerifyAuditBodyBytes is the maximum JSON body size for audit-bundle verification (HTTP handlers).
const MaxVerifyAuditBodyBytes = 10 << 20 // 10 MiB

// VerifyProvenanceJSON verifies AgentField provenance JSON: enhanced or legacy workflow chain,
// or a bare W3C VerifiableCredential document (same shape as UI "download VC document").
// It returns a structured result suitable for APIs and CLI output; it does not write to stdout.
func VerifyProvenanceJSON(vcData []byte, options VerifyOptions) VCVerificationResult {
	result := VCVerificationResult{
		VerifiedAt:        time.Now().UTC().Format(time.RFC3339),
		VerificationSteps: []VerificationStep{},
		DIDResolutions:    []DIDResolutionResult{},
		ComponentResults:  []ComponentVerification{},
	}

	step2 := VerificationStep{Step: 2, Description: "Parsing VC structure"}
	var enhancedChain EnhancedVCChain
	var parseKind string

	if chain, ok := tryParseEnhancedChain(vcData); ok {
		enhancedChain = chain
		normalizeEnhancedChain(&enhancedChain)
		step2.Success = true
		step2.Details = fmt.Sprintf("Parsed enhanced VC chain with %d execution VCs", len(enhancedChain.ExecutionVCs))
		parseKind = "workflow"
	} else if chain, ok := tryParseLegacyWorkflowChain(vcData); ok {
		enhancedChain = chain
		normalizeEnhancedChain(&enhancedChain)
		step2.Success = true
		step2.Details = fmt.Sprintf("Parsed legacy VC chain with %d execution VCs", len(enhancedChain.ExecutionVCs))
		parseKind = "workflow"
	} else if chain, ok := tryParseBareExecutionVC(vcData); ok {
		enhancedChain = chain
		normalizeEnhancedChain(&enhancedChain)
		step2.Success = true
		step2.Details = "Parsed bare W3C VerifiableCredential (execution VC document)"
		parseKind = "credential"
	} else {
		step2.Success = false
		step2.Error = "Invalid VC format: not a recognized AgentField VC structure or W3C VerifiableCredential"
		result.VerificationSteps = append(result.VerificationSteps, step2)
		result.Valid = false
		result.FormatValid = false
		result.Error = step2.Error
		return result
	}
	result.VerificationSteps = append(result.VerificationSteps, step2)

	if parseKind == "workflow" {
		result.Type = "workflow"
		result.WorkflowID = enhancedChain.WorkflowID
	} else {
		result.Type = "credential"
		if enhancedChain.WorkflowID != "" {
			result.WorkflowID = enhancedChain.WorkflowID
		}
	}
	result.FormatValid = true

	step3 := VerificationStep{Step: 3, Description: "Collecting unique DIDs"}
	uniqueDIDs := collectUniqueDIDs(enhancedChain)
	step3.Success = true
	step3.Details = fmt.Sprintf("Found %d unique DIDs", len(uniqueDIDs))
	result.VerificationSteps = append(result.VerificationSteps, step3)

	step4 := VerificationStep{Step: 4, Description: "Resolving DIDs"}
	didResolutions := make(map[string]DIDResolutionInfo)
	resolvedCount := 0

	if len(uniqueDIDs) == 0 {
		step4.Success = true
		step4.Details = "No DIDs to resolve"
		result.VerificationSteps = append(result.VerificationSteps, step4)
	} else {
		for _, did := range uniqueDIDs {
			resolution, err := resolveDID(did, enhancedChain.DIDResolutionBundle, options)
			didResult := DIDResolutionResult{
				DID:    did,
				Method: getDIDMethod(did),
			}

			if err != nil {
				didResult.Success = false
				didResult.Error = err.Error()
			} else {
				didResult.Success = true
				didResult.ResolvedFrom = resolution.ResolvedFrom
				if resolution.WebURL != "" {
					didResult.WebURL = resolution.WebURL
				}
				didResolutions[did] = resolution
				resolvedCount++
			}
			result.DIDResolutions = append(result.DIDResolutions, didResult)
		}

		step4.Success = resolvedCount > 0
		step4.Details = fmt.Sprintf("Resolved %d/%d DIDs", resolvedCount, len(uniqueDIDs))
		if resolvedCount == 0 {
			step4.Error = "Failed to resolve any DIDs"
		}
		result.VerificationSteps = append(result.VerificationSteps, step4)
	}

	step5 := VerificationStep{Step: 5, Description: "Performing comprehensive verification"}
	enhancedVerifier := NewEnhancedVCVerifier(didResolutions, options.Verbose)
	comprehensiveResult := enhancedVerifier.VerifyEnhancedVCChain(enhancedChain)
	result.Comprehensive = comprehensiveResult

	validSignatures := 0
	totalSignatures := len(enhancedChain.ExecutionVCs)
	if enhancedChain.WorkflowVC.VCDocument != nil {
		totalSignatures++
	}

	validCount := 0
	for _, compResult := range comprehensiveResult.ComponentResults {
		if compResult.SignatureValid {
			validSignatures++
		}
		legacyResult := ComponentVerification{
			VCID:           compResult.VCID,
			ExecutionID:    compResult.ExecutionID,
			IssuerDID:      compResult.IssuerDID,
			Valid:          compResult.Valid,
			SignatureValid: compResult.SignatureValid,
			FormatValid:    compResult.FormatValid,
			Status:         compResult.Status,
			DurationMS:     compResult.DurationMS,
			Timestamp:      compResult.Timestamp,
			Error:          compResult.Error,
		}
		result.ComponentResults = append(result.ComponentResults, legacyResult)
		if compResult.Valid {
			validCount++
		}
	}

	step5.Success = comprehensiveResult.Valid
	step5.Details = fmt.Sprintf("Comprehensive verification completed - Score: %.1f/100", comprehensiveResult.OverallScore)
	if !comprehensiveResult.Valid {
		step5.Error = fmt.Sprintf("Found %d critical issues", len(comprehensiveResult.CriticalIssues))
	}
	result.VerificationSteps = append(result.VerificationSteps, step5)

	if options.Verbose && len(comprehensiveResult.CriticalIssues) > 0 {
		step6 := VerificationStep{Step: 6, Description: "Critical Issues Detected"}
		step6.Success = false
		details := "Critical issues found:\n"
		for _, issue := range comprehensiveResult.CriticalIssues {
			details += fmt.Sprintf("  - %s: %s\n", issue.Type, issue.Description)
		}
		step6.Details = details
		result.VerificationSteps = append(result.VerificationSteps, step6)
	}

	result.SignatureValid = comprehensiveResult.SecurityAnalysis.SecurityScore > 80.0
	result.Valid = comprehensiveResult.Valid

	if result.Valid {
		if parseKind == "credential" {
			result.Message = fmt.Sprintf("Verifiable credential verified successfully (Score: %.1f/100)", comprehensiveResult.OverallScore)
		} else {
			result.Message = fmt.Sprintf("Workflow VC chain verified successfully (Score: %.1f/100)", comprehensiveResult.OverallScore)
		}
	} else {
		if parseKind == "credential" {
			result.Message = fmt.Sprintf("Verifiable credential verification failed (Score: %.1f/100)", comprehensiveResult.OverallScore)
		} else {
			result.Message = fmt.Sprintf("Workflow VC chain verification failed (Score: %.1f/100)", comprehensiveResult.OverallScore)
		}
		if len(comprehensiveResult.CriticalIssues) > 0 {
			result.Error = fmt.Sprintf("%d critical issues detected", len(comprehensiveResult.CriticalIssues))
		}
	}

	result.Summary = VerificationSummary{
		TotalComponents: len(enhancedChain.ExecutionVCs),
		ValidComponents: validCount,
		TotalDIDs:       len(uniqueDIDs),
		ResolvedDIDs:    resolvedCount,
		TotalSignatures: totalSignatures,
		ValidSignatures: validSignatures,
	}

	return result
}

func tryParseEnhancedChain(vcData []byte) (EnhancedVCChain, bool) {
	var enhancedChain EnhancedVCChain
	if err := json.Unmarshal(vcData, &enhancedChain); err != nil {
		return EnhancedVCChain{}, false
	}
	if enhancedChain.WorkflowID == "" {
		return EnhancedVCChain{}, false
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(vcData, &raw); err == nil {
		if bundleRaw, ok := raw["did_resolution_bundle"]; ok && len(bundleRaw) > 0 && string(bundleRaw) != "null" {
			var bundle map[string]types.DIDResolutionEntry
			if err := json.Unmarshal(bundleRaw, &bundle); err == nil && len(bundle) > 0 {
				enhancedChain.DIDResolutionBundle = mergeDIDBundle(enhancedChain.DIDResolutionBundle, bundle)
			}
		}
	}
	return enhancedChain, true
}

func tryParseLegacyWorkflowChain(vcData []byte) (EnhancedVCChain, bool) {
	var workflowChain types.WorkflowVCChainResponse
	if err := json.Unmarshal(vcData, &workflowChain); err != nil {
		return EnhancedVCChain{}, false
	}
	if workflowChain.WorkflowID == "" {
		return EnhancedVCChain{}, false
	}
	return convertLegacyChain(workflowChain), true
}

func tryParseBareExecutionVC(vcData []byte) (EnhancedVCChain, bool) {
	execVC, err := executionVCFromBareDocument(json.RawMessage(vcData))
	if err != nil {
		return EnhancedVCChain{}, false
	}
	wfID := execVC.WorkflowID
	if wfID == "" {
		wfID = "standalone-vc"
	}
	chain := EnhancedVCChain{
		WorkflowID:          wfID,
		GeneratedAt:         time.Now().UTC().Format(time.RFC3339),
		TotalExecutions:     1,
		CompletedExecutions: 1,
		WorkflowStatus:      execVC.Status,
		ExecutionVCs:        []types.ExecutionVC{execVC},
		ComponentVCs:        []types.ExecutionVC{execVC},
		WorkflowVC:          types.WorkflowVC{},
	}
	return chain, true
}

func executionVCFromBareDocument(raw json.RawMessage) (types.ExecutionVC, error) {
	var vcDoc types.VCDocument
	if err := json.Unmarshal(raw, &vcDoc); err != nil {
		return types.ExecutionVC{}, err
	}
	hasVC := false
	for _, t := range vcDoc.Type {
		if t == "VerifiableCredential" {
			hasVC = true
			break
		}
	}
	if !hasVC {
		return types.ExecutionVC{}, fmt.Errorf("missing VerifiableCredential type")
	}

	sub := vcDoc.CredentialSubject
	wfID := sub.WorkflowID
	if wfID == "" {
		wfID = "standalone-vc"
	}

	createdAt := time.Now().UTC()
	if vcDoc.IssuanceDate != "" {
		if ts, err := time.Parse(time.RFC3339, vcDoc.IssuanceDate); err == nil {
			createdAt = ts
		}
	}

	vcID := vcDoc.ID
	if vcID == "" {
		vcID = "bare-vc"
	}

	return types.ExecutionVC{
		VCID:        vcID,
		ExecutionID: sub.ExecutionID,
		WorkflowID:  wfID,
		SessionID:   sub.SessionID,
		IssuerDID:   vcDoc.Issuer,
		TargetDID:   sub.Target.DID,
		CallerDID:   sub.Caller.DID,
		VCDocument:  raw,
		Signature:   vcDoc.Proof.ProofValue,
		InputHash:   sub.Execution.InputHash,
		OutputHash:  sub.Execution.OutputHash,
		Status:      sub.Execution.Status,
		CreatedAt:   createdAt,
	}, nil
}

func mergeDIDBundle(
	existing map[string]DIDResolutionInfo,
	entries map[string]types.DIDResolutionEntry,
) map[string]DIDResolutionInfo {
	out := make(map[string]DIDResolutionInfo, len(existing)+len(entries))
	for k, v := range existing {
		out[k] = v
	}
	for did, ent := range entries {
		var jwk map[string]interface{}
		_ = json.Unmarshal(ent.PublicKeyJWK, &jwk)
		out[did] = DIDResolutionInfo{
			DID:          did,
			Method:       ent.Method,
			PublicKeyJWK: jwk,
			ResolvedFrom: ent.ResolvedFrom,
			CachedAt:     ent.ResolvedAt,
		}
	}
	return out
}
