package cli

import (
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/your-org/agentfield/control-plane/pkg/types"
)

// NewVCCommand creates the vc command with subcommands
func NewVCCommand() *cobra.Command {
	vcCmd := &cobra.Command{
		Use:   "vc",
		Short: "Verifiable Credential operations",
		Long:  `Commands for working with AgentField Verifiable Credentials (VCs)`,
	}

	vcCmd.AddCommand(NewVCVerifyCommand())
	return vcCmd
}

// NewVCVerifyCommand creates the vc verify subcommand
func NewVCVerifyCommand() *cobra.Command {
	var outputFormat string
	var resolveWeb bool
	var didResolver string
	var verbose bool

	verifyCmd := &cobra.Command{
		Use:   "verify <vc-file.json>",
		Short: "Verify a AgentField Verifiable Credential",
		Long: `Verify the cryptographic signature and integrity of a AgentField Verifiable Credential.
This command supports offline verification with bundled DIDs and online verification with web resolution.`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			vcFilePath := args[0]
			options := VerifyOptions{
				OutputFormat: outputFormat,
				ResolveWeb:   resolveWeb,
				Resolver:     didResolver,
				Verbose:      verbose,
			}
			return verifyVC(vcFilePath, options)
		},
	}

	verifyCmd.Flags().StringVarP(&outputFormat, "format", "f", "json", "Output format (json, pretty)")
	verifyCmd.Flags().BoolVar(&resolveWeb, "resolve-web", false, "Resolve all DIDs from web")
	verifyCmd.Flags().StringVar(&didResolver, "did-resolver", "", "Custom DID resolver URL")
	verifyCmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "Verbose output with verification steps")
	return verifyCmd
}

// VerifyOptions holds verification configuration
type VerifyOptions struct {
	OutputFormat string
	ResolveWeb   bool
	Resolver     string
	Verbose      bool
}

// DIDResolutionInfo represents DID resolution information
type DIDResolutionInfo struct {
	DID          string                 `json:"did"`
	Method       string                 `json:"method"`
	PublicKeyJWK map[string]interface{} `json:"public_key_jwk"`
	WebURL       string                 `json:"web_url,omitempty"`
	CachedAt     string                 `json:"cached_at,omitempty"`
	ResolvedFrom string                 `json:"resolved_from"`
}

// EnhancedVCChain represents a VC chain with DID resolution bundle
type EnhancedVCChain struct {
	WorkflowID           string                       `json:"workflow_id"`
	GeneratedAt          string                       `json:"generated_at"`
	TotalExecutions      int                          `json:"total_executions"`
	CompletedExecutions  int                          `json:"completed_executions"`
	WorkflowStatus       string                       `json:"workflow_status"`
	ExecutionVCs         []types.ExecutionVC          `json:"execution_vcs"`
	WorkflowVC           types.WorkflowVC             `json:"workflow_vc"`
	DIDResolutionBundle  map[string]DIDResolutionInfo `json:"did_resolution_bundle,omitempty"`
	VerificationMetadata VerificationMetadata         `json:"verification_metadata,omitempty"`
}

// VerificationMetadata contains metadata about the verification process
type VerificationMetadata struct {
	ExportVersion   string `json:"export_version"`
	TotalSignatures int    `json:"total_signatures"`
	BundledDIDs     int    `json:"bundled_dids"`
	ExportTimestamp string `json:"export_timestamp"`
}

// VCVerificationResult represents the comprehensive verification result
type VCVerificationResult struct {
	Valid             bool                    `json:"valid"`
	Type              string                  `json:"type"`
	WorkflowID        string                  `json:"workflow_id,omitempty"`
	SignatureValid    bool                    `json:"signature_valid"`
	FormatValid       bool                    `json:"format_valid"`
	Message           string                  `json:"message"`
	Error             string                  `json:"error,omitempty"`
	VerifiedAt        string                  `json:"verified_at"`
	ComponentResults  []ComponentVerification `json:"component_results,omitempty"`
	DIDResolutions    []DIDResolutionResult   `json:"did_resolutions,omitempty"`
	VerificationSteps []VerificationStep      `json:"verification_steps,omitempty"`
	Summary           VerificationSummary     `json:"summary"`
}

// ComponentVerification represents verification result for a single component
type ComponentVerification struct {
	VCID           string `json:"vc_id"`
	ExecutionID    string `json:"execution_id"`
	IssuerDID      string `json:"issuer_did"`
	Valid          bool   `json:"valid"`
	SignatureValid bool   `json:"signature_valid"`
	FormatValid    bool   `json:"format_valid"`
	Status         string `json:"status"`
	DurationMS     int    `json:"duration_ms"`
	Timestamp      string `json:"timestamp"`
	Error          string `json:"error,omitempty"`
}

// DIDResolutionResult represents the result of DID resolution
type DIDResolutionResult struct {
	DID          string `json:"did"`
	Method       string `json:"method"`
	ResolvedFrom string `json:"resolved_from"`
	Success      bool   `json:"success"`
	Error        string `json:"error,omitempty"`
	WebURL       string `json:"web_url,omitempty"`
}

// VerificationStep represents a single step in the verification process
type VerificationStep struct {
	Step        int    `json:"step"`
	Description string `json:"description"`
	Success     bool   `json:"success"`
	Details     string `json:"details,omitempty"`
	Error       string `json:"error,omitempty"`
}

// VerificationSummary provides a high-level summary
type VerificationSummary struct {
	TotalComponents int `json:"total_components"`
	ValidComponents int `json:"valid_components"`
	TotalDIDs       int `json:"total_dids"`
	ResolvedDIDs    int `json:"resolved_dids"`
	TotalSignatures int `json:"total_signatures"`
	ValidSignatures int `json:"valid_signatures"`
}

func verifyVC(vcFilePath string, options VerifyOptions) error {
	result := VCVerificationResult{
		VerifiedAt:        time.Now().UTC().Format(time.RFC3339),
		VerificationSteps: []VerificationStep{},
		DIDResolutions:    []DIDResolutionResult{},
		ComponentResults:  []ComponentVerification{},
	}

	// Step 1: Read and parse VC file
	step1 := VerificationStep{Step: 1, Description: "Reading VC file"}
	vcData, err := ioutil.ReadFile(vcFilePath)
	if err != nil {
		step1.Success = false
		step1.Error = fmt.Sprintf("Failed to read VC file: %v", err)
		result.VerificationSteps = append(result.VerificationSteps, step1)
		result.Valid = false
		result.Error = step1.Error
		return outputResult(result, options)
	}
	step1.Success = true
	step1.Details = fmt.Sprintf("Read %d bytes from %s", len(vcData), vcFilePath)
	result.VerificationSteps = append(result.VerificationSteps, step1)

	// Step 2: Parse VC structure
	step2 := VerificationStep{Step: 2, Description: "Parsing VC structure"}
	var enhancedChain EnhancedVCChain
	if err := json.Unmarshal(vcData, &enhancedChain); err == nil && enhancedChain.WorkflowID != "" {
		// Enhanced VC chain with DID resolution bundle
		step2.Success = true
		step2.Details = fmt.Sprintf("Parsed enhanced VC chain with %d execution VCs", len(enhancedChain.ExecutionVCs))
		result.Type = "workflow"
		result.WorkflowID = enhancedChain.WorkflowID
		result.FormatValid = true
	} else {
		// Try legacy format
		var workflowChain types.WorkflowVCChainResponse
		if err := json.Unmarshal(vcData, &workflowChain); err == nil && workflowChain.WorkflowID != "" {
			step2.Success = true
			step2.Details = fmt.Sprintf("Parsed legacy VC chain with %d execution VCs", len(workflowChain.ComponentVCs))
			result.Type = "workflow"
			result.WorkflowID = workflowChain.WorkflowID
			result.FormatValid = true
			// Convert to enhanced format
			enhancedChain = convertLegacyChain(workflowChain)
		} else {
			step2.Success = false
			step2.Error = "Invalid VC format: not a recognized AgentField VC structure"
			result.VerificationSteps = append(result.VerificationSteps, step2)
			result.Valid = false
			result.FormatValid = false
			result.Error = step2.Error
			return outputResult(result, options)
		}
	}
	result.VerificationSteps = append(result.VerificationSteps, step2)

	// Step 3: Collect unique DIDs
	step3 := VerificationStep{Step: 3, Description: "Collecting unique DIDs"}
	uniqueDIDs := collectUniqueDIDs(enhancedChain)
	step3.Success = true
	step3.Details = fmt.Sprintf("Found %d unique DIDs", len(uniqueDIDs))
	result.VerificationSteps = append(result.VerificationSteps, step3)

	// Step 4: Resolve DIDs
	step4 := VerificationStep{Step: 4, Description: "Resolving DIDs"}
	didResolutions := make(map[string]DIDResolutionInfo)
	resolvedCount := 0

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

	// Step 5: Enhanced comprehensive verification
	step5 := VerificationStep{Step: 5, Description: "Performing comprehensive verification"}

	// Use the enhanced verifier for comprehensive checks
	enhancedVerifier := NewEnhancedVCVerifier(didResolutions, options.Verbose)
	comprehensiveResult := enhancedVerifier.VerifyEnhancedVCChain(enhancedChain)

	// Convert comprehensive result to legacy format for compatibility
	validSignatures := 0
	totalSignatures := len(enhancedChain.ExecutionVCs)
	if enhancedChain.WorkflowVC.VCDocument != nil {
		totalSignatures++
	}

	for _, compResult := range comprehensiveResult.ComponentResults {
		if compResult.SignatureValid {
			validSignatures++
		}
		// Convert to legacy ComponentVerification format
		legacyResult := ComponentVerification{
			VCID:           compResult.VCID,
			ExecutionID:    compResult.ExecutionID,
			IssuerDID:      compResult.IssuerDID,
			Valid:          compResult.Valid,
			SignatureValid: compResult.SignatureValid,
			FormatValid:    compResult.FormatValid,
			Status:         compResult.Status,
			Error:          compResult.Error,
		}
		result.ComponentResults = append(result.ComponentResults, legacyResult)
	}

	step5.Success = comprehensiveResult.Valid
	step5.Details = fmt.Sprintf("Comprehensive verification completed - Score: %.1f/100", comprehensiveResult.OverallScore)
	if !comprehensiveResult.Valid {
		step5.Error = fmt.Sprintf("Found %d critical issues", len(comprehensiveResult.CriticalIssues))
	}
	result.VerificationSteps = append(result.VerificationSteps, step5)

	// Add detailed verification information if verbose
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

	// Final result based on comprehensive verification
	result.SignatureValid = comprehensiveResult.SecurityAnalysis.SecurityScore > 80.0
	result.Valid = comprehensiveResult.Valid

	if result.Valid {
		result.Message = fmt.Sprintf("Workflow VC chain verified successfully (Score: %.1f/100)", comprehensiveResult.OverallScore)
	} else {
		result.Message = fmt.Sprintf("Workflow VC chain verification failed (Score: %.1f/100)", comprehensiveResult.OverallScore)
		if len(comprehensiveResult.CriticalIssues) > 0 {
			result.Error = fmt.Sprintf("%d critical issues detected", len(comprehensiveResult.CriticalIssues))
		}
	}

	// Summary
	result.Summary = VerificationSummary{
		TotalComponents: len(enhancedChain.ExecutionVCs),
		ValidComponents: len(result.ComponentResults),
		TotalDIDs:       len(uniqueDIDs),
		ResolvedDIDs:    resolvedCount,
		TotalSignatures: totalSignatures,
		ValidSignatures: validSignatures,
	}

	return outputResult(result, options)
}

func collectUniqueDIDs(chain EnhancedVCChain) []string {
	didSet := make(map[string]bool)
	var dids []string

	// Collect from execution VCs
	for _, execVC := range chain.ExecutionVCs {
		var vcDoc types.VCDocument
		if err := json.Unmarshal(execVC.VCDocument, &vcDoc); err == nil {
			if !didSet[vcDoc.Issuer] {
				didSet[vcDoc.Issuer] = true
				dids = append(dids, vcDoc.Issuer)
			}
		}
	}

	// Collect from workflow VC
	if chain.WorkflowVC.VCDocument != nil {
		var workflowVCDoc types.WorkflowVCDocument
		if err := json.Unmarshal(chain.WorkflowVC.VCDocument, &workflowVCDoc); err == nil {
			if !didSet[workflowVCDoc.Issuer] {
				didSet[workflowVCDoc.Issuer] = true
				dids = append(dids, workflowVCDoc.Issuer)
			}
		}
	}

	return dids
}

func resolveDID(did string, bundle map[string]DIDResolutionInfo, options VerifyOptions) (DIDResolutionInfo, error) {
	// 1. did:web always resolves from web
	if strings.HasPrefix(did, "did:web:") {
		return resolveWebDID(did)
	}

	// 2. --resolve-web flag: fetch from web
	if options.ResolveWeb {
		return resolveFromWeb(did, options.Resolver)
	}

	// 3. --did-resolver: use custom endpoint
	if options.Resolver != "" {
		return resolveFromCustom(did, options.Resolver)
	}

	// 4. Fallback: use bundled resolution
	if bundle != nil {
		if resolution, exists := bundle[did]; exists {
			resolution.ResolvedFrom = "bundled"
			return resolution, nil
		}
	}

	return DIDResolutionInfo{}, fmt.Errorf("DID resolution failed: no resolution method available for %s", did)
}

func resolveWebDID(did string) (DIDResolutionInfo, error) {
	// Parse did:web:domain:path format
	parts := strings.Split(did, ":")
	if len(parts) < 3 {
		return DIDResolutionInfo{}, fmt.Errorf("invalid did:web format")
	}

	domain := parts[2]
	path := "/.well-known/did.json"

	if len(parts) > 3 {
		path = "/" + strings.Join(parts[3:], "/") + "/did.json"
	}

	url := fmt.Sprintf("https://%s%s", domain, path)

	resp, err := http.Get(url)
	if err != nil {
		return DIDResolutionInfo{}, fmt.Errorf("failed to fetch DID document: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return DIDResolutionInfo{}, fmt.Errorf("DID document not found: HTTP %d", resp.StatusCode)
	}

	var didDoc map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&didDoc); err != nil {
		return DIDResolutionInfo{}, fmt.Errorf("failed to parse DID document: %v", err)
	}

	// Extract public key from DID document
	publicKeyJWK, err := extractPublicKeyFromDIDDoc(didDoc)
	if err != nil {
		return DIDResolutionInfo{}, err
	}

	return DIDResolutionInfo{
		DID:          did,
		Method:       "web",
		PublicKeyJWK: publicKeyJWK,
		WebURL:       url,
		ResolvedFrom: "web",
	}, nil
}

func resolveFromWeb(did, resolver string) (DIDResolutionInfo, error) {
	// For did:key, we can resolve locally
	if strings.HasPrefix(did, "did:key:") {
		return resolveKeyDID(did)
	}

	// For other methods, would need a universal resolver
	return DIDResolutionInfo{}, fmt.Errorf("web resolution not supported for DID method: %s", getDIDMethod(did))
}

func resolveFromCustom(did, resolver string) (DIDResolutionInfo, error) {
	url := fmt.Sprintf("%s/%s", strings.TrimSuffix(resolver, "/"), did)

	resp, err := http.Get(url)
	if err != nil {
		return DIDResolutionInfo{}, fmt.Errorf("failed to resolve DID: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return DIDResolutionInfo{}, fmt.Errorf("DID resolution failed: HTTP %d", resp.StatusCode)
	}

	var resolution DIDResolutionInfo
	if err := json.NewDecoder(resp.Body).Decode(&resolution); err != nil {
		return DIDResolutionInfo{}, fmt.Errorf("failed to parse resolution response: %v", err)
	}

	resolution.ResolvedFrom = "custom"
	return resolution, nil
}

func resolveKeyDID(did string) (DIDResolutionInfo, error) {
	// Extract the key from did:key format
	if !strings.HasPrefix(did, "did:key:") {
		return DIDResolutionInfo{}, fmt.Errorf("invalid did:key format")
	}

	// For now, return a placeholder - in a full implementation,
	// we would decode the multibase-encoded key
	return DIDResolutionInfo{
		DID:          did,
		Method:       "key",
		ResolvedFrom: "local",
		// PublicKeyJWK would be extracted from the key encoding
	}, fmt.Errorf("did:key resolution not fully implemented")
}

func extractPublicKeyFromDIDDoc(didDoc map[string]interface{}) (map[string]interface{}, error) {
	verificationMethod, ok := didDoc["verificationMethod"].([]interface{})
	if !ok || len(verificationMethod) == 0 {
		return nil, fmt.Errorf("no verification methods found in DID document")
	}

	firstMethod, ok := verificationMethod[0].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid verification method format")
	}

	publicKeyJwk, ok := firstMethod["publicKeyJwk"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("no publicKeyJwk found in verification method")
	}

	return publicKeyJwk, nil
}

func verifyVCSignature(vcDoc types.VCDocument, resolution DIDResolutionInfo) (bool, error) {
	// Create canonical representation for verification
	vcCopy := vcDoc
	vcCopy.Proof = types.VCProof{} // Remove proof for verification

	canonicalBytes, err := json.Marshal(vcCopy)
	if err != nil {
		return false, fmt.Errorf("failed to marshal VC for verification: %w", err)
	}

	// Extract public key from JWK
	xValue, ok := resolution.PublicKeyJWK["x"].(string)
	if !ok {
		return false, fmt.Errorf("invalid public key JWK: missing 'x' parameter")
	}

	publicKeyBytes, err := base64.RawURLEncoding.DecodeString(xValue)
	if err != nil {
		return false, fmt.Errorf("failed to decode public key: %w", err)
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

func verifyWorkflowVCSignature(vcDoc types.WorkflowVCDocument, resolution DIDResolutionInfo) (bool, error) {
	// Create canonical representation for verification
	vcCopy := vcDoc
	vcCopy.Proof = types.VCProof{} // Remove proof for verification

	canonicalBytes, err := json.Marshal(vcCopy)
	if err != nil {
		return false, fmt.Errorf("failed to marshal workflow VC for verification: %w", err)
	}

	// Extract public key from JWK
	xValue, ok := resolution.PublicKeyJWK["x"].(string)
	if !ok {
		return false, fmt.Errorf("invalid public key JWK: missing 'x' parameter")
	}

	publicKeyBytes, err := base64.RawURLEncoding.DecodeString(xValue)
	if err != nil {
		return false, fmt.Errorf("failed to decode public key: %w", err)
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

func getDIDMethod(did string) string {
	parts := strings.Split(did, ":")
	if len(parts) >= 2 {
		return parts[1]
	}
	return "unknown"
}

func convertLegacyChain(legacy types.WorkflowVCChainResponse) EnhancedVCChain {
	return EnhancedVCChain{
		WorkflowID:          legacy.WorkflowID,
		GeneratedAt:         time.Now().UTC().Format(time.RFC3339),
		TotalExecutions:     len(legacy.ComponentVCs),
		CompletedExecutions: len(legacy.ComponentVCs),
		WorkflowStatus:      legacy.Status,
		ExecutionVCs:        legacy.ComponentVCs,
		WorkflowVC:          legacy.WorkflowVC,
	}
}

func outputResult(result VCVerificationResult, options VerifyOptions) error {
	switch options.OutputFormat {
	case "pretty":
		return outputPretty(result, options.Verbose)
	case "json":
		fallthrough
	default:
		return outputJSON(result)
	}
}

func outputJSON(result VCVerificationResult) error {
	jsonData, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal result: %v", err)
	}
	fmt.Println(string(jsonData))

	// Exit with appropriate code
	if !result.Valid {
		os.Exit(1)
	}
	return nil
}

func outputPretty(result VCVerificationResult, verbose bool) error {
	// Status indicator
	status := "❌ INVALID"
	if result.Valid {
		status = "✅ VALID"
	}

	fmt.Printf("AgentField VC Verification: %s\n", status)
	fmt.Printf("Type: %s\n", result.Type)

	if result.WorkflowID != "" {
		fmt.Printf("Workflow ID: %s\n", result.WorkflowID)
	}

	fmt.Printf("Format Valid: %t\n", result.FormatValid)
	fmt.Printf("Signature Valid: %t\n", result.SignatureValid)
	fmt.Printf("Message: %s\n", result.Message)

	if result.Error != "" {
		fmt.Printf("Error: %s\n", result.Error)
	}

	// Summary
	fmt.Printf("\nSummary:\n")
	fmt.Printf("  Components: %d/%d valid\n", result.Summary.ValidComponents, result.Summary.TotalComponents)
	fmt.Printf("  DIDs: %d/%d resolved\n", result.Summary.ResolvedDIDs, result.Summary.TotalDIDs)
	fmt.Printf("  Signatures: %d/%d valid\n", result.Summary.ValidSignatures, result.Summary.TotalSignatures)

	if verbose {
		fmt.Printf("\nVerification Steps:\n")
		for _, step := range result.VerificationSteps {
			status := "✅"
			if !step.Success {
				status = "❌"
			}
			fmt.Printf("  %s Step %d: %s\n", status, step.Step, step.Description)
			if step.Details != "" {
				fmt.Printf("    Details: %s\n", step.Details)
			}
			if step.Error != "" {
				fmt.Printf("    Error: %s\n", step.Error)
			}
		}

		if len(result.DIDResolutions) > 0 {
			fmt.Printf("\nDID Resolutions:\n")
			for _, resolution := range result.DIDResolutions {
				status := "✅"
				if !resolution.Success {
					status = "❌"
				}
				fmt.Printf("  %s %s (%s) - %s\n", status, resolution.DID, resolution.Method, resolution.ResolvedFrom)
				if resolution.Error != "" {
					fmt.Printf("    Error: %s\n", resolution.Error)
				}
			}
		}

		if len(result.ComponentResults) > 0 {
			fmt.Printf("\nComponent Verification:\n")
			for _, comp := range result.ComponentResults {
				status := "✅"
				if !comp.Valid {
					status = "❌"
				}
				fmt.Printf("  %s %s (%s) - %s\n", status, comp.VCID, comp.ExecutionID, comp.Status)
				if comp.Error != "" {
					fmt.Printf("    Error: %s\n", comp.Error)
				}
			}
		}
	}

	fmt.Printf("\nVerified At: %s\n", result.VerifiedAt)

	// Exit with appropriate code
	if !result.Valid {
		os.Exit(1)
	}
	return nil
}
