package cli

import (
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/spf13/cobra"
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
	ComponentVCs         []types.ExecutionVC          `json:"component_vcs,omitempty"`
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
	Valid             bool                             `json:"valid"`
	Type              string                           `json:"type"`
	WorkflowID        string                           `json:"workflow_id,omitempty"`
	SignatureValid    bool                             `json:"signature_valid"`
	FormatValid       bool                             `json:"format_valid"`
	Message           string                           `json:"message"`
	Error             string                           `json:"error,omitempty"`
	VerifiedAt        string                           `json:"verified_at"`
	ComponentResults  []ComponentVerification          `json:"component_results,omitempty"`
	DIDResolutions    []DIDResolutionResult            `json:"did_resolutions,omitempty"`
	VerificationSteps []VerificationStep               `json:"verification_steps,omitempty"`
	Summary           VerificationSummary              `json:"summary"`
	Comprehensive     *ComprehensiveVerificationResult `json:"comprehensive,omitempty"`
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
	step1 := VerificationStep{Step: 1, Description: "Reading VC file"}
	vcData, err := os.ReadFile(vcFilePath)
	if err != nil {
		result := VCVerificationResult{
			VerifiedAt:        time.Now().UTC().Format(time.RFC3339),
			VerificationSteps: []VerificationStep{},
			DIDResolutions:    []DIDResolutionResult{},
			ComponentResults:  []ComponentVerification{},
		}
		step1.Success = false
		step1.Error = fmt.Sprintf("Failed to read VC file: %v", err)
		result.VerificationSteps = append(result.VerificationSteps, step1)
		result.Valid = false
		result.Error = step1.Error
		return outputResult(result, options)
	}
	step1.Success = true
	step1.Details = fmt.Sprintf("Read %d bytes from %s", len(vcData), vcFilePath)

	result := VerifyProvenanceJSON(vcData, options)
	result.VerificationSteps = append([]VerificationStep{step1}, result.VerificationSteps...)
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
	if options.ResolveWeb {
		return DIDResolutionInfo{}, fmt.Errorf("web DID resolution is disabled in this CLI verifier; use bundled DID resolution instead")
	}
	if options.Resolver != "" {
		return DIDResolutionInfo{}, fmt.Errorf("custom DID resolvers are disabled in this CLI verifier; use bundled DID resolution instead")
	}

	// 1. did:web always resolves from web
	if strings.HasPrefix(did, "did:web:") {
		return resolveWebDID(did)
	}

	// 2. Fallback: use bundled resolution
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

	domain, err := url.PathUnescape(parts[2])
	if err != nil {
		return DIDResolutionInfo{}, fmt.Errorf("invalid did:web domain: %w", err)
	}
	path := "/.well-known/did.json"

	if len(parts) > 3 {
		path = "/" + strings.Join(parts[3:], "/") + "/did.json"
	}

	didURL := fmt.Sprintf("https://%s%s", domain, path)
	return DIDResolutionInfo{}, fmt.Errorf("did:web online resolution is disabled in this CLI verifier; use bundled DID resolution for %s (%s)", did, didURL)
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

//nolint:unused // Reserved for future signature verification
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

//nolint:unused // Reserved for future signature verification
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
	chain := EnhancedVCChain{
		WorkflowID:          legacy.WorkflowID,
		GeneratedAt:         time.Now().UTC().Format(time.RFC3339),
		TotalExecutions:     len(legacy.ComponentVCs),
		CompletedExecutions: len(legacy.ComponentVCs),
		WorkflowStatus:      legacy.Status,
		ExecutionVCs:        legacy.ComponentVCs,
		ComponentVCs:        legacy.ComponentVCs,
		WorkflowVC:          legacy.WorkflowVC,
	}
	if len(legacy.DIDResolutionBundle) > 0 {
		chain.DIDResolutionBundle = mergeDIDBundle(chain.DIDResolutionBundle, legacy.DIDResolutionBundle)
	}
	return chain
}

func normalizeEnhancedChain(chain *EnhancedVCChain) {
	if chain == nil {
		return
	}

	// Keep both execution/component aliases in sync for backwards compatibility
	switch {
	case len(chain.ExecutionVCs) == 0 && len(chain.ComponentVCs) > 0:
		chain.ExecutionVCs = chain.ComponentVCs
	case len(chain.ComponentVCs) == 0 && len(chain.ExecutionVCs) > 0:
		chain.ComponentVCs = chain.ExecutionVCs
	}

	// Populate counts if omitted
	if chain.TotalExecutions == 0 {
		chain.TotalExecutions = len(chain.ExecutionVCs)
	}
	if chain.CompletedExecutions == 0 {
		chain.CompletedExecutions = len(chain.ExecutionVCs)
	}

	// Default workflow status from workflow VC if empty
	if chain.WorkflowStatus == "" && chain.WorkflowVC.Status != "" {
		chain.WorkflowStatus = chain.WorkflowVC.Status
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
