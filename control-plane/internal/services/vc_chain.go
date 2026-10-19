package services

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// GetWorkflowVCChain retrieves the complete VC chain for a workflow.
func (s *VCService) GetWorkflowVCChain(workflowID string) (*types.WorkflowVCChainResponse, error) {
	logger.Logger.Debug().Msgf("🔍 GetWorkflowVCChain called for workflow: %s", workflowID)
	logger.Logger.Debug().Msgf("🔍 DID system enabled: %v", s.config.Enabled)

	if !s.config.Enabled {
		logger.Logger.Debug().Msg("🔍 DID system is disabled")
		return nil, fmt.Errorf("DID system is disabled")
	}

	// Get all execution VCs for the workflow
	logger.Logger.Debug().Msgf("🔍 Querying execution VCs for workflow: %s", workflowID)
	executionVCs, err := s.vcStorage.GetExecutionVCsByWorkflow(workflowID)
	if err != nil {
		logger.Logger.Debug().Err(err).Msg("🔍 Failed to get execution VCs")
		return nil, fmt.Errorf("failed to get execution VCs: %w", err)
	}
	logger.Logger.Debug().Msgf("🔍 Found %d execution VCs for workflow %s", len(executionVCs), workflowID)

	// Generate WorkflowVC on-demand with current state
	logger.Logger.Debug().Msgf("🔍 Generating WorkflowVC on-demand for workflow: %s", workflowID)
	workflowVC, err := s.generateWorkflowVCDocument(workflowID, executionVCs)
	if err != nil {
		logger.Logger.Debug().Err(err).Msg("🔍 Failed to generate WorkflowVC")
		return nil, fmt.Errorf("failed to generate workflow VC: %w", err)
	}
	logger.Logger.Debug().Msgf("🔍 Generated WorkflowVC with ID: %s, status: %s", workflowVC.WorkflowVCID, workflowVC.Status)

	// Collect DID resolution bundle for offline verification
	logger.Logger.Debug().Msgf("🔍 Collecting DID resolution bundle for workflow: %s", workflowID)
	didResolutionBundle, unresolvedDIDs := s.collectDIDResolutionBundle(executionVCs, workflowVC)
	if len(unresolvedDIDs) > 0 {
		logger.Logger.Warn().
			Str("workflow_id", workflowID).
			Strs("unresolved_dids", unresolvedDIDs).
			Msg("VC chain references DIDs whose public keys could not be resolved; the chain is not fully verifiable")
	}
	logger.Logger.Debug().Msgf("🔍 Collected %d DID resolution entries", len(didResolutionBundle))

	logger.Logger.Debug().Msgf("🔍 Returning VC chain with %d execution VCs and workflow status: %s", len(executionVCs), workflowVC.Status)
	return &types.WorkflowVCChainResponse{
		WorkflowID:          workflowID,
		ComponentVCs:        executionVCs,
		WorkflowVC:          *workflowVC,
		TotalSteps:          len(executionVCs),
		Status:              workflowVC.Status,
		DIDResolutionBundle: didResolutionBundle,
		UnresolvedDIDs:      unresolvedDIDs,
		Verifiable:          len(unresolvedDIDs) == 0,
	}, nil
}

// collectDIDResolutionBundle collects all unique DIDs from the VC chain and
// resolves their public keys. DIDs that cannot be resolved are returned in
// sorted order instead of being dropped silently.
func (s *VCService) collectDIDResolutionBundle(executionVCs []types.ExecutionVC, workflowVC *types.WorkflowVC) (map[string]types.DIDResolutionEntry, []string) {
	bundle := make(map[string]types.DIDResolutionEntry)
	resolvedAt := time.Now().UTC().Format(time.RFC3339)

	// Collect unique DIDs from execution VCs
	uniqueDIDs := make(map[string]bool)
	unresolvedDIDs := make(map[string]bool)

	for _, vc := range executionVCs {
		uniqueDIDs[vc.IssuerDID] = true
		var document struct {
			Issuer string `json:"issuer"`
		}
		if err := json.Unmarshal(vc.VCDocument, &document); err != nil || document.Issuer == "" {
			// A VC-qualified marker identifies a missing/unreadable document issuer
			// when metadata supplies a signer. If both issuers are empty, the
			// existing empty required-DID entry already accounts for both.
			if vc.IssuerDID != "" {
				unresolvedDIDs[fmt.Sprintf("execution-vc:%s:document-issuer:empty-or-invalid", vc.VCID)] = true
			}
		} else {
			uniqueDIDs[document.Issuer] = true
			// Even a resolvable document issuer cannot stand in for a different
			// metadata signer. Keep the inconsistency visible in unresolved_dids.
			if document.Issuer != vc.IssuerDID {
				unresolvedDIDs[document.Issuer] = true
			}
		}
		if vc.CallerDID != "" {
			uniqueDIDs[vc.CallerDID] = true
		}
		if vc.TargetDID != "" {
			uniqueDIDs[vc.TargetDID] = true
		}
	}
	uniqueDIDs[workflowVC.IssuerDID] = true

	// Resolve each unique DID and collect public keys
	for did := range uniqueDIDs {
		parts := strings.SplitN(did, ":", 3)
		if strings.TrimSpace(did) != did || len(parts) != 3 || parts[0] != "did" || parts[1] == "" || strings.TrimSpace(parts[2]) == "" || strings.ContainsAny(did, " \t\r\n") {
			unresolvedDIDs[did] = true
			continue
		}

		// Determine DID method from the DID string
		method := "key" // Default to "key" method
		if len(did) > 4 && did[:4] == "did:" {
			parts := strings.Split(did, ":")
			if len(parts) >= 2 {
				method = parts[1]
			}
		}

		identity, err := s.didService.ResolveDID(did)
		if err != nil {
			// A control-plane did:key encodes its own public key, so it can
			// still be verified after the issuing identity is gone.
			if publicKeyJWK, keyErr := publicKeyJWKFromDIDKey(did); keyErr == nil {
				bundle[did] = types.DIDResolutionEntry{
					Method:       method,
					PublicKeyJWK: json.RawMessage(publicKeyJWK),
					ResolvedFrom: "did_key",
					ResolvedAt:   resolvedAt,
				}
				continue
			}
			unresolvedDIDs[did] = true
			continue
		}

		// Parse the public key JWK string into a JSON object
		var publicKeyJWK map[string]interface{}
		if err := json.Unmarshal([]byte(identity.PublicKeyJWK), &publicKeyJWK); err != nil {
			unresolvedDIDs[did] = true
			continue
		}

		// Create resolution entry with properly parsed public key JWK
		bundle[did] = types.DIDResolutionEntry{
			Method:       method,
			PublicKeyJWK: json.RawMessage(identity.PublicKeyJWK), // Keep as raw JSON
			ResolvedFrom: "bundled",
			ResolvedAt:   resolvedAt,
		}

	}
	unresolved := make([]string, 0, len(unresolvedDIDs))
	for did := range unresolvedDIDs {
		unresolved = append(unresolved, did)
	}
	sort.Strings(unresolved)
	return bundle, unresolved
}
