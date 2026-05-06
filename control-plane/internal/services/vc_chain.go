package services

import (
	"encoding/json"
	"fmt"
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
	didResolutionBundle, err := s.collectDIDResolutionBundle(executionVCs, workflowVC)
	if err != nil {
		logger.Logger.Debug().Err(err).Msg("🔍 Failed to collect DID resolution bundle")
		// Don't fail the entire request if DID resolution fails - just log and continue without bundle
		didResolutionBundle = make(map[string]types.DIDResolutionEntry)
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
	}, nil
}

// collectDIDResolutionBundle collects all unique DIDs from the VC chain and resolves their public keys.
func (s *VCService) collectDIDResolutionBundle(executionVCs []types.ExecutionVC, workflowVC *types.WorkflowVC) (map[string]types.DIDResolutionEntry, error) {
	bundle := make(map[string]types.DIDResolutionEntry)
	resolvedAt := time.Now().UTC().Format(time.RFC3339)

	// Collect unique DIDs from execution VCs
	uniqueDIDs := make(map[string]bool)

	for _, vc := range executionVCs {
		if vc.IssuerDID != "" && vc.IssuerDID != "did:key:" {
			uniqueDIDs[vc.IssuerDID] = true
		}
		if vc.CallerDID != "" && vc.CallerDID != "did:key:" {
			uniqueDIDs[vc.CallerDID] = true
		}
		if vc.TargetDID != "" && vc.TargetDID != "did:key:" {
			uniqueDIDs[vc.TargetDID] = true
		}
	}

	// Add workflow VC issuer DID
	if workflowVC.IssuerDID != "" && workflowVC.IssuerDID != "did:key:" {
		uniqueDIDs[workflowVC.IssuerDID] = true
	}

	// Resolve each unique DID and collect public keys
	for did := range uniqueDIDs {
		if did == "" || did == "did:key:" || len(strings.TrimSpace(did)) == 0 {
			continue // Skip empty or incomplete DIDs
		}

		identity, err := s.didService.ResolveDID(did)
		if err != nil {
			continue // Skip DIDs that can't be resolved
		}

		// Determine DID method from the DID string
		method := "key" // Default to "key" method
		if len(did) > 4 && did[:4] == "did:" {
			parts := strings.Split(did, ":")
			if len(parts) >= 2 {
				method = parts[1]
			}
		}

		// Parse the public key JWK string into a JSON object
		var publicKeyJWK map[string]interface{}
		if err := json.Unmarshal([]byte(identity.PublicKeyJWK), &publicKeyJWK); err != nil {
			continue // Skip DIDs with invalid public key JWK
		}

		// Create resolution entry with properly parsed public key JWK
		bundle[did] = types.DIDResolutionEntry{
			Method:       method,
			PublicKeyJWK: json.RawMessage(identity.PublicKeyJWK), // Keep as raw JSON
			ResolvedFrom: "bundled",
			ResolvedAt:   resolvedAt,
		}

	}
	return bundle, nil
}
