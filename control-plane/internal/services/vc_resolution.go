package services

import (
	"context"
	"fmt"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// QueryExecutionVCs queries execution VCs with filters.
func (s *VCService) QueryExecutionVCs(filters *types.VCFilters) ([]types.ExecutionVC, error) {
	if !s.config.Enabled {
		return nil, fmt.Errorf("DID system is disabled")
	}
	return s.vcStorage.QueryExecutionVCs(filters)
}

// GetExecutionVCByExecutionID retrieves a single execution VC by execution identifier.
func (s *VCService) GetExecutionVCByExecutionID(executionID string) (*types.ExecutionVC, error) {
	if !s.config.Enabled {
		return nil, fmt.Errorf("DID system is disabled")
	}
	return s.vcStorage.GetExecutionVCByExecutionID(executionID)
}

// ListWorkflowVCs lists all workflow VCs.
func (s *VCService) ListWorkflowVCs() ([]*types.WorkflowVC, error) {
	if !s.config.Enabled {
		return nil, fmt.Errorf("DID system is disabled")
	}
	return s.vcStorage.ListWorkflowVCs()
}

// ListAgentTagVCs returns all non-revoked agent tag VCs.
func (s *VCService) ListAgentTagVCs() ([]*types.AgentTagVCRecord, error) {
	if !s.config.Enabled {
		return nil, fmt.Errorf("DID system is disabled")
	}
	return s.vcStorage.ListAgentTagVCs(context.Background())
}
