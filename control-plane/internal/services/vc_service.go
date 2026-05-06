package services

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/google/uuid"
)

// VCService handles verifiable credential generation, verification, and management.
type VCService struct {
	config     *config.DIDConfig
	didService *DIDService
	vcStorage  *VCStorage
}

// NewVCService creates a new VC service instance with database storage.
func NewVCService(cfg *config.DIDConfig, didService *DIDService, storageProvider storage.StorageProvider) *VCService {
	return &VCService{
		config:     cfg,
		didService: didService,
		vcStorage:  NewVCStorageWithStorage(storageProvider),
	}
}

// Initialize initializes the VC service.
func (s *VCService) Initialize() error {
	if !s.config.Enabled {
		return nil
	}

	return s.vcStorage.Initialize()
}

// GetDIDService returns the DID service instance for DID resolution operations.
func (s *VCService) GetDIDService() *DIDService {
	return s.didService
}

// IsExecutionVCEnabled reports whether execution VC generation should run
// based on DID being enabled and the execution VC requirement flag.
func (s *VCService) IsExecutionVCEnabled() bool {
	if s == nil || s.config == nil {
		return false
	}
	if !s.config.Enabled {
		return false
	}
	return s.config.VCRequirements.RequireVCForExecution
}

// ShouldPersistExecutionVC reports whether execution VCs should be persisted after generation.
func (s *VCService) ShouldPersistExecutionVC() bool {
	if s == nil || s.config == nil {
		return false
	}
	if !s.config.Enabled {
		return false
	}
	return s.config.VCRequirements.PersistExecutionVC
}

// GetWorkflowVCStatusSummaries returns lightweight VC status summaries for the provided workflows.
func (s *VCService) GetWorkflowVCStatusSummaries(workflowIDs []string) (map[string]*types.WorkflowVCStatusSummary, error) {
	summaries := make(map[string]*types.WorkflowVCStatusSummary, len(workflowIDs))
	uniqueIDs := make([]string, 0, len(workflowIDs))
	seen := make(map[string]struct{}, len(workflowIDs))

	for _, id := range workflowIDs {
		if id == "" {
			continue
		}
		if _, exists := summaries[id]; !exists {
			summaries[id] = types.DefaultWorkflowVCStatusSummary(id)
		}
		if _, exists := seen[id]; !exists {
			seen[id] = struct{}{}
			uniqueIDs = append(uniqueIDs, id)
		}
	}

	if len(uniqueIDs) == 0 {
		return summaries, nil
	}

	if s == nil || s.config == nil || !s.config.Enabled || s.vcStorage == nil {
		return summaries, nil
	}

	ctx := context.Background()
	aggregations, err := s.vcStorage.ListWorkflowVCStatusSummaries(ctx, uniqueIDs)
	if err != nil {
		return nil, err
	}

	for _, agg := range aggregations {
		if agg == nil {
			continue
		}

		summary := types.DefaultWorkflowVCStatusSummary(agg.WorkflowID)
		summary.VCCount = agg.VCCount
		summary.VerifiedCount = agg.VerifiedCount
		summary.FailedCount = agg.FailedCount
		summary.HasVCs = agg.VCCount > 0

		if agg.LastCreatedAt != nil {
			summary.LastVCCreated = agg.LastCreatedAt.UTC().Format(time.RFC3339)
		}

		switch {
		case agg.VCCount == 0:
			summary.VerificationStatus = "none"
		case agg.FailedCount > 0:
			summary.VerificationStatus = "failed"
		case agg.VerifiedCount == agg.VCCount:
			summary.VerificationStatus = "verified"
		default:
			summary.VerificationStatus = "pending"
		}

		summaries[agg.WorkflowID] = summary
	}

	return summaries, nil
}

// hashData creates a SHA-256 hash of data.
func (s *VCService) hashData(data []byte) string {
	if !s.config.VCRequirements.HashSensitiveData {
		return ""
	}

	hash := sha256.Sum256(data)
	return base64.RawURLEncoding.EncodeToString(hash[:])
}

// generateVCID generates a unique VC ID using a cryptographically random UUID.
func (s *VCService) generateVCID() string {
	return fmt.Sprintf("vc-%s", uuid.New().String())
}

// marshalDataOrNull marshals data to JSON or returns null JSON if nil/error
func marshalDataOrNull(data interface{}) []byte {
	if data == nil {
		return []byte("null")
	}
	if jsonData, err := json.Marshal(data); err == nil {
		return jsonData
	}
	return []byte("null")
}
