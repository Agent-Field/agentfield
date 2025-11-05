package services

import (
    "context"
    "encoding/json"
    "fmt"

    "github.com/your-org/haxen/control-plane/internal/logger"
    "github.com/your-org/haxen/control-plane/internal/storage"
    "github.com/your-org/haxen/control-plane/pkg/types"
)

// VCStorage manages the storage and retrieval of verifiable credentials.
type VCStorage struct {
    storageProvider storage.StorageProvider
}

// NewVCStorageWithStorage creates a new VC storage instance backed by the configured storage provider.
func NewVCStorageWithStorage(storageProvider storage.StorageProvider) *VCStorage {
    return &VCStorage{storageProvider: storageProvider}
}

// Initialize performs a lightweight sanity check that the storage provider is available.
func (s *VCStorage) Initialize() error {
    if s.storageProvider == nil {
        logger.Logger.Warn().Msg("No storage provider available - VC persistence disabled")
    }
    return nil
}

// StoreExecutionVC persists an execution VC using the backing storage provider.
func (s *VCStorage) StoreExecutionVC(ctx context.Context, vc *types.ExecutionVC) error {
    if s.storageProvider == nil {
        return fmt.Errorf("no storage provider configured for VC storage")
    }

    documentSizeBytes := vc.DocumentSize
    if documentSizeBytes == 0 && len(vc.VCDocument) > 0 {
        documentSizeBytes = int64(len(vc.VCDocument))
    }

    return s.storageProvider.StoreExecutionVC(
        ctx,
        vc.VCID,
        vc.ExecutionID,
        vc.WorkflowID,
        vc.SessionID,
        vc.IssuerDID,
        vc.TargetDID,
        vc.CallerDID,
        vc.InputHash,
        vc.OutputHash,
        vc.Status,
        []byte(vc.VCDocument),
        vc.Signature,
        vc.StorageURI,
        documentSizeBytes,
    )
}

// GetExecutionVC fetches a single execution VC by its VC identifier.
func (s *VCStorage) GetExecutionVC(vcID string) (*types.ExecutionVC, error) {
    if s.storageProvider == nil {
        return nil, fmt.Errorf("no storage provider configured for VC storage")
    }

    ctx := context.Background()
    vcInfo, err := s.storageProvider.GetExecutionVC(ctx, vcID)
    if err != nil {
        return nil, err
    }

    return s.convertVCInfoToExecutionVC(vcInfo)
}

// GetExecutionVCsByWorkflow returns all execution VCs associated with a workflow.
func (s *VCStorage) GetExecutionVCsByWorkflow(workflowID string) ([]types.ExecutionVC, error) {
    filters := types.VCFilters{WorkflowID: &workflowID}
    return s.loadExecutionVCsFromDatabaseWithFilters(filters)
}

// GetExecutionVCsBySession returns all execution VCs associated with a session.
func (s *VCStorage) GetExecutionVCsBySession(sessionID string) ([]types.ExecutionVC, error) {
    filters := types.VCFilters{SessionID: &sessionID}
    return s.loadExecutionVCsFromDatabaseWithFilters(filters)
}

// GetExecutionVCByExecutionID fetches the most recent VC for a specific execution ID.
func (s *VCStorage) GetExecutionVCByExecutionID(executionID string) (*types.ExecutionVC, error) {
    filters := types.VCFilters{ExecutionID: &executionID, Limit: 1}
    vcs, err := s.loadExecutionVCsFromDatabaseWithFilters(filters)
    if err != nil {
        return nil, err
    }
    if len(vcs) == 0 {
        return nil, fmt.Errorf("execution VC not found for execution_id: %s", executionID)
    }
    return &vcs[0], nil
}

// QueryExecutionVCs runs a filtered VC query against the backing store.
func (s *VCStorage) QueryExecutionVCs(filters *types.VCFilters) ([]types.ExecutionVC, error) {
    var applied types.VCFilters
    if filters != nil {
        applied = *filters
    }
    return s.loadExecutionVCsFromDatabaseWithFilters(applied)
}

// StoreWorkflowVC persists workflow-level VC metadata.
func (s *VCStorage) StoreWorkflowVC(ctx context.Context, vc *types.WorkflowVC) error {
    if s.storageProvider == nil {
        return fmt.Errorf("no storage provider configured for VC storage")
    }

    documentSizeBytes := vc.DocumentSize
    if documentSizeBytes == 0 && len(vc.VCDocument) > 0 {
        documentSizeBytes = int64(len(vc.VCDocument))
    }

    return s.storageProvider.StoreWorkflowVC(
        ctx,
        vc.WorkflowVCID,
        vc.WorkflowID,
        vc.SessionID,
        vc.ComponentVCs,
        vc.Status,
        &vc.StartTime,
        vc.EndTime,
        vc.TotalSteps,
        vc.CompletedSteps,
        vc.StorageURI,
        documentSizeBytes,
    )
}

// GetWorkflowVC fetches the latest workflow VC for a workflow identifier.
func (s *VCStorage) GetWorkflowVC(workflowID string) (*types.WorkflowVC, error) {
    if s.storageProvider == nil {
        return nil, fmt.Errorf("no storage provider configured for VC storage")
    }

    ctx := context.Background()
    infos, err := s.storageProvider.ListWorkflowVCs(ctx, workflowID)
    if err != nil {
        return nil, err
    }
    if len(infos) == 0 {
        return nil, fmt.Errorf("workflow VC not found: %s", workflowID)
    }

    return s.convertWorkflowVCInfo(infos[0])
}

// ListWorkflowVCs returns all workflow VCs.
func (s *VCStorage) ListWorkflowVCs() ([]*types.WorkflowVC, error) {
    if s.storageProvider == nil {
        return []*types.WorkflowVC{}, fmt.Errorf("no storage provider configured for VC storage")
    }

    ctx := context.Background()
    infos, err := s.storageProvider.ListWorkflowVCs(ctx, "")
    if err != nil {
        return nil, err
    }

    results := make([]*types.WorkflowVC, 0, len(infos))
    for _, info := range infos {
        vc, err := s.convertWorkflowVCInfo(info)
        if err != nil {
            logger.Logger.Warn().Err(err).Str("workflow_vc_id", info.WorkflowVCID).Msg("failed to convert workflow VC info")
            continue
        }
        results = append(results, vc)
    }

    return results, nil
}

// DeleteExecutionVC is currently a no-op placeholder.
func (s *VCStorage) DeleteExecutionVC(vcID string) error {
    logger.Logger.Debug().Str("vc_id", vcID).Msg("DeleteExecutionVC is not implemented - skipping")
    return nil
}

// DeleteWorkflowVC is currently a no-op placeholder.
func (s *VCStorage) DeleteWorkflowVC(workflowID string) error {
    logger.Logger.Debug().Str("workflow_id", workflowID).Msg("DeleteWorkflowVC is not implemented - skipping")
    return nil
}

// GetVCStats returns simple metrics about stored VCs.
func (s *VCStorage) GetVCStats() map[string]interface{} {
    stats := map[string]interface{}{
        "execution_vcs": 0,
        "workflow_vcs":  0,
    }

    if s.storageProvider == nil {
        return stats
    }

    ctx := context.Background()

    executionInfos, err := s.storageProvider.ListExecutionVCs(ctx, types.VCFilters{})
    if err == nil {
        stats["execution_vcs"] = len(executionInfos)
    } else {
        logger.Logger.Warn().Err(err).Msg("failed to collect execution VC stats")
    }

    workflowInfos, err := s.storageProvider.ListWorkflowVCs(ctx, "")
    if err == nil {
        stats["workflow_vcs"] = len(workflowInfos)
    } else {
        logger.Logger.Warn().Err(err).Msg("failed to collect workflow VC stats")
    }

    return stats
}

// convertVCInfoToExecutionVC hydrates a full ExecutionVC from summary data.
func (s *VCStorage) convertVCInfoToExecutionVC(vcInfo *types.ExecutionVCInfo) (*types.ExecutionVC, error) {
    if vcInfo == nil {
        return nil, fmt.Errorf("execution VC info is nil")
    }

    vcDocument, signature, err := s.getFullVCFromDatabase(vcInfo.VCID)
    if err != nil {
        return nil, fmt.Errorf("failed to load VC document for %s: %w", vcInfo.VCID, err)
    }

    return &types.ExecutionVC{
        VCID:        vcInfo.VCID,
        ExecutionID: vcInfo.ExecutionID,
        WorkflowID:  vcInfo.WorkflowID,
        SessionID:   vcInfo.SessionID,
        IssuerDID:   vcInfo.IssuerDID,
        TargetDID:   vcInfo.TargetDID,
        CallerDID:   vcInfo.CallerDID,
        VCDocument:  vcDocument,
        Signature:   signature,
        StorageURI:  vcInfo.StorageURI,
        DocumentSize: vcInfo.DocumentSize,
        InputHash:   vcInfo.InputHash,
        OutputHash:  vcInfo.OutputHash,
        Status:      vcInfo.Status,
        CreatedAt:   vcInfo.CreatedAt,
    }, nil
}

// convertWorkflowVCInfo hydrates a WorkflowVC struct from stored metadata.
func (s *VCStorage) convertWorkflowVCInfo(info *types.WorkflowVCInfo) (*types.WorkflowVC, error) {
    if info == nil {
        return nil, fmt.Errorf("workflow VC info is nil")
    }

    return &types.WorkflowVC{
        WorkflowID:     info.WorkflowID,
        SessionID:      info.SessionID,
        ComponentVCs:   info.ComponentVCIDs,
        WorkflowVCID:   info.WorkflowVCID,
        Status:         info.Status,
        StartTime:      info.StartTime,
        EndTime:        info.EndTime,
        TotalSteps:     info.TotalSteps,
        CompletedSteps: info.CompletedSteps,
        StorageURI:     info.StorageURI,
        DocumentSize:   info.DocumentSize,
    }, nil
}

// loadExecutionVCsFromDatabaseWithFilters retrieves execution VCs that match the provided filters.
func (s *VCStorage) loadExecutionVCsFromDatabaseWithFilters(filters types.VCFilters) ([]types.ExecutionVC, error) {
    if s.storageProvider == nil {
        return []types.ExecutionVC{}, fmt.Errorf("no storage provider configured for VC storage")
    }

    ctx := context.Background()
    vcInfos, err := s.storageProvider.ListExecutionVCs(ctx, filters)
    if err != nil {
        return nil, fmt.Errorf("failed to list execution VCs from database: %w", err)
    }

    result := make([]types.ExecutionVC, 0, len(vcInfos))
    for _, info := range vcInfos {
        vc, err := s.convertVCInfoToExecutionVC(info)
        if err != nil {
            logger.Logger.Warn().Err(err).Str("vc_id", info.VCID).Msg("failed to convert execution VC info")
            continue
        }
        result = append(result, *vc)
    }

    return result, nil
}

// getFullVCFromDatabase retrieves the full VC document and signature from the storage provider.
func (s *VCStorage) getFullVCFromDatabase(vcID string) (json.RawMessage, string, error) {
	switch provider := s.storageProvider.(type) {
	case *storage.LocalStorage:
		return s.getFullVCFromLocalStorage(provider, vcID)
	default:
		return nil, "", fmt.Errorf("unsupported storage provider for full VC retrieval: %T", s.storageProvider)
	}
}

// getFullVCFromLocalStorage retrieves the VC payload from local SQLite storage.
func (s *VCStorage) getFullVCFromLocalStorage(localStorage *storage.LocalStorage, vcID string) (json.RawMessage, string, error) {
    return localStorage.GetFullExecutionVC(vcID)
}
