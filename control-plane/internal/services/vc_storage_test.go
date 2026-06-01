package services

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

func TestVCStorage_Initialize(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	vcStorage := NewVCStorageWithStorage(provider)
	err := vcStorage.Initialize()
	require.NoError(t, err)

	// Test with nil storage provider
	vcStorageNil := NewVCStorageWithStorage(nil)
	err = vcStorageNil.Initialize()
	require.NoError(t, err) // Should not error, just warn

	_ = ctx
}

func TestVCStorage_StoreExecutionVC_Success(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	vcStorage := NewVCStorageWithStorage(provider)
	require.NoError(t, vcStorage.Initialize())

	vcDoc := types.VCDocument{
		Context: []string{"https://www.w3.org/2018/credentials/v1"},
		Type:    []string{"VerifiableCredential"},
		ID:      "urn:agentfield:vc:test-1",
		Issuer:  "did:key:test",
	}

	vcDocBytes, err := json.Marshal(vcDoc)
	require.NoError(t, err)

	executionVC := &types.ExecutionVC{
		VCID:         "vc-test-1",
		ExecutionID:  "exec-1",
		WorkflowID:   "workflow-1",
		SessionID:    "session-1",
		IssuerDID:    "did:key:test",
		TargetDID:    "did:key:target",
		CallerDID:    "did:key:caller",
		VCDocument:   json.RawMessage(vcDocBytes),
		Signature:    "test-signature",
		StorageURI:   "",
		DocumentSize: int64(len(vcDocBytes)),
		InputHash:    "input-hash",
		OutputHash:   "output-hash",
		Status:       "succeeded",
		CreatedAt:    time.Now(),
	}

	err = vcStorage.StoreExecutionVC(ctx, executionVC)
	require.NoError(t, err)

	// Verify VC was stored
	storedVC, err := provider.GetExecutionVC(ctx, "vc-test-1")
	require.NoError(t, err)
	require.NotNil(t, storedVC)
	require.Equal(t, executionVC.VCID, storedVC.VCID)
	require.Equal(t, executionVC.ExecutionID, storedVC.ExecutionID)
}

func TestVCStorage_StoreExecutionVC_NilProvider(t *testing.T) {
	vcStorage := NewVCStorageWithStorage(nil)

	executionVC := &types.ExecutionVC{
		VCID:        "vc-test",
		ExecutionID: "exec-1",
		VCDocument:  json.RawMessage(`{}`),
	}

	err := vcStorage.StoreExecutionVC(context.Background(), executionVC)
	require.Error(t, err)
	require.Contains(t, err.Error(), "no storage provider configured")
}

func TestVCStorage_StoreExecutionVC_AutoSizeCalculation(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	vcStorage := NewVCStorageWithStorage(provider)
	require.NoError(t, vcStorage.Initialize())

	vcDocBytes := []byte(`{"test": "data"}`)

	executionVC := &types.ExecutionVC{
		VCID:         "vc-auto-size",
		ExecutionID:  "exec-1",
		WorkflowID:   "workflow-1",
		SessionID:    "session-1",
		IssuerDID:    "did:key:test",
		TargetDID:    "",
		CallerDID:    "did:key:caller",
		VCDocument:   json.RawMessage(vcDocBytes),
		Signature:    "sig",
		DocumentSize: 0, // Should be auto-calculated
		Status:       "succeeded",
		CreatedAt:    time.Now(),
	}

	err := vcStorage.StoreExecutionVC(ctx, executionVC)
	require.NoError(t, err)

	// Verify size was calculated
	storedVC, err := provider.GetExecutionVC(ctx, "vc-auto-size")
	require.NoError(t, err)
	require.Equal(t, int64(len(vcDocBytes)), storedVC.DocumentSize)
}

func TestVCStorage_GetExecutionVC_Success(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	vcStorage := NewVCStorageWithStorage(provider)
	require.NoError(t, vcStorage.Initialize())

	vcDoc := types.VCDocument{
		Context: []string{"https://www.w3.org/2018/credentials/v1"},
		Type:    []string{"VerifiableCredential"},
		ID:      "urn:agentfield:vc:get-test",
		Issuer:  "did:key:test",
	}

	vcDocBytes, err := json.Marshal(vcDoc)
	require.NoError(t, err)

	executionVC := &types.ExecutionVC{
		VCID:         "vc-get-test",
		ExecutionID:  "exec-get",
		WorkflowID:   "workflow-1",
		SessionID:    "session-1",
		IssuerDID:    "did:key:test",
		TargetDID:    "",
		CallerDID:    "did:key:caller",
		VCDocument:   json.RawMessage(vcDocBytes),
		Signature:    "test-signature",
		DocumentSize: int64(len(vcDocBytes)),
		Status:       "succeeded",
		CreatedAt:    time.Now(),
	}

	err = vcStorage.StoreExecutionVC(ctx, executionVC)
	require.NoError(t, err)

	// Retrieve VC
	retrievedVC, err := vcStorage.GetExecutionVC("vc-get-test")
	require.NoError(t, err)
	require.NotNil(t, retrievedVC)
	require.Equal(t, executionVC.VCID, retrievedVC.VCID)
	require.Equal(t, executionVC.ExecutionID, retrievedVC.ExecutionID)
	require.Equal(t, executionVC.Signature, retrievedVC.Signature)
}

func TestVCStorage_GetExecutionVC_NilProvider(t *testing.T) {
	vcStorage := NewVCStorageWithStorage(nil)

	vc, err := vcStorage.GetExecutionVC("vc-test")
	require.Error(t, err)
	require.Nil(t, vc)
	require.Contains(t, err.Error(), "no storage provider configured")
}

func TestVCStorage_GetExecutionVCByExecutionID_Success(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	vcStorage := NewVCStorageWithStorage(provider)
	require.NoError(t, vcStorage.Initialize())

	vcDocBytes := []byte(`{"test": "data"}`)

	executionVC := &types.ExecutionVC{
		VCID:         "vc-by-exec-id",
		ExecutionID:  "exec-by-id",
		WorkflowID:   "workflow-1",
		SessionID:    "session-1",
		IssuerDID:    "did:key:test",
		TargetDID:    "",
		CallerDID:    "did:key:caller",
		VCDocument:   json.RawMessage(vcDocBytes),
		Signature:    "sig",
		DocumentSize: int64(len(vcDocBytes)),
		Status:       "succeeded",
		CreatedAt:    time.Now(),
	}

	err := vcStorage.StoreExecutionVC(ctx, executionVC)
	require.NoError(t, err)

	// Get by execution ID
	retrievedVC, err := vcStorage.GetExecutionVCByExecutionID("exec-by-id")
	require.NoError(t, err)
	require.NotNil(t, retrievedVC)
	require.Equal(t, "exec-by-id", retrievedVC.ExecutionID)
	require.Equal(t, "vc-by-exec-id", retrievedVC.VCID)
}

func TestVCStorage_GetExecutionVCByExecutionID_NotFound(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	vcStorage := NewVCStorageWithStorage(provider)
	require.NoError(t, vcStorage.Initialize())

	vc, err := vcStorage.GetExecutionVCByExecutionID("nonexistent-exec-id")
	require.Error(t, err)
	require.Nil(t, vc)
	require.Contains(t, err.Error(), "execution VC not found")
	_ = ctx
}

func TestVCStorage_QueryExecutionVCs_ByWorkflowID(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	vcStorage := NewVCStorageWithStorage(provider)
	require.NoError(t, vcStorage.Initialize())

	vcDocBytes := []byte(`{"test": "data"}`)

	// Store multiple VCs for same workflow
	for i := 1; i <= 3; i++ {
		executionVC := &types.ExecutionVC{
			VCID:         "vc-query-" + string(rune('0'+i)),
			ExecutionID:  "exec-query-" + string(rune('0'+i)),
			WorkflowID:   "workflow-query",
			SessionID:    "session-query",
			IssuerDID:    "did:key:test",
			TargetDID:    "",
			CallerDID:    "did:key:caller",
			VCDocument:   json.RawMessage(vcDocBytes),
			Signature:    "sig",
			DocumentSize: int64(len(vcDocBytes)),
			Status:       "succeeded",
			CreatedAt:    time.Now(),
		}

		err := vcStorage.StoreExecutionVC(ctx, executionVC)
		require.NoError(t, err)
	}

	// Query by workflow ID
	filters := &types.VCFilters{
		WorkflowID: stringPtr("workflow-query"),
	}

	vcs, err := vcStorage.QueryExecutionVCs(filters)
	require.NoError(t, err)
	require.Len(t, vcs, 3)
}

func TestVCStorage_QueryExecutionVCs_BySessionID(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	vcStorage := NewVCStorageWithStorage(provider)
	require.NoError(t, vcStorage.Initialize())

	vcDocBytes := []byte(`{"test": "data"}`)

	executionVC := &types.ExecutionVC{
		VCID:         "vc-session-1",
		ExecutionID:  "exec-session-1",
		WorkflowID:   "workflow-1",
		SessionID:    "session-query",
		IssuerDID:    "did:key:test",
		TargetDID:    "",
		CallerDID:    "did:key:caller",
		VCDocument:   json.RawMessage(vcDocBytes),
		Signature:    "sig",
		DocumentSize: int64(len(vcDocBytes)),
		Status:       "succeeded",
		CreatedAt:    time.Now(),
	}

	err := vcStorage.StoreExecutionVC(ctx, executionVC)
	require.NoError(t, err)

	// Query by session ID
	filters := &types.VCFilters{
		SessionID: stringPtr("session-query"),
	}

	vcs, err := vcStorage.QueryExecutionVCs(filters)
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(vcs), 1)
}

func TestVCStorage_QueryExecutionVCs_NilFilters(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	vcStorage := NewVCStorageWithStorage(provider)
	require.NoError(t, vcStorage.Initialize())

	// Query with nil filters
	vcs, err := vcStorage.QueryExecutionVCs(nil)
	require.NoError(t, err)
	require.NotNil(t, vcs)
	_ = ctx
}

func TestVCStorage_StoreWorkflowVC_Success(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	vcStorage := NewVCStorageWithStorage(provider)
	require.NoError(t, vcStorage.Initialize())

	now := time.Now()
	workflowVC := &types.WorkflowVC{
		WorkflowID:     "workflow-store",
		SessionID:      "session-store",
		ComponentVCs:   []string{"vc-1", "vc-2"},
		WorkflowVCID:   "workflow-vc-store",
		Status:         "succeeded",
		StartTime:      now,
		EndTime:        &now,
		TotalSteps:     2,
		CompletedSteps: 2,
		StorageURI:     "",
		DocumentSize:   0,
	}

	err := vcStorage.StoreWorkflowVC(ctx, workflowVC)
	require.NoError(t, err)

	// Verify workflow VC was stored - GetWorkflowVC looks up by workflow_vc_id
	storedVC, err := provider.GetWorkflowVC(ctx, "workflow-vc-store")
	require.NoError(t, err)
	require.NotNil(t, storedVC)
	require.Equal(t, workflowVC.WorkflowVCID, storedVC.WorkflowVCID)
}

func TestVCStorage_StoreWorkflowVC_NilProvider(t *testing.T) {
	vcStorage := NewVCStorageWithStorage(nil)

	workflowVC := &types.WorkflowVC{
		WorkflowID:   "workflow-test",
		WorkflowVCID: "workflow-vc-test",
	}

	err := vcStorage.StoreWorkflowVC(context.Background(), workflowVC)
	require.Error(t, err)
	require.Contains(t, err.Error(), "no storage provider configured")
}

func TestVCStorage_StoreWorkflowVC_AutoSizeCalculation(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	vcStorage := NewVCStorageWithStorage(provider)
	require.NoError(t, vcStorage.Initialize())

	vcDocBytes := []byte(`{"workflow": "vc"}`)

	workflowVC := &types.WorkflowVC{
		WorkflowID:     "workflow-auto-size",
		SessionID:      "session-1",
		ComponentVCs:   []string{"vc-1"},
		WorkflowVCID:   "workflow-vc-auto",
		Status:         "succeeded",
		StartTime:      time.Now(),
		VCDocument:     json.RawMessage(vcDocBytes),
		DocumentSize:   0, // Should be auto-calculated
		TotalSteps:     1,
		CompletedSteps: 1,
	}

	err := vcStorage.StoreWorkflowVC(ctx, workflowVC)
	require.NoError(t, err)

	// Verify size was calculated - GetWorkflowVC looks up by workflow_vc_id
	storedVC, err := provider.GetWorkflowVC(ctx, "workflow-vc-auto")
	require.NoError(t, err)
	require.Equal(t, int64(len(vcDocBytes)), storedVC.DocumentSize)
}

func TestVCStorage_GetWorkflowVC_Success(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	vcStorage := NewVCStorageWithStorage(provider)
	require.NoError(t, vcStorage.Initialize())

	now := time.Now()
	workflowVC := &types.WorkflowVC{
		WorkflowID:     "workflow-get",
		SessionID:      "session-get",
		ComponentVCs:   []string{"vc-1"},
		WorkflowVCID:   "workflow-vc-get",
		Status:         "succeeded",
		StartTime:      now,
		EndTime:        &now,
		TotalSteps:     1,
		CompletedSteps: 1,
		DocumentSize:   100,
	}

	err := vcStorage.StoreWorkflowVC(ctx, workflowVC)
	require.NoError(t, err)

	// Retrieve workflow VC
	retrievedVC, err := vcStorage.GetWorkflowVC("workflow-get")
	require.NoError(t, err)
	require.NotNil(t, retrievedVC)
	require.Equal(t, workflowVC.WorkflowVCID, retrievedVC.WorkflowVCID)
	require.Equal(t, workflowVC.WorkflowID, retrievedVC.WorkflowID)
}

func TestVCStorage_GetWorkflowVC_NotFound(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	vcStorage := NewVCStorageWithStorage(provider)
	require.NoError(t, vcStorage.Initialize())

	vc, err := vcStorage.GetWorkflowVC("nonexistent-workflow")
	require.Error(t, err)
	require.Nil(t, vc)
	require.Contains(t, err.Error(), "workflow VC not found")
	_ = ctx
}

func TestVCStorage_GetWorkflowVC_NilProvider(t *testing.T) {
	vcStorage := NewVCStorageWithStorage(nil)

	vc, err := vcStorage.GetWorkflowVC("workflow-test")
	require.Error(t, err)
	require.Nil(t, vc)
	require.Contains(t, err.Error(), "no storage provider configured")
}

func TestVCStorage_ListWorkflowVCs_Success(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	vcStorage := NewVCStorageWithStorage(provider)
	require.NoError(t, vcStorage.Initialize())

	// Store multiple workflow VCs
	for i := 1; i <= 2; i++ {
		workflowVC := &types.WorkflowVC{
			WorkflowID:     "workflow-list-" + string(rune('0'+i)),
			SessionID:      "session-list",
			ComponentVCs:   []string{"vc-1"},
			WorkflowVCID:   "workflow-vc-list-" + string(rune('0'+i)),
			Status:         "succeeded",
			StartTime:      time.Now(),
			TotalSteps:     1,
			CompletedSteps: 1,
			DocumentSize:   100,
		}

		err := vcStorage.StoreWorkflowVC(ctx, workflowVC)
		require.NoError(t, err)
	}

	// List workflow VCs
	workflowVCs, err := vcStorage.ListWorkflowVCs()
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(workflowVCs), 2)
}

func TestVCStorage_ListWorkflowVCs_NilProvider(t *testing.T) {
	vcStorage := NewVCStorageWithStorage(nil)

	workflowVCs, err := vcStorage.ListWorkflowVCs()
	require.Error(t, err)
	require.NotNil(t, workflowVCs) // Returns empty slice, not nil
	require.Empty(t, workflowVCs)
	require.Contains(t, err.Error(), "no storage provider configured")
}

func TestVCStorage_ListWorkflowVCStatusSummaries(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	vcStorage := NewVCStorageWithStorage(provider)
	require.NoError(t, vcStorage.Initialize())

	// Test with empty workflow IDs
	summaries, err := vcStorage.ListWorkflowVCStatusSummaries(ctx, []string{})
	require.NoError(t, err)
	require.Empty(t, summaries)

	// Test with workflow IDs that don't exist in storage - should return empty, no error
	summaries, err = vcStorage.ListWorkflowVCStatusSummaries(ctx, []string{"workflow-1", "workflow-2"})
	require.NoError(t, err)
	require.Empty(t, summaries)
}

func TestVCStorage_ListWorkflowVCStatusSummaries_NilProvider(t *testing.T) {
	vcStorage := NewVCStorageWithStorage(nil)

	summaries, err := vcStorage.ListWorkflowVCStatusSummaries(context.Background(), []string{"workflow-1"})
	require.Error(t, err)
	require.Nil(t, summaries)
	require.Contains(t, err.Error(), "no storage provider configured")
}

func TestVCStorage_GetVCStats(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	vcStorage := NewVCStorageWithStorage(provider)
	require.NoError(t, vcStorage.Initialize())

	// Store some VCs
	vcDocBytes := []byte(`{"test": "data"}`)
	executionVC := &types.ExecutionVC{
		VCID:         "vc-stats-1",
		ExecutionID:  "exec-stats-1",
		WorkflowID:   "workflow-1",
		SessionID:    "session-1",
		IssuerDID:    "did:key:test",
		TargetDID:    "",
		CallerDID:    "did:key:caller",
		VCDocument:   json.RawMessage(vcDocBytes),
		Signature:    "sig",
		DocumentSize: int64(len(vcDocBytes)),
		Status:       "succeeded",
		CreatedAt:    time.Now(),
	}

	err := vcStorage.StoreExecutionVC(ctx, executionVC)
	require.NoError(t, err)

	workflowVC := &types.WorkflowVC{
		WorkflowID:     "workflow-stats",
		SessionID:      "session-1",
		ComponentVCs:   []string{"vc-stats-1"},
		WorkflowVCID:   "workflow-vc-stats",
		Status:         "succeeded",
		StartTime:      time.Now(),
		TotalSteps:     1,
		CompletedSteps: 1,
		DocumentSize:   100,
	}

	err = vcStorage.StoreWorkflowVC(ctx, workflowVC)
	require.NoError(t, err)

	// Get stats
	stats := vcStorage.GetVCStats()
	require.NotNil(t, stats)
	require.Contains(t, stats, "execution_vcs")
	require.Contains(t, stats, "workflow_vcs")
	require.GreaterOrEqual(t, stats["execution_vcs"].(int), 1)
	require.GreaterOrEqual(t, stats["workflow_vcs"].(int), 1)
}

func TestVCStorage_GetVCStats_NilProvider(t *testing.T) {
	vcStorage := NewVCStorageWithStorage(nil)

	stats := vcStorage.GetVCStats()
	require.NotNil(t, stats)
	require.Equal(t, 0, stats["execution_vcs"])
	require.Equal(t, 0, stats["workflow_vcs"])
}

func TestVCStorage_DeleteExecutionVC(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	vcStorage := NewVCStorageWithStorage(provider)
	require.NoError(t, vcStorage.Initialize())

	// Delete is currently a no-op, but should not error
	err := vcStorage.DeleteExecutionVC("vc-test")
	require.NoError(t, err)
	_ = ctx
}

func TestVCStorage_DeleteWorkflowVC(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	vcStorage := NewVCStorageWithStorage(provider)
	require.NoError(t, vcStorage.Initialize())

	// Delete is currently a no-op, but should not error
	err := vcStorage.DeleteWorkflowVC("workflow-test")
	require.NoError(t, err)
	_ = ctx
}
