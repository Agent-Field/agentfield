package services

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestVCStorageGetExecutionVCsBySessionAndNilConversion(t *testing.T) {
	provider, ctx := setupTestStorage(t)
	vcStorage := NewVCStorageWithStorage(provider)
	require.NoError(t, vcStorage.Initialize())

	doc := json.RawMessage(`{"session":"a"}`)
	for _, tc := range []struct {
		vcID       string
		executionID string
		sessionID  string
	}{
		{vcID: "vc-session-1", executionID: "exec-session-1", sessionID: "session-a"},
		{vcID: "vc-session-2", executionID: "exec-session-2", sessionID: "session-a"},
		{vcID: "vc-session-3", executionID: "exec-session-3", sessionID: "session-b"},
	} {
		require.NoError(t, vcStorage.StoreExecutionVC(ctx, &types.ExecutionVC{
			VCID:         tc.vcID,
			ExecutionID:  tc.executionID,
			WorkflowID:   "workflow-1",
			SessionID:    tc.sessionID,
			IssuerDID:    "did:key:test",
			CallerDID:    "did:key:caller",
			VCDocument:   doc,
			Signature:    "sig",
			DocumentSize: int64(len(doc)),
			Status:       "succeeded",
			CreatedAt:    time.Now(),
		}))
	}

	records, err := vcStorage.GetExecutionVCsBySession("session-a")
	require.NoError(t, err)
	require.Len(t, records, 2)

	vc, err := vcStorage.convertVCInfoToExecutionVC(nil)
	require.ErrorContains(t, err, "execution VC info is nil")
	require.Nil(t, vc)

	emptyStorage := NewVCStorageWithStorage(nil)
	records, err = emptyStorage.GetExecutionVCsBySession("session-a")
	require.ErrorContains(t, err, "no storage provider configured")
	require.Empty(t, records)

	_, _ = provider, context.Background()
}
