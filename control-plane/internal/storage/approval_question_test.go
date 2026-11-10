package storage

import (
	"context"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

// Contract S8: An approval question is stored per (execution, approval request id)
// and read back whole, asking the same round again updates to the newer words,
// a different round stays a separate row, and an unknown pair reads as nil with
// no error.
func TestApprovalQuestionStorage_S8_PutGetUpdateSeparateRoundsAndUnknown(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	// An unknown (execution, approval_request_id) pair returns nil with no error,
	// matching GetApprovalQuestion's contract for rounds carrying no question.
	missing, err := ls.GetApprovalQuestion(ctx, "exec-1", "apr-missing")
	require.NoError(t, err)
	require.Nil(t, missing)

	round1Initial := types.ApprovalQuestion{
		ExecutionID:       "exec-1",
		ApprovalRequestID: "apr-r1",
		Kind:              types.ApprovalKindPlanReview,
		Question:          "Approve the initial migration plan?",
		Body:              "# Plan v1\n\n1. Create table\n2. Backfill rows\n",
		Options:           []string{"approve", "request_changes", "reject"},
	}
	require.NoError(t, ls.PutApprovalQuestion(ctx, round1Initial))

	gotR1, err := ls.GetApprovalQuestion(ctx, "exec-1", "apr-r1")
	require.NoError(t, err)
	require.Equal(t, &round1Initial, gotR1)

	// Asking the same round again replaces the stored question with the newer words.
	round1Updated := types.ApprovalQuestion{
		ExecutionID:       "exec-1",
		ApprovalRequestID: "apr-r1",
		Kind:              types.ApprovalKindQuestion,
		Question:          "Approve the batched migration plan?",
		Body:              "# Plan v2\n\n1. Create table\n2. Backfill in batches of 1000\n",
		Options:           []string{"ship", "revise"},
	}
	require.NoError(t, ls.PutApprovalQuestion(ctx, round1Updated))

	gotR1Updated, err := ls.GetApprovalQuestion(ctx, "exec-1", "apr-r1")
	require.NoError(t, err)
	require.Equal(t, &round1Updated, gotR1Updated)

	// A second round on the same execution and the same request id on another
	// execution stay independent rows.
	round2 := types.ApprovalQuestion{
		ExecutionID:       "exec-1",
		ApprovalRequestID: "apr-r2",
		Kind:              types.ApprovalKindApproval,
		Question:          "Deploy to staging now?",
		Body:              "All checks are green.",
	}
	require.NoError(t, ls.PutApprovalQuestion(ctx, round2))

	otherExec := types.ApprovalQuestion{
		ExecutionID:       "exec-2",
		ApprovalRequestID: "apr-r1",
		Kind:              types.ApprovalKindQuestion,
		Question:          "Which region?",
		Options:           []string{"us-east-1", "eu-west-1"},
	}
	require.NoError(t, ls.PutApprovalQuestion(ctx, otherExec))

	gotR1Again, err := ls.GetApprovalQuestion(ctx, "exec-1", "apr-r1")
	require.NoError(t, err)
	require.Equal(t, &round1Updated, gotR1Again)

	gotR2, err := ls.GetApprovalQuestion(ctx, "exec-1", "apr-r2")
	require.NoError(t, err)
	require.Equal(t, &round2, gotR2)

	gotOtherExec, err := ls.GetApprovalQuestion(ctx, "exec-2", "apr-r1")
	require.NoError(t, err)
	require.Equal(t, &otherExec, gotOtherExec)

	unknownRound, err := ls.GetApprovalQuestion(ctx, "exec-1", "apr-r99")
	require.NoError(t, err)
	require.Nil(t, unknownRound)

	unknownExec, err := ls.GetApprovalQuestion(ctx, "exec-99", "apr-r1")
	require.NoError(t, err)
	require.Nil(t, unknownExec)
}

// Contract S9: execution_approval_questions is created lazily on the first call
// to a fresh store without a schema migration, and surfaces errors when the
// context or database is closed.
func TestApprovalQuestionStorage_S9_LazyTableCreationWithoutMigration(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	var before int
	err := ls.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'execution_approval_questions'`).Scan(&before)
	require.NoError(t, err)
	require.Equal(t, 0, before)

	got, err := ls.GetApprovalQuestion(ctx, "exec-lazy", "apr-lazy")
	require.NoError(t, err)
	require.Nil(t, got)

	var after int
	err = ls.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'execution_approval_questions'`).Scan(&after)
	require.NoError(t, err)
	require.Equal(t, 1, after)

	// Cancelled contexts after lazy table creation surface SQL errors cleanly.
	cancelledCtx, cancel := context.WithCancel(ctx)
	cancel()
	require.Error(t, ls.PutApprovalQuestion(cancelledCtx, types.ApprovalQuestion{
		ExecutionID:       "exec-lazy",
		ApprovalRequestID: "apr-lazy",
		Kind:              types.ApprovalKindApproval,
	}))
	_, err = ls.GetApprovalQuestion(cancelledCtx, "exec-lazy", "apr-lazy")
	require.Error(t, err)

	// A closed fresh store surfaces and remembers the lazy DDL error on its db handle.
	closedLS, closedCtx := setupLocalStorage(t)
	require.NoError(t, closedLS.Close(closedCtx))
	require.Error(t, closedLS.PutApprovalQuestion(closedCtx, types.ApprovalQuestion{
		ExecutionID:       "exec-closed",
		ApprovalRequestID: "apr-closed",
		Kind:              types.ApprovalKindApproval,
	}))
	_, err = closedLS.GetApprovalQuestion(closedCtx, "exec-closed", "apr-closed")
	require.Error(t, err)
}
