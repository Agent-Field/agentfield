package storage

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// ── THE OFFLINE HOLD ─────────────────────────────────────────────────────────
//
// A node that opted in (types.AgentNode.HoldsExecutionsOffline) keeps its work
// across being offline: a laptop asleep overnight still holds the job it was
// running, and lists it as inflight on its first lease when it wakes. While
// such a node is away nothing renews its executions, so the stale sweep would
// time them out. Instead the cleanup service marks them held: status stays
// running, status_reason becomes types.ExecutionReasonNodeOffline, and
// updated_at is stamped with the moment the hold began.
//
// THE SWEEPS SKIP A HELD ROW. MarkStaleExecutions, MarkStaleWorkflowExecutions
// and RetryStaleWorkflowExecutions all exclude status_reason = node_offline,
// on whichever table they select from, so the hold must be written to both
// tables together — which every function here does.
//
// A hold ends in exactly one of three ways: the node comes back and lists the
// execution (ReleaseHeldExecutions — it carries on), the node comes back and
// does not list it, or the node is removed, or the hold outlives its ceiling
// (FailHeldExecutions — each with its own reason and words).

// heldReasonSQL is the held status_reason as a SQL literal, for the sweeps'
// exclusion predicates. It is a constant, never user input.
const heldReasonSQL = "'" + types.ExecutionReasonNodeOffline + "'"

// holdableStatusesSQL are the statuses a hold may cover: the ones the
// executions stale sweep reaps. A waiting row is already left alone by it.
const holdableStatusesSQL = "('running', 'pending', 'queued')"

// HeldExecution is one execution the control plane is holding for its node.
type HeldExecution struct {
	ExecutionID string
	AgentNodeID string
	// HeldSince is when the hold began (the row's updated_at, which nothing
	// moves while the node is away).
	HeldSince time.Time
}

// HoldOfflineNodeExecutions marks every execution of agentNodeID that has
// been quiet for staleAfter as held, on both tables, and returns how many it
// marked. The caller has already decided the node is opted in and offline;
// this only refuses rows that are fresh, already held, or held through an
// agent restart (types.ExecutionReasonAwaitingAgentRestart, which has its own
// owner).
func (ls *LocalStorage) HoldOfflineNodeExecutions(ctx context.Context, agentNodeID string, staleAfter time.Duration, limit int) (int, error) {
	if strings.TrimSpace(agentNodeID) == "" {
		return 0, fmt.Errorf("agent_node_id is required")
	}
	if limit <= 0 {
		return 0, nil
	}
	db := ls.requireSQLDB()
	cutoff := time.Now().UTC().Add(-staleAfter)
	tsExpr := ls.staleTimestampExpr("COALESCE(updated_at, created_at, started_at)")
	cutoffExpr := ls.staleTimestampExpr("?")
	quiet := `agent_node_id = ?
		  AND status IN ` + holdableStatusesSQL + `
		  AND COALESCE(status_reason, '') NOT IN (` + heldReasonSQL + `, ?)
		  AND ` + tsExpr + ` <= ` + cutoffExpr

	rows, err := db.QueryContext(ctx, `SELECT execution_id FROM executions WHERE `+quiet+` LIMIT ?`,
		agentNodeID, types.ExecutionReasonAwaitingAgentRestart, cutoff, limit)
	if err != nil {
		return 0, fmt.Errorf("query executions to hold: %w", err)
	}
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			rows.Close()
			return 0, fmt.Errorf("scan execution to hold: %w", err)
		}
		ids = append(ids, id)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return 0, fmt.Errorf("iterate executions to hold: %w", err)
	}
	if len(ids) == 0 {
		return 0, nil
	}

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return 0, fmt.Errorf("begin hold transaction: %w", err)
	}
	defer rollbackTx(tx, "HoldOfflineNodeExecutions")

	now := time.Now().UTC()
	held := 0
	for _, id := range ids {
		// The guard is repeated so activity that landed after the select (the
		// node came back and touched it) wins over the hold.
		res, err := tx.ExecContext(ctx, `UPDATE executions SET status_reason = ?, updated_at = ?
			WHERE execution_id = ? AND `+quiet,
			types.ExecutionReasonNodeOffline, now, id, agentNodeID, types.ExecutionReasonAwaitingAgentRestart, cutoff)
		if err != nil {
			return 0, fmt.Errorf("hold execution %s: %w", id, err)
		}
		if n, _ := res.RowsAffected(); n == 0 {
			continue
		}
		if _, err := tx.ExecContext(ctx, `UPDATE workflow_executions SET status_reason = ?, updated_at = ?
			WHERE execution_id = ? AND status IN ('running', 'pending', 'queued', 'waiting')`,
			types.ExecutionReasonNodeOffline, now, id); err != nil {
			return 0, fmt.Errorf("hold workflow execution %s: %w", id, err)
		}
		held++
	}
	if err := tx.Commit(); err != nil {
		return 0, fmt.Errorf("commit hold transaction: %w", err)
	}
	return held, nil
}

// ListHeldExecutions returns the executions currently held, for one node or,
// with agentNodeID empty, for every node.
func (ls *LocalStorage) ListHeldExecutions(ctx context.Context, agentNodeID string) ([]HeldExecution, error) {
	db := ls.requireSQLDB()
	query := `SELECT execution_id, agent_node_id, updated_at FROM executions
		WHERE status_reason = ` + heldReasonSQL + ` AND status IN ` + holdableStatusesSQL
	var args []interface{}
	if agentNodeID != "" {
		query += ` AND agent_node_id = ?`
		args = append(args, agentNodeID)
	}
	rows, err := db.QueryContext(ctx, query+` ORDER BY execution_id`, args...)
	if err != nil {
		return nil, fmt.Errorf("query held executions: %w", err)
	}
	defer rows.Close()
	var out []HeldExecution
	for rows.Next() {
		var h HeldExecution
		var updated interface{}
		if err := rows.Scan(&h.ExecutionID, &h.AgentNodeID, &updated); err != nil {
			return nil, fmt.Errorf("scan held execution: %w", err)
		}
		if h.HeldSince, err = parseDBTime(updated); err != nil {
			return nil, fmt.Errorf("held execution %s updated_at: %w", h.ExecutionID, err)
		}
		out = append(out, h)
	}
	return out, rows.Err()
}

// ReleaseHeldExecutions ends the hold on each listed execution of agentNodeID
// — its node came back still working on it — clearing the reason and
// stamping updated_at as the activity it is. It returns how many it released.
func (ls *LocalStorage) ReleaseHeldExecutions(ctx context.Context, agentNodeID string, ids []string) (int, error) {
	return ls.endHolds(ctx, ids, func(tx *sqlTx, id string, now time.Time) (int64, error) {
		res, err := tx.ExecContext(ctx, `UPDATE executions SET status_reason = NULL, updated_at = ?
			WHERE execution_id = ? AND agent_node_id = ? AND status_reason = `+heldReasonSQL,
			now, id, agentNodeID)
		if err != nil {
			return 0, err
		}
		n, _ := res.RowsAffected()
		if n == 0 {
			return 0, nil
		}
		_, err = tx.ExecContext(ctx, `UPDATE workflow_executions SET status_reason = NULL, updated_at = ?
			WHERE execution_id = ? AND status_reason = `+heldReasonSQL, now, id)
		return n, err
	})
}

// FailHeldExecutions fails each listed execution that is still held, on both
// tables, with reason as status_reason and message as error_message, and
// returns how many it failed. A row that left the hold in the meantime (the
// node came back, or it finished) is left alone.
func (ls *LocalStorage) FailHeldExecutions(ctx context.Context, ids []string, reason, message string) (int, error) {
	return ls.endHolds(ctx, ids, func(tx *sqlTx, id string, now time.Time) (int64, error) {
		res, err := tx.ExecContext(ctx, `UPDATE executions
			SET status = ?, status_reason = ?, error_message = ?, completed_at = ?, updated_at = ?
			WHERE execution_id = ? AND status_reason = `+heldReasonSQL+` AND status IN `+holdableStatusesSQL,
			types.ExecutionStatusFailed, reason, message, now, now, id)
		if err != nil {
			return 0, err
		}
		n, _ := res.RowsAffected()
		if n == 0 {
			return 0, nil
		}
		_, err = tx.ExecContext(ctx, `UPDATE workflow_executions
			SET status = ?, status_reason = ?, error_message = ?, completed_at = ?, updated_at = ?
			WHERE execution_id = ? AND status IN ('running', 'pending', 'queued', 'waiting')`,
			types.ExecutionStatusFailed, reason, message, now, now, id)
		return n, err
	})
}

// endHolds runs end once per id in one transaction and counts the rows it
// moved.
func (ls *LocalStorage) endHolds(ctx context.Context, ids []string, end func(tx *sqlTx, id string, now time.Time) (int64, error)) (int, error) {
	if len(ids) == 0 {
		return 0, nil
	}
	tx, err := ls.requireSQLDB().BeginTx(ctx, nil)
	if err != nil {
		return 0, fmt.Errorf("begin end-hold transaction: %w", err)
	}
	defer rollbackTx(tx, "endHolds")
	now := time.Now().UTC()
	total := 0
	for _, id := range ids {
		n, err := end(tx, id, now)
		if err != nil {
			return 0, fmt.Errorf("end hold on %s: %w", id, err)
		}
		total += int(n)
	}
	if err := tx.Commit(); err != nil {
		return 0, fmt.Errorf("commit end-hold transaction: %w", err)
	}
	return total, nil
}
