package handlers

import (
	"context"
	"strings"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// maxInflightPerLease bounds how many executions one lease (or one link
// heartbeat) may touch.
const maxInflightPerLease = 512

// inflightToucher is the one storage call TouchInflightExecutions needs.
type inflightToucher interface {
	UpdateExecutionRecord(ctx context.Context, executionID string, updater func(*types.Execution) (*types.Execution, error)) (*types.Execution, error)
}

// TouchInflightExecutions bumps updated_at on every listed execution that is
// owned by nodeID and still running, pending or queued, and returns how many
// it touched. It goes through UpdateExecutionRecord, which stamps updated_at
// itself, so SQLite and Postgres take the same path and the stale reaper's SQL
// is unchanged. An execution of another node, or one already terminal, is
// left alone: a node can keep only its own live work alive.
func TouchInflightExecutions(ctx context.Context, store any, nodeID string, ids []string) int {
	toucher, ok := store.(inflightToucher)
	if !ok || len(ids) == 0 {
		return 0
	}
	touched := 0
	seen := make(map[string]struct{}, len(ids))
	for i, id := range ids {
		if i >= maxInflightPerLease {
			break
		}
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		did := false
		_, err := toucher.UpdateExecutionRecord(ctx, id, func(cur *types.Execution) (*types.Execution, error) {
			if cur == nil || cur.AgentNodeID != nodeID {
				return nil, nil
			}
			switch cur.Status {
			case types.ExecutionStatusRunning, types.ExecutionStatusPending, types.ExecutionStatusQueued:
				did = true
				return cur, nil
			}
			return nil, nil
		})
		if err != nil {
			logger.Logger.Debug().Err(err).Str("node_id", nodeID).Str("execution_id", id).Msg("lease: inflight touch skipped")
			continue
		}
		if did {
			touched++
		}
	}
	return touched
}
