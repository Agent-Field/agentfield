package handlers

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// ── A NODE THAT GOES AWAY DOES NOT LOSE ITS WORK BY BEING AWAY ───────────────
//
// A node that registers with metadata.custom.holds_executions_offline = true
// (the codeaf factory, on a laptop that sleeps) promises two things: its work
// survives it being offline, and on its first lease back it lists every
// execution it still holds as inflight. In exchange the stale sweep does not
// time those executions out while it is away. Instead each cleanup pass, before
// the sweeps run, marks the quiet executions of an opted-in, registered,
// offline node as held (storage/offline_hold.go), and the sweeps skip them.
//
// A hold ends in one of four ways, and only the first keeps the execution:
//   - the node's lease lists it as inflight: released, it carries on;
//   - the node's lease does not list it: failed, the node came back without it;
//   - the node is no longer registered: failed, there is nobody to come back;
//   - the hold outlives offline_hold_ceiling: failed, nothing waits forever.
//
// A node that did not opt in is untouched: its executions are reaped by the
// stale sweep exactly as before.

// offlineHolder is the storage the offline hold needs. LocalStorage serves it
// for SQLite and Postgres alike; a store without it simply never holds.
type offlineHolder interface {
	HoldOfflineNodeExecutions(ctx context.Context, agentNodeID string, staleAfter time.Duration, limit int) (int, error)
	ListHeldExecutions(ctx context.Context, agentNodeID string) ([]storage.HeldExecution, error)
	ReleaseHeldExecutions(ctx context.Context, agentNodeID string, ids []string) (int, error)
	FailHeldExecutions(ctx context.Context, ids []string, reason, message string) (int, error)
}

// nodeLostExecutionMessage is the error a held execution fails with when its
// node comes back without it.
func nodeLostExecutionMessage(nodeID string) string {
	return fmt.Sprintf("node %s came back without this execution: it was lost while the node was offline", nodeID)
}

// ResolveOfflineHolds settles every execution held for nodeID against the
// inflight list of the lease that just arrived: a listed one is released and
// carries on, an unlisted one is failed as lost. It returns how many of each.
// A node with nothing held costs one indexed query.
func ResolveOfflineHolds(ctx context.Context, store any, nodeID string, inflight []string) (released, lost int) {
	holder, ok := store.(offlineHolder)
	if !ok {
		return 0, 0
	}
	held, err := holder.ListHeldExecutions(ctx, nodeID)
	if err != nil {
		logger.Logger.Warn().Err(err).Str("node_id", nodeID).Msg("lease: could not read held executions")
		return 0, 0
	}
	if len(held) == 0 {
		return 0, 0
	}
	listed := make(map[string]struct{}, len(inflight))
	for _, id := range inflight {
		listed[strings.TrimSpace(id)] = struct{}{}
	}
	var keep, drop []string
	for _, h := range held {
		if _, ok := listed[h.ExecutionID]; ok {
			keep = append(keep, h.ExecutionID)
		} else {
			drop = append(drop, h.ExecutionID)
		}
	}
	if released, err = holder.ReleaseHeldExecutions(ctx, nodeID, keep); err != nil {
		logger.Logger.Warn().Err(err).Str("node_id", nodeID).Msg("lease: could not release held executions")
	}
	if lost, err = holder.FailHeldExecutions(ctx, drop, types.ExecutionReasonNodeLostExecution, nodeLostExecutionMessage(nodeID)); err != nil {
		logger.Logger.Warn().Err(err).Str("node_id", nodeID).Msg("lease: could not fail lost executions")
	}
	if released > 0 || lost > 0 {
		logger.Logger.Info().Str("node_id", nodeID).Int("released", released).Int("lost", lost).Msg("lease: node back; held executions settled")
	}
	return released, lost
}

// holdOfflineNodes is the cleanup pass's first step when the stale sweep is
// on: it ends the holds that can no longer be honoured and holds the quiet
// executions of every opted-in node that is offline now, so the sweeps that
// follow skip them.
func (ecs *ExecutionCleanupService) holdOfflineNodes(ctx context.Context) {
	holder, ok := ecs.storage.(offlineHolder)
	if !ok {
		return
	}
	agents, err := ecs.storage.ListAgents(ctx, types.AgentFilters{})
	if err != nil {
		// Without the node list nothing can be judged; leave every hold as it
		// is. The sweep that follows still skips held rows.
		logger.Logger.Warn().Err(err).Msg("offline hold: could not list nodes")
		return
	}
	// One node may be registered under several versions; it is opted in when
	// any of them is, and its last word is the latest heartbeat of any.
	type nodeState struct {
		optedIn  bool
		lastSeen time.Time
	}
	nodes := map[string]*nodeState{}
	for _, a := range agents {
		if a == nil {
			continue
		}
		st := nodes[a.ID]
		if st == nil {
			st = &nodeState{}
			nodes[a.ID] = st
		}
		st.optedIn = st.optedIn || a.HoldsExecutionsOffline()
		if a.LastHeartbeat.After(st.lastSeen) {
			st.lastSeen = a.LastHeartbeat
		}
	}

	now := time.Now().UTC()
	ceiling := ecs.config.OfflineHoldCeiling
	if ceiling <= 0 {
		ceiling = config.DefaultOfflineHoldCeiling
	}

	held, err := holder.ListHeldExecutions(ctx, "")
	if err != nil {
		logger.Logger.Warn().Err(err).Msg("offline hold: could not read held executions")
	} else {
		var removed, expired []string
		releasedBy := map[string][]string{}
		for _, h := range held {
			st := nodes[h.AgentNodeID]
			switch {
			case st == nil:
				removed = append(removed, h.ExecutionID)
			case !st.optedIn:
				// The node re-registered without the promise: its work goes
				// back under the ordinary stale rules.
				releasedBy[h.AgentNodeID] = append(releasedBy[h.AgentNodeID], h.ExecutionID)
			case now.Sub(h.HeldSince) > ceiling:
				expired = append(expired, h.ExecutionID)
			}
		}
		ecs.failHeld(ctx, holder, removed, types.ExecutionReasonNodeRemoved, "the node holding this execution is no longer registered")
		ecs.failHeld(ctx, holder, expired, types.ExecutionReasonNodeOfflineTooLong,
			fmt.Sprintf("the node holding this execution stayed offline for more than %s", ceiling))
		for node, ids := range releasedBy {
			if _, err := holder.ReleaseHeldExecutions(ctx, node, ids); err != nil {
				logger.Logger.Warn().Err(err).Str("node_id", node).Msg("offline hold: could not release executions of a node that opted out")
			}
		}
	}

	offlineAfter := ecs.offlineAfter
	if offlineAfter <= 0 {
		offlineAfter = DefaultLeaseTTL
	}
	for id, st := range nodes {
		if !st.optedIn || now.Sub(st.lastSeen) <= offlineAfter {
			continue
		}
		n, err := holder.HoldOfflineNodeExecutions(ctx, id, ecs.config.StaleExecutionTimeout, ecs.config.BatchSize)
		if err != nil {
			logger.Logger.Warn().Err(err).Str("node_id", id).Msg("offline hold: could not hold executions")
			continue
		}
		if n > 0 {
			logger.Logger.Info().Str("node_id", id).Int("held", n).Time("last_seen", st.lastSeen).Msg("offline hold: node offline; its executions are held for it")
		}
	}
}

func (ecs *ExecutionCleanupService) failHeld(ctx context.Context, holder offlineHolder, ids []string, reason, message string) {
	if len(ids) == 0 {
		return
	}
	n, err := holder.FailHeldExecutions(ctx, ids, reason, message)
	if err != nil {
		logger.Logger.Warn().Err(err).Str("reason", reason).Msg("offline hold: could not fail held executions")
		return
	}
	if n > 0 {
		logger.Logger.Info().Str("reason", reason).Int("failed", n).Msg("offline hold: held executions given up")
	}
}
