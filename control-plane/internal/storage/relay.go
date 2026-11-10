package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"
)

// ── THE RELAY: A HARNESS TRANSCRIPT PASSING THROUGH, NOT KEPT ───────────────
//
// WorkAF is the system of record for a codeaf harness run's transcript; the
// control plane is only the pipe between the node that ran it and WorkAF.
// So a transcript line is held here exactly until WorkAF acks it, and no
// longer: RelayAck deletes everything up to the acked seq, and RelayPrune
// drops what nobody acked within the execution-log retention, so a WorkAF
// that was offline can still catch up while it is gone and the table cannot
// grow without bound. The execution itself — its tree, status, timings and
// result — stays where it always was.
//
// THE SEQ IS THE NODE'S. The node numbers each execution's lines from 1 and
// replays every line the control plane has not answered; (execution_id, seq)
// is the key and a replayed line is ignored, so a control plane that died
// between storing a batch and answering it neither loses nor doubles a line.
// A WorkAF reading after its last seq after a restart reads each line once.

// RelayEntry is one line passing through.
type RelayEntry struct {
	Seq  int64           `json:"seq"`
	Kind string          `json:"kind"`
	Data json.RawMessage `json:"data,omitempty"`
	At   time.Time       `json:"at"`
}

var relayTableOnce sync.Map // *sqlDatabase → error

func (ls *LocalStorage) relayTable(ctx context.Context) error {
	if v, ok := relayTableOnce.Load(ls.db); ok {
		if v == nil {
			return nil
		}
		return v.(error)
	}
	_, err := ls.db.ExecContext(ctx, `CREATE TABLE IF NOT EXISTS execution_relay (
		execution_id TEXT NOT NULL,
		seq BIGINT NOT NULL,
		kind TEXT NOT NULL,
		data TEXT,
		recorded_at TIMESTAMP NOT NULL,
		PRIMARY KEY (execution_id, seq))`)
	if err != nil {
		err = fmt.Errorf("create execution_relay: %w", err)
		relayTableOnce.Store(ls.db, err)
		return err
	}
	relayTableOnce.Store(ls.db, nil)
	return nil
}

// RelayPut holds entries for executionID. A seq already held is ignored. It
// answers how many were new.
func (ls *LocalStorage) RelayPut(ctx context.Context, executionID string, entries []RelayEntry) (int, error) {
	if err := ls.relayTable(ctx); err != nil {
		return 0, err
	}
	ls.mu.Lock()
	defer ls.mu.Unlock()
	stored := 0
	now := time.Now().UTC()
	for _, e := range entries {
		res, err := ls.db.ExecContext(ctx, `INSERT INTO execution_relay (execution_id, seq, kind, data, recorded_at)
			VALUES (?, ?, ?, ?, ?) ON CONFLICT (execution_id, seq) DO NOTHING`, executionID, e.Seq, e.Kind, string(e.Data), now)
		if err != nil {
			return stored, fmt.Errorf("relay put: %w", err)
		}
		if n, _ := res.RowsAffected(); n > 0 {
			stored++
		}
	}
	return stored, nil
}

// RelayAfter is executionID's held entries after seq, oldest first.
func (ls *LocalStorage) RelayAfter(ctx context.Context, executionID string, after int64, limit int) ([]RelayEntry, error) {
	if err := ls.relayTable(ctx); err != nil {
		return nil, err
	}
	if limit <= 0 || limit > 1000 {
		limit = 500
	}
	rows, err := ls.db.QueryContext(ctx, `SELECT seq, kind, data, recorded_at FROM execution_relay
		WHERE execution_id = ? AND seq > ? ORDER BY seq LIMIT ?`, executionID, after, limit)
	if err != nil {
		return nil, fmt.Errorf("relay read: %w", err)
	}
	defer rows.Close()
	out := []RelayEntry{}
	for rows.Next() {
		var e RelayEntry
		var data *string
		if err := rows.Scan(&e.Seq, &e.Kind, &data, &e.At); err != nil {
			return nil, err
		}
		if data != nil && *data != "" {
			e.Data = json.RawMessage(*data)
		}
		out = append(out, e)
	}
	return out, rows.Err()
}

// RelayAck drops executionID's entries through seq: WorkAF has them now.
func (ls *LocalStorage) RelayAck(ctx context.Context, executionID string, through int64) (int64, error) {
	if err := ls.relayTable(ctx); err != nil {
		return 0, err
	}
	ls.mu.Lock()
	defer ls.mu.Unlock()
	res, err := ls.db.ExecContext(ctx, `DELETE FROM execution_relay WHERE execution_id = ? AND seq <= ?`, executionID, through)
	if err != nil {
		return 0, fmt.Errorf("relay ack: %w", err)
	}
	n, _ := res.RowsAffected()
	return n, nil
}

// RelayPrune drops every entry recorded before olderThan, acked or not.
func (ls *LocalStorage) RelayPrune(ctx context.Context, olderThan time.Time) error {
	if err := ls.relayTable(ctx); err != nil {
		return err
	}
	ls.mu.Lock()
	defer ls.mu.Unlock()
	_, err := ls.db.ExecContext(ctx, `DELETE FROM execution_relay WHERE recorded_at < ?`, olderThan.UTC())
	return err
}

// RelayCount is how many entries are held, for one execution or ("") all.
func (ls *LocalStorage) RelayCount(ctx context.Context, executionID string) (int64, error) {
	if err := ls.relayTable(ctx); err != nil {
		return 0, err
	}
	q, args := `SELECT COUNT(*) FROM execution_relay`, []any{}
	if executionID != "" {
		q, args = q+` WHERE execution_id = ?`, append(args, executionID)
	}
	var n int64
	err := ls.db.QueryRowContext(ctx, q, args...).Scan(&n)
	return n, err
}

// RelayExecutions is every execution with entries held now, for a subscriber
// that reconnects and must be sent what it has not acked.
func (ls *LocalStorage) RelayExecutions(ctx context.Context) ([]string, error) {
	if err := ls.relayTable(ctx); err != nil {
		return nil, err
	}
	rows, err := ls.db.QueryContext(ctx, `SELECT DISTINCT execution_id FROM execution_relay ORDER BY execution_id`)
	if err != nil {
		return nil, fmt.Errorf("relay executions: %w", err)
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		out = append(out, id)
	}
	return out, rows.Err()
}

// ── WHO A RUN BELONGS TO ────────────────────────────────────────────────────
//
// A run's owner is the node that may watch it: the node that started it (an
// execute carrying X-AgentField-Run-Owner) or the node a trigger routed it to.
// Every execution in the run's tree shares its run_id, so one row scopes the
// whole tree, descendants included. THE FIRST OWNER WINS: a nested call that
// carries the header again cannot move a run to another subscriber. This is
// control data, like the execution tree itself, so it is durable.

func (ls *LocalStorage) runOwnerTable(ctx context.Context) error {
	if err := ls.relayTable(ctx); err != nil {
		return err
	}
	_, err := ls.db.ExecContext(ctx, `CREATE TABLE IF NOT EXISTS run_owners (
		run_id TEXT PRIMARY KEY,
		owner_node_id TEXT NOT NULL,
		created_at TIMESTAMP NOT NULL)`)
	if err != nil {
		return fmt.Errorf("create run_owners: %w", err)
	}
	return nil
}

// SetRunOwner records owner for runID unless the run already has one.
func (ls *LocalStorage) SetRunOwner(ctx context.Context, runID, owner string) error {
	if runID == "" || owner == "" {
		return nil
	}
	if err := ls.runOwnerTable(ctx); err != nil {
		return err
	}
	ls.mu.Lock()
	defer ls.mu.Unlock()
	_, err := ls.db.ExecContext(ctx, `INSERT INTO run_owners (run_id, owner_node_id, created_at) VALUES (?, ?, ?)
		ON CONFLICT (run_id) DO NOTHING`, runID, owner, time.Now().UTC())
	if err != nil {
		return fmt.Errorf("set run owner: %w", err)
	}
	return nil
}

// RunOwner is runID's owner, "" when it has none.
func (ls *LocalStorage) RunOwner(ctx context.Context, runID string) (string, error) {
	if err := ls.runOwnerTable(ctx); err != nil {
		return "", err
	}
	var owner string
	err := ls.db.QueryRowContext(ctx, `SELECT owner_node_id FROM run_owners WHERE run_id = ?`, runID).Scan(&owner)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return "", nil
		}
		return "", fmt.Errorf("run owner: %w", err)
	}
	return owner, nil
}

// RunsOwnedBy is every run owner has owned since since, oldest first.
func (ls *LocalStorage) RunsOwnedBy(ctx context.Context, owner string, since time.Time) ([]string, error) {
	if err := ls.runOwnerTable(ctx); err != nil {
		return nil, err
	}
	rows, err := ls.db.QueryContext(ctx, `SELECT run_id FROM run_owners WHERE owner_node_id = ? AND created_at >= ? ORDER BY created_at`, owner, since.UTC())
	if err != nil {
		return nil, fmt.Errorf("runs owned: %w", err)
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		out = append(out, id)
	}
	return out, rows.Err()
}
