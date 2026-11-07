package storage

import (
	"context"
	"encoding/json"
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
