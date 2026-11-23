package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// ── WHAT A PAUSED RUN IS ASKING ─────────────────────────────────────────────
//
// An approval used to be only an id and a link: the question itself lived in
// the external approval service (hax), so a door with no hax key could say
// nothing better than "the run asks for approval". A person must be able to
// read what they approve, so the control plane now keeps the question beside
// the approval — its kind, a one-line title, the full markdown body (a plan)
// and the option labels — keyed by (execution_id, approval_request_id), one
// row per round. It lives in its own table, created with the schema alongside
// the relay's, so the workflow_executions column lists and their migrations
// stay as they were.

var approvalQuestionsSchemaTable = guardedSchemaTable{
	name: "execution_approval_questions",
	ddl: `CREATE TABLE IF NOT EXISTS execution_approval_questions (
		execution_id TEXT NOT NULL,
		approval_request_id TEXT NOT NULL,
		kind TEXT NOT NULL,
		question TEXT,
		body TEXT,
		options TEXT,
		recorded_at TIMESTAMP NOT NULL,
		PRIMARY KEY (execution_id, approval_request_id))`,
	existsQuery: `SELECT 1 FROM execution_approval_questions LIMIT 0`,
}

// approvalQuestionTable ensures execution_approval_questions exists. THESE
// TABLES ARE CREATED WITH THE SCHEMA, NOT ON FIRST USE; this guard catches
// stores built without autoMigrateSchema, serializes creation per db, and
// caches only success — A FAILED CREATE IS NEVER CACHED.
func (ls *LocalStorage) approvalQuestionTable(ctx context.Context) error {
	return ls.ensureSchemaTable(ctx, approvalQuestionsSchemaTable)
}

// PutApprovalQuestion keeps q for one approval round. A round asked twice
// keeps the newer words.
func (ls *LocalStorage) PutApprovalQuestion(ctx context.Context, q types.ApprovalQuestion) error {
	if err := ls.approvalQuestionTable(ctx); err != nil {
		return err
	}
	opts, err := json.Marshal(q.Options)
	if err != nil {
		return err
	}
	ls.mu.Lock()
	defer ls.mu.Unlock()
	_, err = ls.db.ExecContext(ctx, `INSERT INTO execution_approval_questions
		(execution_id, approval_request_id, kind, question, body, options, recorded_at)
		VALUES (?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT (execution_id, approval_request_id) DO UPDATE SET
		kind = excluded.kind, question = excluded.question, body = excluded.body, options = excluded.options`,
		q.ExecutionID, q.ApprovalRequestID, q.Kind, q.Question, q.Body, string(opts), time.Now().UTC())
	if err != nil {
		return fmt.Errorf("approval question put: %w", err)
	}
	return nil
}

// GetApprovalQuestion is the question asked in one round, nil when that
// round carried none (an old SDK, or hax holding the form).
func (ls *LocalStorage) GetApprovalQuestion(ctx context.Context, executionID, approvalRequestID string) (*types.ApprovalQuestion, error) {
	if err := ls.approvalQuestionTable(ctx); err != nil {
		return nil, err
	}
	var q types.ApprovalQuestion
	var question, body, opts sql.NullString
	err := ls.db.QueryRowContext(ctx, `SELECT execution_id, approval_request_id, kind, question, body, options
		FROM execution_approval_questions WHERE execution_id = ? AND approval_request_id = ?`,
		executionID, approvalRequestID).Scan(&q.ExecutionID, &q.ApprovalRequestID, &q.Kind, &question, &body, &opts)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("approval question get: %w", err)
	}
	q.Question, q.Body = question.String, body.String
	if opts.Valid && opts.String != "" && opts.String != "null" {
		_ = json.Unmarshal([]byte(opts.String), &q.Options)
	}
	return &q, nil
}
