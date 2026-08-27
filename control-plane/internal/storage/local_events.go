package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// TransactionalStorage methods (not fully implemented for local storage yet)
func (ls *LocalStorage) BeginTransaction() (Transaction, error) {
	return nil, fmt.Errorf("transactions not fully implemented for LocalStorage")
}

// StoreWorkflowExecutionEvent inserts an immutable execution event into SQLite.
func (ls *LocalStorage) StoreWorkflowExecutionEvent(ctx context.Context, event *types.WorkflowExecutionEvent) error {
	if event == nil {
		return fmt.Errorf("workflow execution event is nil")
	}

	// Use retry logic for database lock errors
	return ls.retryDatabaseOperation(ctx, event.ExecutionID, func() error {
		tx, err := ls.db.BeginTx(ctx, nil)
		if err != nil {
			return fmt.Errorf("failed to begin transaction: %w", err)
		}
		defer rollbackTx(tx, "StoreWorkflowExecutionEvent:"+event.ExecutionID)

		if err := ls.storeWorkflowExecutionEventTx(ctx, tx, event); err != nil {
			return err
		}

		if err := tx.Commit(); err != nil {
			return fmt.Errorf("failed to commit event transaction: %w", err)
		}

		eventCopy := *event
		ls.workflowExecutionEventBus.Publish(&eventCopy)

		return nil
	})
}

// storeWorkflowExecutionEventTx inserts an execution event within an existing transaction.
// This allows atomic operations where the event storage and execution update happen together.
func (ls *LocalStorage) storeWorkflowExecutionEventTx(ctx context.Context, tx DBTX, event *types.WorkflowExecutionEvent) error {
	payload := string(event.Payload)
	if len(event.Payload) == 0 {
		payload = "{}"
	}

	// Persist recorded_at explicitly: the GORM model's autoCreateTime does
	// not apply to this raw INSERT, and a NULL recorded_at breaks listing.
	if event.RecordedAt.IsZero() {
		event.RecordedAt = time.Now().UTC()
	}

	query := `
		INSERT INTO workflow_execution_events (
			execution_id, workflow_id, run_id, parent_execution_id, sequence, previous_sequence,
			event_type, status, status_reason, payload, emitted_at, recorded_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

	result, err := tx.ExecContext(ctx, query,
		event.ExecutionID,
		event.WorkflowID,
		event.RunID,
		event.ParentExecutionID,
		event.Sequence,
		event.PreviousSequence,
		event.EventType,
		event.Status,
		event.StatusReason,
		payload,
		event.EmittedAt,
		event.RecordedAt,
	)
	if err != nil {
		return fmt.Errorf("failed to insert workflow execution event: %w", err)
	}

	if id, err := result.LastInsertId(); err == nil {
		event.EventID = id
	}

	return nil
}

// ListWorkflowExecutionEvents retrieves execution events ordered by sequence.
func (ls *LocalStorage) ListWorkflowExecutionEvents(ctx context.Context, executionID string, afterSeq *int64, limit int) ([]*types.WorkflowExecutionEvent, error) {
	query := `
		SELECT event_id, execution_id, workflow_id, run_id, parent_execution_id, sequence, previous_sequence,
		       event_type, status, status_reason, payload, emitted_at, recorded_at
		FROM workflow_execution_events
		WHERE execution_id = ?`
	args := []interface{}{executionID}

	if afterSeq != nil {
		query += " AND sequence > ?"
		args = append(args, *afterSeq)
	}

	query += " ORDER BY sequence ASC"
	if limit > 0 {
		query += fmt.Sprintf(" LIMIT %d", limit)
	}

	rows, err := ls.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query workflow execution events: %w", err)
	}
	defer rows.Close()

	var events []*types.WorkflowExecutionEvent
	for rows.Next() {
		evt := &types.WorkflowExecutionEvent{}
		var runID sql.NullString
		var parentID sql.NullString
		var status sql.NullString
		var statusReason sql.NullString
		var payload sql.NullString
		var recordedAt sql.NullTime

		if err := rows.Scan(
			&evt.EventID,
			&evt.ExecutionID,
			&evt.WorkflowID,
			&runID,
			&parentID,
			&evt.Sequence,
			&evt.PreviousSequence,
			&evt.EventType,
			&status,
			&statusReason,
			&payload,
			&evt.EmittedAt,
			&recordedAt,
		); err != nil {
			return nil, fmt.Errorf("failed to scan workflow execution event: %w", err)
		}

		// Rows written before recorded_at was persisted have NULL here;
		// fall back to emitted_at rather than failing the whole listing.
		if recordedAt.Valid {
			evt.RecordedAt = recordedAt.Time
		} else {
			evt.RecordedAt = evt.EmittedAt
		}

		if runID.Valid {
			evt.RunID = &runID.String
		}
		if parentID.Valid {
			evt.ParentExecutionID = &parentID.String
		}
		if status.Valid {
			value := status.String
			evt.Status = &value
		}
		if statusReason.Valid {
			value := statusReason.String
			evt.StatusReason = &value
		}
		if payload.Valid {
			evt.Payload = json.RawMessage(payload.String)
		} else {
			evt.Payload = json.RawMessage("{}")
		}

		events = append(events, evt)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating workflow execution events: %w", err)
	}

	return events, nil
}

// StoreExecutionLogEntry inserts a structured execution log entry and publishes it to subscribers.
func (ls *LocalStorage) StoreExecutionLogEntry(ctx context.Context, entry *types.ExecutionLogEntry) error {
	if entry == nil {
		return fmt.Errorf("execution log entry is nil")
	}
	if strings.TrimSpace(entry.ExecutionID) == "" {
		return fmt.Errorf("execution_id is required")
	}
	if strings.TrimSpace(entry.WorkflowID) == "" {
		entry.WorkflowID = entry.ExecutionID
	}
	if strings.TrimSpace(entry.Level) == "" {
		entry.Level = "info"
	}
	if strings.TrimSpace(entry.Source) == "" {
		entry.Source = "sdk.logger"
	}
	if entry.EmittedAt.IsZero() {
		entry.EmittedAt = time.Now().UTC()
	}
	ls.mu.Lock()
	defer ls.mu.Unlock()

	return ls.retryDatabaseOperation(ctx, entry.ExecutionID, func() error {
		tx, err := ls.db.BeginTx(ctx, nil)
		if err != nil {
			return fmt.Errorf("failed to begin execution log transaction: %w", err)
		}
		defer rollbackTx(tx, "StoreExecutionLogEntry:"+entry.ExecutionID)

		if err := ls.storeExecutionLogEntryTx(ctx, tx, entry); err != nil {
			return err
		}

		if err := tx.Commit(); err != nil {
			return fmt.Errorf("failed to commit execution log transaction: %w", err)
		}

		entryCopy := *entry
		ls.executionLogEventBus.Publish(&entryCopy)
		return nil
	})
}

// StoreExecutionLogEntries atomically stores a batch of structured execution logs for one execution.
func (ls *LocalStorage) StoreExecutionLogEntries(ctx context.Context, executionID string, entries []*types.ExecutionLogEntry) error {
	if len(entries) == 0 {
		return nil
	}

	ls.mu.Lock()
	defer ls.mu.Unlock()

	return ls.retryDatabaseOperation(ctx, executionID, func() error {
		tx, err := ls.db.BeginTx(ctx, nil)
		if err != nil {
			return fmt.Errorf("failed to begin execution log batch transaction: %w", err)
		}
		defer rollbackTx(tx, "StoreExecutionLogEntries:"+executionID)

		for _, entry := range entries {
			if entry == nil {
				continue
			}
			if err := ls.storeExecutionLogEntryTx(ctx, tx, entry); err != nil {
				return err
			}
		}

		if err := tx.Commit(); err != nil {
			return fmt.Errorf("failed to commit execution log batch transaction: %w", err)
		}

		for _, entry := range entries {
			if entry == nil {
				continue
			}
			entryCopy := *entry
			ls.executionLogEventBus.Publish(&entryCopy)
		}
		return nil
	})
}

func (ls *LocalStorage) storeExecutionLogEntryTx(ctx context.Context, tx DBTX, entry *types.ExecutionLogEntry) error {
	var nextSeq int64 = 1
	if err := tx.QueryRowContext(ctx,
		`SELECT COALESCE(MAX(sequence), 0) + 1 FROM execution_logs WHERE execution_id = ?`,
		entry.ExecutionID,
	).Scan(&nextSeq); err != nil {
		return fmt.Errorf("failed to compute execution log sequence: %w", err)
	}
	entry.Sequence = nextSeq

	attributes := "{}"
	if len(entry.Attributes) > 0 {
		attributes = string(entry.Attributes)
	}

	// Persist recorded_at explicitly: the GORM model's autoCreateTime does
	// not apply to this raw INSERT, so omitting the column stores NULL and
	// every read silently takes the legacy emitted_at fallback.
	if entry.RecordedAt.IsZero() {
		entry.RecordedAt = time.Now().UTC()
	}

	query := `
		INSERT INTO execution_logs (
			execution_id, workflow_id, run_id, root_workflow_id, parent_execution_id, sequence,
			agent_node_id, reasoner_id, level, source, event_type, message, attributes,
			system_generated, sdk_language, attempt, span_id, step_id, error_category, emitted_at,
			recorded_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

	result, err := tx.ExecContext(ctx, query,
		entry.ExecutionID,
		entry.WorkflowID,
		entry.RunID,
		entry.RootWorkflowID,
		entry.ParentExecutionID,
		entry.Sequence,
		entry.AgentNodeID,
		entry.ReasonerID,
		entry.Level,
		entry.Source,
		entry.EventType,
		entry.Message,
		attributes,
		entry.SystemGenerated,
		entry.SDKLanguage,
		entry.Attempt,
		entry.SpanID,
		entry.StepID,
		entry.ErrorCategory,
		entry.EmittedAt,
		entry.RecordedAt,
	)
	if err != nil {
		return fmt.Errorf("failed to insert execution log entry: %w", err)
	}
	if id, err := result.LastInsertId(); err == nil {
		entry.EventID = id
	}
	return nil
}

// ListExecutionLogEntries retrieves structured execution logs ordered by sequence.
func (ls *LocalStorage) ListExecutionLogEntries(ctx context.Context, executionID string, afterSeq *int64, limit int, levels []string, nodeIDs []string, sources []string, query string) ([]*types.ExecutionLogEntry, error) {
	baseQuery := `
		SELECT event_id, execution_id, workflow_id, run_id, root_workflow_id, parent_execution_id, sequence,
		       agent_node_id, reasoner_id, level, source, event_type, message, attributes, system_generated,
		       sdk_language, attempt, span_id, step_id, error_category, emitted_at, recorded_at
		FROM execution_logs
		WHERE execution_id = ?`
	args := []interface{}{executionID}

	appendIn := func(column string, values []string) {
		if len(values) == 0 {
			return
		}
		holders := make([]string, 0, len(values))
		for _, value := range values {
			if strings.TrimSpace(value) == "" {
				continue
			}
			holders = append(holders, "?")
			args = append(args, value)
		}
		if len(holders) > 0 {
			baseQuery += " AND " + column + " IN (" + strings.Join(holders, ",") + ")"
		}
	}

	if afterSeq != nil {
		baseQuery += " AND sequence > ?"
		args = append(args, *afterSeq)
	}
	appendIn("level", levels)
	appendIn("agent_node_id", nodeIDs)
	appendIn("source", sources)
	if trimmed := strings.TrimSpace(query); trimmed != "" {
		baseQuery += " AND (message LIKE ? OR attributes LIKE ?)"
		like := "%" + trimmed + "%"
		args = append(args, like, like)
	}

	descendingTail := afterSeq == nil && limit > 0
	if descendingTail {
		baseQuery += " ORDER BY sequence DESC"
		baseQuery += fmt.Sprintf(" LIMIT %d", limit)
	} else {
		baseQuery += " ORDER BY sequence ASC"
		if limit > 0 {
			baseQuery += fmt.Sprintf(" LIMIT %d", limit)
		}
	}

	rows, err := ls.db.QueryContext(ctx, baseQuery, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query execution logs: %w", err)
	}
	defer rows.Close()

	var entries []*types.ExecutionLogEntry
	for rows.Next() {
		entry := &types.ExecutionLogEntry{}
		var runID, rootWorkflowID, parentExecutionID, reasonerID, eventType, sdkLanguage, spanID, stepID, errorCategory sql.NullString
		var attributes sql.NullString
		var attempt sql.NullInt64
		var emittedAt sql.NullTime
		var recordedAt sql.NullTime
		if err := rows.Scan(
			&entry.EventID,
			&entry.ExecutionID,
			&entry.WorkflowID,
			&runID,
			&rootWorkflowID,
			&parentExecutionID,
			&entry.Sequence,
			&entry.AgentNodeID,
			&reasonerID,
			&entry.Level,
			&entry.Source,
			&eventType,
			&entry.Message,
			&attributes,
			&entry.SystemGenerated,
			&sdkLanguage,
			&attempt,
			&spanID,
			&stepID,
			&errorCategory,
			&emittedAt,
			&recordedAt,
		); err != nil {
			return nil, fmt.Errorf("failed to scan execution log entry: %w", err)
		}

		if runID.Valid {
			entry.RunID = &runID.String
		}
		if rootWorkflowID.Valid {
			entry.RootWorkflowID = &rootWorkflowID.String
		}
		if parentExecutionID.Valid {
			entry.ParentExecutionID = &parentExecutionID.String
		}
		if reasonerID.Valid {
			entry.ReasonerID = &reasonerID.String
		}
		if eventType.Valid {
			entry.EventType = &eventType.String
		}
		if sdkLanguage.Valid {
			entry.SDKLanguage = &sdkLanguage.String
		}
		if spanID.Valid {
			entry.SpanID = &spanID.String
		}
		if stepID.Valid {
			entry.StepID = &stepID.String
		}
		if errorCategory.Valid {
			entry.ErrorCategory = &errorCategory.String
		}
		if attempt.Valid {
			value := int(attempt.Int64)
			entry.Attempt = &value
		}
		if emittedAt.Valid {
			entry.EmittedAt = emittedAt.Time
		}
		if recordedAt.Valid {
			entry.RecordedAt = recordedAt.Time
		} else {
			entry.RecordedAt = entry.EmittedAt
		}
		if attributes.Valid {
			entry.Attributes = json.RawMessage(attributes.String)
		} else {
			entry.Attributes = json.RawMessage("{}")
		}

		entries = append(entries, entry)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating execution logs: %w", err)
	}
	if descendingTail {
		for left, right := 0, len(entries)-1; left < right; left, right = left+1, right-1 {
			entries[left], entries[right] = entries[right], entries[left]
		}
	}
	return entries, nil
}

// PruneExecutionLogEntries trims old or excessive execution logs for a single execution.
func (ls *LocalStorage) PruneExecutionLogEntries(ctx context.Context, executionID string, maxEntries int, olderThan time.Time) error {
	if strings.TrimSpace(executionID) == "" {
		return nil
	}
	if !olderThan.IsZero() {
		if _, err := ls.db.ExecContext(ctx,
			`DELETE FROM execution_logs WHERE execution_id = ? AND emitted_at < ?`,
			executionID, olderThan,
		); err != nil {
			return fmt.Errorf("failed to prune execution logs by age: %w", err)
		}
	}
	if maxEntries > 0 {
		if _, err := ls.db.ExecContext(ctx, `
			DELETE FROM execution_logs
			WHERE execution_id = ?
			  AND event_id NOT IN (
			    SELECT event_id FROM execution_logs
			    WHERE execution_id = ?
			    ORDER BY sequence DESC
			    LIMIT ?
			  )`,
			executionID, executionID, maxEntries,
		); err != nil {
			return fmt.Errorf("failed to prune execution logs by count: %w", err)
		}
	}
	return nil
}

// StoreExecutionWebhookEvent records webhook delivery attempts for SQLite deployments.
func (ls *LocalStorage) StoreExecutionWebhookEvent(ctx context.Context, event *types.ExecutionWebhookEvent) error {
	if event == nil {
		return fmt.Errorf("execution webhook event is nil")
	}

	payload := interface{}(nil)
	if len(event.Payload) > 0 {
		payload = string(event.Payload)
	}

	query := `
		INSERT INTO execution_webhook_events (
			execution_id, event_type, status, http_status, payload, response_body, error_message, created_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)`

	_, err := ls.db.ExecContext(ctx, query,
		event.ExecutionID,
		event.EventType,
		event.Status,
		event.HTTPStatus,
		payload,
		event.ResponseBody,
		event.ErrorMessage,
	)
	return err
}

// ListExecutionWebhookEvents returns webhook attempts ordered by creation time.
func (ls *LocalStorage) ListExecutionWebhookEvents(ctx context.Context, executionID string) ([]*types.ExecutionWebhookEvent, error) {
	query := `
		SELECT id, execution_id, event_type, status, http_status, payload, response_body, error_message, created_at
		FROM execution_webhook_events
		WHERE execution_id = ?
		ORDER BY created_at ASC, id ASC`

	rows, err := ls.db.QueryContext(ctx, query, executionID)
	if err != nil {
		return nil, fmt.Errorf("failed to query execution webhook events: %w", err)
	}
	defer rows.Close()

	var events []*types.ExecutionWebhookEvent
	for rows.Next() {
		evt := &types.ExecutionWebhookEvent{}
		var payload sql.NullString
		var response sql.NullString
		var errMsg sql.NullString
		var status sql.NullInt64

		if err := rows.Scan(
			&evt.ID,
			&evt.ExecutionID,
			&evt.EventType,
			&evt.Status,
			&status,
			&payload,
			&response,
			&errMsg,
			&evt.CreatedAt,
		); err != nil {
			return nil, fmt.Errorf("failed to scan execution webhook event: %w", err)
		}

		if status.Valid {
			s := int(status.Int64)
			evt.HTTPStatus = &s
		}
		if payload.Valid {
			evt.Payload = json.RawMessage(payload.String)
		} else {
			evt.Payload = json.RawMessage("{}")
		}
		if response.Valid {
			value := response.String
			evt.ResponseBody = &value
		}
		if errMsg.Valid {
			value := errMsg.String
			evt.ErrorMessage = &value
		}

		events = append(events, evt)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating execution webhook events: %w", err)
	}

	return events, nil
}

// ListExecutionWebhookEventsBatch fetches webhook events for multiple executions in a single query.
func (ls *LocalStorage) ListExecutionWebhookEventsBatch(ctx context.Context, executionIDs []string) (map[string][]*types.ExecutionWebhookEvent, error) {
	results := make(map[string][]*types.ExecutionWebhookEvent)
	if len(executionIDs) == 0 {
		return results, nil
	}

	unique := make([]string, 0, len(executionIDs))
	seen := make(map[string]struct{}, len(executionIDs))
	for _, id := range executionIDs {
		trimmed := strings.TrimSpace(id)
		if trimmed == "" {
			continue
		}
		if _, ok := seen[trimmed]; ok {
			continue
		}
		seen[trimmed] = struct{}{}
		unique = append(unique, trimmed)
	}
	if len(unique) == 0 {
		return results, nil
	}

	placeholders := make([]string, len(unique))
	args := make([]interface{}, len(unique))
	for i, id := range unique {
		placeholders[i] = "?"
		args[i] = id
	}

	query := fmt.Sprintf(`
		SELECT execution_id, id, event_type, status, http_status, payload, response_body, error_message, created_at
		FROM execution_webhook_events
		WHERE execution_id IN (%s)
		ORDER BY execution_id ASC, created_at ASC, id ASC`, strings.Join(placeholders, ","))

	rows, err := ls.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query batch webhook events: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		evt := &types.ExecutionWebhookEvent{}
		var payload sql.NullString
		var response sql.NullString
		var errMsg sql.NullString
		var status sql.NullInt64
		if err := rows.Scan(
			&evt.ExecutionID,
			&evt.ID,
			&evt.EventType,
			&evt.Status,
			&status,
			&payload,
			&response,
			&errMsg,
			&evt.CreatedAt,
		); err != nil {
			return nil, fmt.Errorf("failed to scan batch webhook event: %w", err)
		}

		if status.Valid {
			s := int(status.Int64)
			evt.HTTPStatus = &s
		}
		if payload.Valid {
			evt.Payload = json.RawMessage(payload.String)
		} else {
			evt.Payload = json.RawMessage("{}")
		}
		if response.Valid {
			value := response.String
			evt.ResponseBody = &value
		}
		if errMsg.Valid {
			value := errMsg.String
			evt.ErrorMessage = &value
		}

		results[evt.ExecutionID] = append(results[evt.ExecutionID], evt)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating batch webhook events: %w", err)
	}

	return results, nil
}

// =============================================================================
// DID Document Operations (did:web Resolution)
// =============================================================================

// StoreDIDDocument stores a DID document record.
func (ls *LocalStorage) StoreDIDDocument(ctx context.Context, record *types.DIDDocumentRecord) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store DID document: %w", err)
	}

	query := `
		INSERT INTO did_documents (
			did, agent_id, did_document, public_key_jwk, revoked_at, created_at, updated_at
		) VALUES (?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(did) DO UPDATE SET
			agent_id = excluded.agent_id,
			did_document = excluded.did_document,
			public_key_jwk = excluded.public_key_jwk,
			updated_at = excluded.updated_at`

	_, err := ls.db.ExecContext(ctx, query,
		record.DID, record.AgentID, record.DIDDocument, record.PublicKeyJWK,
		record.RevokedAt, record.CreatedAt, record.UpdatedAt,
	)
	if err != nil {
		return fmt.Errorf("failed to store DID document: %w", err)
	}

	return nil
}

// GetDIDDocument retrieves a DID document by its DID.
func (ls *LocalStorage) GetDIDDocument(ctx context.Context, did string) (*types.DIDDocumentRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get DID document: %w", err)
	}

	query := `
		SELECT did, agent_id, did_document, public_key_jwk, revoked_at, created_at, updated_at
		FROM did_documents WHERE did = ?`

	row := ls.db.QueryRowContext(ctx, query, did)

	record := &types.DIDDocumentRecord{}
	var revokedAt sql.NullTime

	err := row.Scan(
		&record.DID, &record.AgentID, &record.DIDDocument, &record.PublicKeyJWK,
		&revokedAt, &record.CreatedAt, &record.UpdatedAt,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("DID document not found: %s", did)
		}
		return nil, fmt.Errorf("failed to get DID document: %w", err)
	}

	if revokedAt.Valid {
		record.RevokedAt = &revokedAt.Time
	}

	return record, nil
}

// GetDIDDocumentByAgentID retrieves a DID document by agent ID.
func (ls *LocalStorage) GetDIDDocumentByAgentID(ctx context.Context, agentID string) (*types.DIDDocumentRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get DID document by agent ID: %w", err)
	}

	query := `
		SELECT did, agent_id, did_document, public_key_jwk, revoked_at, created_at, updated_at
		FROM did_documents WHERE agent_id = ? AND revoked_at IS NULL
		ORDER BY created_at DESC LIMIT 1`

	row := ls.db.QueryRowContext(ctx, query, agentID)

	record := &types.DIDDocumentRecord{}
	var revokedAt sql.NullTime

	err := row.Scan(
		&record.DID, &record.AgentID, &record.DIDDocument, &record.PublicKeyJWK,
		&revokedAt, &record.CreatedAt, &record.UpdatedAt,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("DID document not found for agent: %s", agentID)
		}
		return nil, fmt.Errorf("failed to get DID document by agent ID: %w", err)
	}

	if revokedAt.Valid {
		record.RevokedAt = &revokedAt.Time
	}

	return record, nil
}

// RevokeDIDDocument revokes a DID document by setting its revoked_at timestamp.
func (ls *LocalStorage) RevokeDIDDocument(ctx context.Context, did string) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during revoke DID document: %w", err)
	}

	query := `UPDATE did_documents SET revoked_at = ?, updated_at = ? WHERE did = ?`

	now := time.Now()
	result, err := ls.db.ExecContext(ctx, query, now, now, did)
	if err != nil {
		return fmt.Errorf("failed to revoke DID document: %w", err)
	}

	rows, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}
	if rows == 0 {
		return fmt.Errorf("DID document not found: %s", did)
	}

	return nil
}

// ListDIDDocuments lists all DID documents.
func (ls *LocalStorage) ListDIDDocuments(ctx context.Context) ([]*types.DIDDocumentRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during list DID documents: %w", err)
	}

	query := `
		SELECT did, agent_id, did_document, public_key_jwk, revoked_at, created_at, updated_at
		FROM did_documents ORDER BY created_at DESC`

	rows, err := ls.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to list DID documents: %w", err)
	}
	defer rows.Close()

	var records []*types.DIDDocumentRecord
	for rows.Next() {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during scan: %w", err)
		}

		record := &types.DIDDocumentRecord{}
		var revokedAt sql.NullTime

		err := rows.Scan(
			&record.DID, &record.AgentID, &record.DIDDocument, &record.PublicKeyJWK,
			&revokedAt, &record.CreatedAt, &record.UpdatedAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan DID document: %w", err)
		}

		if revokedAt.Valid {
			record.RevokedAt = &revokedAt.Time
		}

		records = append(records, record)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating DID documents: %w", err)
	}

	return records, nil
}

// ListAgentsByLifecycleStatus lists agents filtered by lifecycle status.
func (ls *LocalStorage) ListAgentsByLifecycleStatus(ctx context.Context, status types.AgentLifecycleStatus) ([]*types.AgentNode, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during list agents by lifecycle status: %w", err)
	}

	query := `
		SELECT
			id, version, group_id, team_id, base_url, traffic_weight, deployment_type, invocation_url, reasoners, skills,
			communication_config, health_status, lifecycle_status, last_heartbeat,
			registered_at, features, metadata, proposed_tags, approved_tags, COALESCE(instance_id, '')
		FROM agent_nodes WHERE lifecycle_status = ? ORDER BY registered_at DESC`

	rows, err := ls.db.QueryContext(ctx, query, string(status))
	if err != nil {
		return nil, fmt.Errorf("failed to list agents by lifecycle status: %w", err)
	}
	defer rows.Close()

	return ls.scanAgentNodes(ctx, rows)
}

// reconstructAgentLevelTags ensures agent-level ProposedTags and ApprovedTags
// are populated. If the dedicated DB columns were empty (e.g., on older records),
// it reconstructs them from per-reasoner/per-skill fields as a fallback.
func reconstructAgentLevelTags(agent *types.AgentNode) {
	// Only reconstruct if DB columns were empty
	if len(agent.ApprovedTags) == 0 {
		seen := make(map[string]struct{})
		for _, r := range agent.Reasoners {
			for _, t := range r.ApprovedTags {
				if _, exists := seen[t]; !exists {
					seen[t] = struct{}{}
					agent.ApprovedTags = append(agent.ApprovedTags, t)
				}
			}
		}
		for _, sk := range agent.Skills {
			for _, t := range sk.ApprovedTags {
				if _, exists := seen[t]; !exists {
					seen[t] = struct{}{}
					agent.ApprovedTags = append(agent.ApprovedTags, t)
				}
			}
		}
		types.HydrateAgentSessions(agent)
		for _, session := range agent.Sessions {
			for _, t := range session.ApprovedTags {
				if _, exists := seen[t]; !exists {
					seen[t] = struct{}{}
					agent.ApprovedTags = append(agent.ApprovedTags, t)
				}
			}
		}
	}

	if len(agent.ProposedTags) == 0 {
		proposedSeen := make(map[string]struct{})
		for _, r := range agent.Reasoners {
			source := r.ProposedTags
			if len(source) == 0 {
				source = r.Tags
			}
			for _, t := range source {
				if _, exists := proposedSeen[t]; !exists {
					proposedSeen[t] = struct{}{}
					agent.ProposedTags = append(agent.ProposedTags, t)
				}
			}
		}
		for _, sk := range agent.Skills {
			source := sk.ProposedTags
			if len(source) == 0 {
				source = sk.Tags
			}
			for _, t := range source {
				if _, exists := proposedSeen[t]; !exists {
					proposedSeen[t] = struct{}{}
					agent.ProposedTags = append(agent.ProposedTags, t)
				}
			}
		}
		types.HydrateAgentSessions(agent)
		for _, session := range agent.Sessions {
			source := session.ProposedTags
			if len(source) == 0 {
				source = session.Tags
			}
			for _, t := range source {
				if _, exists := proposedSeen[t]; !exists {
					proposedSeen[t] = struct{}{}
					agent.ProposedTags = append(agent.ProposedTags, t)
				}
			}
		}
	}
}

// ============================================================================
// Access Policy Storage
// ============================================================================

// GetAccessPolicies retrieves all enabled access policies, sorted by priority descending.
func (ls *LocalStorage) GetAccessPolicies(ctx context.Context) ([]*types.AccessPolicy, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get access policies: %w", err)
	}

	query := `
		SELECT id, name, caller_tags, target_tags, allow_functions, deny_functions,
		       constraints, action, priority, enabled, description, created_at, updated_at
		FROM access_policies WHERE enabled = true ORDER BY priority DESC, created_at DESC`

	rows, err := ls.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to get access policies: %w", err)
	}
	defer rows.Close()

	var policies []*types.AccessPolicy
	for rows.Next() {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during scan: %w", err)
		}

		policy, err := scanAccessPolicy(rows)
		if err != nil {
			return nil, err
		}
		policies = append(policies, policy)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating access policies: %w", err)
	}

	return policies, nil
}

// GetAccessPolicyByID retrieves a single access policy by its ID.
func (ls *LocalStorage) GetAccessPolicyByID(ctx context.Context, id int64) (*types.AccessPolicy, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get access policy: %w", err)
	}

	query := `
		SELECT id, name, caller_tags, target_tags, allow_functions, deny_functions,
		       constraints, action, priority, enabled, description, created_at, updated_at
		FROM access_policies WHERE id = ?`

	row := ls.db.QueryRowContext(ctx, query, id)

	policy := &types.AccessPolicy{}
	var callerTagsJSON, targetTagsJSON, allowFuncsJSON, denyFuncsJSON, constraintsJSON string
	var description sql.NullString

	err := row.Scan(
		&policy.ID, &policy.Name, &callerTagsJSON, &targetTagsJSON,
		&allowFuncsJSON, &denyFuncsJSON, &constraintsJSON,
		&policy.Action, &policy.Priority, &policy.Enabled, &description,
		&policy.CreatedAt, &policy.UpdatedAt,
	)
	if err != nil {
		return nil, fmt.Errorf("access policy with ID %d not found: %w", id, err)
	}

	if description.Valid {
		policy.Description = &description.String
	}
	if err := unmarshalAccessPolicyJSON(policy, callerTagsJSON, targetTagsJSON, allowFuncsJSON, denyFuncsJSON, constraintsJSON); err != nil {
		return nil, fmt.Errorf("failed to unmarshal access policy %d: %w", id, err)
	}

	return policy, nil
}

// CreateAccessPolicy creates a new access policy.
func (ls *LocalStorage) CreateAccessPolicy(ctx context.Context, policy *types.AccessPolicy) error {
	if ls.mode == "postgres" {
		return ls.createAccessPolicyPostgres(ctx, policy)
	}

	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during create access policy: %w", err)
	}

	callerTagsJSON, targetTagsJSON, allowFuncsJSON, denyFuncsJSON, constraintsJSON, err := marshalAccessPolicyJSON(policy)
	if err != nil {
		return fmt.Errorf("failed to marshal access policy fields: %w", err)
	}

	query := `
		INSERT INTO access_policies (
			name, caller_tags, target_tags, allow_functions, deny_functions,
			constraints, action, priority, enabled, description, created_at, updated_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

	result, err := ls.db.ExecContext(ctx, query,
		policy.Name, callerTagsJSON, targetTagsJSON,
		allowFuncsJSON, denyFuncsJSON, constraintsJSON,
		policy.Action, policy.Priority, policy.Enabled, policy.Description,
		policy.CreatedAt, policy.UpdatedAt,
	)
	if err != nil {
		if strings.Contains(err.Error(), "UNIQUE constraint") {
			return fmt.Errorf("access policy with name %q already exists", policy.Name)
		}
		return fmt.Errorf("failed to create access policy: %w", err)
	}

	id, err := result.LastInsertId()
	if err == nil {
		policy.ID = id
	}

	return nil
}

// createAccessPolicyPostgres creates an access policy using PostgreSQL's RETURNING clause.
func (ls *LocalStorage) createAccessPolicyPostgres(ctx context.Context, policy *types.AccessPolicy) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during create access policy: %w", err)
	}

	callerTagsJSON, targetTagsJSON, allowFuncsJSON, denyFuncsJSON, constraintsJSON, err := marshalAccessPolicyJSON(policy)
	if err != nil {
		return fmt.Errorf("failed to marshal access policy fields: %w", err)
	}

	query := `
		INSERT INTO access_policies (
			name, caller_tags, target_tags, allow_functions, deny_functions,
			constraints, action, priority, enabled, description, created_at, updated_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
		RETURNING id`

	row := ls.db.DB.QueryRowContext(ctx, query,
		policy.Name, callerTagsJSON, targetTagsJSON,
		allowFuncsJSON, denyFuncsJSON, constraintsJSON,
		policy.Action, policy.Priority, policy.Enabled, policy.Description,
		policy.CreatedAt, policy.UpdatedAt,
	)

	if err := row.Scan(&policy.ID); err != nil {
		if strings.Contains(err.Error(), "duplicate key") {
			return fmt.Errorf("access policy with name %q already exists", policy.Name)
		}
		return fmt.Errorf("failed to create access policy: %w", err)
	}

	return nil
}

// UpdateAccessPolicy updates an existing access policy.
func (ls *LocalStorage) UpdateAccessPolicy(ctx context.Context, policy *types.AccessPolicy) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during update access policy: %w", err)
	}

	callerTagsJSON, targetTagsJSON, allowFuncsJSON, denyFuncsJSON, constraintsJSON, err := marshalAccessPolicyJSON(policy)
	if err != nil {
		return fmt.Errorf("failed to marshal access policy fields: %w", err)
	}

	query := `
		UPDATE access_policies SET
			name = ?, caller_tags = ?, target_tags = ?, allow_functions = ?,
			deny_functions = ?, constraints = ?, action = ?, priority = ?,
			enabled = ?, description = ?, updated_at = ?
		WHERE id = ?`

	result, err := ls.db.ExecContext(ctx, query,
		policy.Name, callerTagsJSON, targetTagsJSON,
		allowFuncsJSON, denyFuncsJSON, constraintsJSON,
		policy.Action, policy.Priority, policy.Enabled, policy.Description,
		policy.UpdatedAt, policy.ID,
	)
	if err != nil {
		return fmt.Errorf("failed to update access policy: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}
	if rowsAffected == 0 {
		return fmt.Errorf("access policy with ID %d not found", policy.ID)
	}

	return nil
}

// DeleteAccessPolicy deletes an access policy by ID.
func (ls *LocalStorage) DeleteAccessPolicy(ctx context.Context, id int64) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during delete access policy: %w", err)
	}

	query := `DELETE FROM access_policies WHERE id = ?`

	result, err := ls.db.ExecContext(ctx, query, id)
	if err != nil {
		return fmt.Errorf("failed to delete access policy: %w", err)
	}

	rows, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}
	if rows == 0 {
		return fmt.Errorf("access policy with ID %d not found", id)
	}

	return nil
}

// scanAccessPolicy scans a row into an AccessPolicy struct.
func scanAccessPolicy(rows *sql.Rows) (*types.AccessPolicy, error) {
	policy := &types.AccessPolicy{}
	var callerTagsJSON, targetTagsJSON, allowFuncsJSON, denyFuncsJSON, constraintsJSON string
	var description sql.NullString

	err := rows.Scan(
		&policy.ID, &policy.Name, &callerTagsJSON, &targetTagsJSON,
		&allowFuncsJSON, &denyFuncsJSON, &constraintsJSON,
		&policy.Action, &policy.Priority, &policy.Enabled, &description,
		&policy.CreatedAt, &policy.UpdatedAt,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to scan access policy: %w", err)
	}

	if description.Valid {
		policy.Description = &description.String
	}
	if err := unmarshalAccessPolicyJSON(policy, callerTagsJSON, targetTagsJSON, allowFuncsJSON, denyFuncsJSON, constraintsJSON); err != nil {
		return nil, fmt.Errorf("failed to unmarshal access policy %d: %w", policy.ID, err)
	}

	return policy, nil
}

// unmarshalAccessPolicyJSON populates the JSON fields of an AccessPolicy.
// Returns an error if any JSON field cannot be deserialized, preventing
// corrupted data from silently producing empty policy rules.
func unmarshalAccessPolicyJSON(policy *types.AccessPolicy, callerTags, targetTags, allowFuncs, denyFuncs, constraints string) error {
	if callerTags != "" {
		if err := json.Unmarshal([]byte(callerTags), &policy.CallerTags); err != nil {
			return fmt.Errorf("failed to unmarshal caller_tags: %w", err)
		}
	}
	if targetTags != "" {
		if err := json.Unmarshal([]byte(targetTags), &policy.TargetTags); err != nil {
			return fmt.Errorf("failed to unmarshal target_tags: %w", err)
		}
	}
	if allowFuncs != "" {
		if err := json.Unmarshal([]byte(allowFuncs), &policy.AllowFunctions); err != nil {
			return fmt.Errorf("failed to unmarshal allow_functions: %w", err)
		}
	}
	if denyFuncs != "" {
		if err := json.Unmarshal([]byte(denyFuncs), &policy.DenyFunctions); err != nil {
			return fmt.Errorf("failed to unmarshal deny_functions: %w", err)
		}
	}
	if constraints != "" {
		if err := json.Unmarshal([]byte(constraints), &policy.Constraints); err != nil {
			return fmt.Errorf("failed to unmarshal constraints: %w", err)
		}
	}
	return nil
}

// marshalAccessPolicyJSON serializes the JSON fields of an AccessPolicy for storage.
func marshalAccessPolicyJSON(policy *types.AccessPolicy) (callerTags, targetTags, allowFuncs, denyFuncs, constraints string, err error) {
	ct, err := json.Marshal(policy.CallerTags)
	if err != nil {
		return "", "", "", "", "", fmt.Errorf("caller_tags: %w", err)
	}
	tt, err := json.Marshal(policy.TargetTags)
	if err != nil {
		return "", "", "", "", "", fmt.Errorf("target_tags: %w", err)
	}
	af, err := json.Marshal(policy.AllowFunctions)
	if err != nil {
		return "", "", "", "", "", fmt.Errorf("allow_functions: %w", err)
	}
	df, err := json.Marshal(policy.DenyFunctions)
	if err != nil {
		return "", "", "", "", "", fmt.Errorf("deny_functions: %w", err)
	}
	cn, err := json.Marshal(policy.Constraints)
	if err != nil {
		return "", "", "", "", "", fmt.Errorf("constraints: %w", err)
	}
	return string(ct), string(tt), string(af), string(df), string(cn), nil
}

// ========== Agent Tag VC operations ==========

// StoreAgentTagVC stores or replaces an agent's tag VC.
func (ls *LocalStorage) StoreAgentTagVC(ctx context.Context, agentID, agentDID, vcID, vcDocument, signature string, issuedAt time.Time, expiresAt *time.Time) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store agent tag VC: %w", err)
	}

	query := `
		INSERT INTO agent_tag_vcs (agent_id, agent_did, vc_id, vc_document, signature, issued_at, expires_at, created_at, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(agent_id) DO UPDATE SET
			agent_did = excluded.agent_did,
			vc_id = excluded.vc_id,
			vc_document = excluded.vc_document,
			signature = excluded.signature,
			issued_at = excluded.issued_at,
			expires_at = excluded.expires_at,
			revoked_at = NULL,
			updated_at = excluded.updated_at`

	now := time.Now()
	_, err := ls.db.ExecContext(ctx, query, agentID, agentDID, vcID, vcDocument, signature, issuedAt, expiresAt, now, now)
	if err != nil {
		return fmt.Errorf("failed to store agent tag VC: %w", err)
	}
	return nil
}

// GetAgentTagVC retrieves an agent's tag VC record.
func (ls *LocalStorage) GetAgentTagVC(ctx context.Context, agentID string) (*types.AgentTagVCRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get agent tag VC: %w", err)
	}

	query := `
		SELECT id, agent_id, agent_did, vc_id, vc_document, signature, issued_at, expires_at, revoked_at
		FROM agent_tag_vcs WHERE agent_id = ?`

	row := ls.db.QueryRowContext(ctx, query, agentID)

	record := &types.AgentTagVCRecord{}
	var expiresAt, revokedAt sql.NullTime
	var signature sql.NullString

	err := row.Scan(
		&record.ID, &record.AgentID, &record.AgentDID, &record.VCID,
		&record.VCDocument, &signature, &record.IssuedAt, &expiresAt, &revokedAt,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("agent tag VC not found for agent %s", agentID)
		}
		return nil, fmt.Errorf("failed to get agent tag VC: %w", err)
	}

	if signature.Valid {
		record.Signature = signature.String
	}
	if expiresAt.Valid {
		record.ExpiresAt = &expiresAt.Time
	}
	if revokedAt.Valid {
		record.RevokedAt = &revokedAt.Time
	}

	return record, nil
}

// ListAgentTagVCs returns all non-revoked agent tag VCs.
func (ls *LocalStorage) ListAgentTagVCs(ctx context.Context) ([]*types.AgentTagVCRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during list agent tag VCs: %w", err)
	}

	query := `
		SELECT id, agent_id, agent_did, vc_id, vc_document, signature, issued_at, expires_at, revoked_at
		FROM agent_tag_vcs WHERE revoked_at IS NULL`

	rows, err := ls.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to list agent tag VCs: %w", err)
	}
	defer rows.Close()

	var records []*types.AgentTagVCRecord
	for rows.Next() {
		record := &types.AgentTagVCRecord{}
		var expiresAt, revokedAt sql.NullTime
		var signature sql.NullString

		if err := rows.Scan(
			&record.ID, &record.AgentID, &record.AgentDID, &record.VCID,
			&record.VCDocument, &signature, &record.IssuedAt, &expiresAt, &revokedAt,
		); err != nil {
			return nil, fmt.Errorf("failed to scan agent tag VC: %w", err)
		}

		if signature.Valid {
			record.Signature = signature.String
		}
		if expiresAt.Valid {
			record.ExpiresAt = &expiresAt.Time
		}
		if revokedAt.Valid {
			record.RevokedAt = &revokedAt.Time
		}
		records = append(records, record)
	}

	return records, rows.Err()
}

// RevokeAgentTagVC marks an agent's tag VC as revoked.
func (ls *LocalStorage) RevokeAgentTagVC(ctx context.Context, agentID string) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during revoke agent tag VC: %w", err)
	}

	query := `UPDATE agent_tag_vcs SET revoked_at = ?, updated_at = ? WHERE agent_id = ? AND revoked_at IS NULL`

	now := time.Now()
	result, err := ls.db.ExecContext(ctx, query, now, now, agentID)
	if err != nil {
		return fmt.Errorf("failed to revoke agent tag VC: %w", err)
	}

	rows, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}
	if rows == 0 {
		return fmt.Errorf("no active agent tag VC found for agent %s", agentID)
	}

	return nil
}
