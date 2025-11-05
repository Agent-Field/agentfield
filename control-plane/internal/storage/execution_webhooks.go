package storage

import (
	"github.com/your-org/haxen/control-plane/pkg/types"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// RegisterExecutionWebhook stores or updates the webhook registration for an execution.
func (ls *LocalStorage) RegisterExecutionWebhook(ctx context.Context, webhook *types.ExecutionWebhook) error {
	if webhook == nil {
		return fmt.Errorf("execution webhook registration is nil")
	}
	if strings.TrimSpace(webhook.ExecutionID) == "" {
		return fmt.Errorf("execution id is required for webhook registration")
	}
	if strings.TrimSpace(webhook.URL) == "" {
		return fmt.Errorf("webhook url is required")
	}

	db := ls.requireSQLDB()
	now := time.Now().UTC()
	headersJSON := "{}"
	if len(webhook.Headers) > 0 {
		encoded, err := json.Marshal(webhook.Headers)
		if err != nil {
			return fmt.Errorf("marshal webhook headers: %w", err)
		}
		headersJSON = string(encoded)
	}

	var secret sql.NullString
	if webhook.Secret != nil && strings.TrimSpace(*webhook.Secret) != "" {
		secret = sql.NullString{String: *webhook.Secret, Valid: true}
	}

	nextAttempt := now
	if webhook.NextAttemptAt != nil && !webhook.NextAttemptAt.IsZero() {
		nextAttempt = webhook.NextAttemptAt.UTC()
	}

	_, err := db.ExecContext(ctx, `
		INSERT INTO execution_webhooks (
			execution_id, url, secret, headers, status, attempt_count,
			next_attempt_at, last_attempt_at, last_error, created_at, updated_at
		) VALUES (?, ?, ?, ?, ?, 0, ?, NULL, NULL, ?, ?)
		ON CONFLICT(execution_id) DO UPDATE SET
			url = excluded.url,
			secret = excluded.secret,
			headers = excluded.headers,
			status = excluded.status,
			attempt_count = excluded.attempt_count,
			next_attempt_at = excluded.next_attempt_at,
			last_attempt_at = excluded.last_attempt_at,
			last_error = excluded.last_error,
			updated_at = excluded.updated_at
	`, webhook.ExecutionID, webhook.URL, secret, headersJSON, types.ExecutionWebhookStatusPending, nextAttempt, now, now)
	if err != nil {
		return fmt.Errorf("register execution webhook: %w", err)
	}

	return nil
}

// GetExecutionWebhook fetches the webhook registration for the given execution.
func (ls *LocalStorage) GetExecutionWebhook(ctx context.Context, executionID string) (*types.ExecutionWebhook, error) {
	query := `
		SELECT execution_id, url, secret, headers, status, attempt_count,
		       next_attempt_at, last_attempt_at, last_error, created_at, updated_at
		FROM execution_webhooks
		WHERE execution_id = ?`

	row := ls.requireSQLDB().QueryRowContext(ctx, query, executionID)

	var (
		model                         types.ExecutionWebhook
		rawSecret, rawHeaders, errMsg sql.NullString
		nextAttempt, lastAttempt      sql.NullTime
	)

	if err := row.Scan(
		&model.ExecutionID,
		&model.URL,
		&rawSecret,
		&rawHeaders,
		&model.Status,
		&model.AttemptCount,
		&nextAttempt,
		&lastAttempt,
		&errMsg,
		&model.CreatedAt,
		&model.UpdatedAt,
	); err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, fmt.Errorf("scan execution webhook: %w", err)
	}

	if rawSecret.Valid {
		value := rawSecret.String
		model.Secret = &value
	}

	headers := make(map[string]string)
	if rawHeaders.Valid && strings.TrimSpace(rawHeaders.String) != "" {
		if err := json.Unmarshal([]byte(rawHeaders.String), &headers); err != nil {
			return nil, fmt.Errorf("unmarshal webhook headers: %w", err)
		}
	}
	model.Headers = headers

	if nextAttempt.Valid {
		value := nextAttempt.Time.UTC()
		model.NextAttemptAt = &value
	}
	if lastAttempt.Valid {
		value := lastAttempt.Time.UTC()
		model.LastAttemptAt = &value
	}
	if errMsg.Valid {
		value := errMsg.String
		model.LastError = &value
	}

	return &model, nil
}

// ListDueExecutionWebhooks returns webhook registrations that are ready for delivery.
func (ls *LocalStorage) ListDueExecutionWebhooks(ctx context.Context, limit int) ([]*types.ExecutionWebhook, error) {
	if limit <= 0 {
		limit = 100
	}
	query := `
		SELECT execution_id, url, secret, headers, status, attempt_count,
		       next_attempt_at, last_attempt_at, last_error, created_at, updated_at
		FROM execution_webhooks
		WHERE status = ?
		  AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
		ORDER BY
			CASE WHEN next_attempt_at IS NULL THEN 0 ELSE 1 END,
			next_attempt_at ASC,
			execution_id
		LIMIT ?`

	rows, err := ls.requireSQLDB().QueryContext(ctx, query,
		types.ExecutionWebhookStatusPending,
		time.Now().UTC(),
		limit,
	)
	if err != nil {
		return nil, fmt.Errorf("list due execution webhooks: %w", err)
	}
	defer rows.Close()

	var results []*types.ExecutionWebhook
	for rows.Next() {
		var (
			model                         types.ExecutionWebhook
			rawSecret, rawHeaders, errMsg sql.NullString
			nextAttempt, lastAttempt      sql.NullTime
		)
		if err := rows.Scan(
			&model.ExecutionID,
			&model.URL,
			&rawSecret,
			&rawHeaders,
			&model.Status,
			&model.AttemptCount,
			&nextAttempt,
			&lastAttempt,
			&errMsg,
			&model.CreatedAt,
			&model.UpdatedAt,
		); err != nil {
			return nil, fmt.Errorf("scan due webhook: %w", err)
		}
		if rawSecret.Valid {
			value := rawSecret.String
			model.Secret = &value
		}

		headers := make(map[string]string)
		if rawHeaders.Valid && strings.TrimSpace(rawHeaders.String) != "" {
			if err := json.Unmarshal([]byte(rawHeaders.String), &headers); err != nil {
				return nil, fmt.Errorf("unmarshal webhook headers: %w", err)
			}
		}
		model.Headers = headers

		if nextAttempt.Valid {
			value := nextAttempt.Time.UTC()
			model.NextAttemptAt = &value
		}
		if lastAttempt.Valid {
			value := lastAttempt.Time.UTC()
			model.LastAttemptAt = &value
		}
		if errMsg.Valid {
			value := errMsg.String
			model.LastError = &value
		}
		results = append(results, &model)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate due webhooks: %w", err)
	}

	return results, nil
}

// TryMarkExecutionWebhookInFlight atomically marks a webhook registration as delivering.
func (ls *LocalStorage) TryMarkExecutionWebhookInFlight(ctx context.Context, executionID string, now time.Time) (bool, error) {
	result, err := ls.requireSQLDB().ExecContext(ctx, `
		UPDATE execution_webhooks
		SET status = ?, updated_at = ?
		WHERE execution_id = ?
		  AND status = ?
		  AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
	`, types.ExecutionWebhookStatusDelivering, now.UTC(), executionID,
		types.ExecutionWebhookStatusPending,
		now.UTC(),
	)
	if err != nil {
		return false, fmt.Errorf("mark webhook in flight: %w", err)
	}
	rows, err := result.RowsAffected()
	if err != nil {
		return false, fmt.Errorf("rows affected mark webhook in flight: %w", err)
	}
	return rows > 0, nil
}

// UpdateExecutionWebhookState persists the latest delivery state for a webhook registration.
func (ls *LocalStorage) UpdateExecutionWebhookState(ctx context.Context, executionID string, update types.ExecutionWebhookStateUpdate) error {
	db := ls.requireSQLDB()
	now := time.Now().UTC()

	var (
		nextAttempt sql.NullTime
		lastAttempt sql.NullTime
		lastError   sql.NullString
	)
	if update.NextAttemptAt != nil && !update.NextAttemptAt.IsZero() {
		nextAttempt = sql.NullTime{Time: update.NextAttemptAt.UTC(), Valid: true}
	}
	if update.LastAttemptAt != nil && !update.LastAttemptAt.IsZero() {
		lastAttempt = sql.NullTime{Time: update.LastAttemptAt.UTC(), Valid: true}
	}
	if update.LastError != nil && strings.TrimSpace(*update.LastError) != "" {
		lastError = sql.NullString{String: *update.LastError, Valid: true}
	}

	_, err := db.ExecContext(ctx, `
		UPDATE execution_webhooks
		SET status = ?,
		    attempt_count = ?,
		    next_attempt_at = ?,
		    last_attempt_at = ?,
		    last_error = ?,
		    updated_at = ?
		WHERE execution_id = ?
	`, update.Status, update.AttemptCount, nextAttempt, lastAttempt, lastError, now, executionID)
	if err != nil {
		return fmt.Errorf("update execution webhook state: %w", err)
	}
	return nil
}

// HasExecutionWebhook indicates whether an execution has a registered webhook.
func (ls *LocalStorage) HasExecutionWebhook(ctx context.Context, executionID string) (bool, error) {
	var exists bool
	err := ls.requireSQLDB().QueryRowContext(ctx, `
		SELECT EXISTS(
			SELECT 1 FROM execution_webhooks WHERE execution_id = ?
		)
	`, executionID).Scan(&exists)
	if err != nil {
		return false, fmt.Errorf("check execution webhook: %w", err)
	}
	return exists, nil
}

// ListExecutionWebhooksRegistered returns a map of execution IDs that have webhook registrations.
func (ls *LocalStorage) ListExecutionWebhooksRegistered(ctx context.Context, executionIDs []string) (map[string]bool, error) {
	result := make(map[string]bool, len(executionIDs))
	if len(executionIDs) == 0 {
		return result, nil
	}

	unique := make([]string, 0, len(executionIDs))
	seen := make(map[string]struct{}, len(executionIDs))
	for _, id := range executionIDs {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		unique = append(unique, id)
	}
	if len(unique) == 0 {
		return result, nil
	}

	placeholders := make([]string, len(unique))
	args := make([]interface{}, len(unique))
	for i, id := range unique {
		placeholders[i] = "?"
		args[i] = id
	}

	query := fmt.Sprintf(`
		SELECT execution_id
		FROM execution_webhooks
		WHERE execution_id IN (%s)
	`, strings.Join(placeholders, ","))

	rows, err := ls.requireSQLDB().QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("list execution webhooks registered: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var executionID string
		if err := rows.Scan(&executionID); err != nil {
			return nil, fmt.Errorf("scan execution webhook registration: %w", err)
		}
		result[executionID] = true
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate execution webhook registrations: %w", err)
	}

	return result, nil
}
