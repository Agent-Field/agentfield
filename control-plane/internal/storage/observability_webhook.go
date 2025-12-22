package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

const observabilityWebhookGlobalID = "global"

// GetObservabilityWebhook retrieves the global observability webhook configuration.
// Returns nil if no webhook is configured.
func (ls *LocalStorage) GetObservabilityWebhook(ctx context.Context) (*types.ObservabilityWebhookConfig, error) {
	db := ls.requireSQLDB()

	query := `
		SELECT id, url, secret, headers, enabled, created_at, updated_at
		FROM observability_webhooks
		WHERE id = ?`

	row := db.QueryRowContext(ctx, query, observabilityWebhookGlobalID)

	var (
		config     types.ObservabilityWebhookConfig
		rawSecret  sql.NullString
		rawHeaders sql.NullString
	)

	if err := row.Scan(
		&config.ID,
		&config.URL,
		&rawSecret,
		&rawHeaders,
		&config.Enabled,
		&config.CreatedAt,
		&config.UpdatedAt,
	); err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, fmt.Errorf("scan observability webhook: %w", err)
	}

	if rawSecret.Valid {
		config.Secret = &rawSecret.String
	}

	config.Headers = make(map[string]string)
	if rawHeaders.Valid && rawHeaders.String != "" && rawHeaders.String != "{}" {
		if err := json.Unmarshal([]byte(rawHeaders.String), &config.Headers); err != nil {
			return nil, fmt.Errorf("unmarshal observability webhook headers: %w", err)
		}
	}

	return &config, nil
}

// SetObservabilityWebhook stores or updates the global observability webhook configuration.
// Uses upsert pattern to handle both insert and update.
func (ls *LocalStorage) SetObservabilityWebhook(ctx context.Context, config *types.ObservabilityWebhookConfig) error {
	if config == nil {
		return fmt.Errorf("observability webhook config is nil")
	}
	if config.URL == "" {
		return fmt.Errorf("observability webhook URL is required")
	}

	db := ls.requireSQLDB()
	now := time.Now().UTC()

	// Encode headers to JSON
	headersJSON := "{}"
	if len(config.Headers) > 0 {
		encoded, err := json.Marshal(config.Headers)
		if err != nil {
			return fmt.Errorf("marshal observability webhook headers: %w", err)
		}
		headersJSON = string(encoded)
	}

	// Handle nullable secret
	var secret sql.NullString
	if config.Secret != nil && *config.Secret != "" {
		secret = sql.NullString{String: *config.Secret, Valid: true}
	}

	// Upsert query - works for both SQLite and PostgreSQL
	_, err := db.ExecContext(ctx, `
		INSERT INTO observability_webhooks (id, url, secret, headers, enabled, created_at, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET
			url = excluded.url,
			secret = excluded.secret,
			headers = excluded.headers,
			enabled = excluded.enabled,
			updated_at = excluded.updated_at
	`, observabilityWebhookGlobalID, config.URL, secret, headersJSON, config.Enabled, now, now)
	if err != nil {
		return fmt.Errorf("set observability webhook: %w", err)
	}

	return nil
}

// DeleteObservabilityWebhook removes the global observability webhook configuration.
func (ls *LocalStorage) DeleteObservabilityWebhook(ctx context.Context) error {
	db := ls.requireSQLDB()

	_, err := db.ExecContext(ctx, `DELETE FROM observability_webhooks WHERE id = ?`, observabilityWebhookGlobalID)
	if err != nil {
		return fmt.Errorf("delete observability webhook: %w", err)
	}

	return nil
}

// AddToDeadLetterQueue adds a failed event to the dead letter queue.
func (ls *LocalStorage) AddToDeadLetterQueue(ctx context.Context, event *types.ObservabilityEvent, errorMessage string, retryCount int) error {
	if event == nil {
		return fmt.Errorf("event is nil")
	}

	db := ls.requireSQLDB()

	payload, err := json.Marshal(event.Data)
	if err != nil {
		return fmt.Errorf("marshal event payload: %w", err)
	}

	eventTimestamp, err := time.Parse(time.RFC3339, event.Timestamp)
	if err != nil {
		eventTimestamp = time.Now().UTC()
	}

	_, err = db.ExecContext(ctx, `
		INSERT INTO observability_dead_letter_queue
		(event_type, event_source, event_timestamp, payload, error_message, retry_count, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?)`,
		event.EventType, event.EventSource, eventTimestamp, string(payload), errorMessage, retryCount, time.Now().UTC())
	if err != nil {
		return fmt.Errorf("insert to dead letter queue: %w", err)
	}

	return nil
}

// GetDeadLetterQueueCount returns the number of entries in the dead letter queue.
func (ls *LocalStorage) GetDeadLetterQueueCount(ctx context.Context) (int64, error) {
	db := ls.requireSQLDB()

	var count int64
	err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM observability_dead_letter_queue`).Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("count dead letter queue: %w", err)
	}

	return count, nil
}

// GetDeadLetterQueue returns entries from the dead letter queue with pagination.
func (ls *LocalStorage) GetDeadLetterQueue(ctx context.Context, limit, offset int) ([]types.ObservabilityDeadLetterEntry, error) {
	db := ls.requireSQLDB()

	if limit <= 0 {
		limit = 100
	}
	if offset < 0 {
		offset = 0
	}

	rows, err := db.QueryContext(ctx, `
		SELECT id, event_type, event_source, event_timestamp, payload, error_message, retry_count, created_at
		FROM observability_dead_letter_queue
		ORDER BY created_at ASC
		LIMIT ? OFFSET ?`, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("query dead letter queue: %w", err)
	}
	defer rows.Close()

	var entries []types.ObservabilityDeadLetterEntry
	for rows.Next() {
		var entry types.ObservabilityDeadLetterEntry
		if err := rows.Scan(
			&entry.ID,
			&entry.EventType,
			&entry.EventSource,
			&entry.EventTimestamp,
			&entry.Payload,
			&entry.ErrorMessage,
			&entry.RetryCount,
			&entry.CreatedAt,
		); err != nil {
			return nil, fmt.Errorf("scan dead letter queue entry: %w", err)
		}
		entries = append(entries, entry)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate dead letter queue: %w", err)
	}

	return entries, nil
}

// DeleteFromDeadLetterQueue removes specific entries from the dead letter queue.
func (ls *LocalStorage) DeleteFromDeadLetterQueue(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return nil
	}

	db := ls.requireSQLDB()

	// Build query with placeholders
	query := "DELETE FROM observability_dead_letter_queue WHERE id IN (?"
	args := make([]interface{}, len(ids))
	args[0] = ids[0]
	for i := 1; i < len(ids); i++ {
		query += ",?"
		args[i] = ids[i]
	}
	query += ")"

	_, err := db.ExecContext(ctx, query, args...)
	if err != nil {
		return fmt.Errorf("delete from dead letter queue: %w", err)
	}

	return nil
}

// ClearDeadLetterQueue removes all entries from the dead letter queue.
func (ls *LocalStorage) ClearDeadLetterQueue(ctx context.Context) error {
	db := ls.requireSQLDB()

	_, err := db.ExecContext(ctx, `DELETE FROM observability_dead_letter_queue`)
	if err != nil {
		return fmt.Errorf("clear dead letter queue: %w", err)
	}

	return nil
}
