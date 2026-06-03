package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

type postgresVectorStore struct {
	db     *sqlDatabase
	metric VectorDistanceMetric
}

func newPostgresVectorStore(db *sqlDatabase, metric VectorDistanceMetric) *postgresVectorStore {
	return &postgresVectorStore{
		db:     db,
		metric: metric,
	}
}

func (s *postgresVectorStore) Set(ctx context.Context, record *types.VectorRecord) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := ensureVectorPayload(record); err != nil {
		return err
	}

	meta := normalizeMetadata(record.Metadata)
	metaJSON, err := json.Marshal(meta)
	if err != nil {
		return fmt.Errorf("marshal metadata: %w", err)
	}

	query := `
		INSERT INTO memory_vectors (scope, scope_id, key, embedding, metadata, created_at, updated_at)
		VALUES (?, ?, ?, ?::vector, ?::jsonb, ?, ?)
		ON CONFLICT(scope, scope_id, key) DO UPDATE SET
			embedding = excluded.embedding,
			metadata = excluded.metadata,
			updated_at = excluded.updated_at
	`

	now := nowUTC()
	_, err = s.db.ExecContext(
		ctx,
		query,
		record.Scope,
		record.ScopeID,
		record.Key,
		vectorLiteral(record.Embedding),
		string(metaJSON),
		now,
		now,
	)
	if err != nil {
		return fmt.Errorf("set postgres vector: %w", err)
	}
	return nil
}

func (s *postgresVectorStore) Get(ctx context.Context, scope, scopeID, key string) (*types.VectorRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	query := `
		SELECT embedding::text, metadata, created_at, updated_at
		FROM memory_vectors
		WHERE scope = ? AND scope_id = ? AND key = ?
	`

	var embeddingStr string
	var metadataRaw []byte
	var createdAt, updatedAt time.Time

	err := s.db.QueryRowContext(ctx, query, scope, scopeID, key).Scan(&embeddingStr, &metadataRaw, &createdAt, &updatedAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("get postgres vector: %w", err)
	}

	// Parse postgres vector literal like "[0.1,0.2,0.3]"
	embeddingStr = strings.Trim(embeddingStr, "[]")
	parts := strings.Split(embeddingStr, ",")
	embedding := make([]float32, len(parts))
	for i, p := range parts {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 32)
		if err != nil {
			return nil, fmt.Errorf("parse embedding element: %w", err)
		}
		embedding[i] = float32(v)
	}

	record := &types.VectorRecord{
		Scope:     scope,
		ScopeID:   scopeID,
		Key:       key,
		Embedding: embedding,
		CreatedAt: createdAt,
		UpdatedAt: updatedAt,
	}

	if len(metadataRaw) > 0 {
		if err := json.Unmarshal(metadataRaw, &record.Metadata); err != nil {
			return nil, fmt.Errorf("unmarshal metadata: %w", err)
		}
	}

	return record, nil
}

func (s *postgresVectorStore) Delete(ctx context.Context, scope, scopeID, key string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	_, err := s.db.ExecContext(ctx, `
		DELETE FROM memory_vectors
		WHERE scope = ? AND scope_id = ? AND key = ?
	`, scope, scopeID, key)
	return err
}

func (s *postgresVectorStore) DeleteByPrefix(ctx context.Context, scope, scopeID, prefix string) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}

	// Use || operator to build the LIKE pattern in PostgreSQL
	result, err := s.db.ExecContext(ctx, `
		DELETE FROM memory_vectors
		WHERE scope = ? AND scope_id = ? AND key LIKE ? || '%'
	`, scope, scopeID, prefix)
	if err != nil {
		return 0, err
	}

	rows, err := result.RowsAffected()
	if err != nil {
		return 0, err
	}
	return int(rows), nil
}

func (s *postgresVectorStore) Search(ctx context.Context, scope, scopeID string, query []float32, topK int, filters map[string]interface{}) ([]*types.VectorSearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(query) == 0 {
		return nil, fmt.Errorf("query embedding cannot be empty")
	}

	scoreExpr, distanceExpr := buildPostgresVectorExpressions(s.metric, "query_vec.qv")

	sb := strings.Builder{}
	sb.WriteString("WITH query_vec AS (SELECT ?::vector AS qv) ")
	sb.WriteString("SELECT mv.scope, mv.scope_id, mv.key, mv.metadata, mv.created_at, mv.updated_at, ")
	sb.WriteString(scoreExpr)
	sb.WriteString(" AS score, ")
	sb.WriteString(distanceExpr)
	sb.WriteString(" AS distance FROM memory_vectors mv CROSS JOIN query_vec WHERE mv.scope = ? AND mv.scope_id = ?")

	args := []interface{}{vectorLiteral(query), scope, scopeID}

	for key, val := range filters {
		filterJSON, err := json.Marshal(map[string]interface{}{key: val})
		if err != nil {
			return nil, fmt.Errorf("marshal filter: %w", err)
		}
		sb.WriteString(" AND mv.metadata @> ?::jsonb")
		args = append(args, string(filterJSON))
	}

	sb.WriteString(" ORDER BY ")
	sb.WriteString(distanceExpr)
	sb.WriteString(" ASC")

	if topK <= 0 {
		topK = 10
	}
	sb.WriteString(" LIMIT ?")
	args = append(args, topK)

	rows, err := s.db.QueryContext(ctx, sb.String(), args...)
	if err != nil {
		return nil, fmt.Errorf("query postgres vectors: %w", err)
	}
	defer rows.Close()

	results := make([]*types.VectorSearchResult, 0)
	for rows.Next() {
		result := &types.VectorSearchResult{}
		var metadataRaw []byte
		if err := rows.Scan(
			&result.Scope,
			&result.ScopeID,
			&result.Key,
			&metadataRaw,
			&result.CreatedAt,
			&result.UpdatedAt,
			&result.Score,
			&result.Distance,
		); err != nil {
			return nil, fmt.Errorf("scan postgres vector: %w", err)
		}

		if len(metadataRaw) > 0 {
			if err := json.Unmarshal(metadataRaw, &result.Metadata); err != nil {
				return nil, fmt.Errorf("unmarshal metadata: %w", err)
			}
		} else {
			result.Metadata = map[string]interface{}{}
		}
		results = append(results, result)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return results, nil
}

func vectorLiteral(vec []float32) string {
	values := make([]string, len(vec))
	for i, v := range vec {
		values[i] = strconv.FormatFloat(float64(v), 'f', -1, 32)
	}
	return fmt.Sprintf("[%s]", strings.Join(values, ","))
}

func buildPostgresVectorExpressions(metric VectorDistanceMetric, vectorReference string) (string, string) {
	switch metric {
	case VectorDistanceDot:
		dist := fmt.Sprintf("mv.embedding <#> %s", vectorReference)
		return fmt.Sprintf("-(%s)", dist), dist
	case VectorDistanceL2:
		dist := fmt.Sprintf("mv.embedding <-> %s", vectorReference)
		return fmt.Sprintf("-(%s)", dist), dist
	default:
		dist := fmt.Sprintf("mv.embedding <=> %s", vectorReference)
		return fmt.Sprintf("1 - (%s)", dist), dist
	}
}
