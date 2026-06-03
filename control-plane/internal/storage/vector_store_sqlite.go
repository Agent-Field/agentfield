package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

type sqliteVectorStore struct {
	db     *sqlDatabase
	metric VectorDistanceMetric
}

func newSQLiteVectorStore(db *sqlDatabase, metric VectorDistanceMetric) *sqliteVectorStore {
	return &sqliteVectorStore{
		db:     db,
		metric: metric,
	}
}

func (s *sqliteVectorStore) Set(ctx context.Context, record *types.VectorRecord) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := ensureVectorPayload(record); err != nil {
		return err
	}

	now := nowUTC()
	meta := normalizeMetadata(record.Metadata)
	metaJSON, err := json.Marshal(meta)
	if err != nil {
		return fmt.Errorf("marshal metadata: %w", err)
	}

	query := `
		INSERT INTO memory_vectors (scope, scope_id, key, dimension, embedding, metadata, created_at, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(scope, scope_id, key) DO UPDATE SET
			embedding = excluded.embedding,
			metadata = excluded.metadata,
			dimension = excluded.dimension,
			updated_at = excluded.updated_at
	`

	_, err = s.db.ExecContext(
		ctx,
		query,
		record.Scope,
		record.ScopeID,
		record.Key,
		len(record.Embedding),
		encodeEmbedding(record.Embedding),
		string(metaJSON),
		now,
		now,
	)
	if err != nil {
		return fmt.Errorf("set vector: %w", err)
	}
	return nil
}

func (s *sqliteVectorStore) Get(ctx context.Context, scope, scopeID, key string) (*types.VectorRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	var embeddingBlob []byte
	var metadataRaw sql.NullString
	var createdAt, updatedAt time.Time

	err := s.db.QueryRowContext(ctx, `
		SELECT embedding, metadata, created_at, updated_at
		FROM memory_vectors
		WHERE scope = ? AND scope_id = ? AND key = ?
	`, scope, scopeID, key).Scan(&embeddingBlob, &metadataRaw, &createdAt, &updatedAt)

	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("get vector: %w", err)
	}

	embedding, err := decodeEmbedding(embeddingBlob)
	if err != nil {
		return nil, fmt.Errorf("decode embedding: %w", err)
	}

	metadata := map[string]interface{}{}
	if metadataRaw.Valid && metadataRaw.String != "" {
		if err := json.Unmarshal([]byte(metadataRaw.String), &metadata); err != nil {
			return nil, fmt.Errorf("unmarshal metadata: %w", err)
		}
	}

	return &types.VectorRecord{
		Scope:     scope,
		ScopeID:   scopeID,
		Key:       key,
		Embedding: embedding,
		Metadata:  metadata,
		CreatedAt: createdAt,
		UpdatedAt: updatedAt,
	}, nil
}

func (s *sqliteVectorStore) Delete(ctx context.Context, scope, scopeID, key string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	_, err := s.db.ExecContext(ctx, `
		DELETE FROM memory_vectors
		WHERE scope = ? AND scope_id = ? AND key = ?
	`, scope, scopeID, key)
	return err
}

func (s *sqliteVectorStore) DeleteByPrefix(ctx context.Context, scope, scopeID, prefix string) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}

	result, err := s.db.ExecContext(ctx, `
		DELETE FROM memory_vectors
		WHERE scope = ? AND scope_id = ? AND key LIKE ?
	`, scope, scopeID, prefix+"%")
	if err != nil {
		return 0, err
	}

	rows, err := result.RowsAffected()
	if err != nil {
		return 0, err
	}
	return int(rows), nil
}

func (s *sqliteVectorStore) Search(ctx context.Context, scope, scopeID string, query []float32, topK int, filters map[string]interface{}) ([]*types.VectorSearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(query) == 0 {
		return nil, fmt.Errorf("query embedding cannot be empty")
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT key, embedding, metadata, created_at, updated_at
		FROM memory_vectors
		WHERE scope = ? AND scope_id = ?
	`, scope, scopeID)
	if err != nil {
		return nil, fmt.Errorf("query vectors: %w", err)
	}
	defer rows.Close()

	results := make([]*types.VectorSearchResult, 0)
	for rows.Next() {
		var key string
		var embeddingBlob []byte
		var metadataRaw sql.NullString
		var createdAt, updatedAt time.Time

		if err := rows.Scan(&key, &embeddingBlob, &metadataRaw, &createdAt, &updatedAt); err != nil {
			return nil, fmt.Errorf("scan vector row: %w", err)
		}

		embedding, err := decodeEmbedding(embeddingBlob)
		if err != nil {
			return nil, fmt.Errorf("decode embedding: %w", err)
		}
		if len(embedding) != len(query) {
			continue
		}

		metadata := map[string]interface{}{}
		if metadataRaw.Valid && metadataRaw.String != "" {
			if err := json.Unmarshal([]byte(metadataRaw.String), &metadata); err != nil {
				return nil, fmt.Errorf("unmarshal metadata: %w", err)
			}
		}

		if !metadataMatchesFilters(metadata, filters) {
			continue
		}

		score, distance := computeSimilarity(s.metric, query, embedding)
		results = append(results, &types.VectorSearchResult{
			Scope:     scope,
			ScopeID:   scopeID,
			Key:       key,
			Score:     score,
			Distance:  distance,
			Metadata:  metadata,
			CreatedAt: createdAt,
			UpdatedAt: updatedAt,
		})
	}

	if err := rows.Err(); err != nil {
		return nil, err
	}

	return sortAndLimit(results, topK), nil
}
