package storage

import (
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestVectorHelpers(t *testing.T) {
	t.Run("parse distance metric aliases", func(t *testing.T) {
		cases := []struct {
			input string
			want  VectorDistanceMetric
		}{
			{input: "dot", want: VectorDistanceDot},
			{input: "inner", want: VectorDistanceDot},
			{input: "ip", want: VectorDistanceDot},
			{input: "l2", want: VectorDistanceL2},
			{input: "euclidean", want: VectorDistanceL2},
			{input: " cosine ", want: VectorDistanceCosine},
			{input: "", want: VectorDistanceCosine},
		}

		for _, tc := range cases {
			require.Equal(t, tc.want, parseDistanceMetric(tc.input))
		}
	})

	t.Run("encode and decode embeddings", func(t *testing.T) {
		original := []float32{1.25, -2.5, 3.75}
		encoded := encodeEmbedding(original)
		decoded, err := decodeEmbedding(encoded)
		require.NoError(t, err)
		require.Equal(t, original, decoded)

		_, err = decodeEmbedding([]byte{1, 2, 3})
		require.EqualError(t, err, "invalid embedding length: 3")
	})

	t.Run("validate vector payload requirements", func(t *testing.T) {
		cases := []struct {
			name   string
			record *types.VectorRecord
			err    string
		}{
			{name: "nil", record: nil, err: "vector record cannot be nil"},
			{name: "missing scope", record: &types.VectorRecord{ScopeID: "id", Key: "key", Embedding: []float32{1}}, err: "scope, scope_id, and key are required"},
			{name: "missing embedding", record: &types.VectorRecord{Scope: "scope", ScopeID: "id", Key: "key"}, err: "embedding cannot be empty"},
			{name: "valid", record: &types.VectorRecord{Scope: "scope", ScopeID: "id", Key: "key", Embedding: []float32{1}}},
		}

		for _, tc := range cases {
			err := ensureVectorPayload(tc.record)
			if tc.err == "" {
				require.NoError(t, err, tc.name)
				continue
			}
			require.EqualError(t, err, tc.err, tc.name)
		}
	})

	t.Run("normalize metadata and filter matching", func(t *testing.T) {
		require.Empty(t, normalizeMetadata(nil))

		meta := map[string]interface{}{"count": 7, "kind": "doc"}
		require.Equal(t, meta, normalizeMetadata(meta))
		require.True(t, metadataMatchesFilters(meta, nil))
		require.True(t, metadataMatchesFilters(meta, map[string]interface{}{"count": "7"}))
		require.False(t, metadataMatchesFilters(meta, map[string]interface{}{"missing": "x"}))
		require.False(t, metadataMatchesFilters(meta, map[string]interface{}{"kind": "image"}))
	})

	t.Run("compute similarity and rank results", func(t *testing.T) {
		query := []float32{1, 0}
		candidate := []float32{1, 0}
		zero := []float32{0, 0}

		score, distance := computeSimilarity(VectorDistanceCosine, query, candidate)
		require.Equal(t, 1.0, score)
		require.Equal(t, 0.0, distance)

		score, distance = computeSimilarity(VectorDistanceCosine, query, zero)
		require.Equal(t, 0.0, score)
		require.Equal(t, 1.0, distance)

		score, distance = computeSimilarity(VectorDistanceDot, query, []float32{2, 0})
		require.Equal(t, 2.0, score)
		require.Equal(t, -2.0, distance)

		score, distance = computeSimilarity(VectorDistanceL2, query, []float32{0, 1})
		require.InDelta(t, -1.41421356, score, 0.00001)
		require.InDelta(t, 1.41421356, distance, 0.00001)

		require.Equal(t, 1.0, cosineSimilarity(query, candidate))
		require.Equal(t, 2.0, dotProduct([]float32{1, 1}, []float32{1, 1}))
		require.InDelta(t, 5.0, l2Distance([]float32{0, 0}, []float32{3, 4}), 0.00001)

		results := []*types.VectorSearchResult{
			{Key: "third", Score: 0.5, Distance: 0.7},
			{Key: "first", Score: 0.9, Distance: 0.4},
			{Key: "second", Score: 0.9, Distance: 0.2},
		}
		limited := sortAndLimit(results, 2)
		require.Len(t, limited, 2)
		require.Equal(t, "second", limited[0].Key)
		require.Equal(t, "first", limited[1].Key)
		require.Len(t, sortAndLimit(results, 0), 3)
	})

	t.Run("now utc", func(t *testing.T) {
		require.Equal(t, time.UTC, nowUTC().Location())
	})
}
