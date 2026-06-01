package storage

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

type vectorStore interface {
	Set(ctx context.Context, record *types.VectorRecord) error
	Get(ctx context.Context, scope, scopeID, key string) (*types.VectorRecord, error)
	Delete(ctx context.Context, scope, scopeID, key string) error
	DeleteByPrefix(ctx context.Context, scope, scopeID, prefix string) (int, error)
	Search(ctx context.Context, scope, scopeID string, query []float32, topK int, filters map[string]interface{}) ([]*types.VectorSearchResult, error)
}

type VectorDistanceMetric string

const (
	VectorDistanceCosine VectorDistanceMetric = "cosine"
	VectorDistanceDot    VectorDistanceMetric = "dot"
	VectorDistanceL2     VectorDistanceMetric = "l2"
)

func parseDistanceMetric(val string) VectorDistanceMetric {
	switch strings.ToLower(strings.TrimSpace(val)) {
	case "dot", "inner", "ip":
		return VectorDistanceDot
	case "l2", "euclidean":
		return VectorDistanceL2
	default:
		return VectorDistanceCosine
	}
}

func encodeEmbedding(vec []float32) []byte {
	buf := make([]byte, len(vec)*4)
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

func decodeEmbedding(data []byte) ([]float32, error) {
	if len(data)%4 != 0 {
		return nil, fmt.Errorf("invalid embedding length: %d", len(data))
	}
	vec := make([]float32, len(data)/4)
	for i := range vec {
		vec[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return vec, nil
}

func ensureVectorPayload(record *types.VectorRecord) error {
	if record == nil {
		return errors.New("vector record cannot be nil")
	}
	if record.Scope == "" || record.ScopeID == "" || record.Key == "" {
		return errors.New("scope, scope_id, and key are required")
	}
	if len(record.Embedding) == 0 {
		return errors.New("embedding cannot be empty")
	}
	return nil
}

func normalizeMetadata(meta map[string]interface{}) map[string]interface{} {
	if meta == nil {
		return map[string]interface{}{}
	}
	return meta
}

func cosineSimilarity(a, b []float32) float64 {
	var dot float64
	var normA float64
	var normB float64
	for i := range a {
		dot += float64(a[i] * b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func dotProduct(a, b []float32) float64 {
	var dot float64
	for i := range a {
		dot += float64(a[i] * b[i])
	}
	return dot
}

func l2Distance(a, b []float32) float64 {
	var sum float64
	for i := range a {
		diff := float64(a[i] - b[i])
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func computeSimilarity(metric VectorDistanceMetric, query, candidate []float32) (score float64, distance float64) {
	switch metric {
	case VectorDistanceDot:
		dot := dotProduct(query, candidate)
		return dot, -dot
	case VectorDistanceL2:
		dist := l2Distance(query, candidate)
		return -dist, dist
	default:
		// Cosine distance: score is cosine, distance is 1 - cosine
		cos := cosineSimilarity(query, candidate)
		return cos, 1 - cos
	}
}

func sortAndLimit(results []*types.VectorSearchResult, topK int) []*types.VectorSearchResult {
	sort.Slice(results, func(i, j int) bool {
		if results[i].Score == results[j].Score {
			return results[i].Distance < results[j].Distance
		}
		return results[i].Score > results[j].Score
	})
	if topK > 0 && len(results) > topK {
		return results[:topK]
	}
	return results
}

func metadataMatchesFilters(metadata map[string]interface{}, filters map[string]interface{}) bool {
	if len(filters) == 0 {
		return true
	}
	for key, expected := range filters {
		actual, ok := metadata[key]
		if !ok {
			return false
		}
		if fmt.Sprintf("%v", actual) != fmt.Sprintf("%v", expected) {
			return false
		}
	}
	return true
}

func nowUTC() time.Time {
	return time.Now().UTC()
}
