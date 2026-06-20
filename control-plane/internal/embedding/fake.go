package embedding

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"math"
)

// FakeEmbedder is a deterministic, network-free embedder used when no OpenAI
// API key is configured and in tests. It hashes each input string into a stable
// pseudo-random unit vector at the pinned Dimensions. Identical text always maps
// to an identical vector, so upsert -> search round-trips return exact matches,
// while different text maps to (almost surely) different vectors.
//
// It is NOT semantically meaningful — it exists so the knowledge store works
// locally with zero external dependencies and so tests can assert retrieval and
// scoping behavior without a network call.
type FakeEmbedder struct{}

// NewFakeEmbedder returns a deterministic embedder at the pinned dimension.
func NewFakeEmbedder() *FakeEmbedder { return &FakeEmbedder{} }

// Dimensions returns the pinned embedding dimension.
func (f *FakeEmbedder) Dimensions() int { return Dimensions }

// Embed returns one deterministic unit vector per input string.
func (f *FakeEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = hashVector(t)
	}
	return out, nil
}

// hashVector expands a string into a deterministic, L2-normalized vector of
// length Dimensions using SHA-256 as a seeded PRNG (splitmix64 over counters
// mixed with the digest). Normalization keeps cosine similarity well-behaved.
func hashVector(text string) []float32 {
	digest := sha256.Sum256([]byte(text))
	// Derive a 64-bit seed from the first 8 bytes of the digest.
	seed := binary.LittleEndian.Uint64(digest[:8])

	vec := make([]float32, Dimensions)
	state := seed
	var norm float64
	for i := 0; i < Dimensions; i++ {
		state += 0x9E3779B97F4A7C15
		z := state
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
		z = (z ^ (z >> 27)) * 0x94D049BB133111EB
		z = z ^ (z >> 31)
		// Map to [-1, 1).
		v := float32(float64(z)/float64(math.MaxUint64)*2.0 - 1.0)
		vec[i] = v
		norm += float64(v) * float64(v)
	}
	if norm > 0 {
		inv := float32(1.0 / math.Sqrt(norm))
		for i := range vec {
			vec[i] *= inv
		}
	}
	return vec
}
