// Package embedding provides a pluggable text-embedding layer for the control
// plane's native knowledge (RAG) store.
//
// The control plane pins ONE embedding model and ONE vector dimension for every
// caller. This is deliberate: the underlying vector store (sqlite-vec in dev,
// pgvector in prod) uses fixed-dimension columns/indexes. If different callers
// embedded text with different models/dimensions the index would silently
// corrupt. By centralizing embedding here, every chunk written to the knowledge
// store shares the same model and dimension.
package embedding

import "context"

// Dimensions is the single, pinned embedding dimension used by every Embedder
// implementation in this package. It matches OpenAI's text-embedding-3-small
// (and the legacy Qdrant store this replaces), so existing 1536-dim chunks line
// up. Do NOT make this configurable per-caller — the whole point is one fixed
// dimension for the shared pgvector index.
const Dimensions = 1536

// DefaultModel is the pinned OpenAI embedding model.
const DefaultModel = "text-embedding-3-small"

// Embedder converts text into fixed-dimension float32 vectors.
//
// Implementations MUST return vectors of exactly Dimensions() length, and
// Dimensions() MUST be stable for the lifetime of the process.
type Embedder interface {
	// Embed returns one vector per input string, in order. The returned slice
	// has the same length as texts, and every vector has Dimensions() elements.
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	// Dimensions reports the fixed vector dimension this embedder produces.
	Dimensions() int
}
