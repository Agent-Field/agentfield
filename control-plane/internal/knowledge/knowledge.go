// Package knowledge implements the control plane's native, scope-aware RAG
// knowledge store. Callers send TEXT (not vectors); the service embeds each
// chunk with a pinned model, stores the vector in the existing scoped vector
// store, and answers scoped semantic search queries.
//
// Scoping (mirrors the design's tiered inheritance):
//   - Every chunk is stored under the MOST SPECIFIC scope namespace string:
//     "sender:<senderID>" for sender-tier, "proj:<projectID>" for project-tier,
//     "ws:<workspaceID>" for workspace-tier.
//   - A search reads the ADDITIVE set of namespaces driven by which ids are
//     present in the query scope: always "ws:<workspaceID>", plus
//     "proj:<projectID>" when a project_id is present, plus "sender:<senderID>"
//     when a sender_id is present. So a query in a project owned by a sender
//     sees workspace + project + sender chunks.
//
// Defense in depth: the scope is applied in the vector query (namespace +
// metadata filter) AND every returned chunk's workspace_id/namespace is
// re-verified in Go before it is returned; mismatches are dropped and logged.
// An empty workspace_id is rejected so an unscoped search is structurally
// impossible.
package knowledge

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/Agent-Field/agentfield/control-plane/internal/embedding"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// vectorScope is the fixed storage scope under which all knowledge vectors live.
// The per-tenant namespace is carried in the scope ID (ws:/proj:).
const vectorScope = "knowledge"

// Tier identifies the scope tier of a knowledge operation.
type Tier string

const (
	// TierWorkspace scopes to a workspace's own knowledge only.
	TierWorkspace Tier = "workspace"
	// TierProject scopes to a project's knowledge plus its parent workspace's.
	TierProject Tier = "project"
	// TierSender scopes to a sender's knowledge plus its project + workspace.
	TierSender Tier = "sender"
)

// Scope identifies the tenant scope of a knowledge operation.
type Scope struct {
	Tier        Tier   `json:"tier"`
	WorkspaceID string `json:"workspace_id"`
	ProjectID   string `json:"project_id,omitempty"`
	SenderID    string `json:"sender_id,omitempty"`
}

// Validate enforces the scope invariants. An empty workspace_id is always
// rejected (no unscoped operations); project tier additionally requires a
// project_id, and sender tier additionally requires a sender_id.
func (s Scope) Validate() error {
	if strings.TrimSpace(s.WorkspaceID) == "" {
		return errors.New("workspace_id is required")
	}
	switch s.Tier {
	case TierWorkspace:
		return nil
	case TierProject:
		if strings.TrimSpace(s.ProjectID) == "" {
			return errors.New("project_id is required when tier is project")
		}
		return nil
	case TierSender:
		if strings.TrimSpace(s.SenderID) == "" {
			return errors.New("sender_id is required when tier is sender")
		}
		return nil
	default:
		return fmt.Errorf("invalid tier %q (must be workspace, project or sender)", s.Tier)
	}
}

// workspaceNamespace returns the namespace string for the scope's workspace.
func (s Scope) workspaceNamespace() string { return "ws:" + s.WorkspaceID }

// projectNamespace returns the namespace string for the scope's project.
func (s Scope) projectNamespace() string { return "proj:" + s.ProjectID }

// senderNamespace returns the namespace string for the scope's sender.
func (s Scope) senderNamespace() string { return "sender:" + s.SenderID }

// writeNamespace returns the namespace a chunk is stored under for this scope:
// the most specific namespace by tier (sender > project > workspace).
func (s Scope) writeNamespace() string {
	switch s.Tier {
	case TierSender:
		return s.senderNamespace()
	case TierProject:
		return s.projectNamespace()
	default:
		return s.workspaceNamespace()
	}
}

// searchNamespaces returns the ADDITIVE set of namespaces a search for this
// scope reads from, driven by which ids are present (tier is informational):
// always the workspace namespace, plus the project namespace when a project_id
// is present, plus the sender namespace when a sender_id is present.
func (s Scope) searchNamespaces() []string {
	ns := []string{s.workspaceNamespace()}
	if strings.TrimSpace(s.ProjectID) != "" {
		ns = append(ns, s.projectNamespace())
	}
	if strings.TrimSpace(s.SenderID) != "" {
		ns = append(ns, s.senderNamespace())
	}
	return ns
}

// Chunk is a single unit of text to embed and store.
type Chunk struct {
	Text     string                 `json:"text"`
	Page     *int                   `json:"page,omitempty"`
	Ordinal  *int                   `json:"ordinal,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// SearchHit is a single scoped search result.
type SearchHit struct {
	SourceID string                 `json:"source_id"`
	Text     string                 `json:"text"`
	Score    float64                `json:"score"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// VectorStore is the subset of the control plane's storage that the knowledge
// service depends on. The existing MemoryStorage satisfies it.
type VectorStore interface {
	SetVector(ctx context.Context, record *types.VectorRecord) error
	DeleteVectorsByPrefix(ctx context.Context, scope, scopeID, prefix string) (int, error)
	SimilaritySearch(ctx context.Context, scope, scopeID string, queryEmbedding []float32, topK int, filters map[string]interface{}) ([]*types.VectorSearchResult, error)
}

// Service embeds text and performs scoped storage/retrieval over a VectorStore.
type Service struct {
	store    VectorStore
	embedder embedding.Embedder
}

// NewService builds a knowledge service.
func NewService(store VectorStore, embedder embedding.Embedder) *Service {
	return &Service{store: store, embedder: embedder}
}

// chunkKey is the storage key for a source's chunk. Keys are prefixed with the
// source ID + ":" so all of a source's chunks share a deletable prefix.
func chunkKey(sourceID string, ordinal int) string {
	return fmt.Sprintf("%s:%d", sourceID, ordinal)
}

// sourcePrefix is the key prefix matching every chunk of a source.
func sourcePrefix(sourceID string) string { return sourceID + ":" }

// Upsert embeds and stores every chunk of a source under the scope namespace.
func (s *Service) Upsert(ctx context.Context, scope Scope, sourceID string, chunks []Chunk) (int, error) {
	if err := scope.Validate(); err != nil {
		return 0, err
	}
	if strings.TrimSpace(sourceID) == "" {
		return 0, errors.New("source_id is required")
	}
	if len(chunks) == 0 {
		return 0, errors.New("at least one chunk is required")
	}

	texts := make([]string, len(chunks))
	for i, ch := range chunks {
		if strings.TrimSpace(ch.Text) == "" {
			return 0, fmt.Errorf("chunk %d has empty text", i)
		}
		texts[i] = ch.Text
	}

	vectors, err := s.embedder.Embed(ctx, texts)
	if err != nil {
		return 0, fmt.Errorf("embed chunks: %w", err)
	}
	if len(vectors) != len(chunks) {
		return 0, fmt.Errorf("embedder returned %d vectors for %d chunks", len(vectors), len(chunks))
	}

	namespace := scope.writeNamespace()
	for i, ch := range chunks {
		ordinal := i
		if ch.Ordinal != nil {
			ordinal = *ch.Ordinal
		}

		meta := map[string]interface{}{}
		for k, v := range ch.Metadata {
			meta[k] = v
		}
		// Scope/identity metadata is authoritative and overrides any caller-
		// supplied keys, so it can be trusted during search re-verification.
		meta["source_id"] = sourceID
		meta["workspace_id"] = scope.WorkspaceID
		meta["namespace"] = namespace
		meta["text"] = ch.Text
		meta["ordinal"] = ordinal
		if strings.TrimSpace(scope.ProjectID) != "" {
			meta["project_id"] = scope.ProjectID
		}
		if strings.TrimSpace(scope.SenderID) != "" {
			meta["sender_id"] = scope.SenderID
		}
		if ch.Page != nil {
			meta["page"] = *ch.Page
		}

		record := &types.VectorRecord{
			Scope:     vectorScope,
			ScopeID:   namespace,
			Key:       chunkKey(sourceID, ordinal),
			Embedding: vectors[i],
			Metadata:  meta,
		}
		if err := s.store.SetVector(ctx, record); err != nil {
			return i, fmt.Errorf("store chunk %d: %w", i, err)
		}
	}

	return len(chunks), nil
}

// Search embeds the query and runs a scoped vector search, returning hits in
// descending score order. Results are merged from the additive set of allowed
// namespaces (workspace, plus project and/or sender when those ids are present).
func (s *Service) Search(ctx context.Context, scope Scope, query string, topK int) ([]SearchHit, error) {
	if err := scope.Validate(); err != nil {
		return nil, err
	}
	if strings.TrimSpace(query) == "" {
		return nil, errors.New("query is required")
	}
	if topK <= 0 {
		topK = 10
	}

	vecs, err := s.embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	if len(vecs) != 1 {
		return nil, fmt.Errorf("embedder returned %d vectors for query", len(vecs))
	}
	queryVec := vecs[0]

	allowedNamespaces := scope.searchNamespaces()
	allowed := make(map[string]bool, len(allowedNamespaces))
	for _, ns := range allowedNamespaces {
		allowed[ns] = true
	}

	var merged []*types.VectorSearchResult
	for _, ns := range allowedNamespaces {
		// Scope filter applied IN the query: namespace via scopeID, plus a
		// metadata filter on workspace_id (and namespace) for belt-and-braces.
		filters := map[string]interface{}{
			"namespace":    ns,
			"workspace_id": scope.WorkspaceID,
		}
		hits, err := s.store.SimilaritySearch(ctx, vectorScope, ns, queryVec, topK, filters)
		if err != nil {
			return nil, fmt.Errorf("search namespace %s: %w", ns, err)
		}
		merged = append(merged, hits...)
	}

	// Defense in depth: re-verify every returned chunk's namespace + workspace
	// in Go before returning. Drop and log anything outside the allowed scope.
	out := make([]SearchHit, 0, len(merged))
	for _, r := range merged {
		ns, _ := r.Metadata["namespace"].(string)
		ws, _ := r.Metadata["workspace_id"].(string)
		if !allowed[ns] || ws != scope.WorkspaceID {
			logger.Logger.Warn().
				Str("expected_workspace", scope.WorkspaceID).
				Str("got_workspace", ws).
				Str("namespace", ns).
				Msg("knowledge search dropped out-of-scope chunk")
			continue
		}
		sourceID, _ := r.Metadata["source_id"].(string)
		text, _ := r.Metadata["text"].(string)
		out = append(out, SearchHit{
			SourceID: sourceID,
			Text:     text,
			Score:    r.Score,
			Metadata: scrubInternalMetadata(r.Metadata),
		})
	}

	sortHitsDesc(out)
	if topK > 0 && len(out) > topK {
		out = out[:topK]
	}
	return out, nil
}

// DeleteSource removes every chunk of a source within the given scope.
func (s *Service) DeleteSource(ctx context.Context, scope Scope, sourceID string) (int, error) {
	if err := scope.Validate(); err != nil {
		return 0, err
	}
	if strings.TrimSpace(sourceID) == "" {
		return 0, errors.New("source_id is required")
	}
	namespace := scope.writeNamespace()
	deleted, err := s.store.DeleteVectorsByPrefix(ctx, vectorScope, namespace, sourcePrefix(sourceID))
	if err != nil {
		return 0, fmt.Errorf("delete source %s: %w", sourceID, err)
	}
	return deleted, nil
}

// scrubInternalMetadata returns a copy of metadata without the bookkeeping keys
// the service stores internally, leaving caller-provided metadata intact.
func scrubInternalMetadata(meta map[string]interface{}) map[string]interface{} {
	if meta == nil {
		return nil
	}
	out := make(map[string]interface{}, len(meta))
	for k, v := range meta {
		switch k {
		case "namespace", "text":
			continue
		default:
			out[k] = v
		}
	}
	return out
}

// sortHitsDesc sorts hits by descending score (stable on source/text ties).
func sortHitsDesc(hits []SearchHit) {
	for i := 1; i < len(hits); i++ {
		for j := i; j > 0 && hits[j].Score > hits[j-1].Score; j-- {
			hits[j], hits[j-1] = hits[j-1], hits[j]
		}
	}
}
