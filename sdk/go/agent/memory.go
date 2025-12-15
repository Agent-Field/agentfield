package agent

import (
	"context"
	"encoding/json"
	"sync"
)

// MemoryScope represents different memory isolation levels.
type MemoryScope string

const (
	// ScopeWorkflow isolates memory to the current workflow execution.
	ScopeWorkflow MemoryScope = "workflow"
	// ScopeSession isolates memory to the current session.
	ScopeSession MemoryScope = "session"
	// ScopeUser isolates memory to the current user/actor.
	ScopeUser MemoryScope = "user"
	// ScopeGlobal provides cross-session, cross-workflow storage.
	ScopeGlobal MemoryScope = "global"
)

// MemoryBackend is the pluggable storage interface for memory operations.
// Implementations can use in-memory storage, Redis, databases, or external APIs.
type MemoryBackend interface {
	// Set stores a value at the given scope and key.
	Set(scope MemoryScope, scopeID, key string, value any) error
	// Get retrieves a value; returns (value, found, error).
	Get(scope MemoryScope, scopeID, key string) (any, bool, error)
	// Delete removes a key from storage.
	Delete(scope MemoryScope, scopeID, key string) error
	// List returns all keys in a scope.
	List(scope MemoryScope, scopeID string) ([]string, error)
}

// Memory provides hierarchical state management for agent handlers.
// It supports multiple isolation scopes (workflow, session, user, global)
// with automatic scope ID resolution from execution context.
type Memory struct {
	backend MemoryBackend
}

// NewMemory creates a Memory instance with the given backend.
// If backend is nil, an in-memory backend is used.
func NewMemory(backend MemoryBackend) *Memory {
	if backend == nil {
		backend = NewInMemoryBackend()
	}
	return &Memory{backend: backend}
}

// Set stores a value in the session scope (default scope).
func (m *Memory) Set(ctx context.Context, key string, value any) error {
	execCtx := ExecutionContextFrom(ctx)
	scopeID := execCtx.SessionID
	if scopeID == "" {
		scopeID = execCtx.RunID
	}
	return m.backend.Set(ScopeSession, scopeID, key, value)
}

// Get retrieves a value from the session scope (default scope).
// Returns nil if the key does not exist.
func (m *Memory) Get(ctx context.Context, key string) (any, error) {
	execCtx := ExecutionContextFrom(ctx)
	scopeID := execCtx.SessionID
	if scopeID == "" {
		scopeID = execCtx.RunID
	}
	val, _, err := m.backend.Get(ScopeSession, scopeID, key)
	return val, err
}

// GetWithDefault retrieves a value from the session scope,
// returning the default if the key does not exist.
func (m *Memory) GetWithDefault(ctx context.Context, key string, defaultVal any) (any, error) {
	execCtx := ExecutionContextFrom(ctx)
	scopeID := execCtx.SessionID
	if scopeID == "" {
		scopeID = execCtx.RunID
	}
	val, found, err := m.backend.Get(ScopeSession, scopeID, key)
	if err != nil {
		return nil, err
	}
	if !found {
		return defaultVal, nil
	}
	return val, nil
}

// Delete removes a key from the session scope.
func (m *Memory) Delete(ctx context.Context, key string) error {
	execCtx := ExecutionContextFrom(ctx)
	scopeID := execCtx.SessionID
	if scopeID == "" {
		scopeID = execCtx.RunID
	}
	return m.backend.Delete(ScopeSession, scopeID, key)
}

// List returns all keys in the session scope.
func (m *Memory) List(ctx context.Context) ([]string, error) {
	execCtx := ExecutionContextFrom(ctx)
	scopeID := execCtx.SessionID
	if scopeID == "" {
		scopeID = execCtx.RunID
	}
	return m.backend.List(ScopeSession, scopeID)
}

// WorkflowScope returns a ScopedMemory for workflow-level storage.
// Data is isolated to the current workflow execution.
func (m *Memory) WorkflowScope() *ScopedMemory {
	return &ScopedMemory{
		backend: m.backend,
		scope:   ScopeWorkflow,
		getID: func(ctx context.Context) string {
			execCtx := ExecutionContextFrom(ctx)
			if execCtx.WorkflowID != "" {
				return execCtx.WorkflowID
			}
			return execCtx.RunID
		},
	}
}

// SessionScope returns a ScopedMemory for session-level storage.
// Data persists across workflow executions within the same session.
func (m *Memory) SessionScope() *ScopedMemory {
	return &ScopedMemory{
		backend: m.backend,
		scope:   ScopeSession,
		getID: func(ctx context.Context) string {
			execCtx := ExecutionContextFrom(ctx)
			if execCtx.SessionID != "" {
				return execCtx.SessionID
			}
			return execCtx.RunID
		},
	}
}

// UserScope returns a ScopedMemory for user/actor-level storage.
// Data persists across sessions for the same user.
func (m *Memory) UserScope() *ScopedMemory {
	return &ScopedMemory{
		backend: m.backend,
		scope:   ScopeUser,
		getID: func(ctx context.Context) string {
			execCtx := ExecutionContextFrom(ctx)
			if execCtx.ActorID != "" {
				return execCtx.ActorID
			}
			// Fall back to session if no actor
			if execCtx.SessionID != "" {
				return execCtx.SessionID
			}
			return execCtx.RunID
		},
	}
}

// GlobalScope returns a ScopedMemory for global storage.
// Data is shared across all sessions, users, and workflows.
func (m *Memory) GlobalScope() *ScopedMemory {
	return &ScopedMemory{
		backend: m.backend,
		scope:   ScopeGlobal,
		getID: func(ctx context.Context) string {
			return "global"
		},
	}
}

// ScopedMemory provides memory operations within a specific scope.
type ScopedMemory struct {
	backend MemoryBackend
	scope   MemoryScope
	getID   func(context.Context) string
}

// Set stores a value in this scope.
func (s *ScopedMemory) Set(ctx context.Context, key string, value any) error {
	return s.backend.Set(s.scope, s.getID(ctx), key, value)
}

// Get retrieves a value from this scope.
// Returns nil if the key does not exist.
func (s *ScopedMemory) Get(ctx context.Context, key string) (any, error) {
	val, _, err := s.backend.Get(s.scope, s.getID(ctx), key)
	return val, err
}

// GetWithDefault retrieves a value from this scope,
// returning the default if the key does not exist.
func (s *ScopedMemory) GetWithDefault(ctx context.Context, key string, defaultVal any) (any, error) {
	val, found, err := s.backend.Get(s.scope, s.getID(ctx), key)
	if err != nil {
		return nil, err
	}
	if !found {
		return defaultVal, nil
	}
	return val, nil
}

// Delete removes a key from this scope.
func (s *ScopedMemory) Delete(ctx context.Context, key string) error {
	return s.backend.Delete(s.scope, s.getID(ctx), key)
}

// List returns all keys in this scope.
func (s *ScopedMemory) List(ctx context.Context) ([]string, error) {
	return s.backend.List(s.scope, s.getID(ctx))
}

// GetTyped retrieves a value and unmarshals it into the provided type.
// This is useful when storing complex objects as JSON.
func (s *ScopedMemory) GetTyped(ctx context.Context, key string, dest any) error {
	val, found, err := s.backend.Get(s.scope, s.getID(ctx), key)
	if err != nil {
		return err
	}
	if !found {
		return nil
	}

	// If it's already the right type, try direct assignment
	// Otherwise, marshal/unmarshal through JSON for complex types
	switch v := val.(type) {
	case []byte:
		return json.Unmarshal(v, dest)
	case string:
		return json.Unmarshal([]byte(v), dest)
	default:
		// Round-trip through JSON for type conversion
		data, err := json.Marshal(val)
		if err != nil {
			return err
		}
		return json.Unmarshal(data, dest)
	}
}

// InMemoryBackend provides a thread-safe in-memory implementation of MemoryBackend.
// Data is lost when the process exits.
type InMemoryBackend struct {
	mu   sync.RWMutex
	data map[string]map[string]any // "scope:scopeID" -> key -> value
}

// NewInMemoryBackend creates a new in-memory storage backend.
func NewInMemoryBackend() *InMemoryBackend {
	return &InMemoryBackend{
		data: make(map[string]map[string]any),
	}
}

func (b *InMemoryBackend) compositeKey(scope MemoryScope, scopeID string) string {
	return string(scope) + ":" + scopeID
}

// Set stores a value.
func (b *InMemoryBackend) Set(scope MemoryScope, scopeID, key string, value any) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	ck := b.compositeKey(scope, scopeID)
	if b.data[ck] == nil {
		b.data[ck] = make(map[string]any)
	}
	b.data[ck][key] = value
	return nil
}

// Get retrieves a value.
func (b *InMemoryBackend) Get(scope MemoryScope, scopeID, key string) (any, bool, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	ck := b.compositeKey(scope, scopeID)
	if b.data[ck] == nil {
		return nil, false, nil
	}
	val, found := b.data[ck][key]
	return val, found, nil
}

// Delete removes a key.
func (b *InMemoryBackend) Delete(scope MemoryScope, scopeID, key string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	ck := b.compositeKey(scope, scopeID)
	if b.data[ck] != nil {
		delete(b.data[ck], key)
	}
	return nil
}

// List returns all keys in a scope.
func (b *InMemoryBackend) List(scope MemoryScope, scopeID string) ([]string, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	ck := b.compositeKey(scope, scopeID)
	if b.data[ck] == nil {
		return nil, nil
	}
	keys := make([]string, 0, len(b.data[ck]))
	for key := range b.data[ck] {
		keys = append(keys, key)
	}
	return keys, nil
}

// Clear removes all data from the backend.
// Useful for testing.
func (b *InMemoryBackend) Clear() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.data = make(map[string]map[string]any)
}

// ClearScope removes all data for a specific scope and scopeID.
func (b *InMemoryBackend) ClearScope(scope MemoryScope, scopeID string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	ck := b.compositeKey(scope, scopeID)
	delete(b.data, ck)
}
