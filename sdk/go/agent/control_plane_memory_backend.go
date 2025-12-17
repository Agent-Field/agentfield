package agent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// ControlPlaneMemoryBackend implements MemoryBackend by delegating to the Agentfield control plane
// distributed memory endpoints under `/api/v1/memory/*`.
//
// It preserves the SDK Memory API surface while making storage distributed and scope-aware.
type ControlPlaneMemoryBackend struct {
	baseURL     string
	token       string
	agentNodeID string
	httpClient  *http.Client
}

// NewControlPlaneMemoryBackend creates a distributed memory backend that uses the control plane.
// agentFieldURL should be the control plane base URL (e.g. http://localhost:8080).
func NewControlPlaneMemoryBackend(agentFieldURL, token, agentNodeID string) *ControlPlaneMemoryBackend {
	base := strings.TrimRight(strings.TrimSpace(agentFieldURL), "/")
	return &ControlPlaneMemoryBackend{
		baseURL:     base,
		token:       strings.TrimSpace(token),
		agentNodeID: strings.TrimSpace(agentNodeID),
		httpClient: &http.Client{
			Timeout: 15 * time.Second,
		},
	}
}

type memoryAPIResponse struct {
	Key       string `json:"key"`
	Data      any    `json:"data"`
	Scope     string `json:"scope"`
	ScopeID   string `json:"scope_id"`
	CreatedAt string `json:"created_at"`
	UpdatedAt string `json:"updated_at"`
}

func (b *ControlPlaneMemoryBackend) Set(scope MemoryScope, scopeID, key string, value any) error {
	endpoint, err := url.JoinPath(b.baseURL, "/api/v1/memory/set")
	if err != nil {
		return err
	}

	body := map[string]any{
		"key":   key,
		"data":  value,
		"scope": b.apiScope(scope),
	}
	req, err := http.NewRequest(http.MethodPost, endpoint, mustJSONReader(body))
	if err != nil {
		return err
	}
	b.applyHeaders(req, scope, scopeID)

	resp, err := b.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		msg, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("memory set failed: status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	return nil
}

func (b *ControlPlaneMemoryBackend) Get(scope MemoryScope, scopeID, key string) (any, bool, error) {
	endpoint, err := url.JoinPath(b.baseURL, "/api/v1/memory/get")
	if err != nil {
		return nil, false, err
	}

	body := map[string]any{
		"key":   key,
		"scope": b.apiScope(scope),
	}
	req, err := http.NewRequest(http.MethodPost, endpoint, mustJSONReader(body))
	if err != nil {
		return nil, false, err
	}
	b.applyHeaders(req, scope, scopeID)

	resp, err := b.httpClient.Do(req)
	if err != nil {
		return nil, false, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil, false, nil
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		msg, _ := io.ReadAll(resp.Body)
		return nil, false, fmt.Errorf("memory get failed: status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}

	var mem memoryAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&mem); err != nil {
		return nil, false, err
	}
	return mem.Data, true, nil
}

func (b *ControlPlaneMemoryBackend) Delete(scope MemoryScope, scopeID, key string) error {
	endpoint, err := url.JoinPath(b.baseURL, "/api/v1/memory/delete")
	if err != nil {
		return err
	}

	body := map[string]any{
		"key":   key,
		"scope": b.apiScope(scope),
	}
	req, err := http.NewRequest(http.MethodPost, endpoint, mustJSONReader(body))
	if err != nil {
		return err
	}
	b.applyHeaders(req, scope, scopeID)

	resp, err := b.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil
	}
	if resp.StatusCode != http.StatusNoContent && (resp.StatusCode < 200 || resp.StatusCode >= 300) {
		msg, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("memory delete failed: status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	return nil
}

func (b *ControlPlaneMemoryBackend) List(scope MemoryScope, scopeID string) ([]string, error) {
	endpoint, err := url.JoinPath(b.baseURL, "/api/v1/memory/list")
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest(http.MethodGet, endpoint+"?scope="+url.QueryEscape(b.apiScope(scope)), nil)
	if err != nil {
		return nil, err
	}
	b.applyHeaders(req, scope, scopeID)

	resp, err := b.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		msg, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("memory list failed: status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}

	var memories []memoryAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&memories); err != nil {
		return nil, err
	}

	keys := make([]string, 0, len(memories))
	for _, mem := range memories {
		if strings.TrimSpace(mem.Key) == "" {
			continue
		}
		keys = append(keys, mem.Key)
	}
	return keys, nil
}

func (b *ControlPlaneMemoryBackend) applyHeaders(req *http.Request, scope MemoryScope, scopeID string) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if b.token != "" {
		req.Header.Set("Authorization", "Bearer "+b.token)
	}
	if b.agentNodeID != "" {
		req.Header.Set("X-Agent-Node-ID", b.agentNodeID)
	}

	// Provide the scope ID via headers so the control plane can resolve scope_id consistently.
	switch b.apiScope(scope) {
	case "workflow":
		if scopeID != "" {
			req.Header.Set("X-Workflow-ID", scopeID)
		}
	case "session":
		if scopeID != "" {
			req.Header.Set("X-Session-ID", scopeID)
		}
	case "actor":
		if scopeID != "" {
			req.Header.Set("X-Actor-ID", scopeID)
		}
	case "global":
		// no header required
	}
}

func (b *ControlPlaneMemoryBackend) apiScope(scope MemoryScope) string {
	switch scope {
	case ScopeWorkflow:
		return "workflow"
	case ScopeSession:
		return "session"
	case ScopeUser:
		// API uses "actor" terminology.
		return "actor"
	case ScopeGlobal:
		return "global"
	default:
		return "global"
	}
}

func mustJSONReader(v any) io.Reader {
	data, _ := json.Marshal(v)
	return bytes.NewReader(data)
}
