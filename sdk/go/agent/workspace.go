package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/Agent-Field/agentfield/sdk/go/workspace"
)

// ─────────────────────────────────────────────────────────────────────────────
// Workspace artifacts: node-side transport + per-execution materialize/diff.
//
// A caller may attach a local folder to any reasoner execution. The control
// plane transports the sealed folder to the node as a manifest plus blobs
// (via the endpoints below), the SDK materializes it into a per-execution
// directory, runs the reasoner with that directory exposed through
// WorkspaceDir(ctx), then diffs the folder and attaches the staged changes to
// the result as `workspace_diff`. Reasoner authors write no folder-handling
// code: they read the directory via WorkspaceDir(ctx) and set cmd.Dir on any
// subprocess they spawn. See the frozen contract in
// docs/design/workspace-artifacts.md.
// ─────────────────────────────────────────────────────────────────────────────

// workspaceDirKey is the context key under which the absolute path of the
// per-execution workspace directory is stored.
type workspaceDirKey struct{}

// WorkspaceDir returns the absolute path of the workspace directory materialized
// for the current execution, and true, when the running reasoner was invoked
// with an attached workspace. It returns ("", false) otherwise.
//
// The path is per-execution and carried on the context (never a process-global
// env var or a process-wide chdir, which would race under concurrent requests).
// Reasoner authors should read files relative to this directory and set
// cmd.Dir = dir on any subprocess they spawn so the child runs inside the
// workspace:
//
//	dir, ok := agent.WorkspaceDir(ctx)
//	if ok {
//	    cmd := exec.CommandContext(ctx, "go", "test", "./...")
//	    cmd.Dir = dir
//	    out, err := cmd.CombinedOutput()
//	}
//
// The AGENTFIELD_WORKSPACE env var is intentionally NOT set: a per-execution
// isolation worker (matching the "worker semantics" of the spec) is a
// documented follow-up. In a single Go process the working directory cannot be
// switched per request, so WorkspaceDir + cmd.Dir is the POC contract.
func WorkspaceDir(ctx context.Context) (string, bool) {
	if ctx == nil {
		return "", false
	}
	if dir, ok := ctx.Value(workspaceDirKey{}).(string); ok && dir != "" {
		return dir, true
	}
	return "", false
}

// contextWithWorkspaceDir binds the per-execution workspace directory to ctx.
func contextWithWorkspaceDir(ctx context.Context, dir string) context.Context {
	return context.WithValue(ctx, workspaceDirKey{}, dir)
}

// workspaceStore returns the node's shared content store, bound lazily to
// ~/.agentfield/cas (honoring AGENTFIELD_HOME).
func (a *Agent) workspaceStore() *workspace.CAS {
	a.wsStoreOnce.Do(func() {
		a.wsStore = workspace.NewCAS(workspace.DefaultCASDir())
	})
	return a.wsStore
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto-registered node endpoints
// ─────────────────────────────────────────────────────────────────────────────

// installWorkspaceRoutes registers the three control-plane-initiated workspace
// endpoints on the node's mux:
//
//   - POST /api/v1/workspace/prepare        {"manifest": <manifest>} → {"missing": [...]}
//   - PUT  /api/v1/workspace/blobs/{sha256}  raw bytes → 204 (400 on hash mismatch)
//   - GET  /api/v1/workspace/blobs/{sha256}  → raw bytes (404 if absent)
//
// They share one ContentStore at ~/.agentfield/cas. All three are exercised by
// the control plane before/after dispatch (they work when the control plane is
// behind NAT and the node is remote).
func (a *Agent) installWorkspaceRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/api/v1/workspace/prepare", a.handleWorkspacePrepare)
	mux.HandleFunc("/api/v1/workspace/blobs/", a.handleWorkspaceBlob)
}

func (a *Agent) handleWorkspacePrepare(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()

	var body struct {
		Manifest *workspace.Manifest `json:"manifest"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	if body.Manifest == nil {
		http.Error(w, `expected {"manifest": <manifest>}`, http.StatusBadRequest)
		return
	}

	missing := workspace.MissingBlobs(body.Manifest, a.workspaceStore())
	if missing == nil {
		missing = []string{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"missing": missing})
}

func (a *Agent) handleWorkspaceBlob(w http.ResponseWriter, r *http.Request) {
	sha := strings.TrimPrefix(r.URL.Path, "/api/v1/workspace/blobs/")
	if !isSHA256Hex(sha) {
		http.Error(w, "invalid sha256", http.StatusBadRequest)
		return
	}
	store := a.workspaceStore()

	switch r.Method {
	case http.MethodPut:
		defer r.Body.Close()
		data, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "failed to read blob body", http.StatusBadRequest)
			return
		}
		if err := store.PutVerified(sha, data); err != nil {
			http.Error(w, "blob hash mismatch", http.StatusBadRequest)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	case http.MethodGet:
		if !store.Has(sha) {
			http.Error(w, "blob not found", http.StatusNotFound)
			return
		}
		data, err := store.Get(sha)
		if err != nil {
			http.Error(w, "blob not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(data)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// isSHA256Hex reports whether s is a 64-character lowercase hex string.
func isSHA256Hex(s string) bool {
	if len(s) != 64 {
		return false
	}
	for _, c := range s {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

// ─────────────────────────────────────────────────────────────────────────────
// Execution hook
// ─────────────────────────────────────────────────────────────────────────────

// extractWorkspaceManifest pulls the inline manifest out of a lifted top-level
// `artifacts` envelope (shape: {"workspace": {"manifest": {...}}}). It returns
// nil for any missing or malformed envelope, so the normal (non-workspace)
// execution path is never disturbed.
func extractWorkspaceManifest(artifacts any) *workspace.Manifest {
	envelope, ok := artifacts.(map[string]any)
	if !ok {
		return nil
	}
	ws, ok := envelope["workspace"].(map[string]any)
	if !ok {
		return nil
	}
	manifestRaw, ok := ws["manifest"].(map[string]any)
	if !ok {
		return nil
	}
	if _, ok := manifestRaw["files"].([]any); !ok {
		return nil
	}
	// Re-encode the decoded manifest map and decode it into the typed manifest;
	// this is the simplest lossless conversion from the generic JSON map.
	encoded, err := json.Marshal(manifestRaw)
	if err != nil {
		return nil
	}
	var manifest workspace.Manifest
	if err := json.Unmarshal(encoded, &manifest); err != nil {
		return nil
	}
	return &manifest
}

// invokeWithWorkspace runs a reasoner/skill handler, transparently handling an
// attached workspace. When artifacts carry a workspace manifest the handler runs
// inside a materialized per-execution directory (exposed via WorkspaceDir(ctx))
// and the result gains a `workspace_diff`. When no workspace is attached this is
// a byte-for-byte passthrough to handler(ctx, input) — non-workspace executions
// keep their exact existing behavior.
func (a *Agent) invokeWithWorkspace(ctx context.Context, handler HandlerFunc, input map[string]any, artifacts any) (any, error) {
	manifest := extractWorkspaceManifest(artifacts)
	if manifest == nil {
		return handler(ctx, input)
	}
	execID := executionContextFrom(ctx).ExecutionID
	return a.runInWorkspace(ctx, handler, input, manifest, execID)
}

// runInWorkspace materializes manifest into ~/.agentfield/workspaces/<execID>/,
// runs handler with that directory exposed through WorkspaceDir(ctx), diffs the
// directory against the input manifest (storing new blobs in the CAS), and
// attaches the diff to the result. The directory is removed on success and kept
// (path logged) on failure for inspection.
func (a *Agent) runInWorkspace(ctx context.Context, handler HandlerFunc, input map[string]any, manifest *workspace.Manifest, execID string) (any, error) {
	store := a.workspaceStore()

	if missing := workspace.MissingBlobs(manifest, store); len(missing) > 0 {
		return nil, fmt.Errorf("cannot materialize workspace: missing %d blob(s) from the node content store: %s",
			len(missing), strings.Join(missing, ", "))
	}

	if strings.TrimSpace(execID) == "" {
		execID = generateExecutionID()
	}
	wsDir := filepath.Join(workspace.WorkspacesDir(), execID)

	if err := workspace.Materialize(manifest, wsDir, store); err != nil {
		return nil, fmt.Errorf("materialize workspace: %w", err)
	}

	keepOnError := false
	defer func() {
		if !keepOnError {
			_ = os.RemoveAll(wsDir)
		}
	}()

	result, err := handler(contextWithWorkspaceDir(ctx, wsDir), input)
	if err != nil {
		keepOnError = true
		a.logger.Printf("workspace execution failed; preserving directory for inspection: %s", wsDir)
		return nil, err
	}

	diff, err := workspace.ComputeDiff(manifest, wsDir, store)
	if err != nil {
		return nil, fmt.Errorf("compute workspace diff: %w", err)
	}

	return attachWorkspaceDiff(result, diff), nil
}

// attachWorkspaceDiff attaches the staged workspace_diff to a reasoner result.
// A map result gains a `workspace_diff` key in place; any other result shape is
// wrapped as {"result": <original>, "workspace_diff": ...} so the diff always
// reaches the caller without losing the payload.
func attachWorkspaceDiff(result any, diff *workspace.Diff) any {
	payload := normalizeDiff(diff)
	if m, ok := result.(map[string]any); ok {
		enriched := make(map[string]any, len(m)+1)
		for k, v := range m {
			enriched[k] = v
		}
		enriched["workspace_diff"] = payload
		return enriched
	}
	return map[string]any{"result": result, "workspace_diff": payload}
}

// normalizeDiff guarantees the diff's changed/deleted lists serialize as empty
// JSON arrays rather than null, matching the Python SDK's on-the-wire shape.
func normalizeDiff(diff *workspace.Diff) *workspace.Diff {
	if diff == nil {
		return &workspace.Diff{Changed: []workspace.ChangedFile{}, Deleted: []string{}}
	}
	if diff.Changed == nil {
		diff.Changed = []workspace.ChangedFile{}
	}
	if diff.Deleted == nil {
		diff.Deleted = []string{}
	}
	return diff
}

// liftWorkspaceArtifacts removes the reserved top-level "artifacts" key from a
// decoded request body and returns its value (or nil when absent). The
// artifacts envelope rides alongside the reasoner input fields; lifting it out
// before input handling keeps the reasoner's own input untouched.
func liftWorkspaceArtifacts(body map[string]any) any {
	if body == nil {
		return nil
	}
	artifacts, ok := body["artifacts"]
	if !ok {
		return nil
	}
	delete(body, "artifacts")
	return artifacts
}
