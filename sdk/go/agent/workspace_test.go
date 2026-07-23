package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/Agent-Field/agentfield/sdk/go/workspace"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newWorkspaceAgent(t *testing.T) *Agent {
	t.Helper()
	// Redirect the CAS + workspaces roots into a throwaway home so tests never
	// touch the developer's ~/.agentfield.
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	a, err := New(Config{
		NodeID:  "ws-node",
		Version: "1.0.0",
		Logger:  log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	return a
}

// TestWorkspaceBlobEndpointsRoundTrip exercises prepare → put → get and the
// hash-mismatch / not-found / bad-sha rejections on the auto-registered node
// endpoints.
func TestWorkspaceBlobEndpointsRoundTrip(t *testing.T) {
	a := newWorkspaceAgent(t)
	handler := a.Handler()

	content := []byte("hello workspace")
	sha := workspace.HashBytes(content)

	manifest := &workspace.Manifest{
		Version: 1,
		Files: []workspace.FileEntry{
			{Path: "hello.txt", Size: int64(len(content)), Mode: 0o644, SHA256: sha},
		},
	}

	// prepare: the blob is not yet in the store, so it must be reported missing.
	prepBody, _ := json.Marshal(map[string]any{"manifest": manifest})
	resp := doReq(handler, http.MethodPost, "/api/v1/workspace/prepare", prepBody)
	require.Equal(t, http.StatusOK, resp.Code)
	var prep struct {
		Missing []string `json:"missing"`
	}
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &prep))
	assert.Equal(t, []string{sha}, prep.Missing)

	// put the blob.
	resp = doReq(handler, http.MethodPut, "/api/v1/workspace/blobs/"+sha, content)
	assert.Equal(t, http.StatusNoContent, resp.Code)

	// prepare again: nothing missing now.
	resp = doReq(handler, http.MethodPost, "/api/v1/workspace/prepare", prepBody)
	require.Equal(t, http.StatusOK, resp.Code)
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &prep))
	assert.Empty(t, prep.Missing)

	// get the blob back.
	resp = doReq(handler, http.MethodGet, "/api/v1/workspace/blobs/"+sha, nil)
	require.Equal(t, http.StatusOK, resp.Code)
	assert.Equal(t, content, resp.Body.Bytes())

	// hash-mismatch upload: content does not hash to the claimed key -> 400.
	wrongSha := workspace.HashBytes([]byte("something else"))
	resp = doReq(handler, http.MethodPut, "/api/v1/workspace/blobs/"+wrongSha, content)
	assert.Equal(t, http.StatusBadRequest, resp.Code)

	// unknown blob -> 404.
	absent := workspace.HashBytes([]byte("never stored"))
	resp = doReq(handler, http.MethodGet, "/api/v1/workspace/blobs/"+absent, nil)
	assert.Equal(t, http.StatusNotFound, resp.Code)

	// malformed sha -> 400 on both verbs.
	resp = doReq(handler, http.MethodGet, "/api/v1/workspace/blobs/not-a-sha", nil)
	assert.Equal(t, http.StatusBadRequest, resp.Code)
	resp = doReq(handler, http.MethodPut, "/api/v1/workspace/blobs/not-a-sha", content)
	assert.Equal(t, http.StatusBadRequest, resp.Code)
}

// TestWorkspaceExecutionHook proves a reasoner invoked with an attached
// workspace (a) runs with WorkspaceDir(ctx) exposed, (b) sees clean input with
// the reserved "artifacts" envelope lifted out, and (c) returns a workspace_diff
// reflecting the changes it made.
func TestWorkspaceExecutionHook(t *testing.T) {
	a := newWorkspaceAgent(t)

	// Seal a source folder into the node CAS so its blobs are materializable.
	src := t.TempDir()
	writeTestFile(t, src, "a.txt", "alpha")
	writeTestFile(t, src, "b.txt", "beta")

	cas := workspace.NewCAS(workspace.DefaultCASDir())
	manifest, err := workspace.Seal(src, cas)
	require.NoError(t, err)

	var sawWorkspace bool
	a.RegisterReasoner("edit", func(ctx context.Context, input map[string]any) (any, error) {
		dir, ok := WorkspaceDir(ctx)
		sawWorkspace = ok

		aContent, _ := os.ReadFile(filepath.Join(dir, "a.txt"))
		// Mutate the workspace: modify a.txt, create c.txt, delete b.txt.
		_ = os.WriteFile(filepath.Join(dir, "a.txt"), []byte("alpha-modified"), 0o644)
		_ = os.WriteFile(filepath.Join(dir, "c.txt"), []byte("gamma"), 0o644)
		_ = os.Remove(filepath.Join(dir, "b.txt"))

		// Echo the input keys back so the test can assert "artifacts" never
		// leaked into the reasoner's input.
		keys := make([]string, 0, len(input))
		for k := range input {
			keys = append(keys, k)
		}
		return map[string]any{
			"saw_workspace": ok,
			"workspace_dir": dir,
			"a_content":     string(aContent),
			"input_keys":    keys,
		}, nil
	})

	body, _ := json.Marshal(map[string]any{
		"target": "edit",
		"input":  map[string]any{"foo": "bar"},
		"artifacts": map[string]any{
			"workspace": map[string]any{"manifest": manifest},
		},
	})

	resp := doReq(a.Handler(), http.MethodPost, "/execute", body)
	require.Equal(t, http.StatusOK, resp.Code, resp.Body.String())

	var out map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &out))

	// (a) the reasoner ran inside the materialized workspace.
	assert.True(t, sawWorkspace, "reasoner must observe WorkspaceDir(ctx)")
	assert.Equal(t, "alpha", out["a_content"], "reasoner must read the materialized file content")

	// (b) input validation undisturbed: only the declared field is present.
	keys, _ := out["input_keys"].([]any)
	assert.ElementsMatch(t, []any{"foo"}, keys, "artifacts must be lifted out of reasoner input")

	// (c) the diff came back and reflects the mutations.
	diffRaw, ok := out["workspace_diff"]
	require.True(t, ok, "result must carry workspace_diff")
	diffBytes, _ := json.Marshal(diffRaw)
	var diff workspace.Diff
	require.NoError(t, json.Unmarshal(diffBytes, &diff))

	changed := map[string]bool{}
	for _, c := range diff.Changed {
		changed[c.Path] = true
	}
	assert.True(t, changed["a.txt"], "modified a.txt must be in changed")
	assert.True(t, changed["c.txt"], "new c.txt must be in changed")
	assert.Equal(t, []string{"b.txt"}, diff.Deleted, "deleted b.txt must be reported")

	// The workspace directory is cleaned up on success.
	_, statErr := os.Stat(out["workspace_dir"].(string))
	assert.True(t, os.IsNotExist(statErr), "workspace dir must be removed after a successful execution")
}

// TestNonWorkspaceExecutionUnchanged confirms a plain execution (no artifacts)
// returns the reasoner's result verbatim, with no workspace_diff key.
func TestNonWorkspaceExecutionUnchanged(t *testing.T) {
	a := newWorkspaceAgent(t)
	a.RegisterReasoner("echo", func(ctx context.Context, input map[string]any) (any, error) {
		_, ok := WorkspaceDir(ctx)
		assert.False(t, ok, "no workspace must be exposed for a plain execution")
		return map[string]any{"echo": input["value"]}, nil
	})

	body, _ := json.Marshal(map[string]any{"target": "echo", "input": map[string]any{"value": "hi"}})
	resp := doReq(a.Handler(), http.MethodPost, "/execute", body)
	require.Equal(t, http.StatusOK, resp.Code)

	var out map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &out))
	assert.Equal(t, "hi", out["echo"])
	_, hasDiff := out["workspace_diff"]
	assert.False(t, hasDiff, "plain executions must not gain a workspace_diff")
}

// TestAttachWorkspaceDiffNonMapResult verifies a non-map reasoner result is
// wrapped as {"result": <original>, "workspace_diff": ...}.
func TestAttachWorkspaceDiffNonMapResult(t *testing.T) {
	diff := &workspace.Diff{Changed: []workspace.ChangedFile{{Path: "x", SHA256: "h"}}}
	out := attachWorkspaceDiff("scalar-result", diff)
	m, ok := out.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "scalar-result", m["result"])
	assert.NotNil(t, m["workspace_diff"])
}

func doReq(h http.Handler, method, path string, body []byte) *httptest.ResponseRecorder {
	var r *http.Request
	if body != nil {
		r = httptest.NewRequest(method, path, bytes.NewReader(body))
	} else {
		r = httptest.NewRequest(method, path, nil)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, r)
	return rec
}

func writeTestFile(t *testing.T, dir, rel, content string) {
	t.Helper()
	full := filepath.Join(dir, rel)
	require.NoError(t, os.MkdirAll(filepath.Dir(full), 0o755))
	require.NoError(t, os.WriteFile(full, []byte(content), 0o644))
}
