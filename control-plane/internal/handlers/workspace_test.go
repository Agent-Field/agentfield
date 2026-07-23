package handlers

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/workspace"
	"github.com/gin-gonic/gin"
)

func workspaceTestRouter() *gin.Engine {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	g := r.Group("/api/v1/workspace")
	g.POST("/prepare", WorkspacePrepareHandler())
	g.PUT("/blobs/:sha", WorkspaceBlobPutHandler())
	g.GET("/blobs/:sha", WorkspaceBlobGetHandler())
	g.GET("/staged/:run_id", WorkspaceStagedHandler())
	return r
}

func TestWorkspacePrepareUploadFetch(t *testing.T) {
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	r := workspaceTestRouter()

	content := []byte("hello workspace")
	sha := workspace.HashBytes(content)
	manifest := &workspace.Manifest{
		Version: 1,
		Files:   []workspace.FileEntry{{Path: "a.txt", Size: int64(len(content)), Mode: 0o644, MtimeNS: 1, SHA256: sha}},
	}

	// prepare -> missing contains the blob
	body, _ := json.Marshal(map[string]interface{}{"manifest": manifest})
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/v1/workspace/prepare", bytes.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("prepare status = %d, body %s", w.Code, w.Body.String())
	}
	var prep struct {
		Missing []string `json:"missing"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &prep); err != nil {
		t.Fatalf("decode prepare: %v", err)
	}
	if len(prep.Missing) != 1 || prep.Missing[0] != sha {
		t.Fatalf("expected missing=[%s], got %+v", sha, prep.Missing)
	}

	// upload the blob
	w = httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodPut, "/api/v1/workspace/blobs/"+sha, bytes.NewReader(content)))
	if w.Code != http.StatusNoContent {
		t.Fatalf("blob PUT status = %d, body %s", w.Code, w.Body.String())
	}

	// prepare again -> nothing missing
	w = httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/v1/workspace/prepare", bytes.NewReader(body)))
	var prep2 struct {
		Missing []string `json:"missing"`
	}
	_ = json.Unmarshal(w.Body.Bytes(), &prep2)
	if len(prep2.Missing) != 0 {
		t.Fatalf("expected no missing after upload, got %+v", prep2.Missing)
	}

	// GET the blob back
	w = httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/v1/workspace/blobs/"+sha, nil))
	if w.Code != http.StatusOK {
		t.Fatalf("blob GET status = %d", w.Code)
	}
	got, _ := io.ReadAll(w.Body)
	if !bytes.Equal(got, content) {
		t.Fatalf("blob content mismatch: %q", got)
	}
}

func TestWorkspaceBlobPutRejectsHashMismatch(t *testing.T) {
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	r := workspaceTestRouter()
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodPut, "/api/v1/workspace/blobs/deadbeef", bytes.NewReader([]byte("not deadbeef"))))
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for hash mismatch, got %d", w.Code)
	}
}

func TestWorkspaceStagedFetch(t *testing.T) {
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	r := workspaceTestRouter()
	runID := "run-xyz"

	// 404 before anything is staged
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/v1/workspace/staged/"+runID, nil))
	if w.Code != http.StatusNotFound {
		t.Fatalf("expected 404 before staging, got %d", w.Code)
	}

	// Stage a diff the way the control plane does after an execution.
	if err := workspace.MergeStaged(runID, func(rec *workspace.StagedRecord) {
		rec.ExecutionID = "exec-1"
		rec.ManifestID = "mid"
		rec.Diff = &workspace.Diff{
			Changed: []workspace.ChangedFile{{Path: "out.txt", SHA256: "h", Size: 3, Mode: 0o644}},
			Deleted: []string{"old.txt"},
		}
		rec.NodeBaseURL = "http://node:9000"
	}); err != nil {
		t.Fatalf("stage: %v", err)
	}

	w = httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/v1/workspace/staged/"+runID, nil))
	if w.Code != http.StatusOK {
		t.Fatalf("staged status = %d, body %s", w.Code, w.Body.String())
	}
	var resp workspaceStagedResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode staged: %v", err)
	}
	if resp.RunID != runID || resp.ExecutionID != "exec-1" || resp.Diff == nil {
		t.Fatalf("unexpected staged response: %+v", resp)
	}
	if len(resp.Diff.Changed) != 1 || resp.Diff.Changed[0].Path != "out.txt" {
		t.Fatalf("unexpected diff: %+v", resp.Diff)
	}
	// Internal node routing must not leak to the CLI-facing response.
	if bytes.Contains(w.Body.Bytes(), []byte("node:9000")) {
		t.Fatalf("staged response leaked node base URL: %s", w.Body.String())
	}
}

func TestExtractWorkspaceDiff(t *testing.T) {
	result := []byte(`{"answer":42,"workspace_diff":{"changed":[{"path":"a.txt","sha256":"h","size":1,"mode":420}],"deleted":["b.txt"]}}`)
	diff, stripped, present := extractWorkspaceDiff(result)
	if !present {
		t.Fatalf("expected workspace_diff to be present")
	}
	if diff == nil || len(diff.Changed) != 1 || diff.Changed[0].Path != "a.txt" || len(diff.Deleted) != 1 {
		t.Fatalf("unexpected diff: %+v", diff)
	}
	var decoded map[string]interface{}
	if err := json.Unmarshal(stripped, &decoded); err != nil {
		t.Fatalf("stripped not valid json: %v", err)
	}
	if _, ok := decoded["workspace_diff"]; ok {
		t.Fatalf("workspace_diff should have been stripped: %s", stripped)
	}
	if decoded["answer"].(float64) != 42 {
		t.Fatalf("reasoner output should survive stripping: %v", decoded)
	}

	// No workspace_diff key -> unchanged, not present.
	plain := []byte(`{"answer":1}`)
	if _, _, present := extractWorkspaceDiff(plain); present {
		t.Fatalf("plain result should report no workspace_diff")
	}

	// Non-dict reasoner results are wrapped by the SDK as
	// {"result": <original>, "workspace_diff": {...}}; workspace_diff is still a
	// top-level key, so extraction must find it and leave the wrapped result.
	wrapped := []byte(`{"result":"a plain string","workspace_diff":{"changed":[],"deleted":["x.txt"]}}`)
	diff, stripped, present = extractWorkspaceDiff(wrapped)
	if !present || diff == nil || len(diff.Deleted) != 1 || diff.Deleted[0] != "x.txt" {
		t.Fatalf("wrapped non-dict diff not extracted: present=%v diff=%+v", present, diff)
	}
	var wrappedDecoded map[string]interface{}
	if err := json.Unmarshal(stripped, &wrappedDecoded); err != nil {
		t.Fatalf("stripped wrapped not valid json: %v", err)
	}
	if _, ok := wrappedDecoded["workspace_diff"]; ok {
		t.Fatalf("workspace_diff should be stripped from wrapped result")
	}
	if wrappedDecoded["result"] != "a plain string" {
		t.Fatalf("wrapped result payload should survive: %v", wrappedDecoded)
	}
}
