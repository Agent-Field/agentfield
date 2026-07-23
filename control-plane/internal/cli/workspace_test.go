package cli

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/workspace"
)

func writeTestFile(t *testing.T, dir, rel, content string) {
	t.Helper()
	full := filepath.Join(dir, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", rel, err)
	}
}

func assertTestFile(t *testing.T, dir, rel, want string) {
	t.Helper()
	got, err := os.ReadFile(filepath.Join(dir, filepath.FromSlash(rel)))
	if err != nil {
		t.Fatalf("read %s: %v", rel, err)
	}
	if string(got) != want {
		t.Fatalf("%s = %q, want %q", rel, got, want)
	}
}

func fileExists(dir, rel string) bool {
	_, err := os.Stat(filepath.Join(dir, filepath.FromSlash(rel)))
	return err == nil
}

// startWorkspaceServer stands up a mock of the control-plane workspace endpoints
// (matching the real handlers' contract) so the CLI's HTTP client can be
// exercised over a real socket without an import cycle back into handlers. It is
// backed by the same workspace primitives the real handlers use.
func startWorkspaceServer(t *testing.T) *httptest.Server {
	t.Helper()
	cas := workspace.NewCAS(workspace.DefaultCASDir())
	mux := http.NewServeMux()

	mux.HandleFunc("/api/v1/workspace/prepare", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Manifest *workspace.Manifest `json:"manifest"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		missing := []string{}
		if req.Manifest != nil {
			for _, f := range req.Manifest.Files {
				if !cas.Has(f.SHA256) {
					missing = append(missing, f.SHA256)
				}
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"missing": missing})
	})

	mux.HandleFunc("/api/v1/workspace/blobs/", func(w http.ResponseWriter, r *http.Request) {
		sha := strings.TrimPrefix(r.URL.Path, "/api/v1/workspace/blobs/")
		switch r.Method {
		case http.MethodPut:
			data, _ := io.ReadAll(r.Body)
			if err := cas.PutVerified(sha, data); err != nil {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			w.WriteHeader(http.StatusNoContent)
		case http.MethodGet:
			if !cas.Has(sha) {
				w.WriteHeader(http.StatusNotFound)
				return
			}
			data, _ := cas.Get(sha)
			w.Header().Set("Content-Type", "application/octet-stream")
			_, _ = w.Write(data)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	})

	mux.HandleFunc("/api/v1/workspace/staged/", func(w http.ResponseWriter, r *http.Request) {
		runID := strings.TrimPrefix(r.URL.Path, "/api/v1/workspace/staged/")
		rec, _ := workspace.LoadStaged(runID)
		if rec == nil || rec.Diff == nil {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_ = json.NewEncoder(w).Encode(stagedDiffResponse{
			RunID:       rec.RunID,
			ExecutionID: rec.ExecutionID,
			ManifestID:  rec.ManifestID,
			Diff:        rec.Diff,
		})
	})

	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func TestCLIWorkspaceHTTPRoundTrip(t *testing.T) {
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	srv := startWorkspaceServer(t)
	t.Setenv("AGENTFIELD_SERVER", srv.URL)

	ctx := context.Background()

	// Blob upload/download round trip against the real endpoints.
	content := []byte("cli round trip content")
	sha := workspace.HashBytes(content)
	if err := cpUploadBlob(ctx, sha, content); err != nil {
		t.Fatalf("cpUploadBlob: %v", err)
	}
	got, err := cpDownloadBlob(ctx, sha)
	if err != nil {
		t.Fatalf("cpDownloadBlob: %v", err)
	}
	if string(got) != string(content) {
		t.Fatalf("blob round trip mismatch: %q", got)
	}

	// prepare reports a not-yet-uploaded blob as missing.
	other := []byte("second blob")
	otherSha := workspace.HashBytes(other)
	manifest := &workspace.Manifest{
		Version: 1,
		Files: []workspace.FileEntry{
			{Path: "have.txt", SHA256: sha, Size: int64(len(content)), Mode: 0o644},
			{Path: "missing.txt", SHA256: otherSha, Size: int64(len(other)), Mode: 0o644},
		},
	}
	missing, err := cpPrepareWorkspace(ctx, manifest)
	if err != nil {
		t.Fatalf("cpPrepareWorkspace: %v", err)
	}
	if len(missing) != 1 || missing[0] != otherSha {
		t.Fatalf("expected missing=[%s], got %+v", otherSha, missing)
	}
}

func TestCLIUploadBlobsFallback(t *testing.T) {
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	// startWorkspaceServer serves per-blob PUT but no batch route, so the shared
	// uploader must transparently fall back to parallel PUTs.
	srv := startWorkspaceServer(t)
	t.Setenv("AGENTFIELD_SERVER", srv.URL)

	cas := workspace.NewCAS(workspace.DefaultCASDir())
	shas := make([]string, 0, 6)
	for i := 0; i < 6; i++ {
		sha, err := cas.PutBytes([]byte("cli fallback blob " + string(rune('a'+i))))
		if err != nil {
			t.Fatalf("seed blob: %v", err)
		}
		shas = append(shas, sha)
	}

	stats, err := cpUploadBlobs(context.Background(), cas, shas)
	if err != nil {
		t.Fatalf("cpUploadBlobs: %v", err)
	}
	if stats.Mode != "parallel-fallback" {
		t.Fatalf("mode = %q, want parallel-fallback", stats.Mode)
	}
	if stats.Blobs != len(shas) {
		t.Fatalf("uploaded %d, want %d", stats.Blobs, len(shas))
	}
	// Each blob is now downloadable from the server.
	for _, sha := range shas {
		if _, err := cpDownloadBlob(context.Background(), sha); err != nil {
			t.Fatalf("blob %s not on server after upload: %v", sha, err)
		}
	}
}

func TestCLIDiffAndApplyEndToEnd(t *testing.T) {
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	srv := startWorkspaceServer(t)
	t.Setenv("AGENTFIELD_SERVER", srv.URL)
	ctx := context.Background()
	runID := "run-e2e"

	// Seal an original folder (this is what `af call --dir` does locally).
	orig := t.TempDir()
	writeTestFile(t, orig, "a.txt", "sealed-a")
	writeTestFile(t, orig, "gone.txt", "to-remove")
	cas := workspace.NewCAS(workspace.DefaultCASDir())
	inputManifest, err := workspace.Seal(orig, cas)
	if err != nil {
		t.Fatalf("seal: %v", err)
	}

	// CLI-side staged record (original dir + input manifest).
	if err := stageWorkspaceCall(nil, runID, "exec-e2e", &sealedWorkspace{
		originalDir: orig,
		manifest:    inputManifest,
	}); err != nil {
		t.Fatalf("stageWorkspaceCall: %v", err)
	}

	// Simulate the control plane recording a diff (as processWorkspaceResult
	// does), with the changed blob available in the shared store.
	newA := "node-changed-a"
	shaA, _ := cas.PutBytes([]byte(newA))
	if err := workspace.MergeStaged(runID, func(rec *workspace.StagedRecord) {
		rec.Diff = &workspace.Diff{
			Changed: []workspace.ChangedFile{{Path: "a.txt", SHA256: shaA, Size: int64(len(newA)), Mode: 0o644}},
			Deleted: []string{"gone.txt"},
		}
		rec.NodeBaseURL = "http://node:9000"
	}); err != nil {
		t.Fatalf("merge staged diff: %v", err)
	}

	// `af diff` fetches the diff from the control plane over HTTP.
	if err := runDiff(ctx, runID, "json", io.Discard, io.Discard); err != nil {
		t.Fatalf("runDiff: %v", err)
	}

	// `af apply` fetches diff + blobs from the control plane and writes them back.
	if err := runApply(ctx, runID, false, "json", io.Discard, io.Discard); err != nil {
		t.Fatalf("runApply: %v", err)
	}
	assertTestFile(t, orig, "a.txt", newA)
	if fileExists(orig, "gone.txt") {
		t.Fatalf("gone.txt should have been deleted by apply")
	}
}
