package skillkit

import (
	"crypto/sha256"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestInstallAgentfieldUseEnsuresFurrow(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("CODEX_HOME", filepath.Join(home, "codex"))

	payload := []byte("furrow from skill install")
	sum := sha256.Sum256(payload)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/SHA256SUMS":
			for _, asset := range []string{"furrow-linux-amd64", "furrow-darwin-arm64", "furrow-darwin-amd64"} {
				_, _ = fmt.Fprintf(w, "%x  %s\n", sum, asset)
			}
		case "/furrow-linux-amd64", "/furrow-darwin-arm64", "/furrow-darwin-amd64":
			_, _ = w.Write(payload)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	t.Setenv("AGENTFIELD_FURROW_BASE_URL", server.URL)

	if _, err := Install(InstallOptions{SkillName: "agentfield-use", Targets: []string{"codex"}}); err != nil {
		t.Fatalf("Install(agentfield-use): %v", err)
	}
	info, err := os.Stat(filepath.Join(home, "bin", "furrow"))
	if err != nil {
		t.Fatalf("stat provisioned furrow: %v", err)
	}
	if info.Mode().Perm()&0o111 == 0 {
		t.Fatalf("provisioned furrow mode = %o, want executable", info.Mode().Perm())
	}
}
