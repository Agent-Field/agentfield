package skillkit

import (
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/codeaf"
)

func TestInstallAllEnsuresCodeafExceptDryRun(t *testing.T) {
	old := ensureCodeaf
	t.Cleanup(func() { ensureCodeaf = old })
	calls := 0
	ensureCodeaf = func(opts codeaf.Options, w io.Writer) error {
		calls++
		if w != os.Stderr {
			t.Error("warnings must use stderr")
		}
		return nil
	}
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("CODEX_HOME", filepath.Join(home, "codex"))
	t.Setenv("AGENTFIELD_SKIP_FURROW", "1")
	if _, err := InstallAll(InstallOptions{Targets: []string{"codex"}, DryRun: true}); err != nil {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatal("dry run provisioned")
	}
	if _, err := InstallAll(InstallOptions{Targets: []string{"codex"}}); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("calls=%d, want one for all catalog entries", calls)
	}
}
