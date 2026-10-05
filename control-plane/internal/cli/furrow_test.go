package cli

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/furrow"
)

func TestFurrowEnsureCommand(t *testing.T) {
	t.Setenv("AGENTFIELD_SKIP_FURROW", "1")
	cmd := NewFurrowCommand()
	cmd.SetArgs([]string{"ensure"})
	if err := cmd.Execute(); err != nil {
		t.Fatal(err)
	}
}

func TestFurrowEnsureCommandSurfacesFailure(t *testing.T) {
	home := t.TempDir()
	if err := os.WriteFile(filepath.Join(home, "bin"), []byte("not a directory"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("AGENTFIELD_SKIP_FURROW", "")
	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("HOME", home)

	cmd := NewFurrowCommand()
	cmd.SetArgs([]string{"ensure"})
	err := cmd.Execute()
	if _, supported := furrow.AssetName(runtime.GOOS, runtime.GOARCH); !supported {
		if err != nil {
			t.Fatalf("unsupported platform should be a no-op: %v", err)
		}
		return
	}
	if err == nil || !strings.Contains(err.Error(), "create furrow bin directory") {
		t.Fatalf("error = %v", err)
	}
}
