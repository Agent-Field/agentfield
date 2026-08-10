package cli

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
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
	if err == nil || !strings.Contains(err.Error(), "create furrow bin directory") {
		t.Fatalf("error = %v", err)
	}
}
