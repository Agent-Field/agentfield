package packages

import (
	"os"
	"path/filepath"
	"testing"
)

// Contract: InspectSource parses a local directory's manifest in place, without
// requiring a startable node and without touching any AgentField home.
func TestInspectSource_LocalDirectory(t *testing.T) {
	dir := t.TempDir()
	manifest := `config_version: v1
name: inspect-demo
version: 0.1.0
user_environment:
  required:
    - name: API_KEY
      type: secret
  optional:
    - name: REGION
      default: us-east-1
  require_one_of:
    - id: llm
      options:
        - name: ANTHROPIC_API_KEY
        - name: OPENROUTER_API_KEY
`
	if err := os.WriteFile(filepath.Join(dir, "agentfield-package.yaml"), []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}

	m, err := InspectSource(dir)
	if err != nil {
		t.Fatalf("InspectSource: %v", err)
	}
	if m.Name != "inspect-demo" {
		t.Fatalf("name = %q, want inspect-demo", m.Name)
	}
	if len(m.UserEnvironment.Required) != 1 || m.UserEnvironment.Required[0].Name != "API_KEY" {
		t.Fatalf("required not parsed: %+v", m.UserEnvironment.Required)
	}
	if len(m.UserEnvironment.Optional) != 1 || m.UserEnvironment.Optional[0].Default != "us-east-1" {
		t.Fatalf("optional not parsed: %+v", m.UserEnvironment.Optional)
	}
	if len(m.UserEnvironment.RequireOneOf) != 1 || len(m.UserEnvironment.RequireOneOf[0].Options) != 2 {
		t.Fatalf("require_one_of not parsed: %+v", m.UserEnvironment.RequireOneOf)
	}
}

// Contract: a source with no manifest surfaces a read error rather than panicking.
func TestInspectSource_MissingManifest(t *testing.T) {
	if _, err := InspectSource(t.TempDir()); err == nil {
		t.Fatal("expected an error for a directory with no manifest")
	}
	if _, err := InspectSource(""); err == nil {
		t.Fatal("expected an error for an empty source")
	}
}

// Contract: findManifestDir locates a manifest nested below the root without
// requiring a startable node.
func TestFindManifestDir_WalksToManifest(t *testing.T) {
	root := t.TempDir()
	nested := filepath.Join(root, "sub", "node")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(nested, "agentfield-package.yaml"), []byte("name: x\nversion: 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := findManifestDir(root, "")
	if err != nil {
		t.Fatalf("findManifestDir: %v", err)
	}
	if got != nested {
		t.Fatalf("got %q, want %q", got, nested)
	}
}

func TestFindManifestDir_RejectsSubdirSymlinkOutsideRoot(t *testing.T) {
	root := t.TempDir()
	external := t.TempDir()
	requireManifest := func(dir string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, "agentfield-package.yaml"), []byte("name: x\nversion: 1\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	requireManifest(external)
	link := filepath.Join(root, "selected")
	if err := os.Symlink(external, link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}

	if _, err := findManifestDir(root, "selected"); err == nil {
		t.Fatal("expected symlinked subdirectory outside root to be rejected")
	}
}
