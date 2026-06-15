package prompts

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadAndRender(t *testing.T) {
	path := filepath.Join(t.TempDir(), "prompts.yaml")
	if err := os.WriteFile(path, []byte(`version: test
prompts:
  sample:
    required_variables: [name]
    text: "hello {{name}}"
`), 0o600); err != nil {
		t.Fatalf("write prompts: %v", err)
	}
	store, err := Load(path, "")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if store.Version() != "test" {
		t.Fatalf("version = %q", store.Version())
	}
	rendered, err := store.Render("sample", map[string]string{"name": "Ada"})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if rendered != "hello Ada" {
		t.Fatalf("rendered = %q", rendered)
	}
	if _, err := store.Render("sample", map[string]string{}); err == nil {
		t.Fatal("expected missing variable error")
	}
}
