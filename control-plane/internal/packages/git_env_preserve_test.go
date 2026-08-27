package packages

import (
	"os"
	"path/filepath"
	"testing"
)

func TestC22GitReinstallPreservesExistingDotEnv(t *testing.T) {
	home := t.TempDir()
	repo := filepath.Join(t.TempDir(), "repo")
	writeTestPackage(t, repo, "name: env-demo\nversion: 2.0.0\n")
	setupFakeGit(t, "copy", repo, false)

	dest := seedInstalled(t, home, "env-demo")
	const contents = "OPENAI_API_KEY=keep-me\n"
	if err := os.WriteFile(filepath.Join(dest, ".env"), []byte(contents), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := (&GitInstaller{AgentFieldHome: home}).InstallFromGit("https://gitlab.com/acme/env-demo", true); err != nil {
		t.Fatalf("InstallFromGit: %v", err)
	}
	got, err := os.ReadFile(filepath.Join(dest, ".env"))
	if err != nil {
		t.Fatalf("read preserved .env: %v", err)
	}
	if string(got) != contents {
		t.Fatalf(".env = %q, want %q", got, contents)
	}
}
