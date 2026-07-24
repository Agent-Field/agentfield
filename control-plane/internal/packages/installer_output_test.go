package packages

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// captureStdout runs fn with os.Stdout redirected to a pipe and returns what was
// written. Used to assert on user-facing install output.
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	orig := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	os.Stdout = w
	defer func() { os.Stdout = orig }()

	fn()

	if err := w.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	var buf bytes.Buffer
	if _, err := io.Copy(&buf, r); err != nil {
		t.Fatalf("copy: %v", err)
	}
	_ = r.Close()
	return buf.String()
}

// Contract (item 4): a successful install prints explicit next steps —
// `af run <name>` and `af list` — and, when required variables are missing, the
// correct `af secrets set <VAR> --node <name>` command for each.
func TestInstallPackage_PostInstallNextStepsAndSecrets(t *testing.T) {
	home := t.TempDir()
	installer := &PackageInstaller{AgentFieldHome: home}

	source := filepath.Join(t.TempDir(), "source")
	// No requirements.txt / pyproject.toml and no declared python deps, so the
	// dependency step is a no-op — the install stays hermetic (no pip/network).
	writeTestPackage(t, source, strings.TrimSpace(`
name: nextsteps-node
version: 0.1.0
description: package under test
user_environment:
  required:
    - name: REQUIRED_TOKEN
      description: a token
`)+"\n")

	out := captureStdout(t, func() {
		if err := installer.InstallPackage(source, true); err != nil {
			t.Fatalf("InstallPackage: %v", err)
		}
	})

	for _, want := range []string{
		"af run nextsteps-node",
		"af list",
		"af secrets set REQUIRED_TOKEN --node nextsteps-node",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("post-install output must contain %q, got:\n%s", want, out)
		}
	}
}
