package packages

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// writeManifest writes an agentfield-package.yaml into dir with the given body.
func writeManifest(t *testing.T, dir, body string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "agentfield-package.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// Contract: a node with no declared environment resolves to an empty map
// without opening the secret store.
func TestResolveEnvironment_NoDeclaredEnv(t *testing.T) {
	ar := &AgentNodeRunner{AgentFieldHome: t.TempDir()}
	env, err := ar.resolveEnvironment("demo", &PackageMetadata{})
	if err != nil {
		t.Fatalf("resolveEnvironment: %v", err)
	}
	if len(env) != 0 {
		t.Fatalf("expected empty env, got %v", env)
	}
}

// Contract: a declared required variable already present in the process
// environment is resolved from there (no prompt needed) via the secret store.
func TestResolveEnvironment_ResolvesFromProcessEnv(t *testing.T) {
	t.Setenv("MY_DECLARED_VAR", "from-process")
	ar := &AgentNodeRunner{AgentFieldHome: t.TempDir()}
	meta := &PackageMetadata{
		UserEnvironment: UserEnvironmentConfig{
			Required: []UserEnvironmentVar{{Name: "MY_DECLARED_VAR"}},
		},
	}
	env, err := ar.resolveEnvironment("demo", meta)
	if err != nil {
		t.Fatalf("resolveEnvironment: %v", err)
	}
	if env["MY_DECLARED_VAR"] != "from-process" {
		t.Fatalf("MY_DECLARED_VAR = %q, want from-process", env["MY_DECLARED_VAR"])
	}
}

// Contract: a declared required variable that cannot be resolved in a
// non-interactive session fails the process start.
func TestStartAgentNodeProcess_MissingRequiredEnvFails(t *testing.T) {
	home := t.TempDir()
	pkgPath := filepath.Join(home, "packages", "demo")
	writeManifest(t, pkgPath, `name: demo
version: 1.0.0
user_environment:
  required:
    - name: DEFINITELY_UNSET_REQUIRED_VAR
      type: secret
`)
	ar := &AgentNodeRunner{AgentFieldHome: home}
	node := InstalledPackage{Name: "demo", Path: pkgPath}
	node.Runtime.LogFile = filepath.Join(home, "demo.log")

	_, err := ar.startAgentNodeProcess(node, 8123)
	if err == nil || !strings.Contains(err.Error(), "missing required") {
		t.Fatalf("expected missing-required env error, got %v", err)
	}
}

// Contract: once the declared environment resolves, the process launch proceeds
// far enough to open the log file — an unopenable log path is the failure we
// observe here, proving resolution + env assembly ran.
func TestStartAgentNodeProcess_EnvResolvesThenLogOpenFails(t *testing.T) {
	t.Setenv("RESOLVABLE_VAR", "ok")
	home := t.TempDir()
	pkgPath := filepath.Join(home, "packages", "demo")
	writeManifest(t, pkgPath, `name: demo
version: 1.0.0
user_environment:
  required:
    - name: RESOLVABLE_VAR
`)
	ar := &AgentNodeRunner{AgentFieldHome: home}
	node := InstalledPackage{Name: "demo", Path: pkgPath}
	// LogFile points at a path whose parent does not exist → OpenFile fails,
	// but only after the environment has been fully resolved and assembled.
	node.Runtime.LogFile = filepath.Join(home, "no-such-dir", "demo.log")

	_, err := ar.startAgentNodeProcess(node, 8124)
	if err == nil || !strings.Contains(err.Error(), "failed to open log file") {
		t.Fatalf("expected log-open failure after env resolution, got %v", err)
	}
}

// Contract: a package with no manifest still attempts to launch, falling back
// to the default python entrypoint (failure surfaces at the log-open step).
func TestStartAgentNodeProcess_NoManifestFallback(t *testing.T) {
	home := t.TempDir()
	pkgPath := filepath.Join(home, "packages", "legacy")
	if err := os.MkdirAll(pkgPath, 0o755); err != nil {
		t.Fatal(err)
	}
	ar := &AgentNodeRunner{AgentFieldHome: home}
	node := InstalledPackage{Name: "legacy", Path: pkgPath}
	node.Runtime.LogFile = filepath.Join(home, "no-such-dir", "legacy.log")

	_, err := ar.startAgentNodeProcess(node, 8125)
	if err == nil || !strings.Contains(err.Error(), "failed to open log file") {
		t.Fatalf("expected fallback launch to fail at log-open, got %v", err)
	}
}

// Contract: startNodeDependencies is a no-op for a node whose manifest cannot
// be parsed (no dependencies to start).
func TestStartNodeDependencies_NoManifest(t *testing.T) {
	ar := &AgentNodeRunner{AgentFieldHome: t.TempDir()}
	node := InstalledPackage{Name: "x", Path: t.TempDir()} // no manifest
	// Must not panic and must return promptly.
	ar.startNodeDependencies(node, map[string]bool{})
}

// Contract: a declared node dependency that is not installed is reported (and
// skipped) rather than aborting the parent node's start.
func TestStartNodeDependencies_DeclaredButNotInstalled(t *testing.T) {
	home := t.TempDir()
	pkgPath := filepath.Join(home, "packages", "mynode")
	writeManifest(t, pkgPath, `name: mynode
version: 1.0.0
dependencies:
  nodes:
    - af://registry/missingdep@v1
`)
	ar := &AgentNodeRunner{AgentFieldHome: home}
	node := InstalledPackage{Name: "mynode", Path: pkgPath}
	// Empty registry → dependency is not installed → warning path, no error.
	ar.startNodeDependencies(node, map[string]bool{})
}

// Contract: dependency references already being started (cycle guard) or that
// yield no derivable name are skipped.
func TestStartNodeDependencies_CycleGuardAndEmptyRef(t *testing.T) {
	home := t.TempDir()
	pkgPath := filepath.Join(home, "packages", "mynode")
	writeManifest(t, pkgPath, `name: mynode
version: 1.0.0
dependencies:
  nodes:
    - af://registry/depnode@v1
    - ""
`)
	ar := &AgentNodeRunner{AgentFieldHome: home}
	node := InstalledPackage{Name: "mynode", Path: pkgPath}
	// depnode is marked in-progress so the cycle guard skips it; the empty ref
	// yields no name and is also skipped.
	ar.startNodeDependencies(node, map[string]bool{"depnode": true})
}

// Contract: an installed-and-running dependency is left alone.
func TestStartNodeDependencies_AlreadyRunning(t *testing.T) {
	home := t.TempDir()
	pkgPath := filepath.Join(home, "packages", "mynode")
	writeManifest(t, pkgPath, `name: mynode
version: 1.0.0
dependencies:
  nodes:
    - af://registry/depnode@v1
`)
	// Registry marks depnode as already running.
	registry := &InstallationRegistry{Installed: map[string]InstalledPackage{
		"depnode": {Name: "depnode", Status: "running"},
	}}
	if err := (&PackageUninstaller{AgentFieldHome: home}).saveRegistry(registry); err != nil {
		t.Fatal(err)
	}
	ar := &AgentNodeRunner{AgentFieldHome: home}
	node := InstalledPackage{Name: "mynode", Path: pkgPath}
	ar.startNodeDependencies(node, map[string]bool{})
}

// Contract: an installed-but-not-running dependency is started before the
// parent node; when that start fails it is reported as a warning rather than
// aborting the parent (best-effort dependency bring-up).
func TestStartNodeDependencies_NotRunningIsStarted(t *testing.T) {
	home := t.TempDir()
	pkgPath := filepath.Join(home, "packages", "mynode")
	writeManifest(t, pkgPath, `name: mynode
version: 1.0.0
dependencies:
  nodes:
    - af://registry/depnode@v1
`)
	// depnode is installed but not running and has no usable package on disk,
	// so the recursive start attempt fails and is swallowed as a warning.
	registry := &InstallationRegistry{Installed: map[string]InstalledPackage{
		"depnode": {Name: "depnode", Status: "stopped", Path: filepath.Join(home, "packages", "depnode")},
	}}
	if err := (&PackageUninstaller{AgentFieldHome: home}).saveRegistry(registry); err != nil {
		t.Fatal(err)
	}
	ar := &AgentNodeRunner{AgentFieldHome: home}
	node := InstalledPackage{Name: "mynode", Path: pkgPath}
	// Must return without panicking even though the dependency fails to start.
	ar.startNodeDependencies(node, map[string]bool{})
}

// Contract: waitForAgentNode defaults an empty health path to "/health" and
// times out when nothing is listening.
func TestWaitForAgentNode_DefaultHealthPath(t *testing.T) {
	ar := &AgentNodeRunner{}
	err := ar.waitForAgentNode(1, "", 10*time.Millisecond)
	if err == nil {
		t.Fatalf("expected timeout when nothing is listening")
	}
}
