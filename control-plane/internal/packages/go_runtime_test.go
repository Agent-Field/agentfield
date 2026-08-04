package packages

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// Validation contract for Go-language agent nodes (mirrors the Python path):
//   - a manifest may declare `language: go`; when absent, a go.mod at the root
//     is detected as a Go node, and everything else (no language, no go.mod)
//     stays Python — full backward compatibility.
//   - StartCommand launches a prebuilt binary (entrypoint.start) or defaults to
//     `go run .`.
//   - install builds the Go node with the discovered toolchain; a missing `go`
//     toolchain is an actionable error, not a raw build failure.
//   - a go.mod local replace escaping the package tree is refused with guidance
//     (it would break after the package is copied), unless vendored/overridden.

// writeGoManifest writes an agentfield-package.yaml and a go.mod (with the given
// go directive and optional extra body) into dir, producing a Go-node layout.
func writeGoManifest(t *testing.T, dir, manifest, goDirective, gomodExtra string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "agentfield-package.yaml"), []byte(manifest), 0o644); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	gomod := "module example.com/node\n\ngo " + goDirective + "\n" + gomodExtra
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte(gomod), 0o644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}
}

// stubGo installs a fake `go` on PATH that answers `go version`, materializes the
// `-o <path>` output of `go build`, and no-ops `go build ./...` and `go mod edit`.
// version is what `go version` reports (e.g. "1.21.0").
func stubGo(t *testing.T, version string) {
	t.Helper()
	bin := t.TempDir()
	script := "#!/bin/sh\n" +
		"if [ \"$1\" = \"version\" ]; then echo \"go version go" + version + " linux/amd64\"; exit 0; fi\n" +
		"if [ \"$1\" = \"env\" ] && [ \"$2\" = \"GOTOOLCHAIN\" ]; then echo auto; exit 0; fi\n" +
		"if [ \"$1\" = \"build\" ]; then\n" +
		"  prev=\"\"\n" +
		"  for a in \"$@\"; do\n" +
		"    if [ \"$prev\" = \"-o\" ]; then printf '#!/bin/sh\\nexit 0\\n' > \"$a\"; chmod +x \"$a\"; fi\n" +
		"    prev=\"$a\"\n" +
		"  done\n" +
		"  exit 0\n" +
		"fi\n" +
		"exit 0\n"
	writeExecutable(t, filepath.Join(bin, "go"), script)
	t.Setenv("PATH", bin)
}

func TestParseGoVersion(t *testing.T) {
	cases := []struct {
		in    string
		want  goVersion
		valid bool
	}{
		{"1.21", goVersion{1, 21, 0}, true},
		{"1.21.5", goVersion{1, 21, 5}, true},
		{"go1.25.4", goVersion{1, 25, 4}, true},
		{"1", goVersion{1, 0, 0}, true},
		{" 1.22 ", goVersion{1, 22, 0}, true},
		{"", goVersion{}, false},
		{"go", goVersion{}, false},
		{"abc", goVersion{}, false},
	}
	for _, c := range cases {
		got, ok := parseGoVersion(c.in)
		if ok != c.valid || (ok && got != c.want) {
			t.Errorf("parseGoVersion(%q) = %v,%v; want %v,%v", c.in, got, ok, c.want, c.valid)
		}
	}
}

func TestGoVersionAtLeast(t *testing.T) {
	cases := []struct {
		have, want goVersion
		ok         bool
	}{
		{goVersion{1, 25, 4}, goVersion{1, 21, 0}, true},
		{goVersion{1, 21, 0}, goVersion{1, 21, 0}, true},
		{goVersion{1, 20, 9}, goVersion{1, 21, 0}, false},
		{goVersion{2, 0, 0}, goVersion{1, 30, 0}, true},
		{goVersion{1, 21, 4}, goVersion{1, 21, 5}, false},
	}
	for _, c := range cases {
		if got := c.have.atLeast(c.want); got != c.ok {
			t.Errorf("%v.atLeast(%v) = %v; want %v", c.have, c.want, got, c.ok)
		}
	}
}

func TestReadGoDirective(t *testing.T) {
	t.Run("declared", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, dir, "go.mod", "module x\n\ngo 1.21\n\nrequire foo v1.0.0\n")
		if got := readGoDirective(dir); got != "1.21" {
			t.Fatalf("got %q; want 1.21", got)
		}
	})
	t.Run("no go.mod", func(t *testing.T) {
		if got := readGoDirective(t.TempDir()); got != "" {
			t.Fatalf("got %q; want empty", got)
		}
	})
	t.Run("no go directive", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, dir, "go.mod", "module x\n")
		if got := readGoDirective(dir); got != "" {
			t.Fatalf("got %q; want empty", got)
		}
	})
}

// Contract: language is resolved from the explicit field, then go.mod detection,
// then defaults to Python. Existing Python manifests are unaffected.
func TestLanguageResolutionAndDetection(t *testing.T) {
	t.Run("explicit language: go", func(t *testing.T) {
		dir := t.TempDir()
		writeTestPackage(t, dir, "name: n\nversion: 0.1.0\nlanguage: go\n")
		md, err := ParsePackageMetadata(dir)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		if !md.IsGo() {
			t.Fatalf("explicit language: go must be a Go node")
		}
	})

	t.Run("detected via go.mod when language absent", func(t *testing.T) {
		dir := t.TempDir()
		writeGoManifest(t, dir, "name: n\nversion: 0.1.0\n", "1.21", "")
		md, err := ParsePackageMetadata(dir)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		if !md.IsGo() {
			t.Fatalf("a package with a go.mod must be detected as a Go node")
		}
	})

	t.Run("back-compat: no language, no go.mod is Python", func(t *testing.T) {
		dir := t.TempDir()
		writeTestPackage(t, dir, "name: n\nversion: 0.1.0\nentrypoint:\n  start: python -m n.app\n")
		md, err := ParsePackageMetadata(dir)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		if md.IsGo() {
			t.Fatalf("a Python manifest must not be treated as Go")
		}
	})

	t.Run("explicit language: python wins over a stray go.mod", func(t *testing.T) {
		dir := t.TempDir()
		writeGoManifest(t, dir, "name: n\nversion: 0.1.0\nlanguage: python\n", "1.21", "")
		md, err := ParsePackageMetadata(dir)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		if md.IsGo() {
			t.Fatalf("explicit language: python must override go.mod detection")
		}
	})
}

// Contract: StartCommand for a Go node runs the prebuilt binary, or defaults to
// `go run .` when no entrypoint.start is declared.
func TestStartCommandGo(t *testing.T) {
	t.Run("prebuilt binary path", func(t *testing.T) {
		md := &PackageMetadata{Language: "go", Entrypoint: EntrypointConfig{Start: "bin/swe-planner"}}
		got := md.StartCommand()
		if len(got) != 1 || got[0] != "bin/swe-planner" {
			t.Fatalf("StartCommand() = %v; want [bin/swe-planner]", got)
		}
	})
	t.Run("go run form", func(t *testing.T) {
		md := &PackageMetadata{Language: "go", Entrypoint: EntrypointConfig{Start: "go run ./cmd/swe-planner"}}
		got := md.StartCommand()
		if len(got) != 3 || got[0] != "go" || got[2] != "./cmd/swe-planner" {
			t.Fatalf("StartCommand() = %v; want [go run ./cmd/swe-planner]", got)
		}
	})
	t.Run("default go run . when start empty", func(t *testing.T) {
		md := &PackageMetadata{Language: "go"}
		got := md.StartCommand()
		if len(got) != 3 || got[0] != "go" || got[1] != "run" || got[2] != "." {
			t.Fatalf("StartCommand() = %v; want [go run .]", got)
		}
	})
	t.Run("python default is unchanged for non-Go", func(t *testing.T) {
		md := &PackageMetadata{}
		got := md.StartCommand()
		if len(got) != 2 || got[0] != "python" || got[1] != "main.py" {
			t.Fatalf("StartCommand() = %v; want [python main.py]", got)
		}
	})
}

func TestGoBuildTarget(t *testing.T) {
	cases := []struct {
		name           string
		md             *PackageMetadata
		wantPkg, wantB string
	}{
		{
			name:    "build + binary start",
			md:      &PackageMetadata{Entrypoint: EntrypointConfig{Build: "./cmd/swe-planner", Start: "bin/swe-planner"}},
			wantPkg: "./cmd/swe-planner", wantB: "bin/swe-planner",
		},
		{
			name:    "build only derives bin name",
			md:      &PackageMetadata{Entrypoint: EntrypointConfig{Build: "./cmd/swe-fast"}},
			wantPkg: "./cmd/swe-fast", wantB: filepath.Join("bin", "swe-fast"),
		},
		{
			name:    "no build -> compile check only",
			md:      &PackageMetadata{Entrypoint: EntrypointConfig{Start: "go run ."}},
			wantPkg: "", wantB: "",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			pkg, bin := c.md.goBuildTarget()
			if pkg != c.wantPkg || bin != c.wantB {
				t.Fatalf("goBuildTarget() = (%q,%q); want (%q,%q)", pkg, bin, c.wantPkg, c.wantB)
			}
		})
	}
}

func TestGoBinaryProgram(t *testing.T) {
	dir := "/home/x/.agentfield/packages/n"
	cases := map[string]string{
		"bin/swe-planner":   filepath.Join(dir, "bin/swe-planner"),
		"./bin/swe-planner": filepath.Join(dir, "bin/swe-planner"),
		"go":                "go",
		"/abs/bin/node":     "/abs/bin/node",
		"":                  "",
	}
	for in, want := range cases {
		if got := GoBinaryProgram(dir, in); got != want {
			t.Errorf("GoBinaryProgram(%q) = %q; want %q", in, got, want)
		}
	}
}

// Contract: install builds the Go node with the discovered toolchain, producing
// the declared binary.
func TestInstallGoDependencies_BuildsBinary(t *testing.T) {
	stubGo(t, "1.21.0")
	dir := t.TempDir()
	writeGoManifest(t, dir,
		"name: n\nversion: 0.1.0\nlanguage: go\nentrypoint:\n  build: ./cmd/node\n  start: bin/node\n",
		"1.21", "")
	md, err := ParsePackageMetadata(dir)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if err := InstallGoDependencies(dir, md); err != nil {
		t.Fatalf("InstallGoDependencies: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "bin", "node")); err != nil {
		t.Fatalf("expected built binary bin/node: %v", err)
	}
}

// Contract: a `go run` node has no prebuilt output; install only compile-checks
// and produces no binary, but succeeds.
func TestInstallGoDependencies_GoRunCompileChecks(t *testing.T) {
	stubGo(t, "1.21.0")
	dir := t.TempDir()
	writeGoManifest(t, dir,
		"name: n\nversion: 0.1.0\nlanguage: go\nentrypoint:\n  start: go run ./cmd/node\n",
		"1.21", "")
	md, err := ParsePackageMetadata(dir)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if err := InstallGoDependencies(dir, md); err != nil {
		t.Fatalf("InstallGoDependencies (go run): %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "bin")); !os.IsNotExist(err) {
		t.Fatalf("go run node should not produce a bin/ dir")
	}
}

// Contract: install through the shared dispatcher routes a Go node to the Go
// builder (proving InstallDependencies is language-aware).
func TestInstallDependencies_DispatchesGo(t *testing.T) {
	stubGo(t, "1.21.0")
	dir := t.TempDir()
	writeGoManifest(t, dir,
		"name: n\nversion: 0.1.0\nentrypoint:\n  build: ./cmd/node\n  start: bin/node\n",
		"1.21", "")
	md, err := ParsePackageMetadata(dir)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !md.IsGo() {
		t.Fatalf("detected Go node expected")
	}
	if err := InstallDependencies(dir, md); err != nil {
		t.Fatalf("InstallDependencies: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "bin", "node")); err != nil {
		t.Fatalf("expected Go build to run via dispatcher: %v", err)
	}
}

// Contract: a missing `go` toolchain is an actionable error naming Go, not a raw
// exec failure.
func TestResolveGoToolchain_MissingToolchain(t *testing.T) {
	empty := t.TempDir() // a PATH with no `go`
	t.Setenv("PATH", empty)
	t.Setenv("AGENTFIELD_DISABLE_GO_PROVISIONING", "1")
	dir := t.TempDir()
	writeGoManifest(t, dir, "name: n\nversion: 0.1.0\nlanguage: go\n", "1.21", "")

	_, err := resolveGoToolchain(dir)
	if err == nil {
		t.Fatal("expected an error when no go toolchain is on PATH")
	}
	if !strings.Contains(err.Error(), "go.dev/dl") || !strings.Contains(strings.ToLower(err.Error()), "toolchain") {
		t.Fatalf("error should guide installing Go: %v", err)
	}

	md, _ := ParsePackageMetadata(dir)
	if err := InstallGoDependencies(dir, md); err == nil {
		t.Fatal("InstallGoDependencies should fail without a toolchain")
	}
}

// Contract: an installed toolchain older than the go.mod directive is refused
// with an upgrade hint.
func TestResolveGoToolchain_TooOld(t *testing.T) {
	stubGo(t, "1.20.0") // reports 1.20
	t.Setenv("AGENTFIELD_DISABLE_GO_PROVISIONING", "1")
	dir := t.TempDir()
	writeGoManifest(t, dir, "name: n\nversion: 0.1.0\nlanguage: go\n", "1.99", "") // requires 1.99

	_, err := resolveGoToolchain(dir)
	if err == nil {
		t.Fatal("expected an error when installed Go is older than go.mod requires")
	}
	if !strings.Contains(err.Error(), "1.99") || !strings.Contains(err.Error(), "1.20") {
		t.Fatalf("error should name required (1.99) and found (1.20): %v", err)
	}
}

// Contract: Go 1.21+ with automatic toolchain selection is allowed to honor a
// newer go.mod itself instead of triggering AgentField provisioning.
func TestResolveGoToolchain_AmbientCanAutoSwitch(t *testing.T) {
	stubGo(t, "1.21.0")
	dir := t.TempDir()
	writeGoManifest(t, dir, "name: n\nversion: 0.1.0\nlanguage: go\n", "1.99", "")

	got, err := resolveGoToolchain(dir)
	if err != nil || got != "go" {
		t.Fatalf("resolveGoToolchain = %q, %v; want ambient go", got, err)
	}
}

func TestResolveGoToolchain_AmbientPinnedLocalFallsThrough(t *testing.T) {
	bin := t.TempDir()
	writeExecutable(t, filepath.Join(bin, "go"), "#!/bin/sh\nif [ \"$1\" = version ]; then echo 'go version go1.21.0 linux/amd64'; elif [ \"$1\" = env ]; then echo local; fi\n")
	t.Setenv("PATH", bin)
	t.Setenv("AGENTFIELD_DISABLE_GO_PROVISIONING", "1")
	dir := t.TempDir()
	writeGoManifest(t, dir, "name: n\nversion: 0.1.0\nlanguage: go\n", "1.99", "")

	_, err := resolveGoToolchain(dir)
	if err == nil || !strings.Contains(err.Error(), "1.99") {
		t.Fatalf("pinned-local old Go should be refused: %v", err)
	}
}

// goArchiveFixture creates the layout shipped by official Go archives.
func goArchiveFixture(t *testing.T, entries map[string]string) []byte {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	for name, body := range entries {
		mode := int64(0o644)
		if strings.HasSuffix(name, "/bin/go") {
			mode = 0o755
		}
		if err := tw.WriteHeader(&tar.Header{Name: name, Mode: mode, Size: int64(len(body)), Typeflag: tar.TypeReg}); err != nil {
			t.Fatal(err)
		}
		if _, err := tw.Write([]byte(body)); err != nil {
			t.Fatal(err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func configureGoProvisionFixture(t *testing.T, archive []byte, checksum string) *int {
	t.Helper()
	downloads := 0
	filename := fmt.Sprintf("go9.9.9.%s-%s.tar.gz", runtime.GOOS, runtime.GOARCH)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/index" {
			fmt.Fprintf(w, `[{"version":"go9.9.9","stable":true,"files":[{"filename":%q,"os":%q,"arch":%q,"kind":"archive","sha256":%q}]}]`, filename, runtime.GOOS, runtime.GOARCH, checksum)
			return
		}
		downloads++
		_, _ = w.Write(archive)
	}))
	t.Cleanup(server.Close)
	oldClient, oldIndex, oldBase := goProvisionHTTPClient, goProvisionIndexURL, goProvisionArchiveBaseURL
	goProvisionHTTPClient = server.Client()
	goProvisionIndexURL = server.URL + "/index"
	goProvisionArchiveBaseURL = server.URL + "/"
	t.Cleanup(func() {
		goProvisionHTTPClient, goProvisionIndexURL, goProvisionArchiveBaseURL = oldClient, oldIndex, oldBase
	})
	return &downloads
}

func configureGoProvisionServer(t *testing.T, server *httptest.Server) {
	t.Helper()
	oldClient, oldIndex, oldBase := goProvisionHTTPClient, goProvisionIndexURL, goProvisionArchiveBaseURL
	goProvisionHTTPClient = server.Client()
	goProvisionIndexURL = server.URL + "/index"
	goProvisionArchiveBaseURL = server.URL + "/"
	t.Cleanup(func() {
		goProvisionHTTPClient, goProvisionIndexURL, goProvisionArchiveBaseURL = oldClient, oldIndex, oldBase
	})
}

func assertProvisioningUnavailable(t *testing.T) {
	t.Helper()
	t.Setenv("PATH", t.TempDir())
	t.Setenv("AGENTFIELD_HOME", t.TempDir())

	goCmd, cached, err := provisionGoToolchain()
	if err != nil || goCmd != "" || cached {
		t.Fatalf("provisionGoToolchain() = %q, %v, %v; want empty toolchain without a hard error", goCmd, cached, err)
	}
	dir := t.TempDir()
	writeGoManifest(t, dir, "name: n\nversion: 0.1.0\nlanguage: go\n", "1.21", "")
	_, err = resolveGoToolchain(dir)
	if err == nil || !strings.Contains(err.Error(), "Install Go") || !strings.Contains(err.Error(), "https://go.dev/dl/") {
		t.Fatalf("resolveGoToolchain should retain actionable install-Go guidance, got %v", err)
	}
}

// Contract: an unreachable go.dev index is an ordinary availability failure,
// so resolution retains its actionable install-Go guidance.
func TestProvisionGoToolchain_IndexUnreachableFallsBackToInstallGuidance(t *testing.T) {
	server := httptest.NewServer(http.NotFoundHandler())
	configureGoProvisionServer(t, server)
	server.Close()

	assertProvisioningUnavailable(t)
}

// Contract: a non-200 index response is an ordinary availability failure, so
// resolution retains its actionable install-Go guidance.
func TestProvisionGoToolchain_IndexNonOKFallsBackToInstallGuidance(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "unavailable", http.StatusServiceUnavailable)
	}))
	t.Cleanup(server.Close)
	configureGoProvisionServer(t, server)

	assertProvisioningUnavailable(t)
}

// Contract: malformed index JSON is an ordinary availability failure, so
// resolution retains its actionable install-Go guidance.
func TestProvisionGoToolchain_InvalidIndexJSONFallsBackToInstallGuidance(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, `{"not valid JSON"`)
	}))
	t.Cleanup(server.Close)
	configureGoProvisionServer(t, server)

	assertProvisioningUnavailable(t)
}

// Contract: an index without an archive for the host is an ordinary
// availability failure, so resolution retains its actionable install-Go guidance.
func TestProvisionGoToolchain_NoHostArchiveFallsBackToInstallGuidance(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprintf(w, `[{"version":"go9.9.9","stable":true,"files":[{"os":%q,"arch":%q,"kind":"installer","sha256":"sum"},{"os":"other","arch":"other","kind":"archive","sha256":"sum"}]}]`, runtime.GOOS, runtime.GOARCH)
	}))
	t.Cleanup(server.Close)
	configureGoProvisionServer(t, server)

	assertProvisioningUnavailable(t)
}

// Contract: a non-200 archive response is an ordinary availability failure,
// so resolution retains its actionable install-Go guidance.
func TestProvisionGoToolchain_ArchiveNonOKFallsBackToInstallGuidance(t *testing.T) {
	filename := fmt.Sprintf("go9.9.9.%s-%s.tar.gz", runtime.GOOS, runtime.GOARCH)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/index" {
			fmt.Fprintf(w, `[{"version":"go9.9.9","stable":true,"files":[{"filename":%q,"os":%q,"arch":%q,"kind":"archive","sha256":"sum"}]}]`, filename, runtime.GOOS, runtime.GOARCH)
			return
		}
		http.Error(w, "unavailable", http.StatusBadGateway)
	}))
	t.Cleanup(server.Close)
	configureGoProvisionServer(t, server)

	assertProvisioningUnavailable(t)
}

// Contracts: no ambient Go is provisioned, the resulting build succeeds, and
// a second resolution reuses the completed cache without fetching the archive.
func TestResolveGoToolchain_ProvisionsAndReusesCache(t *testing.T) {
	archive := goArchiveFixture(t, map[string]string{
		"go/bin/go": "#!/bin/sh\nif [ \"$1\" = version ]; then echo 'go version go1.99.0 linux/amd64'; exit 0; fi\nif [ \"$1\" = build ]; then prev=''; for a in \"$@\"; do if [ \"$prev\" = -o ]; then echo built > \"$a\"; fi; prev=\"$a\"; done; fi\n",
	})
	sum := sha256.Sum256(archive)
	downloads := configureGoProvisionFixture(t, archive, hex.EncodeToString(sum[:]))
	t.Setenv("PATH", t.TempDir())
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	dir := t.TempDir()
	writeGoManifest(t, dir, "name: n\nversion: 0.1.0\nlanguage: go\nentrypoint:\n  build: .\n  start: bin/node\n", "1.21", "")
	md, _ := ParsePackageMetadata(dir)
	if err := InstallGoDependencies(dir, md); err != nil {
		t.Fatalf("provisioned install failed: %v", err)
	}
	if _, err := resolveGoToolchain(dir); err != nil {
		t.Fatalf("cached resolution failed: %v", err)
	}
	if *downloads != 1 {
		t.Fatalf("archive downloads = %d; want 1", *downloads)
	}
}

// Contract: checksum verification is a hard gate before extraction.
func TestProvisionGoToolchain_RejectsChecksumMismatch(t *testing.T) {
	archive := goArchiveFixture(t, map[string]string{"go/bin/go": "binary"})
	configureGoProvisionFixture(t, archive, strings.Repeat("0", 64))
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	_, _, err := provisionGoToolchain()
	if err == nil || !strings.Contains(strings.ToLower(err.Error()), "sha256 mismatch") {
		t.Fatalf("expected checksum mismatch, got %v", err)
	}
	if _, statErr := os.Stat(filepath.Join(home, "toolchains", "go9.9.9")); !os.IsNotExist(statErr) {
		t.Fatalf("mismatched archive created install directory: %v", statErr)
	}
}

// Contracts: traversal is rejected without an outside write, and failed
// extraction leaves no final cache directory that a later run can reuse.
func TestProvisionGoToolchain_RejectsTraversalAtomically(t *testing.T) {
	archive := goArchiveFixture(t, map[string]string{"../escaped": "bad", "go/bin/go": "binary"})
	sum := sha256.Sum256(archive)
	configureGoProvisionFixture(t, archive, hex.EncodeToString(sum[:]))
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	_, _, err := provisionGoToolchain()
	if err == nil || !strings.Contains(err.Error(), "unsafe archive path") {
		t.Fatalf("expected unsafe path error, got %v", err)
	}
	if _, statErr := os.Stat(filepath.Join(home, "escaped")); !os.IsNotExist(statErr) {
		t.Fatalf("archive escaped destination: %v", statErr)
	}
	if _, statErr := os.Stat(filepath.Join(home, "toolchains", "go9.9.9")); !os.IsNotExist(statErr) {
		t.Fatalf("failed extraction left final cache: %v", statErr)
	}
}

// Contract: a checksum-valid archive missing the Go executable is rejected
// loudly and never becomes a reusable cached toolchain.
func TestProvisionGoToolchain_RejectsArchiveMissingGoBinaryAtomically(t *testing.T) {
	archive := goArchiveFixture(t, map[string]string{"go/README.md": "not a toolchain"})
	sum := sha256.Sum256(archive)
	configureGoProvisionFixture(t, archive, hex.EncodeToString(sum[:]))
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)

	goCmd, cached, err := provisionGoToolchain()
	if err == nil || !strings.Contains(err.Error(), "does not contain go/bin/"+goBinaryName()) {
		t.Fatalf("expected missing go/bin/%s hard error, got %q, %v, %v", goBinaryName(), goCmd, cached, err)
	}
	if _, statErr := os.Stat(filepath.Join(home, "toolchains", "go9.9.9")); !os.IsNotExist(statErr) {
		t.Fatalf("invalid archive left a final cache directory: %v", statErr)
	}
}

// Contract: failure to query an ambient Go's GOTOOLCHAIN setting means it
// cannot be trusted to auto-switch to the requested version.
func TestGoCanAutoSwitch_EnvFailureCannotAutoSwitch(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "missing-go")
	if goCanAutoSwitch(missing, goVersion{major: 1, minor: 21}) {
		t.Fatal("goCanAutoSwitch returned true when `go env GOTOOLCHAIN` could not run")
	}
}

// Contract: when a Go version cannot be determined, diagnostics still identify
// the exact binary path rather than displaying an empty or invented version.
func TestDisplayGoVersion_UnknownVersionFallsBackToBinaryPath(t *testing.T) {
	goCmd := filepath.Join(t.TempDir(), "unknown-go")
	if got := displayGoVersion(goCmd); got != goCmd {
		t.Fatalf("displayGoVersion(%q) = %q; want the binary path", goCmd, got)
	}
}

// Contract: archive-name validation rejects empty, absolute, and climbing paths
// while preserving an ordinary nested archive entry.
func TestSafeArchiveName_RejectsUnsafeAndAcceptsNested(t *testing.T) {
	cases := []struct {
		name    string
		want    string
		wantErr bool
	}{
		{name: "", wantErr: true},
		{name: string(filepath.Separator) + "absolute", wantErr: true},
		{name: "../../outside", wantErr: true},
		{name: "go/bin/" + goBinaryName(), want: filepath.Join("go", "bin", goBinaryName())},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := safeArchiveName(tc.name)
			if (err != nil) != tc.wantErr || got != tc.want {
				t.Fatalf("safeArchiveName(%q) = %q, %v; want %q, error=%v", tc.name, got, err, tc.want, tc.wantErr)
			}
		})
	}
}

// Contract: cached Go detection requires a real regular executable that can
// run, rejecting missing paths, directories, and non-executable files.
func TestUsableGoBinary_RequiresRunnableRegularExecutable(t *testing.T) {
	dir := t.TempDir()
	if usableGoBinary(filepath.Join(dir, "missing")) {
		t.Fatal("missing path reported as a usable Go binary")
	}
	if usableGoBinary(dir) {
		t.Fatal("directory reported as a usable Go binary")
	}

	if runtime.GOOS == "windows" {
		runnable, err := os.Executable()
		if err != nil {
			t.Fatal(err)
		}
		if !usableGoBinary(runnable) {
			t.Fatalf("running test executable %q should be usable", runnable)
		}
		return
	}

	nonExecutable := filepath.Join(dir, "non-executable-go")
	if err := os.WriteFile(nonExecutable, []byte("#!/bin/sh\nexit 0\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if usableGoBinary(nonExecutable) {
		t.Fatal("regular file without execute permission reported as usable")
	}
	runnable := filepath.Join(dir, "runnable-go")
	writeExecutable(t, runnable, "#!/bin/sh\nexit 0\n")
	if !usableGoBinary(runnable) {
		t.Fatal("runnable regular executable reported as unusable")
	}
}

func TestExtractGoZip_SuccessAndTraversalRefusal(t *testing.T) {
	makeZip := func(name, body string) string {
		t.Helper()
		path := filepath.Join(t.TempDir(), "fixture.zip")
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		zw := zip.NewWriter(f)
		h := &zip.FileHeader{Name: name, Method: zip.Store}
		h.SetMode(0o755)
		w, err := zw.CreateHeader(h)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(body))
		if err := zw.Close(); err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		return path
	}

	dst := t.TempDir()
	if err := extractGoArchive(makeZip("go/bin/go.exe", "binary"), dst); err != nil {
		t.Fatalf("extract valid zip: %v", err)
	}
	if got, err := os.ReadFile(filepath.Join(dst, "go", "bin", "go.exe")); err != nil || string(got) != "binary" {
		t.Fatalf("valid zip output = %q, %v", got, err)
	}

	escapeRoot := t.TempDir()
	badDst := filepath.Join(escapeRoot, "destination")
	if err := os.Mkdir(badDst, 0o755); err != nil {
		t.Fatal(err)
	}
	err := extractGoArchive(makeZip("../escaped", "bad"), badDst)
	if err == nil || !strings.Contains(err.Error(), "unsafe archive path") {
		t.Fatalf("expected zip traversal refusal, got %v", err)
	}
	if _, err := os.Stat(filepath.Join(escapeRoot, "escaped")); !os.IsNotExist(err) {
		t.Fatalf("zip traversal wrote outside destination: %v", err)
	}
}

func TestDiscoverGoArchive_NoMatchingStableBuild(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, `[{"version":"go10.0.0","stable":false,"files":[]},{"version":"go9.9.9","stable":true,"files":[{"os":"other","arch":"other","kind":"archive","sha256":"sum"}]}]`)
	}))
	defer server.Close()
	oldClient, oldIndex := goProvisionHTTPClient, goProvisionIndexURL
	goProvisionHTTPClient, goProvisionIndexURL = server.Client(), server.URL
	defer func() { goProvisionHTTPClient, goProvisionIndexURL = oldClient, oldIndex }()
	_, _, err := discoverGoArchive()
	if err == nil || !strings.Contains(err.Error(), "no stable Go archive") {
		t.Fatalf("expected no-match error, got %v", err)
	}
}

// Contract: a satisfying toolchain passes the version gate.
func TestResolveGoToolchain_Satisfies(t *testing.T) {
	stubGo(t, "1.25.4")
	dir := t.TempDir()
	writeGoManifest(t, dir, "name: n\nversion: 0.1.0\nlanguage: go\n", "1.21", "")
	got, err := resolveGoToolchain(dir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "go" {
		t.Fatalf("got %q; want go", got)
	}
}

// Contract: a go.mod local replace that escapes the package tree is refused with
// guidance (vendor / override) rather than a confusing raw build failure.
func TestGoReplaceDirective_OutOfTreeRefused(t *testing.T) {
	stubGo(t, "1.21.0")
	dir := t.TempDir()
	writeGoManifest(t, dir,
		"name: n\nversion: 0.1.0\nlanguage: go\nentrypoint:\n  build: ./cmd/node\n  start: bin/node\n",
		"1.21",
		"\nreplace example.com/sdk => ../../other/sdk\n")

	md, _ := ParsePackageMetadata(dir)
	err := InstallGoDependencies(dir, md)
	if err == nil {
		t.Fatal("expected refusal for an out-of-tree replace directive")
	}
	if !strings.Contains(err.Error(), "vendor") || !strings.Contains(err.Error(), "replace") {
		t.Fatalf("error should explain vendoring/replace: %v", err)
	}
}

// Contract: an in-tree replace (pointing inside the package) is fine, and a
// vendored package with an out-of-tree replace builds (vendor ships the dep).
func TestGoReplaceDirective_InTreeAndVendoredAllowed(t *testing.T) {
	t.Run("in-tree replace allowed", func(t *testing.T) {
		stubGo(t, "1.21.0")
		dir := t.TempDir()
		writeGoManifest(t, dir,
			"name: n\nversion: 0.1.0\nlanguage: go\nentrypoint:\n  build: ./cmd/node\n  start: bin/node\n",
			"1.21", "\nreplace example.com/sub => ./internal/sub\n")
		md, _ := ParsePackageMetadata(dir)
		if err := InstallGoDependencies(dir, md); err != nil {
			t.Fatalf("in-tree replace should build: %v", err)
		}
	})

	t.Run("vendored out-of-tree replace allowed", func(t *testing.T) {
		stubGo(t, "1.21.0")
		dir := t.TempDir()
		writeGoManifest(t, dir,
			"name: n\nversion: 0.1.0\nlanguage: go\nentrypoint:\n  build: ./cmd/node\n  start: bin/node\n",
			"1.21", "\nreplace example.com/sdk => ../../other/sdk\n")
		if err := os.MkdirAll(filepath.Join(dir, "vendor"), 0o755); err != nil {
			t.Fatal(err)
		}
		md, _ := ParsePackageMetadata(dir)
		if err := InstallGoDependencies(dir, md); err != nil {
			t.Fatalf("vendored package should build despite out-of-tree replace: %v", err)
		}
	})
}

// Contract: the AGENTFIELD_GO_REPLACE override bypasses the out-of-tree guard so
// an out-of-tree dependency can be repointed at install time.
func TestGoReplaceOverride_Bypasses(t *testing.T) {
	stubGo(t, "1.21.0")
	dir := t.TempDir()
	writeGoManifest(t, dir,
		"name: n\nversion: 0.1.0\nlanguage: go\nentrypoint:\n  build: ./cmd/node\n  start: bin/node\n",
		"1.21", "\nreplace example.com/sdk => ../../other/sdk\n")
	t.Setenv("AGENTFIELD_GO_REPLACE", "example.com/sdk=/opt/sdk")
	md, _ := ParsePackageMetadata(dir)
	if err := InstallGoDependencies(dir, md); err != nil {
		t.Fatalf("override should allow the build to proceed: %v", err)
	}
}

// Contract: a Go node with a go.mod passes ValidatePackage even without an
// explicit entrypoint.start or a main.py.
func TestValidatePackage_GoModOnly(t *testing.T) {
	dir := t.TempDir()
	writeGoManifest(t, dir, "name: n\nversion: 0.1.0\n", "1.21", "")
	if err := ValidatePackage(dir); err != nil {
		t.Fatalf("a Go module package should validate: %v", err)
	}
}
