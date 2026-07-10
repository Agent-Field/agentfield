package packages

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

// This file is the Go-toolchain counterpart to pyinterp.go. Where pyinterp.go
// resolves a Python interpreter that satisfies a node's requires-python, this
// resolves the `go` toolchain that builds a Go node and drives the install-time
// build. The two follow the same philosophy: discover the toolchain, fail with
// an *actionable* message when it is missing or too old, and never block on
// something we cannot parse.

// goVersion is a parsed (major, minor, patch) Go toolchain version. Go releases
// are identified by major.minor (e.g. 1.21); patch is tracked but rarely gates a
// build.
type goVersion struct {
	major, minor, patch int
}

func (v goVersion) String() string {
	return fmt.Sprintf("%d.%d.%d", v.major, v.minor, v.patch)
}

// atLeast reports whether v is greater than or equal to o.
func (v goVersion) atLeast(o goVersion) bool {
	for _, d := range []int{v.major - o.major, v.minor - o.minor, v.patch - o.patch} {
		if d < 0 {
			return false
		}
		if d > 0 {
			return true
		}
	}
	return true
}

// parseGoVersion parses a Go version like "1.21", "1.21.5", or "go1.21.5" into a
// goVersion. A missing minor/patch defaults to 0. It reports false only when the
// major component is absent or non-numeric, so an unparseable value never blocks
// a build (the caller treats false as "unknown, don't gate").
func parseGoVersion(s string) (goVersion, bool) {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "go")
	if s == "" {
		return goVersion{}, false
	}
	parts := strings.Split(s, ".")
	nums := [3]int{}
	for i := 0; i < len(parts) && i < 3; i++ {
		digits := leadingDigits(parts[i])
		if digits == "" {
			if i == 0 {
				return goVersion{}, false
			}
			break
		}
		n, err := strconv.Atoi(digits)
		if err != nil {
			return goVersion{}, false
		}
		nums[i] = n
	}
	return goVersion{nums[0], nums[1], nums[2]}, true
}

// readGoDirective returns the version from the `go X.Y[.Z]` directive in the
// package's go.mod, or "" when there is no go.mod or no go directive. This is the
// minimum toolchain version the module was written against.
func readGoDirective(packagePath string) string {
	f, err := os.Open(filepath.Join(packagePath, "go.mod"))
	if err != nil {
		return ""
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		// Match a bare "go X.Y" directive, not "go run" text or "toolchain".
		if strings.HasPrefix(line, "go ") {
			ver := strings.TrimSpace(strings.TrimPrefix(line, "go "))
			// Guard against comments trailing the directive.
			if i := strings.IndexAny(ver, " \t/"); i >= 0 {
				ver = ver[:i]
			}
			if _, ok := parseGoVersion(ver); ok {
				return ver
			}
		}
	}
	return ""
}

// installedGoVersion runs `go version` and returns the parsed toolchain version.
func installedGoVersion(goCmd string) (goVersion, bool) {
	out, err := exec.Command(goCmd, "version").Output()
	if err != nil {
		return goVersion{}, false
	}
	// Output form: "go version go1.21.5 linux/amd64".
	for _, tok := range strings.Fields(string(out)) {
		if strings.HasPrefix(tok, "go1") || strings.HasPrefix(tok, "go2") {
			if v, ok := parseGoVersion(tok); ok {
				return v, true
			}
		}
	}
	return goVersion{}, false
}

// resolveGoToolchain locates the `go` toolchain used to build a Go node. It
// returns:
//   - (goCmd, nil) when a usable `go` is on PATH (and satisfies the go.mod
//     directive, if any).
//   - ("", err) with an actionable message when `go` is absent, or present but
//     older than the version the module's go.mod requires.
//
// Mirrors pyinterp.go's resolveVenvInterpreter: discover, gate on the declared
// minimum, and explain how to fix a miss rather than failing later inside the
// raw build.
func resolveGoToolchain(packagePath string) (string, error) {
	goCmd := firstOnPath("go")
	if goCmd == "" {
		return "", fmt.Errorf(
			"this agent node is a Go node, but no `go` toolchain was found on PATH.\n" +
				"Install Go and ensure `go` is on PATH, then run `af install` again:\n" +
				"  • macOS:   brew install go\n" +
				"  • Ubuntu:  sudo apt-get install golang-go  (or the official tarball)\n" +
				"  • or download the installer from https://go.dev/dl/")
	}

	want := readGoDirective(packagePath)
	if want != "" {
		wantV, wantOK := parseGoVersion(want)
		haveV, haveOK := installedGoVersion(goCmd)
		if wantOK && haveOK && !haveV.atLeast(wantV) {
			return "", fmt.Errorf(
				"this agent node requires Go %s or newer (from its go.mod), but `go` on PATH is %s "+
					"— upgrade Go (https://go.dev/dl/) and run `af install` again",
				want, haveV)
		}
	}
	return goCmd, nil
}

// InstallGoDependencies builds a Go agent node at install time so `af run`
// launches a compiled binary instead of recompiling on every start. It is the
// Go analogue of InstallPythonDependencies: discover the toolchain, resolve the
// module's dependencies (via `go build`, which downloads modules as needed), and
// leave the package ready to launch.
//
// Build target (from goBuildTarget):
//   - entrypoint.build set, entrypoint.start a binary path -> `go build -o
//     <package>/<start> <build>`, producing the runnable binary.
//   - otherwise (a `go run ...` / empty entrypoint) -> `go build ./...`, a
//     compile check only; the binary is produced by `go run` at launch.
//
// A vendored module (vendor/ present) builds with -mod=vendor so it is fully
// hermetic after the package is copied into ~/.agentfield/packages.
func InstallGoDependencies(packagePath string, metadata *PackageMetadata) error {
	goCmd, err := resolveGoToolchain(packagePath)
	if err != nil {
		return err
	}

	// Guard against local replace directives that escape the package tree: after
	// the node is copied into ~/.agentfield/packages, a "replace => ../../x" path
	// no longer resolves. Fail early with guidance rather than a raw build error.
	if err := checkGoReplaceDirectives(packagePath); err != nil {
		return err
	}
	// Apply any build-time replace overrides (AGENTFIELD_GO_REPLACE) so an
	// out-of-tree dependency can be repointed at install time.
	if err := applyGoReplaceOverrides(goCmd, packagePath); err != nil {
		return err
	}

	buildPkg, outBin := metadata.goBuildTarget()

	args := []string{"build"}
	if hasVendorDir(packagePath) {
		args = append(args, "-mod=vendor")
	}
	if outBin != "" {
		outAbs := filepath.Join(packagePath, outBin)
		if err := os.MkdirAll(filepath.Dir(outAbs), 0o755); err != nil {
			return fmt.Errorf("failed to create Go build output directory: %w", err)
		}
		args = append(args, "-o", outAbs, buildPkg)
	} else {
		// Compile-check the whole module; the binary is produced by `go run`.
		args = append(args, "./...")
	}

	cmd := exec.Command(goCmd, args...)
	cmd.Dir = packagePath
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("failed to build Go node (go %s): %w\nOutput: %s", strings.Join(args, " "), err, out)
	}
	return nil
}

// goBuildTarget returns the Go package to compile and the package-relative
// output binary path for this node. When entrypoint.build is declared, the node
// is prebuilt: the build package is compiled to entrypoint.start (a bare binary
// path). When build is empty, there is no prebuilt output (outBin == "") and the
// caller compile-checks the module instead, leaving launch to `go run`.
func (m *PackageMetadata) goBuildTarget() (buildPkg, outBin string) {
	build := strings.TrimSpace(m.Entrypoint.Build)
	if build == "" {
		return "", ""
	}
	start := strings.Fields(m.Entrypoint.Start)
	// The output is the start token when start is a bare binary path (not a
	// `go run ...` command). Otherwise derive a sensible default under bin/.
	if len(start) > 0 && start[0] != "go" {
		return build, withExeSuffix(start[0])
	}
	return build, defaultGoBinName(build)
}

// defaultGoBinName derives a bin/<name> output path from a Go build package spec
// like "./cmd/swe-planner" -> "bin/swe-planner" ("bin/swe-planner.exe" on
// Windows).
func defaultGoBinName(buildPkg string) string {
	name := filepath.Base(strings.TrimSpace(buildPkg))
	if name == "" || name == "." || name == "/" || name == "..." {
		name = "app"
	}
	return withExeSuffix(filepath.Join("bin", name))
}

// withExeSuffix appends ".exe" to a binary path on Windows (manifests declare
// unix-style paths like "bin/swe-planner"; the compiled output must carry the
// conventional executable extension there). A no-op on other platforms and on
// paths that already end in .exe.
func withExeSuffix(path string) string {
	return withExeSuffixFor(runtime.GOOS, path)
}

// withExeSuffixFor is withExeSuffix for an explicit GOOS, pure so the Windows
// branch is unit-testable on any platform.
func withExeSuffixFor(goos, path string) string {
	if goos == "windows" && !strings.EqualFold(filepath.Ext(path), ".exe") {
		return path + ".exe"
	}
	return path
}

// hasVendorDir reports whether the package ships a Go vendor/ directory, which
// makes the build hermetic (and immune to out-of-tree replace directives).
func hasVendorDir(packagePath string) bool {
	info, err := os.Stat(filepath.Join(packagePath, "vendor"))
	return err == nil && info.IsDir()
}

// outOfTreeReplaces returns the go.mod local replace directives whose target is
// a filesystem path that escapes the package directory. These are the ones that
// break after the package is copied into ~/.agentfield/packages. Module-version
// replacements (no path RHS) and in-tree paths are not returned.
func outOfTreeReplaces(packagePath string) []string {
	data, err := os.ReadFile(filepath.Join(packagePath, "go.mod"))
	if err != nil {
		return nil
	}
	var bad []string
	inBlock := false
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "//") {
			continue
		}
		switch {
		case strings.HasPrefix(line, "replace ("):
			inBlock = true
			continue
		case inBlock && line == ")":
			inBlock = false
			continue
		case strings.HasPrefix(line, "replace "):
			if spec := replaceEscapesTree(packagePath, strings.TrimPrefix(line, "replace ")); spec != "" {
				bad = append(bad, spec)
			}
		case inBlock:
			if spec := replaceEscapesTree(packagePath, line); spec != "" {
				bad = append(bad, spec)
			}
		}
	}
	return bad
}

// replaceEscapesTree parses one replace body ("old => new" or "old v => new v")
// and returns the directive text when its RHS is a filesystem path outside
// packagePath, or "" otherwise.
func replaceEscapesTree(packagePath, body string) string {
	body = strings.TrimSpace(body)
	parts := strings.SplitN(body, "=>", 2)
	if len(parts) != 2 {
		return ""
	}
	rhs := strings.TrimSpace(parts[1])
	// The target may be "path" or "path version"; a path starts with . or / (a
	// module-version replacement like "example.com/x v1.2.3" is not a path).
	target := strings.Fields(rhs)
	if len(target) == 0 {
		return ""
	}
	p := target[0]
	if !strings.HasPrefix(p, ".") && !strings.HasPrefix(p, "/") {
		return "" // module-version replacement, not a local path
	}
	abs := p
	if !filepath.IsAbs(p) {
		abs = filepath.Join(packagePath, p)
	}
	rel, err := filepath.Rel(packagePath, filepath.Clean(abs))
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return strings.TrimSpace(body)
	}
	return ""
}

// checkGoReplaceDirectives fails when go.mod has a local replace directive that
// points outside the package tree, since that path won't resolve once the node
// is copied into ~/.agentfield/packages. It is a no-op when the package is
// vendored (vendor/ ships the dependency), when an AGENTFIELD_GO_REPLACE override
// is supplied, or when every replace is in-tree / a module-version replacement.
func checkGoReplaceDirectives(packagePath string) error {
	if hasVendorDir(packagePath) || strings.TrimSpace(os.Getenv("AGENTFIELD_GO_REPLACE")) != "" {
		return nil
	}
	bad := outOfTreeReplaces(packagePath)
	if len(bad) == 0 {
		return nil
	}
	return fmt.Errorf(
		"this Go node's go.mod has local replace directive(s) that point outside the package and won't resolve after install:\n"+
			"  replace %s\n"+
			"Fix it one of these ways, then reinstall:\n"+
			"  • vendor the module so it ships with the node:  go mod vendor   (recommended)\n"+
			"  • or publish/tag the replaced module and use a versioned require instead of a local replace\n"+
			"  • or repoint it at install time:  AGENTFIELD_GO_REPLACE=\"<module>=<abs-path-or-module@version>\" af install",
		strings.Join(bad, "\n  replace "))
}

// applyGoReplaceOverrides applies each entry of the AGENTFIELD_GO_REPLACE env var
// (comma-separated "old=new" pairs, in `go mod edit -replace` syntax) to the
// package's go.mod before building. This is the build-arg-style escape hatch for
// out-of-tree replace directives when vendoring is not an option. A no-op when
// the variable is unset.
func applyGoReplaceOverrides(goCmd, packagePath string) error {
	raw := strings.TrimSpace(os.Getenv("AGENTFIELD_GO_REPLACE"))
	if raw == "" {
		return nil
	}
	for _, entry := range strings.Split(raw, ",") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		cmd := exec.Command(goCmd, "mod", "edit", "-replace="+entry)
		cmd.Dir = packagePath
		if out, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("failed to apply Go replace override %q: %w\nOutput: %s", entry, err, out)
		}
	}
	return nil
}

// GoBinaryProgram resolves a Go node's start program token to an absolute path
// when it is a package-relative binary (e.g. "bin/swe-planner" or
// "./bin/swe-planner"), so exec launches it regardless of the parent process's
// working directory. A bare command such as "go" (the `go run` form) or an
// already-absolute path is returned unchanged. It is the Go counterpart to the
// venv-python substitution the runner does for Python nodes.
func GoBinaryProgram(packageDir, program string) string {
	return goBinaryProgramFor(runtime.GOOS, packageDir, program)
}

// goBinaryProgramFor is GoBinaryProgram for an explicit GOOS, pure so the
// Windows .exe fallback is unit-testable on any platform.
func goBinaryProgramFor(goos, packageDir, program string) string {
	if program == "" || program == "go" || filepath.IsAbs(program) {
		return program
	}
	if strings.ContainsRune(program, '/') || strings.ContainsRune(program, filepath.Separator) {
		resolved := filepath.Join(packageDir, program)
		// Manifests declare unix-style binary paths ("bin/swe-planner"); on
		// Windows the install-time build produced "bin/swe-planner.exe", so
		// resolve to that when the extensionless path is absent.
		if !fileExists(resolved) {
			if withExe := withExeSuffixFor(goos, resolved); withExe != resolved && fileExists(withExe) {
				return withExe
			}
		}
		return resolved
	}
	return program
}
