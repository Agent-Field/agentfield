package packages

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	toml "github.com/pelletier/go-toml/v2"
)

// pyVersion is a parsed (major, minor, patch) Python version.
type pyVersion struct {
	major, minor, patch int
}

func (v pyVersion) String() string {
	return fmt.Sprintf("%d.%d.%d", v.major, v.minor, v.patch)
}

// compare returns -1, 0, or 1 as v is less than, equal to, or greater than o.
func (v pyVersion) compare(o pyVersion) int {
	for _, d := range []int{v.major - o.major, v.minor - o.minor, v.patch - o.patch} {
		if d < 0 {
			return -1
		}
		if d > 0 {
			return 1
		}
	}
	return 0
}

// parsePyVersion parses a dotted version like "3.12" or "3.12.5" into a
// pyVersion. A missing minor/patch defaults to 0. Non-numeric suffixes on a
// component (e.g. "3.12rc1") are ignored. It reports false only when the major
// component is absent or non-numeric.
func parsePyVersion(s string) (pyVersion, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return pyVersion{}, false
	}
	parts := strings.Split(s, ".")
	nums := [3]int{}
	for i := 0; i < len(parts) && i < 3; i++ {
		digits := leadingDigits(parts[i])
		if digits == "" {
			if i == 0 {
				return pyVersion{}, false
			}
			break
		}
		n, err := strconv.Atoi(digits)
		if err != nil {
			return pyVersion{}, false
		}
		nums[i] = n
	}
	return pyVersion{nums[0], nums[1], nums[2]}, true
}

// leadingDigits returns the leading run of ASCII digits in s.
func leadingDigits(s string) string {
	i := 0
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	return s[:i]
}

// constraintSatisfied reports whether version v satisfies a PEP 440-style
// requires-python constraint. The constraint is a comma-separated list of
// clauses, each an operator (>=, >, <=, <, ==, ===, !=, ~=) followed by a
// version, with an optional ".*" wildcard on == / !=. An empty or
// unparseable constraint is treated as satisfied so that a manifest we cannot
// understand never blocks installation.
func constraintSatisfied(v pyVersion, requires string) bool {
	requires = strings.TrimSpace(requires)
	if requires == "" {
		return true
	}
	for _, clause := range strings.Split(requires, ",") {
		clause = strings.TrimSpace(clause)
		if clause == "" {
			continue
		}
		if !clauseSatisfied(v, clause) {
			return false
		}
	}
	return true
}

// clauseSatisfied evaluates a single constraint clause against v.
func clauseSatisfied(v pyVersion, clause string) bool {
	op := "=="
	rest := clause
	// Order matters: match the longest operators first.
	for _, o := range []string{"===", ">=", "<=", "==", "!=", "~=", ">", "<"} {
		if strings.HasPrefix(clause, o) {
			op = o
			rest = strings.TrimSpace(clause[len(o):])
			break
		}
	}

	wildcard := strings.HasSuffix(rest, ".*")
	rest = strings.TrimSuffix(rest, ".*")
	want, ok := parsePyVersion(rest)
	if !ok {
		return true // permissive: don't block on something we can't parse
	}
	cmp := v.compare(want)

	switch op {
	case ">=":
		return cmp >= 0
	case ">":
		return cmp > 0
	case "<=":
		return cmp <= 0
	case "<":
		return cmp < 0
	case "!=":
		if wildcard {
			return v.major != want.major || v.minor != want.minor
		}
		return cmp != 0
	case "==", "===":
		if wildcard {
			return v.major == want.major && v.minor == want.minor
		}
		return cmp == 0
	case "~=":
		// ~=X.Y  => >=X.Y and ==X.*   ; ~=X.Y.Z => >=X.Y.Z and ==X.Y.*
		if cmp < 0 {
			return false
		}
		if strings.Count(rest, ".") >= 2 {
			return v.major == want.major && v.minor == want.minor
		}
		return v.major == want.major
	}
	return true
}

// readRequiresPython returns the requires-python constraint declared in the
// package's pyproject.toml, or "" when there is no pyproject.toml, no
// [project].requires-python field, or the file cannot be parsed.
func readRequiresPython(packagePath string) string {
	data, err := os.ReadFile(filepath.Join(packagePath, "pyproject.toml"))
	if err != nil {
		return ""
	}
	var pp struct {
		Project struct {
			RequiresPython string `toml:"requires-python"`
		} `toml:"project"`
	}
	if err := toml.Unmarshal(data, &pp); err != nil {
		return ""
	}
	return strings.TrimSpace(pp.Project.RequiresPython)
}

// interpreterVersion runs the interpreter and returns its reported version.
func interpreterVersion(interp string) (pyVersion, bool) {
	out, err := exec.Command(interp, "-c", "import sys;print('%d.%d.%d'%sys.version_info[:3])").Output()
	if err != nil {
		return pyVersion{}, false
	}
	return parsePyVersion(strings.TrimSpace(string(out)))
}

// uvRequest derives a bare "major.minor" interpreter request from a
// requires-python constraint, taken from the first lower-bound / equality
// clause (>=, ==, ===, ~=, >). Returns "" when no such clause is present.
// A bare major.minor is the request form uv accepts most reliably; the caller
// re-checks the resolved interpreter against the full constraint afterward.
func uvRequest(requires string) string {
	for _, clause := range strings.Split(requires, ",") {
		clause = strings.TrimSpace(clause)
		for _, o := range []string{"===", ">=", "==", "~=", ">"} {
			if strings.HasPrefix(clause, o) {
				rest := strings.TrimSuffix(strings.TrimSpace(clause[len(o):]), ".*")
				if v, ok := parsePyVersion(rest); ok {
					return fmt.Sprintf("%d.%d", v.major, v.minor)
				}
			}
		}
	}
	return ""
}

// resolveVenvInterpreter picks the interpreter to build a node's venv with,
// honoring the package's requires-python. It returns:
//   - ("", nil) when the package declares no constraint — the caller keeps its
//     legacy python3 -> python fallback, so behavior is unchanged.
//   - (interp, nil) with a concrete interpreter (command or path) to use.
//   - ("", err) with an actionable message when a constraint is declared but no
//     compatible interpreter is available and none can be provisioned.
//
// Resolution order when a constraint is present: the ambient python3/python if
// it already satisfies, then a uv-provisioned interpreter (uv auto-downloads a
// standalone build), then a matching pyenv-installed version.
func resolveVenvInterpreter(packagePath string) (string, error) {
	requires := readRequiresPython(packagePath)
	if requires == "" {
		return "", nil
	}

	// 1. Ambient interpreter already satisfies?
	ambient := firstOnPath("python3", "python")
	if ambient != "" {
		if v, ok := interpreterVersion(ambient); ok && constraintSatisfied(v, requires) {
			return ambient, nil
		}
	}

	// 2. Provision via uv (downloads a standalone interpreter if needed).
	if interp := provisionViaUv(requires); interp != "" {
		// clearLine() wipes the active dependency spinner's line so this notice
		// lands cleanly on its own line instead of being appended to it.
		fmt.Printf("%s  🐍 Provisioned Python %s via uv (node requires %q)\n", clearLine(), displayVersion(interp), requires)
		return interp, nil
	}

	// 3. Discover a matching pyenv-installed version.
	if interp := provisionViaPyenv(requires); interp != "" {
		fmt.Printf("%s  🐍 Using pyenv Python %s (node requires %q)\n", clearLine(), displayVersion(interp), requires)
		return interp, nil
	}

	found := "no python3 found on PATH"
	if ambient != "" {
		if v, ok := interpreterVersion(ambient); ok {
			found = fmt.Sprintf("found %s (Python %s)", ambient, v)
		}
	}
	req := uvRequest(requires)
	if req == "" {
		req = requires
	}
	return "", fmt.Errorf(
		"this agent node requires Python %s, but no compatible interpreter is available (%s).\n"+
			"Fix it by installing a matching Python, then reinstall:\n"+
			"  • with uv (auto-downloads it):  uv python install %s\n"+
			"  • or via pyenv:                 pyenv install %s\n"+
			"  • or install Python %s from your OS package manager\n"+
			"then ensure it is on PATH (or that uv/pyenv can see it) and run `af install` again",
		requires, found, req, req, req)
}

// firstOnPath returns the first of the given commands found on PATH, or "".
func firstOnPath(cmds ...string) string {
	for _, c := range cmds {
		if _, err := exec.LookPath(c); err == nil {
			return c
		}
	}
	return ""
}

// displayVersion returns interp's version as a string for logging, or the
// interpreter path itself when the version cannot be determined.
func displayVersion(interp string) string {
	if v, ok := interpreterVersion(interp); ok {
		return v.String()
	}
	return interp
}

// provisionViaUv asks uv for an interpreter satisfying requires, downloading a
// standalone build if necessary. Returns the interpreter path, or "" when uv is
// unavailable or cannot provide a satisfying interpreter.
func provisionViaUv(requires string) string {
	if _, err := exec.LookPath("uv"); err != nil {
		return ""
	}
	req := uvRequest(requires)
	if req == "" {
		return ""
	}
	// Ensure a matching interpreter exists (no-op if already present).
	_ = exec.Command("uv", "python", "install", req).Run()
	out, err := exec.Command("uv", "python", "find", req).Output()
	if err != nil {
		return ""
	}
	interp := strings.TrimSpace(string(out))
	if interp == "" {
		return ""
	}
	if v, ok := interpreterVersion(interp); ok && constraintSatisfied(v, requires) {
		return interp
	}
	return ""
}

// provisionViaPyenv returns the highest pyenv-installed interpreter that
// satisfies requires, or "" when pyenv is unavailable or has no match. It only
// discovers already-installed versions; it does not trigger a source build.
func provisionViaPyenv(requires string) string {
	if _, err := exec.LookPath("pyenv"); err != nil {
		return ""
	}
	versionsOut, err := exec.Command("pyenv", "versions", "--bare").Output()
	if err != nil {
		return ""
	}
	rootOut, err := exec.Command("pyenv", "root").Output()
	if err != nil {
		return ""
	}
	pyenvRoot := strings.TrimSpace(string(rootOut))

	var best string
	var bestV pyVersion
	for _, line := range strings.Split(string(versionsOut), "\n") {
		name := strings.TrimSpace(line)
		if name == "" {
			continue
		}
		v, ok := parsePyVersion(name)
		if !ok || !constraintSatisfied(v, requires) {
			continue
		}
		if best != "" && v.compare(bestV) <= 0 {
			continue
		}
		interp := filepath.Join(pyenvRoot, "versions", name, "bin", "python")
		if fileExists(interp) {
			best, bestV = interp, v
		}
	}
	return best
}
