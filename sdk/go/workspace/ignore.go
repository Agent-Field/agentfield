package workspace

import (
	"bufio"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// DefaultExcludes are the folder/file patterns always ignored when sealing,
// independent of any .gitignore. They match the spec's default exclude list.
var DefaultExcludes = []string{
	".git/",
	"node_modules/",
	".venv/",
	"venv/",
	"__pycache__/",
	"target/",
	"dist/",
	"build/",
	".DS_Store",
}

// ignorePattern is one compiled .gitignore-style rule.
type ignorePattern struct {
	negate   bool // pattern started with '!'
	dirOnly  bool // pattern ended with '/'
	anchored bool // pattern is rooted at the workspace root (contained a slash)
	segments []string
	raw      string
}

// Matcher evaluates .gitignore-style rules plus the built-in default excludes.
// The zero value is not usable; construct with NewMatcher.
type Matcher struct {
	patterns []ignorePattern
}

// NewMatcher builds a matcher from the default excludes followed by the given
// pattern lines (typically the contents of a root .gitignore). Later patterns
// take precedence, so a trailing "!keep" line can re-include a path a default
// or earlier rule excluded — mirroring git's last-match-wins semantics.
func NewMatcher(gitignoreLines []string) *Matcher {
	m := &Matcher{}
	for _, line := range DefaultExcludes {
		if p, ok := compilePattern(line); ok {
			m.patterns = append(m.patterns, p)
		}
	}
	for _, line := range gitignoreLines {
		if p, ok := compilePattern(line); ok {
			m.patterns = append(m.patterns, p)
		}
	}
	return m
}

// LoadMatcher reads a .gitignore file at root (if present) and returns a matcher
// combining the default excludes with its rules. A missing .gitignore is not an
// error; only the defaults apply.
func LoadMatcher(root string) (*Matcher, error) {
	lines, err := readGitignore(filepath.Join(root, ".gitignore"))
	if err != nil {
		return nil, err
	}
	return NewMatcher(lines), nil
}

func readGitignore(gitignorePath string) ([]string, error) {
	f, err := os.Open(gitignorePath) //nolint:gosec // path derived from the sealed workspace root
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()

	var lines []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	return lines, scanner.Err()
}

func compilePattern(line string) (ignorePattern, bool) {
	trimmed := strings.TrimRight(line, " ")
	if trimmed == "" || strings.HasPrefix(trimmed, "#") {
		return ignorePattern{}, false
	}

	p := ignorePattern{raw: trimmed}
	if strings.HasPrefix(trimmed, "!") {
		p.negate = true
		trimmed = trimmed[1:]
	}
	if strings.HasSuffix(trimmed, "/") {
		p.dirOnly = true
		trimmed = strings.TrimSuffix(trimmed, "/")
	}
	// A slash anywhere except a trailing one anchors the pattern to the root.
	if strings.Contains(trimmed, "/") {
		p.anchored = true
	}
	trimmed = strings.TrimPrefix(trimmed, "/")
	if trimmed == "" {
		return ignorePattern{}, false
	}
	p.segments = strings.Split(trimmed, "/")
	return p, true
}

// Ignored reports whether a relative, slash-separated path should be excluded.
// isDir must indicate whether the path is a directory so that directory-only
// patterns (trailing slash) are honored. Evaluation is last-match-wins.
func (m *Matcher) Ignored(relPath string, isDir bool) bool {
	relPath = strings.Trim(filepathToSlash(relPath), "/")
	if relPath == "" {
		return false
	}
	segments := strings.Split(relPath, "/")

	ignored := false
	matched := false
	for _, p := range m.patterns {
		if p.matches(segments, isDir) {
			ignored = !p.negate
			matched = true
		}
	}
	_ = matched
	return ignored
}

// matches reports whether the pattern applies to the given path segments.
func (p ignorePattern) matches(segments []string, isDir bool) bool {
	if p.dirOnly {
		// A directory-only rule matches the directory itself or anything nested
		// beneath it, but never a plain file at the leaf position.
		if p.anchored {
			return p.matchAnchoredDir(segments)
		}
		return p.matchBasenameDir(segments)
	}
	if p.anchored {
		return p.matchAnchored(segments, len(segments))
	}
	// Unanchored file pattern: match the basename against the final segment, and
	// also allow matching any ancestor segment so an excluded directory prunes
	// its whole subtree.
	last := segments[len(segments)-1]
	if segmentMatch(p.segments[len(p.segments)-1], last) && p.matchBasenameTail(segments) {
		return true
	}
	return p.matchBasenameDir(segments)
}

// matchBasenameTail matches the (possibly multi-segment) unanchored pattern
// against the tail of the path segments.
func (p ignorePattern) matchBasenameTail(segments []string) bool {
	if len(p.segments) > len(segments) {
		return false
	}
	offset := len(segments) - len(p.segments)
	for i, seg := range p.segments {
		if !segmentMatch(seg, segments[offset+i]) {
			return false
		}
	}
	return true
}

// matchBasenameDir returns true when any ancestor segment matches an unanchored
// single-segment directory pattern (pruning the subtree).
func (p ignorePattern) matchBasenameDir(segments []string) bool {
	if len(p.segments) != 1 {
		return p.matchBasenameTail(segments)
	}
	for _, seg := range segments {
		if segmentMatch(p.segments[0], seg) {
			return true
		}
	}
	return false
}

// matchAnchored matches an anchored pattern against the leading path segments,
// supporting "**" as a match for zero or more segments.
func (p ignorePattern) matchAnchored(segments []string, limit int) bool {
	return matchSegments(p.segments, segments[:limit])
}

func (p ignorePattern) matchAnchoredDir(segments []string) bool {
	// Match the pattern against every prefix of the path so that a rule like
	// "build/" prunes build/ and everything under it.
	for i := 1; i <= len(segments); i++ {
		if matchSegments(p.segments, segments[:i]) {
			return true
		}
	}
	return false
}

// matchSegments matches pattern segments against path segments with "**" support.
func matchSegments(pattern, pathSegs []string) bool {
	if len(pattern) == 0 {
		return len(pathSegs) == 0
	}
	if pattern[0] == "**" {
		// "**" consumes zero or more path segments.
		for i := 0; i <= len(pathSegs); i++ {
			if matchSegments(pattern[1:], pathSegs[i:]) {
				return true
			}
		}
		return false
	}
	if len(pathSegs) == 0 {
		return false
	}
	if !segmentMatch(pattern[0], pathSegs[0]) {
		return false
	}
	return matchSegments(pattern[1:], pathSegs[1:])
}

// segmentMatch matches a single glob segment (supporting *, ?, [] via path.Match)
// against a single path segment.
func segmentMatch(pattern, name string) bool {
	if pattern == "*" || pattern == name {
		return true
	}
	ok, err := path.Match(pattern, name)
	if err != nil {
		return false
	}
	return ok
}

// filepathToSlash converts OS separators to forward slashes without importing
// path/filepath at every call site.
func filepathToSlash(p string) string {
	return strings.ReplaceAll(p, "\\", "/")
}
