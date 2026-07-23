package workspace

import "testing"

func TestDefaultExcludes(t *testing.T) {
	m := NewMatcher(nil)
	cases := []struct {
		path    string
		isDir   bool
		ignored bool
	}{
		{".git", true, true},
		{".git/config", false, true},
		{"node_modules", true, true},
		{"node_modules/left-pad/index.js", false, true},
		{"pkg/__pycache__", true, true},
		{"pkg/__pycache__/mod.cpython-311.pyc", false, true},
		{".venv/bin/python", false, true},
		{"venv/bin/python", false, true},
		{"target/debug/app", false, true},
		{"dist/bundle.js", false, true},
		{"build/out.o", false, true},
		{".DS_Store", false, true},
		{"src/.DS_Store", false, true},
		{"src/main.go", false, false},
		{"README.md", false, false},
		{"builder/keep.go", false, false}, // must not match "build/"
	}
	for _, c := range cases {
		if got := m.Ignored(c.path, c.isDir); got != c.ignored {
			t.Errorf("Ignored(%q, dir=%v) = %v, want %v", c.path, c.isDir, got, c.ignored)
		}
	}
}

func TestGitignorePatterns(t *testing.T) {
	lines := []string{
		"# a comment",
		"",
		"*.log",
		"secrets.env",
		"/rootonly.txt",
		"cache/",
		"docs/*.tmp",
		"!keep.log",
	}
	m := NewMatcher(lines)
	cases := []struct {
		path    string
		isDir   bool
		ignored bool
	}{
		{"app.log", false, true},
		{"logs/app.log", false, true}, // *.log matches basename at any depth
		{"keep.log", false, false},    // negation re-includes
		{"secrets.env", false, true},
		{"rootonly.txt", false, true},
		{"sub/rootonly.txt", false, false}, // anchored to root only
		{"cache", true, true},
		{"cache/data.bin", false, true},
		{"docs/notes.tmp", false, true},
		{"docs/deep/notes.tmp", false, false}, // docs/*.tmp is a single level
		{"main.go", false, false},
	}
	for _, c := range cases {
		if got := m.Ignored(c.path, c.isDir); got != c.ignored {
			t.Errorf("Ignored(%q) = %v, want %v", c.path, got, c.ignored)
		}
	}
}

func TestDoubleStarPattern(t *testing.T) {
	m := NewMatcher([]string{"a/**/z.txt"})
	if !m.Ignored("a/z.txt", false) {
		t.Errorf("a/**/z.txt should match a/z.txt")
	}
	if !m.Ignored("a/b/c/z.txt", false) {
		t.Errorf("a/**/z.txt should match a/b/c/z.txt")
	}
	if m.Ignored("a/z.md", false) {
		t.Errorf("a/**/z.txt should not match a/z.md")
	}
}
