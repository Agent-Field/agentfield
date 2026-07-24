package packages

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// Contract: compiled Go node binaries get the conventional .exe extension on
// Windows and stay untouched elsewhere; already-suffixed paths (any case) are
// never double-suffixed.
func TestWithExeSuffixFor(t *testing.T) {
	cases := []struct {
		name string
		goos string
		in   string
		want string
	}{
		{"windows adds exe", "windows", "bin/swe-planner", "bin/swe-planner.exe"},
		{"windows keeps existing exe", "windows", "bin/swe-planner.exe", "bin/swe-planner.exe"},
		{"windows keeps uppercase exe", "windows", "bin/APP.EXE", "bin/APP.EXE"},
		{"linux untouched", "linux", "bin/swe-planner", "bin/swe-planner"},
		{"darwin untouched", "darwin", "bin/swe-planner", "bin/swe-planner"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := withExeSuffixFor(tc.goos, tc.in); got != tc.want {
				t.Fatalf("withExeSuffixFor(%q, %q) = %q; want %q", tc.goos, tc.in, got, tc.want)
			}
		})
	}
}

// Contract: an unusable build-package base falls back to bin/app (suffixed on
// Windows like every other derived binary name).
func TestDefaultGoBinNameFallback(t *testing.T) {
	want := withExeSuffix(filepath.Join("bin", "app"))
	if got := defaultGoBinName("..."); got != want {
		t.Fatalf("defaultGoBinName(\"...\") = %q; want %q", got, want)
	}
}

// Contract: on Windows, a manifest's unix-style start path ("bin/app")
// resolves to the .exe the install-time build produced when the extensionless
// file is absent; everywhere else (and whenever the extensionless file exists)
// the plain resolved path wins.
func TestGoBinaryProgramForWindowsExeFallback(t *testing.T) {
	writeFile := func(t *testing.T, dir, rel string) {
		t.Helper()
		p := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("bin"), 0o755); err != nil {
			t.Fatal(err)
		}
	}

	t.Run("windows falls back to built exe", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, dir, filepath.Join("bin", "app.exe"))
		want := filepath.Join(dir, "bin", "app.exe")
		if got := goBinaryProgramFor("windows", dir, "bin/app"); got != want {
			t.Fatalf("goBinaryProgramFor = %q; want %q", got, want)
		}
	})

	t.Run("windows prefers extensionless file when present", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, dir, filepath.Join("bin", "app"))
		writeFile(t, dir, filepath.Join("bin", "app.exe"))
		want := filepath.Join(dir, "bin", "app")
		if got := goBinaryProgramFor("windows", dir, "bin/app"); got != want {
			t.Fatalf("goBinaryProgramFor = %q; want %q", got, want)
		}
	})

	t.Run("windows with neither file returns resolved path", func(t *testing.T) {
		dir := t.TempDir()
		want := filepath.Join(dir, "bin", "app")
		if got := goBinaryProgramFor("windows", dir, "bin/app"); got != want {
			t.Fatalf("goBinaryProgramFor = %q; want %q", got, want)
		}
	})

	t.Run("non-windows never substitutes exe", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, dir, filepath.Join("bin", "app.exe"))
		want := filepath.Join(dir, "bin", "app")
		if got := goBinaryProgramFor("linux", dir, "bin/app"); got != want {
			t.Fatalf("goBinaryProgramFor = %q; want %q", got, want)
		}
	})

	t.Run("bare and special program tokens pass through", func(t *testing.T) {
		dir := t.TempDir()
		for _, program := range []string{"", "go", "app"} {
			if got := goBinaryProgramFor("windows", dir, program); got != program {
				t.Fatalf("goBinaryProgramFor(%q) = %q; want it unchanged", program, got)
			}
		}
		abs := filepath.Join(dir, "bin", "app")
		if got := goBinaryProgramFor("windows", dir, abs); got != abs {
			t.Fatalf("goBinaryProgramFor(abs) = %q; want %q", got, abs)
		}
	})

	t.Run("exported wrapper uses host GOOS", func(t *testing.T) {
		dir := t.TempDir()
		if got, want := GoBinaryProgram(dir, "bin/app"), goBinaryProgramFor(runtime.GOOS, dir, "bin/app"); got != want {
			t.Fatalf("GoBinaryProgram = %q; want %q", got, want)
		}
	})
}
