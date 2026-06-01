package packages

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeExecutable(t *testing.T, path, contents string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(contents), 0755); err != nil {
		t.Fatalf("write executable %s: %v", path, err)
	}
}

func prependPath(t *testing.T, dir string) {
	t.Helper()
	orig := os.Getenv("PATH")
	t.Setenv("PATH", dir+string(os.PathListSeparator)+orig)
}

func setupFakePythonTools(t *testing.T, binDir string, failPython3, failPython, failRequirements bool, failDep string) string {
	t.Helper()

	pipScript := `#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$FAKE_PIP_LOG"
if [ "${FAKE_PIP_FAIL_REQUIREMENTS:-}" = "1" ] && [ "${2:-}" = "-r" ]; then
  echo "requirements failure" >&2
  exit 1
fi
if [ -n "${FAKE_PIP_FAIL_DEP:-}" ] && [ "${2:-}" = "$FAKE_PIP_FAIL_DEP" ]; then
  echo "dependency failure" >&2
  exit 1
fi
exit 0
`

	pythonImpl := `#!/usr/bin/env bash
set -eu
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "venv" ]; then
  venv_path="${3:-}"
  mkdir -p "$venv_path/bin"
  cat > "$venv_path/bin/pip" <<'EOF'
` + pipScript + `EOF
  chmod +x "$venv_path/bin/pip"
  exit 0
fi
echo "unexpected python args: $*" >&2
exit 1
`

	python3Script := pythonImpl
	if failPython3 {
		python3Script = "#!/usr/bin/env bash\nexit 1\n"
	}
	pythonScript := pythonImpl
	if failPython {
		pythonScript = "#!/usr/bin/env bash\nexit 1\n"
	}

	writeExecutable(t, filepath.Join(binDir, "python3"), python3Script)
	writeExecutable(t, filepath.Join(binDir, "python"), pythonScript)

	pipLog := filepath.Join(t.TempDir(), "pip.log")
	t.Setenv("FAKE_PIP_LOG", pipLog)
	if failRequirements {
		t.Setenv("FAKE_PIP_FAIL_REQUIREMENTS", "1")
	}
	if failDep != "" {
		t.Setenv("FAKE_PIP_FAIL_DEP", failDep)
	}
	return pipLog
}

func TestInstallDependenciesAdditionalCoverage(t *testing.T) {
	tests := []struct {
		name             string
		withRequirements bool
		pythonDeps       []string
		systemDeps       []string
		failPython3      bool
		failPython       bool
		failRequirements bool
		failDep          string
		wantErr          string
		wantPipCalls     []string
	}{
		{
			name:       "system dependencies only",
			systemDeps: []string{"curl"},
		},
		{
			name:             "fallback to python installs requirements and deps",
			withRequirements: true,
			pythonDeps:       []string{"dep1"},
			systemDeps:       []string{"jq"},
			failPython3:      true,
			wantPipCalls: []string{
				"install --upgrade pip",
				"install -r",
				"install dep1",
			},
		},
		{
			name:             "venv creation failure",
			withRequirements: true,
			failPython3:      true,
			failPython:       true,
			wantErr:          "failed to create virtual environment",
		},
		{
			name:             "requirements install failure",
			withRequirements: true,
			failRequirements: true,
			wantErr:          "failed to install requirements.txt dependencies",
		},
		{
			name:        "dependency install failure",
			pythonDeps:  []string{"baddep"},
			failDep:     "baddep",
			wantErr:     "failed to install dependency baddep",
			wantPipCalls: []string{
				"install --upgrade pip",
				"install baddep",
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			installer := &PackageInstaller{AgentFieldHome: t.TempDir()}
			pkgDir := t.TempDir()
			if tc.withRequirements {
				if err := os.WriteFile(filepath.Join(pkgDir, "requirements.txt"), []byte("demo==1.0.0\n"), 0644); err != nil {
					t.Fatalf("write requirements.txt: %v", err)
				}
			}

			binDir := filepath.Join(t.TempDir(), "bin")
			if err := os.MkdirAll(binDir, 0755); err != nil {
				t.Fatalf("mkdir bin: %v", err)
			}
			pipLog := setupFakePythonTools(t, binDir, tc.failPython3, tc.failPython, tc.failRequirements, tc.failDep)
			prependPath(t, binDir)

			err := installer.installDependencies(pkgDir, &PackageMetadata{
				Dependencies: DependencyConfig{
					Python: tc.pythonDeps,
					System: tc.systemDeps,
				},
			})
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("installDependencies error = %v, want %q", err, tc.wantErr)
				}
			} else if err != nil {
				t.Fatalf("installDependencies: %v", err)
			}

			data, readErr := os.ReadFile(pipLog)
			logOutput := string(data)
			if readErr != nil && len(tc.wantPipCalls) > 0 {
				t.Fatalf("read pip log: %v", readErr)
			}
			for _, want := range tc.wantPipCalls {
				if !strings.Contains(logOutput, want) {
					t.Fatalf("pip log %q missing %q", logOutput, want)
				}
			}
		})
	}
}
