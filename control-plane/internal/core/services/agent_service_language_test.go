package services

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
)

// writeVenv materializes a Unix-style virtualenv under dir and returns its root
// and bin directory. A Go-node test that plants one exercises the venv-found
// branches; one that does not exercises the system-Python fallback, which is the
// branch a real Go install reaches (installing a Go node never builds a venv).
func writeVenv(t *testing.T, dir string) (venvPath, venvBin, python string) {
	t.Helper()
	venvPath = filepath.Join(dir, "venv")
	venvBin = filepath.Join(venvPath, "bin")
	require.NoError(t, os.MkdirAll(venvBin, 0o755))
	python = filepath.Join(venvBin, "python")
	require.NoError(t, os.WriteFile(python, []byte("#!/bin/sh\nexit 0\n"), 0o755))
	return venvPath, venvBin, python
}

func buildConfigCapturingStdout(t *testing.T, dir, name string, port int) (interfaces.ProcessConfig, string) {
	t.Helper()
	service := &DefaultAgentService{}
	var (
		cfg interfaces.ProcessConfig
		err error
	)
	out := captureStdout(t, func() {
		cfg, err = service.buildProcessConfig(packages.InstalledPackage{
			Name:    name,
			Path:    dir,
			Runtime: packages.RuntimeInfo{LogFile: filepath.Join(dir, name+".log")},
		}, port)
	})
	require.NoError(t, err)
	return cfg, out
}

// Contract: starting a Go node says nothing about Python, injects no Python env,
// and still launches the resolved Go binary. Covers a manifest that declares
// `language: go` and one that omits it but ships a go.mod (parse-time detection).
func TestBuildProcessConfigGoNodeSkipsPythonResolution(t *testing.T) {
	cases := []struct {
		name     string
		manifest string
		goMod    bool
		venv     bool
	}{
		{
			name:     "explicit language go",
			manifest: "name: go-node\nversion: 0.1.0\nlanguage: go\nentrypoint:\n  start: bin/node\n",
			venv:     true,
		},
		{
			name:     "language omitted with go.mod",
			manifest: "name: go-node\nversion: 0.1.0\nentrypoint:\n  start: bin/node\n",
			goMod:    true,
			venv:     true,
		},
		{
			// The shape `af install` actually produces: a Go node has no venv, so
			// the unguarded code fell through to the system-Python fallback. This
			// is the case that reported the bug, and the only one that can fail on
			// the "not found"/"system Python" assertions below.
			name:     "no venv, as a real Go install leaves it",
			manifest: "name: go-node\nversion: 0.1.0\nlanguage: go\nentrypoint:\n  start: bin/node\n",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// A non-empty PYTHONHOME in the inherited environment keeps the
			// "PYTHONHOME=" assertion below about what the venv block appends.
			t.Setenv("PYTHONHOME", "inherited")

			dir := t.TempDir()
			writeManifest(t, dir, tc.manifest)
			if tc.goMod {
				require.NoError(t, os.WriteFile(filepath.Join(dir, "go.mod"),
					[]byte("module example.com/node\n\ngo 1.21\n"), 0o644))
			}
			venvPath := filepath.Join(dir, "venv")
			venvBin := filepath.Join(venvPath, "bin")
			if tc.venv {
				venvPath, venvBin, _ = writeVenv(t, dir)
			}
			require.NoError(t, os.MkdirAll(filepath.Join(dir, "bin"), 0o755))
			binary := filepath.Join(dir, "bin", "node")
			require.NoError(t, os.WriteFile(binary, []byte("#!/bin/sh\nexit 0\n"), 0o755))

			cfg, out := buildConfigCapturingStdout(t, dir, "go-node", 8201)

			// Contract 1: no Python/venv chatter.
			for _, phrase := range []string{
				"Using virtual environment",
				"Virtual environment not found",
				"Virtual environment fully activated",
				"system Python",
			} {
				assert.NotContains(t, out, phrase, "Go node printed a Python message")
			}

			// Contract 2: no Python env injected as a side effect. Collected rather
			// than asserted entry-by-entry so a failure names the offender instead
			// of dumping the inherited environment.
			var injected []string
			for _, entry := range cfg.Env {
				switch {
				case entry == "VIRTUAL_ENV="+venvPath,
					entry == "PYTHONHOME=",
					entry == "PYTHONPATH="+filepath.Join(venvPath, "lib"),
					strings.HasPrefix(entry, "PATH="+venvBin):
					injected = append(injected, entry)
				}
			}
			assert.Empty(t, injected, "Go node had Python env injected")

			// Contract 5: the launch command is the resolved Go binary.
			assert.Equal(t, binary, cfg.Command)
			assert.Empty(t, cfg.Args)
		})
	}
}

// Contract: a Python node keeps its interpreter resolution — venv activation
// when one exists, system-Python fallback plus the warning when it does not.
func TestBuildProcessConfigPythonNodeKeepsInterpreterResolution(t *testing.T) {
	const manifest = "name: py-node\nversion: 0.1.0\nlanguage: python\nentrypoint:\n  start: python main.py\n"

	t.Run("with venv", func(t *testing.T) {
		t.Setenv("PYTHONHOME", "inherited")

		dir := t.TempDir()
		writeManifest(t, dir, manifest)
		venvPath, venvBin, python := writeVenv(t, dir)

		cfg, out := buildConfigCapturingStdout(t, dir, "py-node", 8202)

		assert.Contains(t, out, "Using virtual environment: "+venvPath)
		assert.Equal(t, python, cfg.Command)
		assert.Equal(t, []string{"main.py"}, cfg.Args)
		assert.Contains(t, cfg.Env, "VIRTUAL_ENV="+venvPath)
		assertEnvWithPrefix(t, cfg.Env, "PATH=", venvBin)
		assert.Contains(t, cfg.Env, "PYTHONHOME=")
		assert.Contains(t, cfg.Env, "PYTHONPATH="+filepath.Join(venvPath, "lib"))
	})

	t.Run("without venv", func(t *testing.T) {
		dir := t.TempDir()
		writeManifest(t, dir, manifest)

		stubBin := filepath.Join(dir, "stub")
		require.NoError(t, os.MkdirAll(stubBin, 0o755))
		stubPython := filepath.Join(stubBin, "python3")
		require.NoError(t, os.WriteFile(stubPython, []byte("#!/bin/sh\nexit 0\n"), 0o755))
		t.Setenv("PATH", stubBin)

		cfg, out := buildConfigCapturingStdout(t, dir, "py-node", 8203)

		assert.Contains(t, out, "Virtual environment not found")
		assert.Equal(t, stubPython, cfg.Command)
	})
}
