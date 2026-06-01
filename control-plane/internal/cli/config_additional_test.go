package cli

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
)

func TestPackageConfigManagerEnvFiles(t *testing.T) {
	t.Run("load env strips quotes and comments", func(t *testing.T) {
		pkgDir := t.TempDir()
		require.NoError(t, os.WriteFile(filepath.Join(pkgDir, ".env"), []byte(strings.Join([]string{
			"# comment",
			"PLAIN=value",
			`DOUBLE="quoted value"`,
			"SINGLE='quoted-secret'",
			"",
		}, "\n")), 0o600))

		pcm := &PackageConfigManager{}
		envVars, err := pcm.loadEnvFile(pkgDir)
		require.NoError(t, err)
		require.Equal(t, map[string]string{
			"PLAIN":  "value",
			"DOUBLE": "quoted value",
			"SINGLE": "quoted-secret",
		}, envVars)
	})

	t.Run("save env quotes special values", func(t *testing.T) {
		pkgDir := t.TempDir()
		pcm := &PackageConfigManager{}

		err := pcm.saveEnvFile(pkgDir, map[string]string{
			"API_KEY": "abc 123",
			"PLAIN":   "value",
			"QUOTE":   `a"b`,
		})
		require.NoError(t, err)

		data, err := os.ReadFile(filepath.Join(pkgDir, ".env"))
		require.NoError(t, err)
		content := string(data)
		require.Contains(t, content, "PLAIN=value")
		require.Contains(t, content, `API_KEY="abc 123"`)
		require.Contains(t, content, `QUOTE="a\"b"`)
	})
}

func TestPackageConfigManagerLoadAndMutateVariables(t *testing.T) {
	homeDir := t.TempDir()
	pkgDir := filepath.Join(homeDir, "packages", "demo")
	require.NoError(t, os.MkdirAll(pkgDir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(homeDir, "installed.yaml"), []byte(`
installed:
  demo:
    name: demo
    path: `+pkgDir+`
`), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"), []byte(`
name: demo
user_environment:
  required:
    - name: API_KEY
      description: api key
      validation: "^[A-Z]+$"
  optional:
    - name: REGION
      description: region
      default: us-east-1
`), 0o644))

	pcm := &PackageConfigManager{AgentFieldHome: homeDir}

	metadata, loadedPath, err := pcm.loadPackageMetadata("demo")
	require.NoError(t, err)
	require.Equal(t, pkgDir, loadedPath)
	require.Equal(t, "API_KEY", metadata.UserEnvironment.Required[0].Name)

	err = pcm.SetVariable("demo", "API_KEY", "ABC")
	require.NoError(t, err)
	envVars, err := pcm.loadEnvFile(pkgDir)
	require.NoError(t, err)
	require.Equal(t, "ABC", envVars["API_KEY"])

	err = pcm.SetVariable("demo", "API_KEY", "abc")
	require.ErrorContains(t, err, "value does not match required format")

	err = pcm.SetVariable("demo", "UNKNOWN", "value")
	require.ErrorContains(t, err, "unknown environment variable")

	err = pcm.SetVariable("demo", "API_KEY", "ANY")
	require.NoError(t, err)
	err = pcm.UnsetVariable("demo", "API_KEY")
	require.NoError(t, err)
	envVars, err = pcm.loadEnvFile(pkgDir)
	require.NoError(t, err)
	require.NotContains(t, envVars, "API_KEY")
}

func TestPackageConfigManagerListConfigAndErrors(t *testing.T) {
	t.Run("list config prints required and optional values", func(t *testing.T) {
		homeDir := t.TempDir()
		pkgDir := filepath.Join(homeDir, "pkg")
		require.NoError(t, os.MkdirAll(pkgDir, 0o755))
		require.NoError(t, os.WriteFile(filepath.Join(homeDir, "installed.yaml"), []byte(`
installed:
  demo:
    name: demo
    path: `+pkgDir+`
`), 0o644))
		require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"), []byte(`
name: demo
user_environment:
  required:
    - name: SECRET_KEY
      description: top secret
      type: secret
  optional:
    - name: REGION
      description: deployment region
      default: us-west-2
    - name: EMPTY
      description: unset optional
`), 0o644))
		require.NoError(t, os.WriteFile(filepath.Join(pkgDir, ".env"), []byte("SECRET_KEY=supersecret\n"), 0o600))

		pcm := &PackageConfigManager{AgentFieldHome: homeDir}
		output := captureOutput(t, func() {
			require.NoError(t, pcm.ListConfig("demo"))
		})
		require.Contains(t, output, "Required Variables")
		require.Contains(t, output, "SECRET_KEY")
		require.Contains(t, output, "supe***cret")
		require.Contains(t, output, "REGION: us-west-2 (default)")
		require.Contains(t, output, "EMPTY: (not set)")
	})

	t.Run("load metadata surfaces parse and read errors", func(t *testing.T) {
		homeDir := t.TempDir()
		pcm := &PackageConfigManager{AgentFieldHome: homeDir}

		_, _, err := pcm.loadPackageMetadata("missing")
		require.ErrorContains(t, err, "not installed")

		require.NoError(t, os.WriteFile(filepath.Join(homeDir, "installed.yaml"), []byte("installed: ["), 0o644))
		_, _, err = pcm.loadPackageMetadata("missing")
		require.ErrorContains(t, err, "failed to parse registry")

		homeDir = t.TempDir()
		pkgDir := filepath.Join(homeDir, "pkg")
		require.NoError(t, os.MkdirAll(pkgDir, 0o755))
		require.NoError(t, os.WriteFile(filepath.Join(homeDir, "installed.yaml"), []byte(`
installed:
  demo:
    name: demo
    path: `+pkgDir+`
`), 0o644))
		pcm = &PackageConfigManager{AgentFieldHome: homeDir}
		_, _, err = pcm.loadPackageMetadata("demo")
		require.ErrorContains(t, err, "failed to read package metadata")

		require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"), []byte("name: ["), 0o644))
		_, _, err = pcm.loadPackageMetadata("demo")
		require.ErrorContains(t, err, "failed to parse package metadata")
	})

	t.Run("unset variable without env file fails", func(t *testing.T) {
		homeDir := t.TempDir()
		pkgDir := filepath.Join(homeDir, "pkg")
		require.NoError(t, os.MkdirAll(pkgDir, 0o755))
		require.NoError(t, os.WriteFile(filepath.Join(homeDir, "installed.yaml"), []byte(`
installed:
  demo:
    name: demo
    path: `+pkgDir+`
`), 0o644))
		require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"), []byte("name: demo\n"), 0o644))

		pcm := &PackageConfigManager{AgentFieldHome: homeDir}
		err := pcm.UnsetVariable("demo", "MISSING")
		require.ErrorContains(t, err, "no environment file found")
	})
}

func TestPackageConfigManagerInteractiveAndCommand(t *testing.T) {
	t.Run("interactive config saves required and optional values", func(t *testing.T) {
		homeDir := t.TempDir()
		pkgDir := filepath.Join(homeDir, "pkg")
		require.NoError(t, os.MkdirAll(pkgDir, 0o755))
		require.NoError(t, os.WriteFile(filepath.Join(homeDir, "installed.yaml"), []byte(`
installed:
  demo:
    name: demo
    path: `+pkgDir+`
`), 0o644))
		require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"), []byte(`
name: demo
user_environment:
  required:
    - name: API_KEY
      description: api key
`), 0o644))

		pcm := &PackageConfigManager{AgentFieldHome: homeDir}
		withStdin(t, "secret\n", func() {
			output := captureOutput(t, func() {
				require.NoError(t, pcm.InteractiveConfig("demo"))
			})
			require.Contains(t, output, "Configuring environment variables")
			require.Contains(t, output, "Environment configuration saved")
		})

		envVars, err := pcm.loadEnvFile(pkgDir)
		require.NoError(t, err)
		require.Equal(t, "secret", envVars["API_KEY"])
	})

	t.Run("run config command list set unset", func(t *testing.T) {
		homeDir := t.TempDir()
		t.Setenv("AGENTFIELD_HOME", homeDir)
		pkgDir := filepath.Join(homeDir, "pkg")
		require.NoError(t, os.MkdirAll(pkgDir, 0o755))
		require.NoError(t, os.WriteFile(filepath.Join(homeDir, "installed.yaml"), []byte(`
installed:
  demo:
    name: demo
    path: `+pkgDir+`
`), 0o644))
		require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"), []byte(`
name: demo
user_environment:
  required:
    - name: API_KEY
      description: api key
`), 0o644))
		require.NoError(t, os.WriteFile(filepath.Join(pkgDir, ".env"), []byte("API_KEY=old\n"), 0o600))

		defer func(oldList bool, oldSet, oldUnset string) {
			configList, configSet, configUnset = oldList, oldSet, oldUnset
		}(configList, configSet, configUnset)

		configList, configSet, configUnset = true, "", ""
		output := captureOutput(t, func() {
			runConfigCommand(nil, []string{"demo"})
		})
		require.Contains(t, output, "Environment configuration for: demo")

		configList, configSet, configUnset = false, "API_KEY=updated", ""
		output = captureOutput(t, func() {
			runConfigCommand(nil, []string{"demo"})
		})
		require.Contains(t, output, "Set API_KEY for package demo")

		configList, configSet, configUnset = false, "", "API_KEY"
		output = captureOutput(t, func() {
			runConfigCommand(nil, []string{"demo"})
		})
		require.Contains(t, output, "Unset API_KEY for package demo")
	})
}

func TestPromptForVariableBranches(t *testing.T) {
	pcm := &PackageConfigManager{}

	tests := []struct {
		name         string
		envVar       packages.UserEnvironmentVar
		currentValue string
		input        string
		want         string
		wantErr      string
	}{
		{
			name:         "empty input keeps current value",
			envVar:       packages.UserEnvironmentVar{Name: "NAME", Description: "desc"},
			currentValue: "current",
			input:        "\n",
			want:         "current",
		},
		{
			name:         "empty input uses default",
			envVar:       packages.UserEnvironmentVar{Name: "NAME", Description: "desc", Default: "fallback"},
			currentValue: "",
			input:        "\n",
			want:         "fallback",
		},
		{
			name:         "invalid regex returns error",
			envVar:       packages.UserEnvironmentVar{Name: "NAME", Description: "desc", Validation: "["},
			currentValue: "",
			input:        "anything\n",
			wantErr:      "invalid validation pattern",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			withStdin(t, tt.input, func() {
				gotOutput := captureOutput(t, func() {
					value, err := pcm.promptForVariable(tt.envVar, tt.currentValue)
					if tt.wantErr != "" {
						require.ErrorContains(t, err, tt.wantErr)
						return
					}
					require.NoError(t, err)
					require.Equal(t, tt.want, value)
				})
				require.Contains(t, gotOutput, tt.envVar.Name)
			})
		})
	}
}

func TestMaskSecret(t *testing.T) {
	require.Equal(t, "********", maskSecret("12345678"))
	require.Equal(t, "abcd****wxyz", maskSecret("abcd1234wxyz"))
}

func captureOutput(t *testing.T, fn func()) string {
	t.Helper()
	oldStdout := os.Stdout
	r, w, err := os.Pipe()
	require.NoError(t, err)
	os.Stdout = w
	defer func() {
		os.Stdout = oldStdout
	}()

	fn()

	require.NoError(t, w.Close())
	var buf bytes.Buffer
	_, err = io.Copy(&buf, r)
	require.NoError(t, err)
	require.NoError(t, r.Close())
	return buf.String()
}

func withStdin(t *testing.T, input string, fn func()) {
	t.Helper()
	oldStdin := os.Stdin
	r, w, err := os.Pipe()
	require.NoError(t, err)
	_, err = io.WriteString(w, input)
	require.NoError(t, err)
	require.NoError(t, w.Close())
	os.Stdin = r
	defer func() {
		os.Stdin = oldStdin
		require.NoError(t, r.Close())
	}()
	fn()
}
