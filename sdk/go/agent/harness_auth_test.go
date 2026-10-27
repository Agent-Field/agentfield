package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/sdk/go/harness"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type codeAFAgentAuthContract struct {
	Warning string `json:"warning"`
	Refusal string `json:"refusal"`
	Cases   []struct {
		Name    string             `json:"name"`
		Allow   bool               `json:"allow"`
		Env     map[string]string  `json:"env"`
		Overlay map[string]string  `json:"overlay"`
		Files   map[string]*string `json:"files"`
	} `json:"cases"`
}

func readCodeAFAgentAuthContract(t *testing.T) codeAFAgentAuthContract {
	t.Helper()
	data, err := os.ReadFile("../../codeaf-auth-contract.json")
	require.NoError(t, err)
	var contract codeAFAgentAuthContract
	require.NoError(t, json.Unmarshal(data, &contract))
	return contract
}

// A fresh process gives each Agent test an unspent harness warning budget
// without exporting a production API solely to reset sync.Once in tests.
func runCodeAFAuthInFreshProcess(t *testing.T) bool {
	t.Helper()
	if os.Getenv("AGENTFIELD_TEST_CODEAF_AUTH_PROCESS") == t.Name() {
		return false
	}
	command := exec.Command(os.Args[0], "-test.run=^"+regexp.QuoteMeta(t.Name())+"$", "-test.count=1")
	command.Env = append(os.Environ(), "AGENTFIELD_TEST_CODEAF_AUTH_PROCESS="+t.Name())
	output, err := command.CombinedOutput()
	require.NoError(t, err, "%s", output)
	return true
}

type codeAFAgentNoDownload struct{ t *testing.T }

func (transport codeAFAgentNoDownload) RoundTrip(*http.Request) (*http.Response, error) {
	transport.t.Error("download attempted")
	return nil, errors.New("downloads forbidden by CodeAF auth tests")
}

func isolateCodeAFAgentAuth(t *testing.T) (string, *bytes.Buffer) {
	t.Helper()
	for _, entry := range os.Environ() {
		key, _, _ := strings.Cut(entry, "=")
		if strings.HasPrefix(key, "CODEAF_") || strings.HasPrefix(key, "AFORGE_") || strings.HasPrefix(key, "AGENTFIELD_") || key == "OPENROUTER_API_KEY" || key == "OPENAI_API_KEY" {
			t.Setenv(key, "")
			require.NoError(t, os.Unsetenv(key))
		}
	}
	directory := t.TempDir()
	t.Setenv("HOME", filepath.Join(directory, "home"))
	t.Setenv("PATH", directory)
	t.Setenv("AGENTFIELD_HOME", filepath.Join(directory, "agentfield"))
	t.Setenv("CODEAF_TELEMETRY", "off")
	var output bytes.Buffer
	previousWriter, previousTransport := log.Writer(), http.DefaultTransport
	log.SetOutput(&output)
	http.DefaultTransport = codeAFAgentNoDownload{t}
	t.Cleanup(func() {
		log.SetOutput(previousWriter)
		http.DefaultTransport = previousTransport
	})
	return directory, &output
}

func TestCodeAFK6ConstructionWarning(t *testing.T) {
	contract := readCodeAFAgentAuthContract(t)
	for _, tc := range []struct {
		name, provider, envProvider, key string
		configured, legacy, warn         bool
	}{
		{name: "default", configured: true, warn: true},
		{name: "explicit", provider: "codeaf", configured: true, warn: true},
		{name: "alias", provider: "aforge", configured: true, warn: true},
		{name: "key", provider: "codeaf", key: "key", configured: true},
		{name: "other", provider: "claude-code", configured: true},
		{name: "no-config"},
		{name: "legacy", provider: "aforge", configured: true, legacy: true},
		{name: "env-codeaf", envProvider: "codeaf", configured: true, warn: true},
		{name: "env-other", envProvider: "claude-code", configured: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if runCodeAFAuthInFreshProcess(t) {
				return
			}
			_, output := isolateCodeAFAgentAuth(t)
			t.Setenv("OPENROUTER_API_KEY", tc.key)
			t.Setenv("AGENTFIELD_HARNESS_PROVIDER", tc.envProvider)
			if tc.legacy {
				t.Setenv("AFORGE_BIN", "/fake/legacy")
			}
			cfg := Config{NodeID: "auth-test", Version: "1", Logger: log.New(output, "", 0)}
			if tc.configured {
				cfg.HarnessConfig = &HarnessConfig{Provider: tc.provider}
			}
			for i := 0; i < 2; i++ {
				_, err := New(cfg)
				require.NoError(t, err)
			}
			expected := 0
			if tc.warn {
				expected = 1
			}
			assert.Equal(t, expected, strings.Count(output.String(), contract.Warning))
		})
	}
}

func codeAFAgentFakeBinary(t *testing.T, directory string) (string, string) {
	t.Helper()
	binary, marker := filepath.Join(directory, "fake-codeaf"), filepath.Join(directory, "ran")
	require.NoError(t, os.WriteFile(binary, []byte(fmt.Sprintf("#!/bin/sh\n/bin/cat >/dev/null\nprintf ran > '%s'\nprintf '%%s\\n' '{\"ok\":true,\"stop\":\"done\",\"answer\":\"answer\"}'\n", marker)), 0o755))
	return binary, marker
}

func TestCodeAFW1SharedProcessWarningBudget(t *testing.T) {
	contract := readCodeAFAgentAuthContract(t)
	for _, tc := range []struct {
		name                 string
		constructions, calls int
		configured           bool
		key                  string
	}{
		{name: "construction-then-calls", constructions: 1, calls: 2, configured: true},
		{name: "two-agents", constructions: 2, calls: 1, configured: true},
		{name: "first-refusal", constructions: 1, calls: 2},
		{name: "call-clears-key", constructions: 1, calls: 1, configured: true, key: "key"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if runCodeAFAuthInFreshProcess(t) {
				return
			}
			directory, output := isolateCodeAFAgentAuth(t)
			t.Setenv("OPENROUTER_API_KEY", tc.key)
			binary, marker := codeAFAgentFakeBinary(t, directory)
			cfg := Config{NodeID: "auth-test", Version: "1", Logger: log.New(output, "", 0)}
			if tc.configured {
				cfg.HarnessConfig = &HarnessConfig{Provider: harness.ProviderCodeAF, BinPath: binary}
			}
			var agent *Agent
			for i := 0; i < tc.constructions; i++ {
				var err error
				agent, err = New(cfg)
				require.NoError(t, err)
			}
			expected := 0
			if tc.configured && tc.key == "" {
				expected = 1
			}
			assert.Equal(t, expected, strings.Count(output.String(), contract.Warning))
			for call := 0; call < tc.calls; call++ {
				_, err := agent.Harness(context.Background(), "task", nil, nil, harness.Options{Env: map[string]string{"OPENROUTER_API_KEY": ""}})
				var unavailable *harness.ProviderUnavailableError
				require.ErrorAs(t, err, &unavailable)
				assert.ErrorIs(t, err, harness.ErrProviderUnavailable)
				assert.True(t, unavailable.NotConfigured)
				assert.Equal(t, contract.Refusal, err.Error())
				assert.Equal(t, harness.ProviderCodeAF, unavailable.Provider)
				assert.Equal(t, []string{"OPENROUTER_API_KEY"}, unavailable.MissingAuthEnv)
				assert.Equal(t, "af codeaf ensure", unavailable.InstallCommand)
				assert.Equal(t, 1, strings.Count(output.String(), contract.Warning))
			}
			cfg.HarnessConfig = &HarnessConfig{Env: map[string]string{"OPENROUTER_API_KEY": ""}}
			_, err := New(cfg)
			require.NoError(t, err)
			assert.Equal(t, 1, strings.Count(output.String(), contract.Warning))
			assert.NoFileExists(t, marker)
		})
	}
}

func TestCodeAFW3ConstructionUsesConfigEnvironment(t *testing.T) {
	contract := readCodeAFAgentAuthContract(t)
	for _, tc := range contract.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			if runCodeAFAuthInFreshProcess(t) {
				return
			}
			directory, output := isolateCodeAFAgentAuth(t)
			expand := strings.NewReplacer("$STATE", filepath.Join(directory, "agentfield", "codeaf"), "$PROFILE", filepath.Join(directory, "profile"), "$HOME", filepath.Join(directory, "home")).Replace
			for key, value := range tc.Env {
				t.Setenv(key, expand(value))
			}
			for file, contents := range tc.Files {
				target := expand(file)
				require.NoError(t, os.MkdirAll(filepath.Dir(target), 0o755))
				if contents == nil {
					require.NoError(t, os.Mkdir(target, 0o755))
				} else {
					require.NoError(t, os.WriteFile(target, []byte(*contents), 0o600))
				}
			}
			overlay := map[string]string{}
			for key, value := range tc.Overlay {
				overlay[key] = expand(value)
			}
			_, err := New(Config{
				NodeID: "auth-test", Version: "1", Logger: log.New(output, "", 0),
				HarnessConfig: &HarnessConfig{Env: overlay},
			})
			require.NoError(t, err)
			expected := 0
			if !tc.Allow {
				expected = 1
			}
			assert.Equal(t, expected, strings.Count(output.String(), contract.Warning))
		})
	}
}

func TestCodeAFW3KeyOnlyInConfigEnvLaunchesStub(t *testing.T) {
	if runCodeAFAuthInFreshProcess(t) {
		return
	}
	contract := readCodeAFAgentAuthContract(t)
	directory, output := isolateCodeAFAgentAuth(t)
	binary, marker := codeAFAgentFakeBinary(t, directory)
	agent, err := New(Config{
		NodeID: "auth-test", Version: "1", Logger: log.New(output, "", 0),
		HarnessConfig: &HarnessConfig{Provider: harness.ProviderCodeAF, BinPath: binary, Env: map[string]string{"OPENROUTER_API_KEY": "config-key"}},
	})
	require.NoError(t, err)
	assert.NotContains(t, output.String(), contract.Warning)
	result, err := agent.Harness(context.Background(), "task", nil, nil, harness.Options{})
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, "answer", result.Result)
	assert.FileExists(t, marker)
	assert.NotContains(t, output.String(), contract.Warning)
}
