package harness

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type authContractCase struct {
	Name    string             `json:"name"`
	Allow   bool               `json:"allow"`
	Env     map[string]string  `json:"env"`
	Overlay map[string]string  `json:"overlay"`
	Files   map[string]*string `json:"files"`
}

type authContract struct {
	Refusal string             `json:"refusal"`
	Warning string             `json:"warning"`
	Cases   []authContractCase `json:"cases"`
}

type authNoDownload struct{ t *testing.T }

func (transport authNoDownload) RoundTrip(*http.Request) (*http.Response, error) {
	transport.t.Error("download attempted")
	return nil, errors.New("downloads forbidden by K1-K7 tests")
}

func readAuthContract(t *testing.T) authContract {
	t.Helper()
	data, err := os.ReadFile("../../codeaf-auth-contract.json")
	require.NoError(t, err)
	var contract authContract
	require.NoError(t, json.Unmarshal(data, &contract))
	return contract
}

func isolateCodeAFAuth(t *testing.T) string {
	t.Helper()
	isolateCodeAF(t)
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
	codeAFMissingKeyOnce = sync.Once{}
	previous := http.DefaultTransport
	http.DefaultTransport = authNoDownload{t}
	t.Cleanup(func() { http.DefaultTransport = previous })
	return directory
}

func authFakeBinary(t *testing.T, directory string) (string, string) {
	t.Helper()
	binary, marker := filepath.Join(directory, "fake-codeaf"), filepath.Join(directory, "ran")
	require.NoError(t, os.WriteFile(binary, []byte(fmt.Sprintf("#!/bin/sh\n/bin/cat >/dev/null\nprintf ran > '%s'\nprintf '%%s\\n' '{\"ok\":true,\"stop\":\"done\",\"answer\":\"answer\"}'\n", marker)), 0o755))
	return binary, marker
}

func assertAuthRefusal(t *testing.T, err error, contract authContract) {
	t.Helper()
	var unavailable *ProviderUnavailableError
	require.ErrorAs(t, err, &unavailable)
	assert.ErrorIs(t, err, ErrProviderUnavailable)
	assert.Equal(t, contract.Refusal, err.Error())
	assert.Equal(t, ProviderCodeAF, unavailable.Provider)
	assert.True(t, unavailable.NotConfigured)
	assert.Equal(t, []string{"OPENROUTER_API_KEY"}, unavailable.MissingAuthEnv)
	assert.Equal(t, "af codeaf ensure", unavailable.InstallCommand)
}

func TestCodeAFW4K1K3K4K7SharedKeyLadder(t *testing.T) {
	contract := readAuthContract(t)
	for _, tc := range contract.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			directory := isolateCodeAFAuth(t)
			logs := captureCodeAFLogs(t)
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
			binary, marker := authFakeBinary(t, directory)
			p := NewCodeAFProvider(binary)
			// Any attempt at automatic resolution (and therefore installation) fails.
			p.resolveBinary = func(ctx context.Context, override string) (codeAFBinary, error) {
				require.Equal(t, binary, override, "automatic resolution/installation attempted")
				return resolveCodeAF(ctx, override)
			}
			overlay := map[string]string{}
			for key, value := range tc.Overlay {
				overlay[key] = expand(value)
			}
			result, err := p.Execute(context.Background(), "task", Options{Env: overlay})
			if tc.Allow {
				require.NoError(t, err)
				require.NotNil(t, result)
				assert.Equal(t, "answer", result.Result)
				assert.FileExists(t, marker)
				assert.NotContains(t, logs.String(), contract.Warning)
			} else {
				assertAuthRefusal(t, err, contract)
				var unavailable *ProviderUnavailableError
				require.True(t, errors.As(err, &unavailable))
				assert.Equal(t, binary, unavailable.Binary)
				assert.NoFileExists(t, marker)
				assert.Equal(t, 1, strings.Count(logs.String(), contract.Warning))
			}
		})
	}
}

func TestCodeAFW2K1K2BeforeResolution(t *testing.T) {
	contract := readAuthContract(t)
	for _, provider := range []string{"", "codeaf", "aforge"} {
		for _, pin := range []string{"automatic", "env", "option"} {
			t.Run(provider+"/"+pin, func(t *testing.T) {
				directory := isolateCodeAFAuth(t)
				logs := captureCodeAFLogs(t)
				binary, marker := authFakeBinary(t, directory)
				if pin == "env" {
					t.Setenv("CODEAF_BIN", binary)
				}
				built, err := BuildProvider(provider, "")
				require.NoError(t, err)
				p := built.(*CodeAFProvider)
				p.resolveBinary = func(context.Context, string) (codeAFBinary, error) {
					t.Fatal("binary resolution/installation attempted")
					return codeAFBinary{}, nil
				}
				opts := Options{}
				if pin == "option" {
					opts.BinPath = binary
				}
				for call := 0; call < 3; call++ {
					_, err := p.Execute(context.Background(), "task", opts)
					assertAuthRefusal(t, err, contract)
				}
				assert.NoFileExists(t, marker)
				assert.Equal(t, 1, strings.Count(logs.String(), contract.Warning))
			})
		}
	}
}

func TestCodeAFK5LegacyPinWithoutKey(t *testing.T) {
	for _, pin := range []string{"env", "option"} {
		t.Run(pin, func(t *testing.T) {
			directory := isolateCodeAFAuth(t)
			logs := captureCodeAFLogs(t)
			binary, marker := authFakeBinary(t, directory)
			override := binary
			if pin == "env" {
				override = ""
				t.Setenv("AFORGE_BIN", binary)
			}
			p, err := BuildProvider("aforge", override)
			require.NoError(t, err)
			_, err = p.Execute(context.Background(), "task", Options{})
			require.NoError(t, err)
			assert.FileExists(t, marker)
			assert.NotContains(t, logs.String(), readAuthContract(t).Warning)
		})
	}
}
