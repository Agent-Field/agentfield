package agent

import (
	"bytes"
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCodeAFK6ConstructionWarning(t *testing.T) {
	data, err := os.ReadFile("../../codeaf-auth-contract.json")
	require.NoError(t, err)
	var contract struct {
		Warning string `json:"warning"`
	}
	require.NoError(t, json.Unmarshal(data, &contract))
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
			for _, entry := range os.Environ() {
				key, _, _ := strings.Cut(entry, "=")
				if strings.HasPrefix(key, "CODEAF_") || strings.HasPrefix(key, "AFORGE_") || strings.HasPrefix(key, "AGENTFIELD_") || key == "OPENROUTER_API_KEY" || key == "OPENAI_API_KEY" {
					t.Setenv(key, "")
					require.NoError(t, os.Unsetenv(key))
				}
			}
			directory := t.TempDir()
			t.Setenv("HOME", filepath.Join(directory, "home"))
			t.Setenv("AGENTFIELD_HOME", filepath.Join(directory, "agentfield"))
			t.Setenv("CODEAF_TELEMETRY", "off")
			t.Setenv("OPENROUTER_API_KEY", tc.key)
			t.Setenv("AGENTFIELD_HARNESS_PROVIDER", tc.envProvider)
			if tc.legacy {
				t.Setenv("AFORGE_BIN", "/fake/legacy")
			}
			codeAFConfigWarning = sync.Once{}
			var output bytes.Buffer
			cfg := Config{NodeID: "auth-test", Version: "1", Logger: log.New(&output, "", 0)}
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
