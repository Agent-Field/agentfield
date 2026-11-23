package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestServerHostYAMLAndEnvironment(t *testing.T) {
	for _, tc := range []struct {
		name, yamlHost, envHost, want string
		setEnv                        bool
	}{
		{name: "yaml", yamlHost: "127.0.0.1", want: "127.0.0.1"},
		{name: "environment", yamlHost: "127.0.0.2", envHost: "127.0.0.1", want: "127.0.0.1", setEnv: true},
		{name: "explicit wildcard", yamlHost: "127.0.0.1", setEnv: true},
		{name: "default wildcard"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			previous, set := os.LookupEnv("AGENTFIELD_HOST")
			require.NoError(t, os.Unsetenv("AGENTFIELD_HOST"))
			t.Cleanup(func() {
				if set {
					_ = os.Setenv("AGENTFIELD_HOST", previous)
				} else {
					_ = os.Unsetenv("AGENTFIELD_HOST")
				}
			})
			if tc.setEnv {
				t.Setenv("AGENTFIELD_HOST", tc.envHost)
			}
			path := filepath.Join(t.TempDir(), "agentfield.yaml")
			require.NoError(t, os.WriteFile(path, []byte("agentfield:\n  host: "+tc.yamlHost+"\n"), 0o600))
			cfg, err := LoadConfig(path)
			require.NoError(t, err)
			encoded, err := yaml.Marshal(cfg)
			require.NoError(t, err)
			var value map[string]map[string]any
			require.NoError(t, yaml.Unmarshal(encoded, &value))
			require.Equal(t, tc.want, value["agentfield"]["host"])
		})
	}
}
