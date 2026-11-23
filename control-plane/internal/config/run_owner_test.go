package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestRunOwnerConfigContractA3A6A11(t *testing.T) {
	t.Setenv("AGENTFIELD_RUN_OWNER_REQUIRE_PROOF", "")
	t.Setenv("AGENTFIELD_RUN_OWNER_PROOF_KEY", "")
	path := filepath.Join(t.TempDir(), "agentfield.yaml")
	require.NoError(t, os.WriteFile(path, []byte("agentfield: {}\n"), 0600))
	cfg, err := LoadConfig(path)
	require.NoError(t, err)
	require.Equal(t, RunOwnerConfig{}, cfg.AgentField.RunOwner)

	const content = "agentfield:\n  run_owner:\n    require_proof: true\n    proof_key: '  YAML signing key  '\n"
	require.NoError(t, os.WriteFile(path, []byte(content), 0600))
	cfg, err = LoadConfig(path)
	require.NoError(t, err)
	require.Equal(t, RunOwnerConfig{RequireProof: true, ProofKey: "  YAML signing key  "}, cfg.AgentField.RunOwner)

	t.Setenv("AGENTFIELD_RUN_OWNER_REQUIRE_PROOF", "false")
	t.Setenv("AGENTFIELD_RUN_OWNER_PROOF_KEY", "  env signing key  ")
	cfg, err = LoadConfig(path)
	require.NoError(t, err)
	require.Equal(t, RunOwnerConfig{ProofKey: "  env signing key  "}, cfg.AgentField.RunOwner)

	t.Setenv("AGENTFIELD_RUN_OWNER_REQUIRE_PROOF", "true")
	ApplyEnvOverrides(cfg)
	require.True(t, cfg.AgentField.RunOwner.RequireProof)
}
