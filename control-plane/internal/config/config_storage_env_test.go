package config

import (
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/stretchr/testify/require"
)

func TestApplyEnvOverrides_StorageLocal(t *testing.T) {
	t.Setenv("AGENTFIELD_STORAGE_LOCAL_DATABASE_PATH", " /data/agentfield.db ")
	t.Setenv("AGENTFIELD_STORAGE_LOCAL_KV_STORE_PATH", "/data/agentfield.bolt")

	cfg := &Config{}
	ApplyEnvOverrides(cfg)

	require.Equal(t, "/data/agentfield.db", cfg.Storage.Local.DatabasePath)
	require.Equal(t, "/data/agentfield.bolt", cfg.Storage.Local.KVStorePath)
}

func TestApplyEnvOverrides_StorageLocalUnsetKeepsConfig(t *testing.T) {
	t.Setenv("AGENTFIELD_STORAGE_LOCAL_DATABASE_PATH", "")
	t.Setenv("AGENTFIELD_STORAGE_LOCAL_KV_STORE_PATH", "")

	cfg := &Config{}
	cfg.Storage.Local = storage.LocalStorageConfig{DatabasePath: "/from-file.db", KVStorePath: "/from-file.bolt"}
	ApplyEnvOverrides(cfg)

	require.Equal(t, "/from-file.db", cfg.Storage.Local.DatabasePath)
	require.Equal(t, "/from-file.bolt", cfg.Storage.Local.KVStorePath)
}
