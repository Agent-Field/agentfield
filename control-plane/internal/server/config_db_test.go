package server

import (
	"context"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

type configStoreStub struct {
	storage.StorageProvider
	entry *storage.ConfigEntry
	err   error
}

func (s *configStoreStub) GetConfig(_ context.Context, key string) (*storage.ConfigEntry, error) {
	if key != dbConfigKey {
		return nil, nil
	}
	if s.err != nil {
		return nil, s.err
	}
	return s.entry, nil
}

func baseConfigForDBTests() config.Config {
	return config.Config{
		AgentField: config.AgentFieldConfig{
			Port: 8080,
			NodeHealth: config.NodeHealthConfig{
				CheckInterval: 10 * time.Second,
				CheckTimeout:  5 * time.Second,
			},
			ExecutionCleanup: config.ExecutionCleanupConfig{
				Enabled:                true,
				RetentionPeriod:        24 * time.Hour,
				CleanupInterval:        time.Hour,
				BatchSize:              100,
				PreserveRecentDuration: 30 * time.Minute,
				StaleExecutionTimeout:  15 * time.Minute,
			},
			Approval: config.ApprovalConfig{
				WebhookSecret:      "file-secret",
				DefaultExpiryHours: 24,
			},
			NodeLogProxy: config.NodeLogProxyConfig{
				ConnectTimeout:    2 * time.Second,
				StreamIdleTimeout: time.Minute,
				MaxStreamDuration: 10 * time.Minute,
				MaxTailLines:      500,
			},
			ExecutionLogs: config.ExecutionLogsConfig{
				RetentionPeriod:        12 * time.Hour,
				MaxEntriesPerExecution: 1000,
				MaxTailEntries:         100,
				StreamIdleTimeout:      45 * time.Second,
				MaxStreamDuration:      5 * time.Minute,
			},
		},
		Features: config.FeatureConfig{
			DID: config.DIDConfig{
				Method: "did:key",
			},
			Connector: config.ConnectorConfig{
				Enabled: true,
				Token:   "file-token",
				Capabilities: map[string]config.ConnectorCapability{
					"calendar": {Enabled: true, ReadOnly: true},
				},
			},
		},
		Storage: config.StorageConfig{
			Mode: "local",
			Local: storage.LocalStorageConfig{
				DatabasePath: "file.db",
				KVStorePath:  "file.bolt",
			},
		},
		UI: config.UIConfig{
			Enabled:  true,
			Mode:     "embedded",
			DistPath: "ui/dist",
			DevPort:  3000,
		},
		API: config.APIConfig{
			CORS: config.CORSConfig{
				AllowedOrigins: []string{"https://file.example"},
			},
			Auth: config.AuthConfig{
				APIKey: "file-api-key",
			},
		},
	}
}

func TestMergeDBConfigPreservesStorageSection(t *testing.T) {
	cfg := baseConfigForDBTests()
	originalStorage := cfg.Storage

	dbCfg := &config.Config{
		Storage: config.StorageConfig{
			Mode: "postgres",
			Postgres: storage.PostgresStorageConfig{
				Host: "db.internal",
				Port: 5432,
			},
		},
		AgentField: config.AgentFieldConfig{Port: 9090},
	}

	mergeDBConfig(&cfg, dbCfg)

	require.Equal(t, 9090, cfg.AgentField.Port)
	require.Equal(t, originalStorage, cfg.Storage)
}

func TestMergeDBConfigAppliesNonZeroDBValues(t *testing.T) {
	cfg := baseConfigForDBTests()

	dbCfg := &config.Config{
		AgentField: config.AgentFieldConfig{
			Port: 9090,
			NodeHealth: config.NodeHealthConfig{
				CheckInterval: 15 * time.Second,
				CheckTimeout:  7 * time.Second,
			},
			ExecutionCleanup: config.ExecutionCleanupConfig{
				Enabled:                false,
				RetentionPeriod:        48 * time.Hour,
				CleanupInterval:        2 * time.Hour,
				BatchSize:              200,
				PreserveRecentDuration: time.Hour,
				StaleExecutionTimeout:  30 * time.Minute,
			},
			Approval: config.ApprovalConfig{
				WebhookSecret:      "db-secret",
				DefaultExpiryHours: 72,
			},
			NodeLogProxy: config.NodeLogProxyConfig{
				ConnectTimeout:    4 * time.Second,
				StreamIdleTimeout: 90 * time.Second,
				MaxStreamDuration: 20 * time.Minute,
				MaxTailLines:      900,
			},
			ExecutionLogs: config.ExecutionLogsConfig{
				RetentionPeriod:        72 * time.Hour,
				MaxEntriesPerExecution: 3000,
				MaxTailEntries:         400,
				StreamIdleTimeout:      90 * time.Second,
				MaxStreamDuration:      20 * time.Minute,
			},
		},
		Features: config.FeatureConfig{
			DID: config.DIDConfig{Method: "did:web"},
		},
		UI: config.UIConfig{
			Enabled:  false,
			Mode:     "dev",
			DistPath: "db/dist",
			DevPort:  5173,
		},
		API: config.APIConfig{
			CORS: config.CORSConfig{
				AllowedOrigins: []string{"https://db.example"},
			},
		},
	}

	mergeDBConfig(&cfg, dbCfg)

	require.Equal(t, 9090, cfg.AgentField.Port)
	require.Equal(t, dbCfg.AgentField.NodeHealth, cfg.AgentField.NodeHealth)
	require.Equal(t, dbCfg.AgentField.ExecutionCleanup, cfg.AgentField.ExecutionCleanup)
	require.Equal(t, dbCfg.AgentField.Approval, cfg.AgentField.Approval)
	require.Equal(t, dbCfg.AgentField.NodeLogProxy, cfg.AgentField.NodeLogProxy)
	require.Equal(t, dbCfg.AgentField.ExecutionLogs, cfg.AgentField.ExecutionLogs)
	require.Equal(t, "did:web", cfg.Features.DID.Method)
	require.Equal(t, dbCfg.UI, cfg.UI)
	require.Equal(t, []string{"https://db.example"}, cfg.API.CORS.AllowedOrigins)
	require.Equal(t, "file-api-key", cfg.API.Auth.APIKey)
}

func TestOverlayDBConfigMissingEntryReturnsGracefully(t *testing.T) {
	cfg := baseConfigForDBTests()
	original := cfg

	err := overlayDBConfig(&cfg, &configStoreStub{})
	require.NoError(t, err)
	require.Equal(t, original, cfg)
}

func TestOverlayDBConfigInvalidYAMLDoesNotMutateLoadedConfig(t *testing.T) {
	cfg := baseConfigForDBTests()
	original := cfg

	err := overlayDBConfig(&cfg, &configStoreStub{
		entry: &storage.ConfigEntry{
			Key:       dbConfigKey,
			Value:     "agentfield: [invalid",
			Version:   2,
			UpdatedAt: time.Date(2026, 4, 7, 12, 0, 0, 0, time.UTC),
		},
	})

	require.Error(t, err)
	require.Contains(t, err.Error(), "failed to parse database config YAML")
	require.Equal(t, original, cfg)
}

func TestOverlayDBConfigRoundTripPreservesStorageAndMergesExpected(t *testing.T) {
	cfg := baseConfigForDBTests()

	dbCfg := config.Config{
		AgentField: config.AgentFieldConfig{
			Port: 7070,
			NodeHealth: config.NodeHealthConfig{
				CheckInterval: 20 * time.Second,
				CheckTimeout:  9 * time.Second,
			},
			ExecutionCleanup: config.ExecutionCleanupConfig{
				Enabled:                false,
				RetentionPeriod:        96 * time.Hour,
				CleanupInterval:        3 * time.Hour,
				BatchSize:              500,
				PreserveRecentDuration: 2 * time.Hour,
				StaleExecutionTimeout:  45 * time.Minute,
			},
			Approval: config.ApprovalConfig{
				WebhookSecret:      "db-webhook-secret",
				DefaultExpiryHours: 96,
			},
			NodeLogProxy: config.NodeLogProxyConfig{
				ConnectTimeout:    5 * time.Second,
				StreamIdleTimeout: 75 * time.Second,
				MaxStreamDuration: 25 * time.Minute,
				MaxTailLines:      1200,
			},
			ExecutionLogs: config.ExecutionLogsConfig{
				RetentionPeriod:        168 * time.Hour,
				MaxEntriesPerExecution: 6000,
				MaxTailEntries:         600,
				StreamIdleTimeout:      2 * time.Minute,
				MaxStreamDuration:      30 * time.Minute,
			},
		},
		Features: config.FeatureConfig{
			DID: config.DIDConfig{Method: "did:web"},
			Connector: config.ConnectorConfig{
				Enabled: true,
				Token:   "db-should-not-win",
			},
		},
		Storage: config.StorageConfig{
			Mode: "postgres",
			Postgres: storage.PostgresStorageConfig{
				Host: "db.internal",
				Port: 5432,
			},
		},
		UI: config.UIConfig{
			Enabled:  false,
			Mode:     "separate",
			DistPath: "db-ui",
			DevPort:  4173,
		},
		API: config.APIConfig{
			CORS: config.CORSConfig{
				AllowedOrigins: []string{"https://db.example", "https://console.example"},
			},
		},
	}

	payload, err := yaml.Marshal(dbCfg)
	require.NoError(t, err)

	err = overlayDBConfig(&cfg, &configStoreStub{
		entry: &storage.ConfigEntry{
			Key:       dbConfigKey,
			Value:     string(payload),
			Version:   7,
			UpdatedAt: time.Date(2026, 4, 7, 12, 0, 0, 0, time.UTC),
		},
	})
	require.NoError(t, err)

	expected := baseConfigForDBTests()
	expected.AgentField = dbCfg.AgentField
	expected.Features.DID = dbCfg.Features.DID
	expected.UI = dbCfg.UI
	expected.API.CORS = dbCfg.API.CORS

	// Compare via YAML round-trip so nil-vs-empty-slice differences from
	// unmarshalling do not produce false negatives.
	expectedYAML, err := yaml.Marshal(expected)
	require.NoError(t, err)
	actualYAML, err := yaml.Marshal(cfg)
	require.NoError(t, err)
	require.Equal(t, string(expectedYAML), string(actualYAML))
}
