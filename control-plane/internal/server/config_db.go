package server

import (
	"context"
	"fmt"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"gopkg.in/yaml.v3"
)

const dbConfigKey = "agentfield.yaml"

// overlayDBConfig loads config from the database and merges it into the
// existing config. The storage section is preserved from the original config
// to avoid the bootstrap problem (DB connection settings can't come from DB).
// Precedence: env vars > DB config > file config > defaults.
func overlayDBConfig(cfg *config.Config, store storage.StorageProvider) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	entry, err := store.GetConfig(ctx, dbConfigKey)
	if err != nil {
		return fmt.Errorf("failed to read config from database: %w", err)
	}
	if entry == nil {
		fmt.Println("[config] No database config found (key: agentfield.yaml), using file/env config only.")
		return nil
	}

	// Preserve the storage config — it must always come from file/env (bootstrap)
	savedStorage := cfg.Storage

	// Parse the DB-stored YAML into a config struct
	var dbCfg config.Config
	if err := yaml.Unmarshal([]byte(entry.Value), &dbCfg); err != nil {
		return fmt.Errorf("failed to parse database config YAML: %w", err)
	}

	// Overlay non-zero DB values onto the existing config
	mergeDBConfig(cfg, &dbCfg)

	// Restore storage config (never overridden from DB)
	cfg.Storage = savedStorage

	// The command loaders apply environment variables before server construction,
	// while the database overlay necessarily happens during construction. Reapply
	// them here so the documented env > DB precedence remains true, including on
	// runtime config reloads.
	config.ApplyEnvOverrides(cfg)

	fmt.Printf("[config] Loaded config from database (key: %s, version: %d, updated: %s)\n",
		entry.Key, entry.Version, entry.UpdatedAt.Format(time.RFC3339))
	return nil
}

// mergeDBConfig selectively merges DB config values into the target config.
// Only non-zero/non-empty values from the DB config are applied.
func mergeDBConfig(target, dbCfg *config.Config) {
	// AgentField settings
	if dbCfg.AgentField.Port != 0 {
		target.AgentField.Port = dbCfg.AgentField.Port
	}
	mergeDBNodeHealthConfig(&target.AgentField.NodeHealth, dbCfg.AgentField.NodeHealth)
	// ARD exposure is intentionally not merged from DB config. File/env config
	// defines the deployment guardrails; runtime opt-in state lives in ard.state.
	// Merge execution cleanup field-by-field to avoid zeroing out unset fields
	if dbCfg.AgentField.ExecutionCleanup.RetentionPeriod != 0 {
		target.AgentField.ExecutionCleanup.RetentionPeriod = dbCfg.AgentField.ExecutionCleanup.RetentionPeriod
	}
	if dbCfg.AgentField.ExecutionCleanup.CleanupInterval != 0 {
		target.AgentField.ExecutionCleanup.CleanupInterval = dbCfg.AgentField.ExecutionCleanup.CleanupInterval
	}
	if dbCfg.AgentField.ExecutionCleanup.BatchSize != 0 {
		target.AgentField.ExecutionCleanup.BatchSize = dbCfg.AgentField.ExecutionCleanup.BatchSize
	}
	if dbCfg.AgentField.ExecutionCleanup.PreserveRecentDuration != 0 {
		target.AgentField.ExecutionCleanup.PreserveRecentDuration = dbCfg.AgentField.ExecutionCleanup.PreserveRecentDuration
	}
	if dbCfg.AgentField.ExecutionCleanup.StaleExecutionTimeout != 0 {
		target.AgentField.ExecutionCleanup.StaleExecutionTimeout = dbCfg.AgentField.ExecutionCleanup.StaleExecutionTimeout
	}
	// Enabled is a bool — only override if cleanup config is present in DB at all
	if dbCfg.AgentField.ExecutionCleanup.RetentionPeriod != 0 || dbCfg.AgentField.ExecutionCleanup.CleanupInterval != 0 {
		target.AgentField.ExecutionCleanup.Enabled = dbCfg.AgentField.ExecutionCleanup.Enabled
	}
	if dbCfg.AgentField.Approval.WebhookSecret != "" || dbCfg.AgentField.Approval.DefaultExpiryHours != 0 {
		target.AgentField.Approval = dbCfg.AgentField.Approval
	}
	if dbCfg.AgentField.NodeLogProxy.ConnectTimeout != 0 {
		target.AgentField.NodeLogProxy.ConnectTimeout = dbCfg.AgentField.NodeLogProxy.ConnectTimeout
	}
	if dbCfg.AgentField.NodeLogProxy.StreamIdleTimeout != 0 {
		target.AgentField.NodeLogProxy.StreamIdleTimeout = dbCfg.AgentField.NodeLogProxy.StreamIdleTimeout
	}
	if dbCfg.AgentField.NodeLogProxy.MaxStreamDuration != 0 {
		target.AgentField.NodeLogProxy.MaxStreamDuration = dbCfg.AgentField.NodeLogProxy.MaxStreamDuration
	}
	if dbCfg.AgentField.NodeLogProxy.MaxTailLines != 0 {
		target.AgentField.NodeLogProxy.MaxTailLines = dbCfg.AgentField.NodeLogProxy.MaxTailLines
	}
	if dbCfg.AgentField.ExecutionLogs.RetentionPeriod != 0 {
		target.AgentField.ExecutionLogs.RetentionPeriod = dbCfg.AgentField.ExecutionLogs.RetentionPeriod
	}
	if dbCfg.AgentField.ExecutionLogs.MaxEntriesPerExecution != 0 {
		target.AgentField.ExecutionLogs.MaxEntriesPerExecution = dbCfg.AgentField.ExecutionLogs.MaxEntriesPerExecution
	}
	if dbCfg.AgentField.ExecutionLogs.MaxTailEntries != 0 {
		target.AgentField.ExecutionLogs.MaxTailEntries = dbCfg.AgentField.ExecutionLogs.MaxTailEntries
	}
	if dbCfg.AgentField.ExecutionLogs.StreamIdleTimeout != 0 {
		target.AgentField.ExecutionLogs.StreamIdleTimeout = dbCfg.AgentField.ExecutionLogs.StreamIdleTimeout
	}
	if dbCfg.AgentField.ExecutionLogs.MaxStreamDuration != 0 {
		target.AgentField.ExecutionLogs.MaxStreamDuration = dbCfg.AgentField.ExecutionLogs.MaxStreamDuration
	}

	// Features
	if dbCfg.Features.DID.Method != "" {
		target.Features.DID = dbCfg.Features.DID
	}
	// MCP's default is enabled, so the pointer distinguishes an omitted setting
	// from an explicit database-backed `features.mcp.enabled: false` kill switch.
	if dbCfg.Features.MCP.Enabled != nil {
		target.Features.MCP.Enabled = dbCfg.Features.MCP.Enabled
	}
	// NOTE: Connector config (token, capabilities) is intentionally NOT merged
	// from DB. These are security-sensitive and must come from file/env config,
	// similar to how storage config is protected from the bootstrap problem.

	// API settings (but never override API key from DB for security)
	if len(dbCfg.API.CORS.AllowedOrigins) > 0 {
		target.API.CORS = dbCfg.API.CORS
	}

	// UI settings
	if dbCfg.UI.Mode != "" {
		target.UI = dbCfg.UI
	}
}

func mergeDBNodeHealthConfig(target *config.NodeHealthConfig, dbCfg config.NodeHealthConfig) {
	if dbCfg.CheckInterval != 0 {
		target.CheckInterval = dbCfg.CheckInterval
	}
	if dbCfg.CheckTimeout != 0 {
		target.CheckTimeout = dbCfg.CheckTimeout
	}
	if dbCfg.ConsecutiveFailures != 0 {
		target.ConsecutiveFailures = dbCfg.ConsecutiveFailures
	}
	if dbCfg.RecoveryDebounce != 0 {
		target.RecoveryDebounce = dbCfg.RecoveryDebounce
	}
	if dbCfg.HeartbeatStaleThreshold != 0 {
		target.HeartbeatStaleThreshold = dbCfg.HeartbeatStaleThreshold
	}
	if dbCfg.AgentRestartGrace != 0 {
		target.AgentRestartGrace = dbCfg.AgentRestartGrace
	}
	if dbCfg.AgentDrainGrace != 0 {
		target.AgentDrainGrace = dbCfg.AgentDrainGrace
	}
	if dbCfg.AgentOrphanReapEnabled != nil {
		enabled := *dbCfg.AgentOrphanReapEnabled
		target.AgentOrphanReapEnabled = &enabled
	}
}
