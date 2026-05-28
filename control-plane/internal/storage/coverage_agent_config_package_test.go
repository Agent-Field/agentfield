package storage

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func testAgentPackage(now time.Time) *types.AgentPackage {
	description := "demo package"
	author := "agentfield"
	repository := "https://example.com/repo.git"
	return &types.AgentPackage{
		ID:                  "pkg-demo",
		Name:                "Demo Package",
		Version:             "1.0.0",
		Description:         &description,
		Author:              &author,
		Repository:          &repository,
		InstallPath:         "/tmp/demo",
		ConfigurationSchema: json.RawMessage(`{"type":"object"}`),
		Status:              types.PackageStatusInstalled,
		ConfigurationStatus: types.ConfigurationStatusActive,
		InstalledAt:         now,
		UpdatedAt:           now,
		Metadata: types.PackageMetadata{
			Dependencies: []string{"dep-a"},
			Custom:       map[string]interface{}{"x": "y"},
		},
	}
}

func testAgentConfiguration(now time.Time) *types.AgentConfiguration {
	createdBy := "alice"
	updatedBy := "alice"
	return &types.AgentConfiguration{
		AgentID:         "agent-1",
		PackageID:       "pkg-demo",
		Configuration:   map[string]interface{}{"token": "abc", "retries": 2},
		EncryptedFields: []string{"token"},
		Status:          types.ConfigurationStatusDraft,
		Version:         1,
		CreatedAt:       now,
		UpdatedAt:       now,
		CreatedBy:       &createdBy,
		UpdatedBy:       &updatedBy,
	}
}

func TestAgentConfigurationAndPackageCoverage(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	now := time.Now().UTC()

	t.Run("configuration model converters handle success and failure", func(t *testing.T) {
		cfg := testAgentConfiguration(now)
		model, err := agentConfigurationToModel(cfg)
		require.NoError(t, err)
		require.Equal(t, cfg.AgentID, model.AgentID)

		roundTrip, err := agentConfigurationFromModel(model)
		require.NoError(t, err)
		require.Equal(t, cfg.AgentID, roundTrip.AgentID)
		require.Equal(t, cfg.EncryptedFields, roundTrip.EncryptedFields)

		badConfig := *cfg
		badConfig.Configuration = map[string]interface{}{"bad": make(chan int)}
		_, err = agentConfigurationToModel(&badConfig)
		require.ErrorContains(t, err, "failed to marshal configuration")

		_, err = agentConfigurationFromModel(&AgentConfigurationModel{
			AgentID:       "agent-1",
			PackageID:     "pkg-demo",
			Status:        string(types.ConfigurationStatusDraft),
			Configuration: []byte("{"),
		})
		require.ErrorContains(t, err, "failed to unmarshal configuration")

		_, err = agentConfigurationFromModel(&AgentConfigurationModel{
			AgentID:         "agent-1",
			PackageID:       "pkg-demo",
			Status:          string(types.ConfigurationStatusDraft),
			Configuration:   []byte(`{}`),
			EncryptedFields: []byte("{"),
		})
		require.ErrorContains(t, err, "failed to unmarshal encrypted fields")
	})

	t.Run("agent package lifecycle and queries", func(t *testing.T) {
		pkg := testAgentPackage(now)
		require.NoError(t, ls.StoreAgentPackage(ctx, pkg))

		stored, err := ls.GetAgentPackage(ctx, pkg.ID)
		require.NoError(t, err)
		require.Equal(t, pkg.Name, stored.Name)
		require.Equal(t, pkg.Metadata.Dependencies, stored.Metadata.Dependencies)

		nameLike := "Demo"
		status := types.PackageStatusInstalled
		cfgStatus := types.ConfigurationStatusActive
		pkgs, err := ls.QueryAgentPackages(ctx, types.PackageFilters{
			Name:                &nameLike,
			Status:              &status,
			ConfigurationStatus: &cfgStatus,
			Limit:               1,
		})
		require.NoError(t, err)
		require.Len(t, pkgs, 1)

		pkg.Version = "1.1.0"
		pkg.UpdatedAt = now.Add(time.Minute)
		pkg.Status = types.PackageStatusRunning
		require.NoError(t, ls.UpdateAgentPackage(ctx, pkg))

		updated, err := ls.GetAgentPackage(ctx, pkg.ID)
		require.NoError(t, err)
		require.Equal(t, "1.1.0", updated.Version)
		require.Equal(t, types.PackageStatusRunning, updated.Status)

		require.NoError(t, ls.DeleteAgentPackage(ctx, pkg.ID))
		_, err = ls.GetAgentPackage(ctx, pkg.ID)
		require.EqualError(t, err, "package with ID 'pkg-demo' not found")
		require.EqualError(t, ls.DeleteAgentPackage(ctx, pkg.ID), "package with ID 'pkg-demo' not found")
		require.EqualError(t, ls.UpdateAgentPackage(ctx, pkg), "package with ID 'pkg-demo' not found")

		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		require.EqualError(t, ls.StoreAgentPackage(cancelled, pkg), "context cancelled during store agent package: context canceled")
		_, err = ls.GetAgentPackage(cancelled, "pkg-demo")
		require.EqualError(t, err, "context cancelled during get agent package: context canceled")
		_, err = ls.QueryAgentPackages(cancelled, types.PackageFilters{})
		require.EqualError(t, err, "context cancelled during query agent packages: context canceled")
		require.EqualError(t, ls.UpdateAgentPackage(cancelled, pkg), "context canceled")
		require.EqualError(t, ls.DeleteAgentPackage(cancelled, pkg.ID), "context canceled")
	})

	t.Run("agent configuration lifecycle validation and queries", func(t *testing.T) {
		pkg := testAgentPackage(now)
		require.NoError(t, ls.StoreAgentPackage(ctx, pkg))

		cfg := testAgentConfiguration(now)
		err := ls.StoreAgentConfiguration(ctx, cfg)
		require.ErrorContains(t, err, "failed to store agent configuration")

		gormDB, err := ls.gormWithContext(ctx)
		require.NoError(t, err)
		model, err := agentConfigurationToModel(cfg)
		require.NoError(t, err)
		require.NoError(t, gormDB.Create(model).Error)
		cfg.ID = model.ID

		stored, err := ls.GetAgentConfiguration(ctx, cfg.AgentID, cfg.PackageID)
		require.NoError(t, err)
		require.Equal(t, cfg.AgentID, stored.AgentID)
		require.Equal(t, "abc", stored.Configuration["token"])

		status := types.ConfigurationStatusDraft
		createdBy := "alice"
		configs, err := ls.QueryAgentConfigurations(ctx, types.ConfigurationFilters{
			AgentID:   &cfg.AgentID,
			PackageID: &cfg.PackageID,
			Status:    &status,
			CreatedBy: &createdBy,
			Limit:     1,
		})
		require.NoError(t, err)
		require.Len(t, configs, 1)

		updatedBy := "bob"
		cfg.Configuration["retries"] = 3
		cfg.UpdatedBy = &updatedBy
		cfg.UpdatedAt = now.Add(time.Minute)
		cfg.Status = types.ConfigurationStatusActive
		require.NoError(t, ls.UpdateAgentConfiguration(ctx, cfg))

		updated, err := ls.GetAgentConfiguration(ctx, cfg.AgentID, cfg.PackageID)
		require.NoError(t, err)
		require.Equal(t, float64(3), updated.Configuration["retries"])
		require.Equal(t, types.ConfigurationStatusActive, updated.Status)

		result, err := ls.ValidateAgentConfiguration(ctx, cfg.AgentID, cfg.PackageID, map[string]interface{}{"token": "abc"})
		require.NoError(t, err)
		require.True(t, result.Valid)
		require.Empty(t, result.Errors)

		require.NoError(t, ls.DeleteAgentConfiguration(ctx, cfg.AgentID, cfg.PackageID))
		_, err = ls.GetAgentConfiguration(ctx, cfg.AgentID, cfg.PackageID)
		require.EqualError(t, err, "configuration for agent 'agent-1' and package 'pkg-demo' not found")
		require.EqualError(t, ls.DeleteAgentConfiguration(ctx, cfg.AgentID, cfg.PackageID), "configuration for agent 'agent-1' and package 'pkg-demo' not found")
		require.EqualError(t, ls.UpdateAgentConfiguration(ctx, cfg), "configuration for agent 'agent-1' and package 'pkg-demo' not found")

		missingPkgResult, err := ls.ValidateAgentConfiguration(ctx, cfg.AgentID, "missing", map[string]interface{}{})
		require.NoError(t, err)
		require.False(t, missingPkgResult.Valid)
		require.NotEmpty(t, missingPkgResult.Errors)

		badPkg := testAgentPackage(now)
		badPkg.ID = "pkg-bad-schema"
		badPkg.ConfigurationSchema = json.RawMessage("{")
		require.NoError(t, ls.StoreAgentPackage(ctx, badPkg))
		invalidSchemaResult, err := ls.ValidateAgentConfiguration(ctx, cfg.AgentID, badPkg.ID, map[string]interface{}{})
		require.NoError(t, err)
		require.False(t, invalidSchemaResult.Valid)
		require.NotEmpty(t, invalidSchemaResult.Errors)

		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		require.EqualError(t, ls.StoreAgentConfiguration(cancelled, cfg), "context cancelled during store agent configuration: context canceled")
		_, err = ls.GetAgentConfiguration(cancelled, cfg.AgentID, cfg.PackageID)
		require.EqualError(t, err, "context canceled")
		_, err = ls.QueryAgentConfigurations(cancelled, types.ConfigurationFilters{})
		require.EqualError(t, err, "context cancelled during query agent configurations: context canceled")
		require.EqualError(t, ls.UpdateAgentConfiguration(cancelled, cfg), "context canceled")
		require.EqualError(t, ls.DeleteAgentConfiguration(cancelled, cfg.AgentID, cfg.PackageID), "context canceled")
		_, err = ls.ValidateAgentConfiguration(cancelled, cfg.AgentID, cfg.PackageID, nil)
		require.EqualError(t, err, "context cancelled during validate agent configuration: context canceled")
	})
}
