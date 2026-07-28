package config

import (
	"testing"
)

// TestMCPConfig_IsEnabledDefault covers the IsEnabled default (nil => true) and
// the explicit true/false cases. The embedded MCP server ships on by default.
func TestMCPConfig_IsEnabledDefault(t *testing.T) {
	var nilCfg MCPConfig
	if !nilCfg.IsEnabled() {
		t.Error("IsEnabled() with nil Enabled must default to true")
	}

	tru := true
	if !(MCPConfig{Enabled: &tru}).IsEnabled() {
		t.Error("IsEnabled() with *true must be true")
	}

	fls := false
	if (MCPConfig{Enabled: &fls}).IsEnabled() {
		t.Error("IsEnabled() with *false must be false")
	}
}

// TestApplyEnvOverrides_MCPEnabled covers the AGENTFIELD_MCP_ENABLED override:
// a falsey value flips the toggle to explicit false, a truthy value to explicit
// true, and an unset var leaves it nil (default-on).
func TestApplyEnvOverrides_MCPEnabled(t *testing.T) {
	t.Run("disable", func(t *testing.T) {
		cfg := &Config{}
		t.Setenv("AGENTFIELD_MCP_ENABLED", "false")
		ApplyEnvOverrides(cfg)
		if cfg.Features.MCP.Enabled == nil || *cfg.Features.MCP.Enabled {
			t.Errorf("Enabled = %v, want explicit false", cfg.Features.MCP.Enabled)
		}
		if cfg.Features.MCP.IsEnabled() {
			t.Error("IsEnabled() must be false after AGENTFIELD_MCP_ENABLED=false")
		}
	})

	t.Run("enable", func(t *testing.T) {
		cfg := &Config{}
		t.Setenv("AGENTFIELD_MCP_ENABLED", "1")
		ApplyEnvOverrides(cfg)
		if cfg.Features.MCP.Enabled == nil || !*cfg.Features.MCP.Enabled {
			t.Errorf("Enabled = %v, want explicit true", cfg.Features.MCP.Enabled)
		}
	})

	t.Run("unset defaults on", func(t *testing.T) {
		cfg := &Config{}
		ApplyEnvOverrides(cfg)
		if cfg.Features.MCP.Enabled != nil {
			t.Errorf("Enabled = %v, want nil (default-on) when env unset", cfg.Features.MCP.Enabled)
		}
		if !cfg.Features.MCP.IsEnabled() {
			t.Error("IsEnabled() must default to true when env unset")
		}
	})
}
