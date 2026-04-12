package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestEffectiveNodeLogProxy(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		in   NodeLogProxyConfig
		want NodeLogProxyConfig
	}{
		{
			name: "applies defaults for zero values",
			in:   NodeLogProxyConfig{},
			want: NodeLogProxyConfig{
				ConnectTimeout:    5 * time.Second,
				StreamIdleTimeout: 60 * time.Second,
				MaxStreamDuration: 15 * time.Minute,
				MaxTailLines:      10000,
			},
		},
		{
			name: "preserves explicit values",
			in: NodeLogProxyConfig{
				ConnectTimeout:    2 * time.Second,
				StreamIdleTimeout: 3 * time.Second,
				MaxStreamDuration: 4 * time.Minute,
				MaxTailLines:      55,
			},
			want: NodeLogProxyConfig{
				ConnectTimeout:    2 * time.Second,
				StreamIdleTimeout: 3 * time.Second,
				MaxStreamDuration: 4 * time.Minute,
				MaxTailLines:      55,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := EffectiveNodeLogProxy(tt.in); got != tt.want {
				t.Fatalf("unexpected proxy config: got %+v want %+v", got, tt.want)
			}
		})
	}
}

func TestEffectiveExecutionLogs(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		in   ExecutionLogsConfig
		want ExecutionLogsConfig
	}{
		{
			name: "applies defaults for zero values",
			in:   ExecutionLogsConfig{},
			want: ExecutionLogsConfig{
				RetentionPeriod:        24 * time.Hour,
				MaxEntriesPerExecution: 5000,
				MaxTailEntries:         1000,
				StreamIdleTimeout:      60 * time.Second,
				MaxStreamDuration:      15 * time.Minute,
			},
		},
		{
			name: "preserves explicit values",
			in: ExecutionLogsConfig{
				RetentionPeriod:        12 * time.Hour,
				MaxEntriesPerExecution: 7,
				MaxTailEntries:         8,
				StreamIdleTimeout:      9 * time.Second,
				MaxStreamDuration:      10 * time.Minute,
			},
			want: ExecutionLogsConfig{
				RetentionPeriod:        12 * time.Hour,
				MaxEntriesPerExecution: 7,
				MaxTailEntries:         8,
				StreamIdleTimeout:      9 * time.Second,
				MaxStreamDuration:      10 * time.Minute,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := EffectiveExecutionLogs(tt.in); got != tt.want {
				t.Fatalf("unexpected execution log config: got %+v want %+v", got, tt.want)
			}
		})
	}
}

func TestLoadConfig(t *testing.T) {
	t.Run("loads explicit config path", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "custom.yaml")
		if err := os.WriteFile(path, []byte("agentfield:\n  port: 7777\n"), 0o644); err != nil {
			t.Fatalf("write config: %v", err)
		}

		cfg, err := LoadConfig(path)
		if err != nil {
			t.Fatalf("LoadConfig returned error: %v", err)
		}
		if cfg.AgentField.Port != 7777 {
			t.Fatalf("expected port 7777, got %d", cfg.AgentField.Port)
		}
	})

	t.Run("falls back to config directory default path", func(t *testing.T) {
		dir := t.TempDir()
		t.Chdir(dir)
		if err := os.Mkdir("config", 0o755); err != nil {
			t.Fatalf("mkdir config: %v", err)
		}
		if err := os.WriteFile(filepath.Join("config", "agentfield.yaml"), []byte("agentfield:\n  port: 8081\n"), 0o644); err != nil {
			t.Fatalf("write fallback config: %v", err)
		}
		t.Setenv("AGENTFIELD_API_KEY", "from-env")

		cfg, err := LoadConfig("")
		if err != nil {
			t.Fatalf("LoadConfig returned error: %v", err)
		}
		if cfg.AgentField.Port != 8081 {
			t.Fatalf("expected port 8081, got %d", cfg.AgentField.Port)
		}
		if cfg.API.Auth.APIKey != "from-env" {
			t.Fatalf("expected env override to be applied, got %q", cfg.API.Auth.APIKey)
		}
	})

	t.Run("errors when config file is missing", func(t *testing.T) {
		_, err := LoadConfig(filepath.Join(t.TempDir(), "missing.yaml"))
		if err == nil || !strings.Contains(err.Error(), "configuration file not found") {
			t.Fatalf("expected missing file error, got %v", err)
		}
	})

	t.Run("errors when path is unreadable as file", func(t *testing.T) {
		dir := t.TempDir()
		_, err := LoadConfig(dir)
		if err == nil || !strings.Contains(err.Error(), "failed to read configuration file") {
			t.Fatalf("expected read error, got %v", err)
		}
	})

	t.Run("errors on malformed yaml", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "broken.yaml")
		if err := os.WriteFile(path, []byte("agentfield: [\n"), 0o644); err != nil {
			t.Fatalf("write broken config: %v", err)
		}

		_, err := LoadConfig(path)
		if err == nil || !strings.Contains(err.Error(), "failed to parse configuration file") {
			t.Fatalf("expected parse error, got %v", err)
		}
	})
}

func TestApplyEnvOverrides(t *testing.T) {
	cfg := &Config{
		API: APIConfig{
			Auth: AuthConfig{APIKey: "initial"},
		},
		Features: FeatureConfig{
			DID: DIDConfig{
				Authorization: AuthorizationConfig{
					Domain:        "initial-domain",
					AdminToken:    "initial-admin",
					InternalToken: "initial-internal",
				},
			},
			Tracing: TracingConfig{
				Endpoint:    "initial-endpoint",
				ServiceName: "initial-service",
			},
		},
		AgentField: AgentFieldConfig{
			Registration: RegistrationConfig{
				ServerlessDiscoveryAllowedHosts: []string{"keep-me"},
			},
			LLMHealth: LLMHealthConfig{
				Endpoints: []LLMEndpoint{{Name: "existing", URL: "http://existing"}},
			},
			NodeHealth: NodeHealthConfig{
				CheckInterval:           time.Second,
				CheckTimeout:            time.Second,
				ConsecutiveFailures:     1,
				RecoveryDebounce:        time.Second,
				HeartbeatStaleThreshold: time.Second,
			},
			ExecutionQueue: ExecutionQueueConfig{MaxConcurrentPerAgent: 1},
			ExecutionCleanup: ExecutionCleanupConfig{
				MaxRetries:   1,
				RetryBackoff: time.Second,
			},
			NodeLogProxy: NodeLogProxyConfig{
				ConnectTimeout:    time.Second,
				StreamIdleTimeout: time.Second,
				MaxStreamDuration: time.Minute,
				MaxTailLines:      1,
			},
			ExecutionLogs: ExecutionLogsConfig{
				RetentionPeriod:        time.Hour,
				MaxEntriesPerExecution: 1,
				MaxTailEntries:         1,
				StreamIdleTimeout:      time.Second,
				MaxStreamDuration:      time.Minute,
			},
			Approval: ApprovalConfig{
				WebhookSecret:      "initial-secret",
				DefaultExpiryHours: 1,
			},
		},
	}

	env := map[string]string{
		"AGENTFIELD_API_KEY":                                         "legacy-key",
		"AGENTFIELD_API_AUTH_API_KEY":                                "nested-key",
		"AGENTFIELD_REGISTRATION_SERVERLESS_DISCOVERY_ALLOWED_HOSTS": " a.example.com, ,b.example.com ",
		"AGENTFIELD_HEALTH_CHECK_INTERVAL":                           "45s",
		"AGENTFIELD_HEALTH_CHECK_TIMEOUT":                            "7s",
		"AGENTFIELD_HEALTH_CONSECUTIVE_FAILURES":                     "6",
		"AGENTFIELD_HEALTH_RECOVERY_DEBOUNCE":                        "9s",
		"AGENTFIELD_HEARTBEAT_STALE_THRESHOLD":                       "11s",
		"AGENTFIELD_LLM_HEALTH_ENABLED":                              "1",
		"AGENTFIELD_LLM_HEALTH_CHECK_INTERVAL":                       "12s",
		"AGENTFIELD_LLM_HEALTH_CHECK_TIMEOUT":                        "13s",
		"AGENTFIELD_LLM_HEALTH_FAILURE_THRESHOLD":                    "14",
		"AGENTFIELD_LLM_HEALTH_RECOVERY_TIMEOUT":                     "15s",
		"AGENTFIELD_LLM_HEALTH_ENDPOINT":                             "http://llm.local/health",
		"AGENTFIELD_LLM_HEALTH_ENDPOINT_NAME":                        "litellm",
		"AGENTFIELD_MAX_CONCURRENT_PER_AGENT":                        "17",
		"AGENTFIELD_EXECUTION_MAX_RETRIES":                           "18",
		"AGENTFIELD_EXECUTION_RETRY_BACKOFF":                         "19s",
		"AGENTFIELD_AUTHORIZATION_ENABLED":                           "true",
		"AGENTFIELD_AUTHORIZATION_DID_AUTH_ENABLED":                  "1",
		"AGENTFIELD_AUTHORIZATION_DOMAIN":                            "auth.local",
		"AGENTFIELD_AUTHORIZATION_ADMIN_TOKEN":                       "admin-token",
		"AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN":                    "internal-token",
		"AGENTFIELD_NODE_LOG_PROXY_CONNECT_TIMEOUT":                  "21s",
		"AGENTFIELD_NODE_LOG_PROXY_STREAM_IDLE_TIMEOUT":              "22s",
		"AGENTFIELD_NODE_LOG_PROXY_MAX_DURATION":                     "23s",
		"AGENTFIELD_NODE_LOG_MAX_TAIL_LINES":                         "24",
		"AGENTFIELD_EXECUTION_LOG_RETENTION_PERIOD":                  "25h",
		"AGENTFIELD_EXECUTION_LOG_MAX_ENTRIES_PER_EXECUTION":         "26",
		"AGENTFIELD_EXECUTION_LOG_MAX_TAIL_ENTRIES":                  "27",
		"AGENTFIELD_EXECUTION_LOG_STREAM_IDLE_TIMEOUT":               "28s",
		"AGENTFIELD_EXECUTION_LOG_MAX_DURATION":                      "29s",
		"AGENTFIELD_APPROVAL_WEBHOOK_SECRET":                         "webhook-secret",
		"AGENTFIELD_APPROVAL_DEFAULT_EXPIRY_HOURS":                   "30",
		"AGENTFIELD_TRACING_ENABLED":                                 "1",
		"OTEL_EXPORTER_OTLP_ENDPOINT":                                "http://otel.local:4318",
		"OTEL_SERVICE_NAME":                                          "control-plane",
		"AGENTFIELD_TRACING_INSECURE":                                "true",
		"AGENTFIELD_CONNECTOR_ENABLED":                               "1",
		"AGENTFIELD_CONNECTOR_TOKEN":                                 "connector-token",
		"AGENTFIELD_CONNECTOR_CAP_POLICY_MANAGEMENT":                 "true",
		"AGENTFIELD_CONNECTOR_CAP_TAG_MANAGEMENT":                    "readonly",
		"AGENTFIELD_CONNECTOR_CAP_DID_MANAGEMENT":                    "false",
	}
	for k, v := range env {
		t.Setenv(k, v)
	}

	ApplyEnvOverrides(cfg)

	if cfg.API.Auth.APIKey != "nested-key" {
		t.Fatalf("expected nested api key to win, got %q", cfg.API.Auth.APIKey)
	}
	if got := cfg.AgentField.Registration.ServerlessDiscoveryAllowedHosts; len(got) != 2 || got[0] != "a.example.com" || got[1] != "b.example.com" {
		t.Fatalf("unexpected allowed hosts: %#v", got)
	}
	if cfg.AgentField.NodeHealth.CheckInterval != 45*time.Second ||
		cfg.AgentField.NodeHealth.CheckTimeout != 7*time.Second ||
		cfg.AgentField.NodeHealth.ConsecutiveFailures != 6 ||
		cfg.AgentField.NodeHealth.RecoveryDebounce != 9*time.Second ||
		cfg.AgentField.NodeHealth.HeartbeatStaleThreshold != 11*time.Second {
		t.Fatalf("unexpected node health overrides: %+v", cfg.AgentField.NodeHealth)
	}
	if !cfg.AgentField.LLMHealth.Enabled ||
		cfg.AgentField.LLMHealth.CheckInterval != 12*time.Second ||
		cfg.AgentField.LLMHealth.CheckTimeout != 13*time.Second ||
		cfg.AgentField.LLMHealth.FailureThreshold != 14 ||
		cfg.AgentField.LLMHealth.RecoveryTimeout != 15*time.Second {
		t.Fatalf("unexpected llm health overrides: %+v", cfg.AgentField.LLMHealth)
	}
	if got := cfg.AgentField.LLMHealth.Endpoints; len(got) != 2 || got[1].Name != "litellm" || got[1].URL != "http://llm.local/health" {
		t.Fatalf("unexpected llm endpoints: %#v", got)
	}
	if cfg.AgentField.ExecutionQueue.MaxConcurrentPerAgent != 17 {
		t.Fatalf("expected max concurrent override, got %d", cfg.AgentField.ExecutionQueue.MaxConcurrentPerAgent)
	}
	if cfg.AgentField.ExecutionCleanup.MaxRetries != 18 || cfg.AgentField.ExecutionCleanup.RetryBackoff != 19*time.Second {
		t.Fatalf("unexpected execution cleanup overrides: %+v", cfg.AgentField.ExecutionCleanup)
	}
	if !cfg.Features.DID.Authorization.Enabled ||
		!cfg.Features.DID.Authorization.DIDAuthEnabled ||
		cfg.Features.DID.Authorization.Domain != "auth.local" ||
		cfg.Features.DID.Authorization.AdminToken != "admin-token" ||
		cfg.Features.DID.Authorization.InternalToken != "internal-token" {
		t.Fatalf("unexpected authorization overrides: %+v", cfg.Features.DID.Authorization)
	}
	if cfg.AgentField.NodeLogProxy.ConnectTimeout != 21*time.Second ||
		cfg.AgentField.NodeLogProxy.StreamIdleTimeout != 22*time.Second ||
		cfg.AgentField.NodeLogProxy.MaxStreamDuration != 23*time.Second ||
		cfg.AgentField.NodeLogProxy.MaxTailLines != 24 {
		t.Fatalf("unexpected node log proxy overrides: %+v", cfg.AgentField.NodeLogProxy)
	}
	if cfg.AgentField.ExecutionLogs.RetentionPeriod != 25*time.Hour ||
		cfg.AgentField.ExecutionLogs.MaxEntriesPerExecution != 26 ||
		cfg.AgentField.ExecutionLogs.MaxTailEntries != 27 ||
		cfg.AgentField.ExecutionLogs.StreamIdleTimeout != 28*time.Second ||
		cfg.AgentField.ExecutionLogs.MaxStreamDuration != 29*time.Second {
		t.Fatalf("unexpected execution log overrides: %+v", cfg.AgentField.ExecutionLogs)
	}
	if cfg.AgentField.Approval.WebhookSecret != "webhook-secret" || cfg.AgentField.Approval.DefaultExpiryHours != 30 {
		t.Fatalf("unexpected approval overrides: %+v", cfg.AgentField.Approval)
	}
	if !cfg.Features.Tracing.Enabled ||
		cfg.Features.Tracing.Endpoint != "http://otel.local:4318" ||
		cfg.Features.Tracing.ServiceName != "control-plane" ||
		!cfg.Features.Tracing.Insecure {
		t.Fatalf("unexpected tracing overrides: %+v", cfg.Features.Tracing)
	}
	if !cfg.Features.Connector.Enabled || cfg.Features.Connector.Token != "connector-token" {
		t.Fatalf("unexpected connector overrides: %+v", cfg.Features.Connector)
	}
	if got := cfg.Features.Connector.Capabilities["policy_management"]; !got.Enabled || got.ReadOnly {
		t.Fatalf("unexpected policy_management capability: %+v", got)
	}
	if got := cfg.Features.Connector.Capabilities["tag_management"]; !got.Enabled || !got.ReadOnly {
		t.Fatalf("unexpected tag_management capability: %+v", got)
	}
	if got := cfg.Features.Connector.Capabilities["did_management"]; got.Enabled {
		t.Fatalf("unexpected did_management capability: %+v", got)
	}
}

func TestApplyEnvOverridesIgnoresInvalidValues(t *testing.T) {
	cfg := &Config{
		AgentField: AgentFieldConfig{
			NodeHealth: NodeHealthConfig{
				CheckInterval:       3 * time.Second,
				CheckTimeout:        4 * time.Second,
				ConsecutiveFailures: 5,
			},
			ExecutionQueue:   ExecutionQueueConfig{MaxConcurrentPerAgent: 6},
			ExecutionCleanup: ExecutionCleanupConfig{MaxRetries: 7, RetryBackoff: 8 * time.Second},
			NodeLogProxy: NodeLogProxyConfig{
				ConnectTimeout: 9 * time.Second,
				MaxTailLines:   10,
			},
			ExecutionLogs: ExecutionLogsConfig{
				RetentionPeriod:        11 * time.Hour,
				MaxEntriesPerExecution: 12,
			},
			Approval: ApprovalConfig{DefaultExpiryHours: 13},
		},
		Features: FeatureConfig{
			Tracing:   TracingConfig{Insecure: true},
			Connector: ConnectorConfig{Enabled: true},
		},
	}

	for k, v := range map[string]string{
		"AGENTFIELD_HEALTH_CHECK_INTERVAL":                   "not-a-duration",
		"AGENTFIELD_HEALTH_CHECK_TIMEOUT":                    "still-bad",
		"AGENTFIELD_HEALTH_CONSECUTIVE_FAILURES":             "bad-int",
		"AGENTFIELD_LLM_HEALTH_CHECK_INTERVAL":               "bad",
		"AGENTFIELD_LLM_HEALTH_FAILURE_THRESHOLD":            "bad",
		"AGENTFIELD_MAX_CONCURRENT_PER_AGENT":                "bad",
		"AGENTFIELD_EXECUTION_MAX_RETRIES":                   "bad",
		"AGENTFIELD_EXECUTION_RETRY_BACKOFF":                 "bad",
		"AGENTFIELD_NODE_LOG_PROXY_CONNECT_TIMEOUT":          "bad",
		"AGENTFIELD_NODE_LOG_MAX_TAIL_LINES":                 "bad",
		"AGENTFIELD_EXECUTION_LOG_RETENTION_PERIOD":          "bad",
		"AGENTFIELD_EXECUTION_LOG_MAX_ENTRIES_PER_EXECUTION": "bad",
		"AGENTFIELD_APPROVAL_DEFAULT_EXPIRY_HOURS":           "bad",
		"AGENTFIELD_TRACING_INSECURE":                        "0",
		"AGENTFIELD_CONNECTOR_ENABLED":                       "false",
	} {
		t.Setenv(k, v)
	}

	ApplyEnvOverrides(cfg)

	if cfg.AgentField.NodeHealth.CheckInterval != 3*time.Second ||
		cfg.AgentField.NodeHealth.CheckTimeout != 4*time.Second ||
		cfg.AgentField.NodeHealth.ConsecutiveFailures != 5 {
		t.Fatalf("invalid node health values should not override: %+v", cfg.AgentField.NodeHealth)
	}
	if cfg.AgentField.ExecutionQueue.MaxConcurrentPerAgent != 6 {
		t.Fatalf("invalid max concurrent override applied: %+v", cfg.AgentField.ExecutionQueue)
	}
	if cfg.AgentField.ExecutionCleanup.MaxRetries != 7 || cfg.AgentField.ExecutionCleanup.RetryBackoff != 8*time.Second {
		t.Fatalf("invalid execution cleanup override applied: %+v", cfg.AgentField.ExecutionCleanup)
	}
	if cfg.AgentField.NodeLogProxy.ConnectTimeout != 9*time.Second || cfg.AgentField.NodeLogProxy.MaxTailLines != 10 {
		t.Fatalf("invalid node log proxy override applied: %+v", cfg.AgentField.NodeLogProxy)
	}
	if cfg.AgentField.ExecutionLogs.RetentionPeriod != 11*time.Hour || cfg.AgentField.ExecutionLogs.MaxEntriesPerExecution != 12 {
		t.Fatalf("invalid execution logs override applied: %+v", cfg.AgentField.ExecutionLogs)
	}
	if cfg.AgentField.Approval.DefaultExpiryHours != 13 {
		t.Fatalf("invalid approval override applied: %+v", cfg.AgentField.Approval)
	}
	if cfg.Features.Tracing.Insecure {
		t.Fatalf("expected insecure to become false for non-true value")
	}
	if cfg.Features.Connector.Enabled {
		t.Fatalf("expected connector enabled to become false for non-true value")
	}
}
