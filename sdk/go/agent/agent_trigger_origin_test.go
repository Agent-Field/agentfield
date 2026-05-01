package agent

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/sdk/go/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestWithEventTrigger_StampsCallerFileAndLine verifies that WithEventTrigger
// captures the file and line where the option is invoked.
func TestWithEventTrigger_StampsCallerFileAndLine(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	agent.RegisterReasoner("paymentHandler",
		func(ctx context.Context, input map[string]any) (any, error) {
			return nil, nil
		},
		WithEventTrigger("stripe", "payment_intent.succeeded"),
	)

	reasoner := agent.reasoners["paymentHandler"]
	require.NotNil(t, reasoner)
	require.Len(t, reasoner.Triggers, 1)

	binding := reasoner.Triggers[0]
	assert.Equal(t, "stripe", binding.Source)
	assert.NotEmpty(t, binding.CodeOrigin, "CodeOrigin should be stamped")

	// Verify file and line in CodeOrigin
	assert.Contains(t, binding.CodeOrigin, "agent_trigger_origin_test.go")
	assert.Contains(t, binding.CodeOrigin, ":")

	// Parse and validate line number is reasonable
	parts := strings.Split(binding.CodeOrigin, ":")
	assert.Len(t, parts, 2, "CodeOrigin should be in format file:line")
}

// TestWithScheduleTrigger_StampsCallerFileAndLine verifies that WithScheduleTrigger
// captures the file and line where the option is invoked.
func TestWithScheduleTrigger_StampsCallerFileAndLine(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	agent.RegisterReasoner("dailyTask",
		func(ctx context.Context, input map[string]any) (any, error) {
			return nil, nil
		},
		WithScheduleTrigger("0 2 * * *"),
	)

	reasoner := agent.reasoners["dailyTask"]
	require.NotNil(t, reasoner)
	require.Len(t, reasoner.Triggers, 1)

	binding := reasoner.Triggers[0]
	assert.Equal(t, "cron", binding.Source)
	assert.NotEmpty(t, binding.CodeOrigin, "CodeOrigin should be stamped")

	// Verify file in CodeOrigin
	assert.Contains(t, binding.CodeOrigin, "agent_trigger_origin_test.go")
	assert.Contains(t, binding.CodeOrigin, ":")
}

// TestWithTriggers_PreservesUserSuppliedOrigin verifies that WithTriggers does NOT
// overwrite a user-supplied CodeOrigin value when passed via EventTrigger.
func TestWithTriggers_PreservesUserSuppliedOrigin(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	// Build a custom EventTrigger using triggerToBinding's conversion,
	// then test that WithTriggers preserves the explicit CodeOrigin when set manually
	agent.RegisterReasoner("customTriggered",
		func(ctx context.Context, input map[string]any) (any, error) {
			return nil, nil
		},
		WithTriggers(EventTrigger{Source: "stripe", Types: []string{"charge.succeeded"}}),
	)

	reasoner := agent.reasoners["customTriggered"]
	require.NotNil(t, reasoner)
	require.Len(t, reasoner.Triggers, 1)

	binding := reasoner.Triggers[0]
	// WithTriggers should stamp a CodeOrigin when the EventTrigger doesn't have one
	assert.NotEmpty(t, binding.CodeOrigin, "CodeOrigin should be stamped")
	assert.Contains(t, binding.CodeOrigin, "agent_trigger_origin_test.go")
}

// TestWithTriggers_MultipleTypes verifies that WithTriggers correctly handles
// multiple trigger types and stamps each with CodeOrigin.
func TestWithTriggers_MultipleTypes(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	agent.RegisterReasoner("multiSource",
		func(ctx context.Context, input map[string]any) (any, error) {
			return nil, nil
		},
		WithTriggers(
			EventTrigger{Source: "stripe", Types: []string{"charge.succeeded"}},
			ScheduleTrigger{Cron: "0 * * * *"},
		),
	)

	reasoner := agent.reasoners["multiSource"]
	require.NotNil(t, reasoner)
	require.Len(t, reasoner.Triggers, 2)

	// Both bindings should have CodeOrigin
	for i, binding := range reasoner.Triggers {
		assert.NotEmpty(t, binding.CodeOrigin, "Binding %d should have CodeOrigin", i)
		assert.Contains(t, binding.CodeOrigin, "agent_trigger_origin_test.go")
	}
}

// TestWithEventTrigger_CodeOriginJSON verifies that CodeOrigin is included in
// the JSON wire format when set.
func TestWithEventTrigger_CodeOriginJSON(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	agent.RegisterReasoner("jsonTest",
		func(ctx context.Context, input map[string]any) (any, error) {
			return nil, nil
		},
		WithEventTrigger("stripe", "charge.succeeded"),
	)

	reasoner := agent.reasoners["jsonTest"]
	require.NotNil(t, reasoner)
	require.Len(t, reasoner.Triggers, 1)

	binding := reasoner.Triggers[0]

	// Marshal to JSON and verify CodeOrigin is present
	jsonData, err := json.Marshal(binding)
	require.NoError(t, err)

	var unmarshaled types.TriggerBinding
	err = json.Unmarshal(jsonData, &unmarshaled)
	require.NoError(t, err)

	assert.NotEmpty(t, unmarshaled.CodeOrigin)
	assert.Contains(t, unmarshaled.CodeOrigin, "agent_trigger_origin_test.go")
}

// TestWithEventTrigger_CodeOriginOmittedWhenEmpty verifies that CodeOrigin is
// omitted from JSON when it is empty (omitempty tag).
func TestWithEventTrigger_CodeOriginOmittedWhenEmpty(t *testing.T) {
	// Create a binding with empty CodeOrigin
	binding := types.TriggerBinding{
		Source:     "webhook",
		CodeOrigin: "",
	}

	jsonData, err := json.Marshal(binding)
	require.NoError(t, err)

	// CodeOrigin should not appear in JSON due to omitempty tag
	assert.NotContains(t, string(jsonData), "code_origin")

	// Unmarshal and verify
	var unmarshaled types.TriggerBinding
	err = json.Unmarshal(jsonData, &unmarshaled)
	require.NoError(t, err)

	assert.Empty(t, unmarshaled.CodeOrigin)
}

// TestMultipleTriggers_AllStamped verifies that when multiple triggers are declared
// via separate options, each gets its own CodeOrigin at the call site.
func TestMultipleTriggers_AllStamped(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	agent.RegisterReasoner("multiTrigger",
		func(ctx context.Context, input map[string]any) (any, error) {
			return nil, nil
		},
		WithEventTrigger("stripe", "payment.success"),
		WithEventTrigger("github", "push"),
		WithScheduleTrigger("0 * * * *"),
	)

	reasoner := agent.reasoners["multiTrigger"]
	require.NotNil(t, reasoner)
	require.Len(t, reasoner.Triggers, 3)

	// Each should have a CodeOrigin pointing to this test file
	for i, binding := range reasoner.Triggers {
		assert.NotEmpty(t, binding.CodeOrigin, "Trigger %d should have CodeOrigin", i)
		assert.Contains(t, binding.CodeOrigin, "agent_trigger_origin_test.go")
	}
}

// TestWithTriggerSecretEnv_DoesNotAffectCodeOrigin verifies that WithTriggerSecretEnv
// does not clear or modify the CodeOrigin of the last trigger.
func TestWithTriggerSecretEnv_DoesNotAffectCodeOrigin(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	agent.RegisterReasoner("secretTest",
		func(ctx context.Context, input map[string]any) (any, error) {
			return nil, nil
		},
		WithEventTrigger("stripe", "charge.succeeded"),
		WithTriggerSecretEnv("STRIPE_SECRET"),
	)

	reasoner := agent.reasoners["secretTest"]
	require.NotNil(t, reasoner)
	require.Len(t, reasoner.Triggers, 1)

	binding := reasoner.Triggers[0]
	assert.NotEmpty(t, binding.CodeOrigin, "CodeOrigin should persist after WithTriggerSecretEnv")
	assert.Equal(t, "STRIPE_SECRET", binding.SecretEnvVar)
	assert.Contains(t, binding.CodeOrigin, "agent_trigger_origin_test.go")
}

// TestWithTriggerConfig_DoesNotAffectCodeOrigin verifies that WithTriggerConfig
// does not clear or modify the CodeOrigin of the last trigger.
func TestWithTriggerConfig_DoesNotAffectCodeOrigin(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	agent.RegisterReasoner("configTest",
		func(ctx context.Context, input map[string]any) (any, error) {
			return nil, nil
		},
		WithEventTrigger("webhook", "event"),
		WithTriggerConfig(map[string]any{"path": "/webhook"}),
	)

	reasoner := agent.reasoners["configTest"]
	require.NotNil(t, reasoner)
	require.Len(t, reasoner.Triggers, 1)

	binding := reasoner.Triggers[0]
	assert.NotEmpty(t, binding.CodeOrigin, "CodeOrigin should persist after WithTriggerConfig")
	assert.NotEmpty(t, binding.Config, "Config should be set")
	assert.Contains(t, binding.CodeOrigin, "agent_trigger_origin_test.go")
}

func TestTriggerOptions_EdgeBranches(t *testing.T) {
	t.Run("captureCodeOrigin returns empty when stack frame is unavailable", func(t *testing.T) {
		assert.Empty(t, captureCodeOrigin(100000))
	})

	t.Run("WithTriggers ignores unknown trigger values", func(t *testing.T) {
		reasoner := &Reasoner{}
		WithTriggers(struct{ Unsupported bool }{Unsupported: true})(reasoner)
		assert.Empty(t, reasoner.Triggers)
	})

	t.Run("WithTriggerSecretEnv and config are noops without a trigger", func(t *testing.T) {
		reasoner := &Reasoner{}
		WithTriggerSecretEnv("SECRET")(reasoner)
		WithTriggerConfig(map[string]any{"path": "/hook"})(reasoner)
		assert.Empty(t, reasoner.Triggers)
	})

	t.Run("WithTriggerConfig ignores unmarshalable config", func(t *testing.T) {
		reasoner := &Reasoner{}
		WithEventTrigger("generic_bearer", "push")(reasoner)
		WithTriggerConfig(map[string]any{"bad": func() {}})(reasoner)
		require.Len(t, reasoner.Triggers, 1)
		assert.Empty(t, reasoner.Triggers[0].Config)
	})

	t.Run("triggerToBinding covers config, timezone, and unknown types", func(t *testing.T) {
		eventBinding, ok := triggerToBinding(EventTrigger{
			Source:    "generic_bearer",
			Types:     []string{"push"},
			SecretEnv: "TOKEN",
			Config:    map[string]any{"path": "/hook"},
		})
		require.True(t, ok)
		assert.Equal(t, "generic_bearer", eventBinding.Source)
		assert.JSONEq(t, `{"path":"/hook"}`, string(eventBinding.Config))
		assert.Equal(t, "TOKEN", eventBinding.SecretEnvVar)

		scheduleBinding, ok := triggerToBinding(ScheduleTrigger{Cron: "*/5 * * * *", Timezone: "America/Toronto"})
		require.True(t, ok)
		assert.Equal(t, "cron", scheduleBinding.Source)
		assert.Contains(t, string(scheduleBinding.Config), "America/Toronto")

		_, ok = triggerToBinding(42)
		assert.False(t, ok)
	})
}
