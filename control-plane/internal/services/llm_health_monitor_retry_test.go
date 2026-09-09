package services

import (
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/stretchr/testify/assert"
)

func TestLLMHealthMonitor_RetryAfterSeconds(t *testing.T) {
	base := time.Date(2026, time.August, 31, 12, 0, 0, 0, time.UTC)
	now := base
	monitor := NewLLMHealthMonitor(config.LLMHealthConfig{
		CheckInterval:    6 * time.Second,
		RecoveryTimeout:  10 * time.Second,
		FailureThreshold: 1,
		Endpoints:        []config.LLMEndpoint{{Name: "primary"}},
	}, nil)
	monitor.now = func() time.Time { return now }

	assert.Equal(t, 10, monitor.RetryAfterSeconds("unknown"))
	assert.Equal(t, 10, monitor.RetryAfterSeconds("primary"))

	monitor.mu.Lock()
	// Checks are scheduled at +6s, +12s, ...; a 10-second recovery timeout
	// therefore cannot transition this circuit before the +12s tick.
	monitor.nextCheckAt = base.Add(6 * time.Second)
	monitor.handleFailure(monitor.endpoints["primary"])
	monitor.mu.Unlock()
	assert.Equal(t, 12, monitor.RetryAfterSeconds("primary"))

	now = base.Add(5*time.Second + 100*time.Millisecond)
	assert.Equal(t, 7, monitor.RetryAfterSeconds("primary"))
	now = base.Add(11*time.Second + 100*time.Millisecond)
	assert.Equal(t, 1, monitor.RetryAfterSeconds("primary"))

	// If an open status predates scheduler bookkeeping, stay conservative by
	// including one full check interval instead of reverting to the raw timeout.
	now = base
	monitor.mu.Lock()
	monitor.endpoints["primary"].nextProbeAt = time.Time{}
	monitor.endpoints["primary"].circuitOpenedAt = base
	monitor.mu.Unlock()
	assert.Equal(t, 16, monitor.RetryAfterSeconds("primary"))

	defaults := NewLLMHealthMonitor(config.LLMHealthConfig{}, nil)
	assert.Equal(t, 30, defaults.RetryAfterSeconds("unknown"))
	assert.Equal(t, 30, (*LLMHealthMonitor)(nil).RetryAfterSeconds("unknown"))
}
