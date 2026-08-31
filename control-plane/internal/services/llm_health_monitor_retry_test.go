package services

import (
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/stretchr/testify/assert"
)

func TestLLMHealthMonitor_RetryAfterSeconds(t *testing.T) {
	monitor := NewLLMHealthMonitor(config.LLMHealthConfig{
		RecoveryTimeout: 10 * time.Second,
		Endpoints:       []config.LLMEndpoint{{Name: "primary"}},
	}, nil)

	assert.Equal(t, 10, monitor.RetryAfterSeconds("unknown"))
	assert.Equal(t, 10, monitor.RetryAfterSeconds("primary"))

	monitor.mu.Lock()
	monitor.endpoints["primary"].CircuitState = CircuitOpen
	monitor.endpoints["primary"].circuitOpenedAt = time.Now().Add(-4 * time.Second)
	monitor.mu.Unlock()
	remaining := monitor.RetryAfterSeconds("primary")
	assert.GreaterOrEqual(t, remaining, 5)
	assert.LessOrEqual(t, remaining, 6)

	monitor.mu.Lock()
	monitor.endpoints["primary"].circuitOpenedAt = time.Now().Add(-20 * time.Second)
	monitor.mu.Unlock()
	assert.Equal(t, 1, monitor.RetryAfterSeconds("primary"))

	defaults := NewLLMHealthMonitor(config.LLMHealthConfig{}, nil)
	assert.Equal(t, 30, defaults.RetryAfterSeconds("unknown"))
	assert.Equal(t, 30, (*LLMHealthMonitor)(nil).RetryAfterSeconds("unknown"))
}
