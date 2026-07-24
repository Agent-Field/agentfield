package handlers

import (
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
)

// AgentConcurrencyLimiter tracks and enforces per-agent concurrent execution limits.
type AgentConcurrencyLimiter struct {
	maxPerAgent int
	counts      sync.Map // map[string]*int64
}

var (
	concurrencyLimiterOnce sync.Once
	concurrencyLimiter     *AgentConcurrencyLimiter
)

// InitConcurrencyLimiter initializes the global concurrency limiter.
// Must be called once at server startup. maxPerAgent <= 0 means unlimited.
func InitConcurrencyLimiter(maxPerAgent int) {
	concurrencyLimiterOnce.Do(func() {
		concurrencyLimiter = &AgentConcurrencyLimiter{
			maxPerAgent: maxPerAgent,
		}
		if maxPerAgent > 0 {
			logger.Logger.Info().
				Int("max_concurrent_per_agent", maxPerAgent).
				Msg("Per-agent concurrency limiter enabled")
		}
	})
}

// GetConcurrencyLimiter returns the global concurrency limiter.
func GetConcurrencyLimiter() *AgentConcurrencyLimiter {
	return concurrencyLimiter
}

// Acquire attempts to reserve an execution slot for the given agent.
// Returns nil on success, or an error if the agent is at capacity.
func (l *AgentConcurrencyLimiter) Acquire(agentNodeID string) error {
	if l == nil || l.maxPerAgent <= 0 {
		return nil // Unlimited
	}

	actual, _ := l.counts.LoadOrStore(agentNodeID, new(int64))
	counter := actual.(*int64)

	current := atomic.AddInt64(counter, 1)
	if current > int64(l.maxPerAgent) {
		// Roll back
		atomic.AddInt64(counter, -1)
		return fmt.Errorf("agent %s has reached max concurrent executions (%d)", agentNodeID, l.maxPerAgent)
	}

	return nil
}

// Release frees an execution slot for the given agent.
func (l *AgentConcurrencyLimiter) Release(agentNodeID string) {
	if l == nil || l.maxPerAgent <= 0 {
		return
	}

	actual, ok := l.counts.Load(agentNodeID)
	if !ok {
		return
	}
	counter := actual.(*int64)
	newVal := atomic.AddInt64(counter, -1)
	if newVal < 0 {
		// Guard against underflow from mismatched release calls
		atomic.StoreInt64(counter, 0)
	}
}

// GetRunningCount returns the current number of running executions for an agent.
func (l *AgentConcurrencyLimiter) GetRunningCount(agentNodeID string) int64 {
	if l == nil {
		return 0
	}
	actual, ok := l.counts.Load(agentNodeID)
	if !ok {
		return 0
	}
	return atomic.LoadInt64(actual.(*int64))
}

// MaxPerAgent returns the configured max concurrent executions per agent.
// Returns 0 if unlimited.
func (l *AgentConcurrencyLimiter) MaxPerAgent() int {
	if l == nil {
		return 0
	}
	return l.maxPerAgent
}

// GetAllCounts returns a snapshot of running counts for all agents.
func (l *AgentConcurrencyLimiter) GetAllCounts() map[string]int64 {
	result := make(map[string]int64)
	if l == nil {
		return result
	}
	l.counts.Range(func(key, value any) bool {
		agentID := key.(string)
		counter := value.(*int64)
		count := atomic.LoadInt64(counter)
		if count > 0 {
			result[agentID] = count
		}
		return true
	})
	return result
}
