package events

import (
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"sync"
)

// Usage is observed at the persisted SDK-entry boundary, never from parent
// execution rollups. The observer sanitizes and durably queues before returning.
var usageObserver struct {
	sync.RWMutex
	fn func(string, int, *types.ExecutionUsage)
}

func SetUsageObserver(fn func(string, int, *types.ExecutionUsage)) {
	usageObserver.Lock()
	defer usageObserver.Unlock()
	usageObserver.fn = fn
}

func PublishUsage(executionID string, entry int, row *types.ExecutionUsage) {
	usageObserver.RLock()
	defer usageObserver.RUnlock()
	if usageObserver.fn != nil {
		usageObserver.fn(executionID, entry, row)
	}
}
