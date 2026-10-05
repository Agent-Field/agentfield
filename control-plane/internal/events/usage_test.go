package events

import (
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

func TestUsageObserverDeliveryReplacementAndUnsubscribe(t *testing.T) {
	defer SetUsageObserver(nil)
	SetUsageObserver(nil)
	PublishUsage("unobserved", 0, nil)
	row := &types.ExecutionUsage{InputTokens: 7}
	calls := 0
	SetUsageObserver(func(id string, index int, got *types.ExecutionUsage) {
		calls++
		if id != "execution" || index != 3 || got != row {
			t.Fatalf("observer changed receipt identity: %s %d %p", id, index, got)
		}
	})
	PublishUsage("execution", 3, row)
	replacementCalls := 0
	SetUsageObserver(func(string, int, *types.ExecutionUsage) { replacementCalls++ })
	PublishUsage("replacement", 0, row)
	SetUsageObserver(nil)
	PublishUsage("after-stop", 0, row)
	if calls != 1 || replacementCalls != 1 {
		t.Fatalf("observer delivery old=%d replacement=%d", calls, replacementCalls)
	}
}
