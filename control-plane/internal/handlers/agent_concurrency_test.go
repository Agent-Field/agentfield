package handlers

import (
	"sync"
	"testing"
)

func TestAgentConcurrencyLimiter_AcquireRelease(t *testing.T) {
	limiter := &AgentConcurrencyLimiter{maxPerAgent: 3}

	// Should be able to acquire 3 slots
	for i := 0; i < 3; i++ {
		if err := limiter.Acquire("agent-1"); err != nil {
			t.Fatalf("acquire %d failed: %v", i, err)
		}
	}

	// 4th should fail
	if err := limiter.Acquire("agent-1"); err == nil {
		t.Fatal("expected error on 4th acquire, got nil")
	}

	// Different agent should still work
	if err := limiter.Acquire("agent-2"); err != nil {
		t.Fatalf("acquire for different agent failed: %v", err)
	}

	// Release one slot for agent-1
	limiter.Release("agent-1")

	// Now agent-1 should be able to acquire again
	if err := limiter.Acquire("agent-1"); err != nil {
		t.Fatal("expected acquire to succeed after release")
	}

	// Verify counts
	if count := limiter.GetRunningCount("agent-1"); count != 3 {
		t.Fatalf("expected 3 running for agent-1, got %d", count)
	}
	if count := limiter.GetRunningCount("agent-2"); count != 1 {
		t.Fatalf("expected 1 running for agent-2, got %d", count)
	}
}

func TestAgentConcurrencyLimiter_Unlimited(t *testing.T) {
	limiter := &AgentConcurrencyLimiter{maxPerAgent: 0}

	// Should always succeed with 0 (unlimited)
	for i := 0; i < 100; i++ {
		if err := limiter.Acquire("agent-1"); err != nil {
			t.Fatalf("acquire failed with unlimited: %v", err)
		}
	}
}

func TestAgentConcurrencyLimiter_NilSafe(t *testing.T) {
	var limiter *AgentConcurrencyLimiter

	// All operations on nil limiter should be safe
	if err := limiter.Acquire("agent-1"); err != nil {
		t.Fatalf("nil acquire should return nil, got: %v", err)
	}
	limiter.Release("agent-1")
	if count := limiter.GetRunningCount("agent-1"); count != 0 {
		t.Fatalf("nil count should return 0, got %d", count)
	}
	counts := limiter.GetAllCounts()
	if len(counts) != 0 {
		t.Fatalf("nil GetAllCounts should return empty, got %v", counts)
	}
}

func TestAgentConcurrencyLimiter_ConcurrentAccess(t *testing.T) {
	limiter := &AgentConcurrencyLimiter{maxPerAgent: 10}

	var wg sync.WaitGroup
	acquired := make(chan struct{}, 100)
	rejected := make(chan struct{}, 100)

	// Launch 20 goroutines trying to acquire for the same agent
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := limiter.Acquire("agent-1"); err != nil {
				rejected <- struct{}{}
			} else {
				acquired <- struct{}{}
			}
		}()
	}

	wg.Wait()
	close(acquired)
	close(rejected)

	acquiredCount := 0
	for range acquired {
		acquiredCount++
	}
	rejectedCount := 0
	for range rejected {
		rejectedCount++
	}

	if acquiredCount != 10 {
		t.Fatalf("expected exactly 10 acquired, got %d", acquiredCount)
	}
	if rejectedCount != 10 {
		t.Fatalf("expected exactly 10 rejected, got %d", rejectedCount)
	}
}

func TestAgentConcurrencyLimiter_ReleaseUnderflowProtection(t *testing.T) {
	limiter := &AgentConcurrencyLimiter{maxPerAgent: 5}

	// Release without acquire should not go negative
	limiter.Release("agent-1")
	if count := limiter.GetRunningCount("agent-1"); count != 0 {
		t.Fatalf("expected 0 after underflow protection, got %d", count)
	}

	// Should still be able to acquire normally
	if err := limiter.Acquire("agent-1"); err != nil {
		t.Fatalf("acquire after underflow should work: %v", err)
	}
}

func TestAgentConcurrencyLimiter_GetAllCounts(t *testing.T) {
	limiter := &AgentConcurrencyLimiter{maxPerAgent: 10}

	limiter.Acquire("agent-a")
	limiter.Acquire("agent-a")
	limiter.Acquire("agent-b")

	counts := limiter.GetAllCounts()
	if counts["agent-a"] != 2 {
		t.Fatalf("expected 2 for agent-a, got %d", counts["agent-a"])
	}
	if counts["agent-b"] != 1 {
		t.Fatalf("expected 1 for agent-b, got %d", counts["agent-b"])
	}
}
