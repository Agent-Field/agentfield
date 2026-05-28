package middleware

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestSignatureCacheSeenAndExpiry(t *testing.T) {
	cache := &signatureCache{
		entries: make(map[string]time.Time),
		ttl:     5 * time.Millisecond,
		stop:    make(chan struct{}),
	}

	require.False(t, cache.seen("sig-a"))
	require.True(t, cache.seen("sig-a"))

	cache.entries["sig-a"] = time.Now().Add(-time.Millisecond)
	require.False(t, cache.seen("sig-a"))
}

func TestSignatureCacheCleanupAndClose(t *testing.T) {
	cache := &signatureCache{
		entries: map[string]time.Time{
			"expired": time.Now().Add(-time.Millisecond),
			"active":  time.Now().Add(50 * time.Millisecond),
		},
		ttl:  5 * time.Millisecond,
		stop: make(chan struct{}),
	}

	done := make(chan struct{})
	go func() {
		cache.cleanup()
		close(done)
	}()

	time.Sleep(20 * time.Millisecond)

	cache.mu.Lock()
	_, expiredExists := cache.entries["expired"]
	_, activeExists := cache.entries["active"]
	cache.mu.Unlock()

	require.False(t, expiredExists)
	require.True(t, activeExists)

	cache.Close()

	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Fatal("cleanup goroutine did not stop after Close")
	}
}
