package harness

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestOpenCodeConcurrencyLimit(t *testing.T) {
	t.Setenv("OPENCODE_MAX_CONCURRENT", "2")

	// reset global semaphore (important!)
	openCodeSemaphore = nil
	semOnce = sync.Once{}

	var current int64
	var maxSeen int64
	var wg sync.WaitGroup

	p := NewOpenCodeProvider("", "")
	p.runCLI = func(ctx context.Context, cmd []string, env map[string]string, cwd string, timeout int) (*CLIResult, error) {
		c := atomic.AddInt64(&current, 1)

		// track max concurrency
		for {
			m := atomic.LoadInt64(&maxSeen)
			if c > m {
				if atomic.CompareAndSwapInt64(&maxSeen, m, c) {
					break
				}
				continue
			}
			break
		}

		// simulate work
		time.Sleep(100 * time.Millisecond)

		atomic.AddInt64(&current, -1)
		return &CLIResult{}, nil
	}

	// launch 5 concurrent calls
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = p.Execute(context.Background(), "test", Options{})
		}()
	}

	wg.Wait()

	if maxSeen > 2 {
		t.Fatalf("expected max 2 concurrent executions, got %d", maxSeen)
	}
}

func TestOpenCodeContextCancellation(t *testing.T) {
	t.Setenv("OPENCODE_MAX_CONCURRENT", "1")

	openCodeSemaphore = nil
	semOnce = sync.Once{}

	block := make(chan struct{})
	defer close(block)

	p := NewOpenCodeProvider("", "")

	p.runCLI = func(ctx context.Context, cmd []string, env map[string]string, cwd string, timeout int) (*CLIResult, error) {
		<-block
		return &CLIResult{}, nil
	}

	// occupy the slot
	go func() {
		_, _ = p.Execute(context.Background(), "test", Options{})
	}()

	time.Sleep(50 * time.Millisecond)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := p.Execute(ctx, "test", Options{})
	if err == nil {
		t.Fatalf("expected context cancellation error")
	}
}
