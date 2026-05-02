package harness

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
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

// TestOpenCodeConcurrencyLimit_RealSubprocess exercises the semaphore against
// real RunCLI subprocesses (no runCLI mock). The fake opencode binary records
// a start/end timestamp for every invocation; afterwards we sweep the timeline
// to compute the maximum number of subprocesses that were actually overlapping
// in wall-clock time and assert it never exceeds the configured limit.
func TestOpenCodeConcurrencyLimit_RealSubprocess(t *testing.T) {
	const limit = 2
	const callers = 6

	t.Setenv("OPENCODE_MAX_CONCURRENT", fmt.Sprintf("%d", limit))
	openCodeSemaphore = nil
	semOnce = sync.Once{}

	dir := t.TempDir()
	logDir := t.TempDir()

	// Each invocation writes a "<start_ns> <end_ns>" line to a file named by
	// $$ (subprocess pid) so we can later compute the overlap count.
	script := writeTestScript(t, dir, "opencode", fmt.Sprintf(
		"#!/bin/sh\n"+
			"start=$(date +%%s%%N)\n"+
			"sleep 0.2\n"+
			"end=$(date +%%s%%N)\n"+
			"echo \"$start $end\" > %q/$$.span\n"+
			"echo ok\n",
		logDir,
	))

	p := NewOpenCodeProvider(script, "")

	var wg sync.WaitGroup
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			raw, err := p.Execute(context.Background(), "prompt", Options{})
			if err != nil {
				t.Errorf("Execute: %v", err)
				return
			}
			if raw.IsError {
				t.Errorf("Execute IsError: %s", raw.ErrorMessage)
			}
		}()
	}
	wg.Wait()

	type span struct{ start, end int64 }
	entries, err := os.ReadDir(logDir)
	if err != nil {
		t.Fatalf("read logDir: %v", err)
	}
	if len(entries) != callers {
		t.Fatalf("expected %d span files, got %d", callers, len(entries))
	}

	spans := make([]span, 0, len(entries))
	for _, e := range entries {
		b, err := os.ReadFile(filepath.Join(logDir, e.Name()))
		if err != nil {
			t.Fatalf("read span: %v", err)
		}
		var s, en int64
		if _, err := fmt.Sscanf(string(b), "%d %d", &s, &en); err != nil {
			t.Fatalf("parse span %q: %v", string(b), err)
		}
		spans = append(spans, span{s, en})
	}

	// Sweep-line: at every start +1, at every end -1. Track running max.
	type ev struct {
		t     int64
		delta int
	}
	evs := make([]ev, 0, 2*len(spans))
	for _, s := range spans {
		evs = append(evs, ev{s.start, +1}, ev{s.end, -1})
	}
	sort.Slice(evs, func(i, j int) bool {
		if evs[i].t != evs[j].t {
			return evs[i].t < evs[j].t
		}
		// Process -1 before +1 at the same instant so back-to-back spans
		// don't look like overlap.
		return evs[i].delta < evs[j].delta
	})
	cur, max := 0, 0
	for _, e := range evs {
		cur += e.delta
		if cur > max {
			max = cur
		}
	}

	if max > limit {
		t.Fatalf("observed %d concurrent subprocesses, want <= %d", max, limit)
	}
	if max == 0 {
		t.Fatalf("no overlap observed at all — test did not actually run concurrently")
	}
}

// TestOpenCodeSemaphore_ReleasedOnSubprocessFailure verifies that a slot is
// returned to the pool even when the subprocess exits non-zero / errors,
// rather than leaking forever. With limit=1, two sequential failing calls
// must both complete; if the first leaked the slot, the second would block.
func TestOpenCodeSemaphore_ReleasedOnSubprocessFailure(t *testing.T) {
	t.Setenv("OPENCODE_MAX_CONCURRENT", "1")
	openCodeSemaphore = nil
	semOnce = sync.Once{}

	dir := t.TempDir()
	script := writeTestScript(t, dir, "opencode",
		"#!/bin/sh\necho 'boom' >&2\nexit 7\n")

	p := NewOpenCodeProvider(script, "")

	for i := 0; i < 3; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		raw, err := p.Execute(ctx, "prompt", Options{})
		cancel()
		if err != nil {
			t.Fatalf("call %d unexpected err: %v", i, err)
		}
		if !raw.IsError {
			t.Fatalf("call %d expected IsError for non-zero exit", i)
		}
	}
}
