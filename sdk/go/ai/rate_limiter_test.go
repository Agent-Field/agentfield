package ai

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newTestRateLimiter builds a limiter with a fixed clock and a synchronous
// sleep stub so tests are deterministic and fast. maxRetries is applied
// verbatim (including 0) so circuit-breaker behaviour can be tested without
// retry interference.
func newTestRateLimiter(cfg RateLimiterConfig, now func() time.Time) (*RateLimiter, *[]time.Duration) {
	rl := NewRateLimiter(cfg)
	rl.maxRetries = cfg.MaxRetries // honor 0 (no retries) for deterministic tests
	var slept []time.Duration
	rl.now = now
	rl.sleep = func(ctx context.Context, d time.Duration) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		slept = append(slept, d)
		return nil
	}
	return rl, &slept
}

func TestNewRateLimiterAppliesDefaults(t *testing.T) {
	rl := NewRateLimiter(RateLimiterConfig{})
	assert.Equal(t, defaultRateLimitMaxRetries, rl.maxRetries)
	assert.Equal(t, defaultRateLimitBaseDelay, rl.baseDelay)
	assert.Equal(t, defaultRateLimitMaxDelay, rl.maxDelay)
	assert.Equal(t, defaultRateLimitJitterFactor, rl.jitterFactor)
	assert.Equal(t, defaultCircuitBreakerThreshold, rl.circuitBreakerThreshold)
	assert.Equal(t, defaultCircuitBreakerTimeout, rl.circuitBreakerTimeout)
}

func TestNewRateLimiterHonorsExplicitValues(t *testing.T) {
	rl := NewRateLimiter(RateLimiterConfig{
		MaxRetries:              3,
		BaseDelay:               time.Second,
		MaxDelay:                10 * time.Second,
		JitterFactor:            0.1,
		CircuitBreakerThreshold: 2,
		CircuitBreakerTimeout:   time.Minute,
	})
	assert.Equal(t, 3, rl.maxRetries)
	assert.Equal(t, time.Second, rl.baseDelay)
	assert.Equal(t, 10*time.Second, rl.maxDelay)
	assert.Equal(t, 0.1, rl.jitterFactor)
	assert.Equal(t, 2, rl.circuitBreakerThreshold)
	assert.Equal(t, time.Minute, rl.circuitBreakerTimeout)
}

func TestExecuteSucceedsWithoutRetry(t *testing.T) {
	rl, slept := newTestRateLimiter(RateLimiterConfig{MaxRetries: 3}, time.Now)
	calls := 0
	err := rl.Execute(context.Background(), func() error {
		calls++
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, 1, calls)
	assert.Empty(t, *slept)
	assert.Equal(t, CircuitClosed, rl.State())
}

func TestExecuteRetriesOnRateLimitThenSucceeds(t *testing.T) {
	rl, slept := newTestRateLimiter(RateLimiterConfig{MaxRetries: 3, JitterFactor: 0.0001}, time.Now)
	calls := 0
	err := rl.Execute(context.Background(), func() error {
		calls++
		if calls < 3 {
			return &APIError{StatusCode: 429, Message: "rate limited"}
		}
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, 3, calls)
	assert.Len(t, *slept, 2) // two retries before the third succeeds
}

func TestExecuteReturnsNonRateLimitErrorImmediately(t *testing.T) {
	rl, slept := newTestRateLimiter(RateLimiterConfig{MaxRetries: 3}, time.Now)
	sentinel := errors.New("boom")
	calls := 0
	err := rl.Execute(context.Background(), func() error {
		calls++
		return sentinel
	})
	require.ErrorIs(t, err, sentinel)
	assert.Equal(t, 1, calls)
	assert.Empty(t, *slept)
}

func TestExecuteExhaustsRetries(t *testing.T) {
	rl, slept := newTestRateLimiter(RateLimiterConfig{MaxRetries: 2, JitterFactor: 0.0001}, time.Now)
	calls := 0
	err := rl.Execute(context.Background(), func() error {
		calls++
		return &APIError{StatusCode: 503}
	})
	require.ErrorIs(t, err, ErrMaxRetriesExceeded)
	assert.Equal(t, 3, calls) // initial + 2 retries
	assert.Len(t, *slept, 2)
}

func TestExecuteHonorsContextCancellation(t *testing.T) {
	rl := NewRateLimiter(RateLimiterConfig{MaxRetries: 3})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	calls := 0
	err := rl.Execute(ctx, func() error {
		calls++
		return &APIError{StatusCode: 429}
	})
	require.ErrorIs(t, err, context.Canceled)
	assert.Equal(t, 1, calls) // one attempt, then cancelled during sleep
}

func TestCircuitBreakerOpensAndRejects(t *testing.T) {
	fixed := time.Unix(1000, 0)
	rl, _ := newTestRateLimiter(RateLimiterConfig{
		MaxRetries:              0,
		CircuitBreakerThreshold: 2,
		CircuitBreakerTimeout:   time.Minute,
	}, func() time.Time { return fixed })

	// Two failures should open the circuit (threshold=2, no retries).
	for i := 0; i < 2; i++ {
		err := rl.Execute(context.Background(), func() error {
			return &APIError{StatusCode: 429}
		})
		require.Error(t, err)
	}

	assert.Equal(t, CircuitOpen, rl.State())

	// Next call is rejected without invoking fn.
	called := false
	err := rl.Execute(context.Background(), func() error {
		called = true
		return nil
	})
	require.ErrorIs(t, err, ErrCircuitOpen)
	assert.False(t, called)
}

func TestCircuitBreakerHalfOpenAfterTimeout(t *testing.T) {
	current := time.Unix(1000, 0)
	rl, _ := newTestRateLimiter(RateLimiterConfig{
		MaxRetries:              0,
		CircuitBreakerThreshold: 1,
		CircuitBreakerTimeout:   30 * time.Second,
	}, func() time.Time { return current })

	// Open the circuit.
	err := rl.Execute(context.Background(), func() error {
		return &APIError{StatusCode: 429}
	})
	require.Error(t, err)
	assert.Equal(t, CircuitOpen, rl.State())

	// Advance past the timeout: circuit becomes half-open.
	current = current.Add(31 * time.Second)
	assert.Equal(t, CircuitHalfOpen, rl.State())

	// A successful probe closes the circuit.
	err = rl.Execute(context.Background(), func() error {
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, CircuitClosed, rl.State())
}

func TestCircuitBreakerHalfOpenAdmitsSingleProbe(t *testing.T) {
	current := time.Unix(1000, 0)
	rl, _ := newTestRateLimiter(RateLimiterConfig{
		MaxRetries:              0,
		CircuitBreakerThreshold: 1,
		CircuitBreakerTimeout:   30 * time.Second,
	}, func() time.Time { return current })

	// Open the circuit.
	require.Error(t, rl.Execute(context.Background(), func() error {
		return &APIError{StatusCode: 429}
	}))

	// Advance past the timeout: half-open, one probe should be admitted.
	current = current.Add(31 * time.Second)
	require.Equal(t, CircuitHalfOpen, rl.State())

	// Hold the probe in flight and, from within it, confirm every concurrent
	// caller is fast-failed with ErrCircuitOpen (only one probe is admitted).
	probeAdmitted := false
	err := rl.Execute(context.Background(), func() error {
		probeAdmitted = true
		for i := 0; i < 5; i++ {
			inner := rl.Execute(context.Background(), func() error {
				t.Fatal("a second request was admitted while the probe was in flight")
				return nil
			})
			require.ErrorIs(t, inner, ErrCircuitOpen)
		}
		return &APIError{StatusCode: 429} // probe fails
	})
	require.Error(t, err)
	require.True(t, probeAdmitted)

	// A failed probe re-opens the circuit (timer restarted from now).
	assert.Equal(t, CircuitOpen, rl.State())
}

func TestCircuitBreakerFailedProbeReopens(t *testing.T) {
	current := time.Unix(1000, 0)
	rl, _ := newTestRateLimiter(RateLimiterConfig{
		MaxRetries:              0,
		CircuitBreakerThreshold: 1,
		CircuitBreakerTimeout:   30 * time.Second,
	}, func() time.Time { return current })

	require.Error(t, rl.Execute(context.Background(), func() error {
		return &APIError{StatusCode: 429}
	}))

	current = current.Add(31 * time.Second)
	// Failed probe.
	require.Error(t, rl.Execute(context.Background(), func() error {
		return &APIError{StatusCode: 429}
	}))
	// Circuit is open again and the next immediate caller is rejected.
	assert.Equal(t, CircuitOpen, rl.State())
	err := rl.Execute(context.Background(), func() error {
		t.Fatal("caller admitted while circuit re-opened")
		return nil
	})
	require.ErrorIs(t, err, ErrCircuitOpen)
}

func TestCircuitBreakerProbeReleasedOnNonRateLimitError(t *testing.T) {
	current := time.Unix(1000, 0)
	rl, _ := newTestRateLimiter(RateLimiterConfig{
		MaxRetries:              0,
		CircuitBreakerThreshold: 1,
		CircuitBreakerTimeout:   30 * time.Second,
	}, func() time.Time { return current })

	require.Error(t, rl.Execute(context.Background(), func() error {
		return &APIError{StatusCode: 429}
	}))

	current = current.Add(31 * time.Second)
	// The probe hits a non-rate-limit error: it must not consume the probe slot,
	// so a following caller can still take the half-open probe.
	require.ErrorContains(t, rl.Execute(context.Background(), func() error {
		return errors.New("unrelated failure")
	}), "unrelated failure")

	// Still half-open, probe slot free: a fresh probe is admitted and succeeds.
	require.Equal(t, CircuitHalfOpen, rl.State())
	require.NoError(t, rl.Execute(context.Background(), func() error {
		return nil
	}))
	assert.Equal(t, CircuitClosed, rl.State())
}

func TestCircuitBreakerDisabled(t *testing.T) {
	rl, _ := newTestRateLimiter(RateLimiterConfig{
		MaxRetries:              0,
		CircuitBreakerThreshold: -1,
	}, time.Now)
	// Many failures never open the circuit.
	for i := 0; i < 10; i++ {
		_ = rl.Execute(context.Background(), func() error {
			return &APIError{StatusCode: 429}
		})
	}
	assert.Equal(t, CircuitClosed, rl.State())
}

func TestSuccessResetsFailureCount(t *testing.T) {
	rl, _ := newTestRateLimiter(RateLimiterConfig{
		MaxRetries:              0,
		CircuitBreakerThreshold: 3,
		CircuitBreakerTimeout:   time.Minute,
	}, time.Now)

	_ = rl.Execute(context.Background(), func() error { return &APIError{StatusCode: 429} })
	_ = rl.Execute(context.Background(), func() error { return &APIError{StatusCode: 429} })
	// Success resets the counter before threshold is hit.
	_ = rl.Execute(context.Background(), func() error { return nil })
	// Two more failures should not open the circuit (counter was reset).
	_ = rl.Execute(context.Background(), func() error { return &APIError{StatusCode: 429} })
	_ = rl.Execute(context.Background(), func() error { return &APIError{StatusCode: 429} })
	assert.Equal(t, CircuitClosed, rl.State())
}

func TestNilRateLimiterExecutesDirectly(t *testing.T) {
	var rl *RateLimiter
	calls := 0
	err := rl.Execute(context.Background(), func() error {
		calls++
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, 1, calls)
	assert.Equal(t, CircuitClosed, rl.State())
}

func TestBackoffDelayExponentialGrowth(t *testing.T) {
	rl := NewRateLimiter(RateLimiterConfig{
		BaseDelay:    time.Second,
		MaxDelay:     time.Minute,
		JitterFactor: 0.0001, // near-zero jitter for a stable assertion
	})
	d0 := rl.backoffDelay(0, 0)
	d1 := rl.backoffDelay(1, 0)
	d2 := rl.backoffDelay(2, 0)

	assert.InDelta(t, time.Second, d0, float64(50*time.Millisecond))
	assert.InDelta(t, 2*time.Second, d1, float64(50*time.Millisecond))
	assert.InDelta(t, 4*time.Second, d2, float64(50*time.Millisecond))
}

func TestBackoffDelayCapsAtMaxDelay(t *testing.T) {
	rl := NewRateLimiter(RateLimiterConfig{
		BaseDelay:    time.Second,
		MaxDelay:     5 * time.Second,
		JitterFactor: 0.0001,
	})
	d := rl.backoffDelay(10, 0) // 2^10 * 1s would be huge; must cap
	assert.LessOrEqual(t, d, 5*time.Second+time.Millisecond)
}

func TestBackoffDelayUsesRetryAfter(t *testing.T) {
	rl := NewRateLimiter(RateLimiterConfig{
		BaseDelay:    time.Second,
		MaxDelay:     time.Minute,
		JitterFactor: 0.0001,
	})
	d := rl.backoffDelay(0, 3*time.Second)
	assert.InDelta(t, 3*time.Second, d, float64(50*time.Millisecond))
}

func TestBackoffDelayIgnoresUnreasonableRetryAfter(t *testing.T) {
	rl := NewRateLimiter(RateLimiterConfig{
		BaseDelay:    time.Second,
		MaxDelay:     5 * time.Second,
		JitterFactor: 0.0001,
	})
	// retryAfter exceeds maxDelay: fall back to exponential backoff.
	d := rl.backoffDelay(0, time.Hour)
	assert.InDelta(t, time.Second, d, float64(50*time.Millisecond))
}

func TestBackoffDelayEnforcesMinimum(t *testing.T) {
	rl := NewRateLimiter(RateLimiterConfig{
		BaseDelay:    time.Nanosecond,
		MaxDelay:     time.Minute,
		JitterFactor: 0.0001,
	})
	d := rl.backoffDelay(0, 0)
	assert.GreaterOrEqual(t, d, rateLimiterMinDelay)
}

func TestIsRateLimitError(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"429", &APIError{StatusCode: 429}, true},
		{"503", &APIError{StatusCode: 503}, true},
		{"500", &APIError{StatusCode: 500}, false},
		{"keyword rate limit", errors.New("Rate Limit exceeded"), true},
		{"keyword too many requests", errors.New("429 too many requests"), true},
		{"keyword throttled", errors.New("request was throttled"), true},
		{"keyword tpm", errors.New("TPM exceeded for this key"), true},
		{"unrelated", errors.New("connection refused"), false},
		{"wrapped 429", fmt.Errorf("call failed: %w", &APIError{StatusCode: 429}), true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, isRateLimitError(tc.err))
		})
	}
}

func TestRetryAfterFromError(t *testing.T) {
	assert.Equal(t, time.Duration(0), retryAfterFromError(errors.New("plain")))
	assert.Equal(t, time.Duration(0), retryAfterFromError(&APIError{StatusCode: 429}))
	assert.Equal(t, 5*time.Second, retryAfterFromError(&APIError{StatusCode: 429, Code: "5"}))
	assert.Equal(t, time.Duration(0), retryAfterFromError(&APIError{StatusCode: 429, Code: "not-a-number"}))
}

func TestCircuitStateString(t *testing.T) {
	assert.Equal(t, "closed", CircuitClosed.String())
	assert.Equal(t, "open", CircuitOpen.String())
	assert.Equal(t, "half-open", CircuitHalfOpen.String())
	assert.Equal(t, "unknown", CircuitState(99).String())
}

func TestSleepWithContext(t *testing.T) {
	// Zero duration returns immediately with the context error (nil here).
	require.NoError(t, sleepWithContext(context.Background(), 0))

	// Real short sleep completes.
	start := time.Now()
	require.NoError(t, sleepWithContext(context.Background(), 10*time.Millisecond))
	assert.GreaterOrEqual(t, time.Since(start), 5*time.Millisecond)

	// Cancelled context returns promptly.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	require.ErrorIs(t, sleepWithContext(ctx, time.Hour), context.Canceled)
}

func TestContainerSeedStable(t *testing.T) {
	assert.Equal(t, containerSeed(), containerSeed())
}

// TestCircuitBreakerHalfOpenConcurrentProbe verifies that, under real goroutine
// contention, exactly one caller is admitted as the half-open probe once the
// open timeout has elapsed. All others must fast-fail with ErrCircuitOpen.
func TestCircuitBreakerHalfOpenConcurrentProbe(t *testing.T) {
	current := time.Unix(1000, 0)
	rl, _ := newTestRateLimiter(RateLimiterConfig{
		MaxRetries:              0,
		CircuitBreakerThreshold: 1,
		CircuitBreakerTimeout:   30 * time.Second,
	}, func() time.Time { return current })

	// Open the circuit.
	require.Error(t, rl.Execute(context.Background(), func() error {
		return &APIError{StatusCode: 429}
	}))

	// Advance past the timeout so the breaker is half-open.
	current = current.Add(31 * time.Second)

	const callers = 50
	var admitted int32
	var rejected int32
	release := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(callers)

	for i := 0; i < callers; i++ {
		go func() {
			defer wg.Done()
			err := rl.Execute(context.Background(), func() error {
				// The admitted probe blocks here so all other goroutines reach
				// admit() while the probe slot is held.
				atomic.AddInt32(&admitted, 1)
				<-release
				return nil
			})
			if err != nil {
				require.ErrorIs(t, err, ErrCircuitOpen)
				atomic.AddInt32(&rejected, 1)
			}
		}()
	}

	// Give every goroutine time to pass (or be rejected by) admit(), then let
	// the single probe complete.
	require.Eventually(t, func() bool {
		return atomic.LoadInt32(&admitted)+atomic.LoadInt32(&rejected) == callers
	}, 2*time.Second, time.Millisecond)
	close(release)
	wg.Wait()

	assert.Equal(t, int32(1), atomic.LoadInt32(&admitted), "exactly one probe should be admitted")
	assert.Equal(t, int32(callers-1), atomic.LoadInt32(&rejected), "all other callers should fast-fail")
	assert.Equal(t, CircuitClosed, rl.State(), "successful probe closes the circuit")
}
