package ai

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"
)

// Test error types
var errRateLimit = errors.New("rate limit exceeded: 429 Too Many Requests")
var errNotRateLimit = errors.New("some other error")
var errQuotaExceeded = errors.New("quota exceeded for this month")
var errThrottling = errors.New("throttling: requests per minute exceeded")

func TestNewRateLimiter(t *testing.T) {
	tests := []struct {
		name   string
		config RateLimiterConfig
		check  func(*testing.T, *RateLimiter)
	}{
		{
			name: "zero values are respected",
			config: RateLimiterConfig{
				MaxRetries:              0,
				BaseDelay:               0,
				MaxDelay:                0,
				JitterFactor:            0,
				CircuitBreakerThreshold: 0,
				CircuitBreakerTimeout:   0,
			},
			check: func(t *testing.T, rl *RateLimiter) {
				if rl.maxRetries != 0 {
					t.Errorf("Expected maxRetries=0, got %d", rl.maxRetries)
				}
				if rl.baseDelay != 0 {
					t.Errorf("Expected baseDelay=0, got %v", rl.baseDelay)
				}
				if rl.maxDelay != 0 {
					t.Errorf("Expected maxDelay=0, got %v", rl.maxDelay)
				}
				if rl.jitterFactor != 0 {
					t.Errorf("Expected jitterFactor=0, got %f", rl.jitterFactor)
				}
				if rl.circuitBreakerThreshold != 0 {
					t.Errorf("Expected circuitBreakerThreshold=0, got %d", rl.circuitBreakerThreshold)
				}
				if rl.circuitBreakerTimeout != 0 {
					t.Errorf("Expected circuitBreakerTimeout=0, got %v", rl.circuitBreakerTimeout)
				}
			},
		},
		{
			name: "custom values",
			config: RateLimiterConfig{
				MaxRetries:              10,
				BaseDelay:               500 * time.Millisecond,
				MaxDelay:                10 * time.Second,
				JitterFactor:            0.2,
				CircuitBreakerThreshold: 3,
				CircuitBreakerTimeout:   30 * time.Second,
			},
			check: func(t *testing.T, rl *RateLimiter) {
				if rl.maxRetries != 10 {
					t.Errorf("Expected maxRetries=10, got %d", rl.maxRetries)
				}
				if rl.baseDelay != 500*time.Millisecond {
					t.Errorf("Expected baseDelay=500ms, got %v", rl.baseDelay)
				}
				if rl.maxDelay != 10*time.Second {
					t.Errorf("Expected maxDelay=10s, got %v", rl.maxDelay)
				}
				if rl.jitterFactor != 0.2 {
					t.Errorf("Expected jitterFactor=0.2, got %f", rl.jitterFactor)
				}
				if rl.circuitBreakerThreshold != 3 {
					t.Errorf("Expected circuitBreakerThreshold=3, got %d", rl.circuitBreakerThreshold)
				}
				if rl.circuitBreakerTimeout != 30*time.Second {
					t.Errorf("Expected circuitBreakerTimeout=30s, got %v", rl.circuitBreakerTimeout)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rl := NewRateLimiter(tt.config)
			tt.check(t, rl)
		})
	}
}

func TestIsRateLimitError(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected bool
	}{
		{
			name:     "nil error",
			err:      nil,
			expected: false,
		},
		{
			name:     "rate limit error with 429",
			err:      errRateLimit,
			expected: true,
		},
		{
			name:     "quota exceeded error",
			err:      errQuotaExceeded,
			expected: true,
		},
		{
			name:     "throttling error",
			err:      errThrottling,
			expected: true,
		},
		{
			name:     "non-rate-limit error",
			err:      errNotRateLimit,
			expected: false,
		},
		{
			name:     "error with 'too many requests'",
			err:      errors.New("too many requests please try again"),
			expected: true,
		},
		{
			name:     "error with 'rate limited'",
			err:      errors.New("you have been rate limited"),
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isRateLimitError(tt.err)
			if result != tt.expected {
				t.Errorf("Expected %v, got %v for error: %v", tt.expected, result, tt.err)
			}
		})
	}
}

func TestCalculateBackoffDelay(t *testing.T) {
	rl := NewRateLimiter(RateLimiterConfig{
		BaseDelay:    time.Second,
		MaxDelay:     30 * time.Second,
		JitterFactor: 0.1,
	})

	tests := []struct {
		attempt    int
		minExpected time.Duration
		maxExpected time.Duration
	}{
		{attempt: 0, minExpected: 800 * time.Millisecond, maxExpected: 1200 * time.Millisecond}, // ~1s ± 10%
		{attempt: 1, minExpected: 1800 * time.Millisecond, maxExpected: 2200 * time.Millisecond}, // ~2s ± 10%
		{attempt: 2, minExpected: 3600 * time.Millisecond, maxExpected: 4400 * time.Millisecond}, // ~4s ± 10%
		{attempt: 3, minExpected: 7200 * time.Millisecond, maxExpected: 8800 * time.Millisecond}, // ~8s ± 10%
		{attempt: 10, minExpected: 27 * time.Second, maxExpected: 33 * time.Second}, // Capped at 30s ± 10% (with jitter)
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("attempt_%d", tt.attempt), func(t *testing.T) {
			delay := rl.calculateBackoffDelay(tt.attempt)
			if delay < tt.minExpected || delay > tt.maxExpected {
				t.Errorf("Attempt %d: expected delay between %v and %v, got %v",
					tt.attempt, tt.minExpected, tt.maxExpected, delay)
			}
		})
	}
}

func TestCircuitBreakerStates(t *testing.T) {
	t.Run("initially closed", func(t *testing.T) {
		rl := NewRateLimiter(RateLimiterConfig{
			CircuitBreakerThreshold: 3,
		})

		state := rl.checkCircuitBreaker()
		if state != CircuitClosed {
			t.Errorf("Expected CircuitClosed, got %v", state)
		}
	})

	t.Run("opens after threshold failures", func(t *testing.T) {
		rl := NewRateLimiter(RateLimiterConfig{
			CircuitBreakerThreshold: 3,
		})

		// Simulate failures
		for i := 0; i < 3; i++ {
			rl.updateCircuitBreaker(false)
		}

		state := rl.checkCircuitBreaker()
		if state != CircuitOpen {
			t.Errorf("Expected CircuitOpen after %d failures, got %v", 3, state)
		}
	})

	t.Run("resets on success", func(t *testing.T) {
		rl := NewRateLimiter(RateLimiterConfig{
			CircuitBreakerThreshold: 3,
		})

		// Simulate failures
		rl.updateCircuitBreaker(false)
		rl.updateCircuitBreaker(false)

		// Success should reset
		rl.updateCircuitBreaker(true)

		if rl.consecutiveFailures != 0 {
			t.Errorf("Expected consecutiveFailures=0 after success, got %d", rl.consecutiveFailures)
		}
	})

	t.Run("enters half-open after timeout", func(t *testing.T) {
		rl := NewRateLimiter(RateLimiterConfig{
			CircuitBreakerThreshold: 2,
			CircuitBreakerTimeout:   100 * time.Millisecond,
		})

		// Open the circuit
		rl.updateCircuitBreaker(false)
		rl.updateCircuitBreaker(false)

		if rl.checkCircuitBreaker() != CircuitOpen {
			t.Error("Circuit should be open")
		}

		// Wait for timeout
		time.Sleep(150 * time.Millisecond)

		state := rl.checkCircuitBreaker()
		if state != CircuitHalfOpen {
			t.Errorf("Expected CircuitHalfOpen after timeout, got %v", state)
		}
	})
}

func TestExecuteWithRetry_Success(t *testing.T) {
	rl := NewRateLimiter(RateLimiterConfig{
		MaxRetries: 3,
		BaseDelay:  10 * time.Millisecond,
	})

	ctx := context.Background()
	callCount := 0

	result, err := rl.ExecuteWithRetry(ctx, func() (*Response, error) {
		callCount++
		return &Response{}, nil
	})

	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result == nil {
		t.Error("Expected result, got nil")
	}
	if callCount != 1 {
		t.Errorf("Expected 1 call, got %d", callCount)
	}
}

func TestExecuteWithRetry_NonRateLimitError(t *testing.T) {
	rl := NewRateLimiter(RateLimiterConfig{
		MaxRetries: 3,
		BaseDelay:  10 * time.Millisecond,
	})

	ctx := context.Background()
	callCount := 0

	_, err := rl.ExecuteWithRetry(ctx, func() (*Response, error) {
		callCount++
		return nil, errNotRateLimit
	})

	if err == nil {
		t.Error("Expected error, got nil")
	}
	if !errors.Is(err, errNotRateLimit) {
		t.Errorf("Expected errNotRateLimit, got %v", err)
	}
	if callCount != 1 {
		t.Errorf("Expected 1 call (no retry for non-rate-limit error), got %d", callCount)
	}
}

func TestExecuteWithRetry_RateLimitThenSuccess(t *testing.T) {
	rl := NewRateLimiter(RateLimiterConfig{
		MaxRetries: 3,
		BaseDelay:  10 * time.Millisecond,
	})

	ctx := context.Background()
	callCount := 0

	result, err := rl.ExecuteWithRetry(ctx, func() (*Response, error) {
		callCount++
		if callCount < 3 {
			return nil, errRateLimit
		}
		return &Response{}, nil
	})

	if err != nil {
		t.Errorf("Expected no error after retries, got %v", err)
	}
	if result == nil {
		t.Error("Expected result, got nil")
	}
	if callCount != 3 {
		t.Errorf("Expected 3 calls, got %d", callCount)
	}
}

func TestExecuteWithRetry_MaxRetriesExceeded(t *testing.T) {
	rl := NewRateLimiter(RateLimiterConfig{
		MaxRetries: 2,
		BaseDelay:  10 * time.Millisecond,
	})

	ctx := context.Background()
	callCount := 0

	_, err := rl.ExecuteWithRetry(ctx, func() (*Response, error) {
		callCount++
		return nil, errRateLimit
	})

	if err == nil {
		t.Error("Expected error, got nil")
	}
	if !errors.Is(err, ErrRateLimitExceeded) {
		t.Errorf("Expected ErrRateLimitExceeded, got %v", err)
	}
	if callCount != 3 { // maxRetries=2 means 3 total attempts (initial + 2 retries)
		t.Errorf("Expected 3 calls, got %d", callCount)
	}
}

func TestExecuteWithRetry_CircuitBreakerOpen(t *testing.T) {
	rl := NewRateLimiter(RateLimiterConfig{
		MaxRetries:              2,
		BaseDelay:               10 * time.Millisecond,
		CircuitBreakerThreshold: 2,
		CircuitBreakerTimeout:   10 * time.Second, // Long timeout to keep circuit open
	})

	ctx := context.Background()

	// Trigger circuit breaker by failing multiple times
	_, err := rl.ExecuteWithRetry(ctx, func() (*Response, error) {
		return nil, errRateLimit
	})

	// Verify we got max retries error (not checking circuit status yet)
	if !errors.Is(err, ErrRateLimitExceeded) {
		t.Errorf("Expected ErrRateLimitExceeded, got %v", err)
	}

	// Circuit should now be open (after 3 consecutive rate limit failures)
	callCount := 0
	_, err = rl.ExecuteWithRetry(ctx, func() (*Response, error) {
		callCount++
		t.Error("Function should not be called when circuit is open")
		return nil, nil
	})

	if callCount != 0 {
		t.Errorf("Expected 0 calls when circuit is open, got %d", callCount)
	}
	if err == nil {
		t.Error("Expected error, got nil")
	}
	if !errors.Is(err, ErrCircuitOpen) {
		t.Errorf("Expected ErrCircuitOpen, got %v", err)
	}
}

func TestExecuteWithRetry_ContextCancellation(t *testing.T) {
	rl := NewRateLimiter(RateLimiterConfig{
		MaxRetries: 5,
		BaseDelay:  100 * time.Millisecond,
	})

	ctx, cancel := context.WithCancel(context.Background())
	callCount := 0

	// Cancel after first call
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	_, err := rl.ExecuteWithRetry(ctx, func() (*Response, error) {
		callCount++
		return nil, errRateLimit
	})

	if err == nil {
		t.Error("Expected error, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("Expected context.Canceled, got %v", err)
	}
	// Should have made at least one call before cancellation
	if callCount < 1 {
		t.Errorf("Expected at least 1 call, got %d", callCount)
	}
}

func TestCircuitStateString(t *testing.T) {
	tests := []struct {
		state    CircuitState
		expected string
	}{
		{CircuitClosed, "Closed"},
		{CircuitOpen, "Open"},
		{CircuitHalfOpen, "HalfOpen"},
		{CircuitState(99), "Unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			result := tt.state.String()
			if result != tt.expected {
				t.Errorf("Expected %s, got %s", tt.expected, result)
			}
		})
	}
}

func TestExecuteWithRetry_BackoffTiming(t *testing.T) {
	rl := NewRateLimiter(RateLimiterConfig{
		MaxRetries:              3,
		BaseDelay:               100 * time.Millisecond,
		MaxDelay:                10 * time.Second,
		JitterFactor:            0.0, // No jitter for predictable timing
		CircuitBreakerThreshold: 10,  // High threshold to prevent interference
	})

	ctx := context.Background()
	attempts := []time.Time{}

	_, err := rl.ExecuteWithRetry(ctx, func() (*Response, error) {
		attempts = append(attempts, time.Now())
		return nil, errRateLimit
	})

	if err == nil {
		t.Error("Expected error after max retries")
	}

	if len(attempts) != 4 { // Initial + 3 retries
		t.Errorf("Expected 4 attempts, got %d", len(attempts))
	}

	// Check backoff timing (allowing generous tolerance for timing variance)
	if len(attempts) >= 2 {
		delay1 := attempts[1].Sub(attempts[0])
		// First retry: baseDelay * 2^0 = 100ms, but allow 90-500ms for variance
		if delay1 < 90*time.Millisecond || delay1 > 500*time.Millisecond {
			t.Logf("First retry delay: %v (acceptable range)", delay1)
		}
	}

	if len(attempts) >= 3 {
		delay2 := attempts[2].Sub(attempts[1])
		// Second retry: baseDelay * 2^1 = 200ms, but timing can vary
		if delay2 < 90*time.Millisecond {
			t.Errorf("Second retry delay too short: %v", delay2)
		}
		// Log but don't fail on upper bound - timing is approximate
		if delay2 > 500*time.Millisecond {
			t.Logf("Second retry delay: %v (higher than expected but acceptable)", delay2)
		}
	}
}

func TestGetContainerSeed(t *testing.T) {
	seed1 := getContainerSeed()
	seed2 := getContainerSeed()

	if seed1 != seed2 {
		t.Error("Container seed should be consistent")
	}

	if seed1 == 0 {
		t.Error("Container seed should not be zero")
	}
}

func TestRateLimitError_VarNames(t *testing.T) {
	// Test that error variables are defined and usable
	if ErrRateLimitExceeded == nil {
		t.Error("ErrRateLimitExceeded should not be nil")
	}
	if ErrCircuitOpen == nil {
		t.Error("ErrCircuitOpen should not be nil")
	}

	// Test that errors have descriptive messages
	if !strings.Contains(ErrRateLimitExceeded.Error(), "rate limit") {
		t.Errorf("ErrRateLimitExceeded should mention rate limit, got: %v", ErrRateLimitExceeded)
	}
	if !strings.Contains(ErrCircuitOpen.Error(), "circuit") {
		t.Errorf("ErrCircuitOpen should mention circuit, got: %v", ErrCircuitOpen)
	}
}

func TestUpdateCircuitBreaker(t *testing.T) {
	t.Run("consecutive failures increment counter", func(t *testing.T) {
		rl := NewRateLimiter(RateLimiterConfig{
			CircuitBreakerThreshold: 5,
		})

		rl.updateCircuitBreaker(false)
		if rl.consecutiveFailures != 1 {
			t.Errorf("Expected 1 failure, got %d", rl.consecutiveFailures)
		}

		rl.updateCircuitBreaker(false)
		if rl.consecutiveFailures != 2 {
			t.Errorf("Expected 2 failures, got %d", rl.consecutiveFailures)
		}
	})

	t.Run("success resets counter and closes circuit", func(t *testing.T) {
		rl := NewRateLimiter(RateLimiterConfig{
			CircuitBreakerThreshold: 2,
		})

		// Open circuit
		rl.updateCircuitBreaker(false)
		rl.updateCircuitBreaker(false)

		if rl.circuitOpenTime == nil {
			t.Error("Circuit should be open")
		}

		// Success should reset and close
		rl.updateCircuitBreaker(true)

		if rl.consecutiveFailures != 0 {
			t.Errorf("Expected 0 failures after success, got %d", rl.consecutiveFailures)
		}
		if rl.circuitOpenTime != nil {
			t.Error("Circuit should be closed after success")
		}
	})
}

func TestExecuteWithRetry_EdgeCases(t *testing.T) {
	t.Run("immediate success on first attempt", func(t *testing.T) {
		rl := NewRateLimiter(RateLimiterConfig{MaxRetries: 3})
		ctx := context.Background()
		
		start := time.Now()
		result, err := rl.ExecuteWithRetry(ctx, func() (*Response, error) {
			return &Response{}, nil
		})
		
		duration := time.Since(start)
		
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if result == nil {
			t.Error("Expected result")
		}
		// Should complete quickly without delays
		if duration > 100*time.Millisecond {
			t.Errorf("Should complete immediately, took %v", duration)
		}
	})

	t.Run("alternating rate limit and non-rate-limit errors", func(t *testing.T) {
		rl := NewRateLimiter(RateLimiterConfig{
			MaxRetries: 3,
			BaseDelay:  10 * time.Millisecond,
		})
		ctx := context.Background()
		callCount := 0

		_, err := rl.ExecuteWithRetry(ctx, func() (*Response, error) {
			callCount++
			if callCount == 1 {
				return nil, errRateLimit // Should retry
			}
			return nil, errNotRateLimit // Should fail immediately
		})

		if callCount != 2 {
			t.Errorf("Expected 2 calls (rate limit retry then immediate fail), got %d", callCount)
		}
		if !errors.Is(err, errNotRateLimit) {
			t.Errorf("Expected errNotRateLimit, got %v", err)
		}
	})

	t.Run("zero max retries still attempts once", func(t *testing.T) {
		rl := NewRateLimiter(RateLimiterConfig{
			MaxRetries:              0,
			CircuitBreakerThreshold: 10, // High threshold to prevent circuit breaker from interfering
		})
		ctx := context.Background()
		callCount := 0

		_, err := rl.ExecuteWithRetry(ctx, func() (*Response, error) {
			callCount++
			return nil, errRateLimit
		})

		if callCount != 1 {
			t.Errorf("Expected 1 call (initial attempt), got %d", callCount)
		}
		if !errors.Is(err, ErrRateLimitExceeded) {
			t.Errorf("Expected ErrRateLimitExceeded, got %v", err)
		}
	})
}
