package ai

import (
	"context"
	"crypto/md5"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// RateLimitError represents an error due to rate limiting or circuit breaker.
var ErrRateLimitExceeded = errors.New("rate limit retries exhausted")
var ErrCircuitOpen = errors.New("circuit breaker is open")

// CircuitState represents the state of the circuit breaker.
type CircuitState int

const (
	// CircuitClosed means requests are allowed.
	CircuitClosed CircuitState = iota
	// CircuitOpen means requests are blocked.
	CircuitOpen
	// CircuitHalfOpen means a test request is allowed.
	CircuitHalfOpen
)

// String returns the string representation of CircuitState.
func (s CircuitState) String() string {
	switch s {
	case CircuitClosed:
		return "Closed"
	case CircuitOpen:
		return "Open"
	case CircuitHalfOpen:
		return "HalfOpen"
	default:
		return "Unknown"
	}
}

// RateLimiter provides exponential backoff retry logic with circuit breaker pattern.
// It is safe for concurrent use by multiple goroutines.
type RateLimiter struct {
	maxRetries              int
	baseDelay               time.Duration
	maxDelay                time.Duration
	jitterFactor            float64
	circuitBreakerThreshold int
	circuitBreakerTimeout   time.Duration

	// Callbacks for observability
	onCircuitOpen  func()
	onCircuitClose func()

	// Circuit breaker state (protected by mu)
	mu                  sync.Mutex
	consecutiveFailures int
	circuitOpenTime     *time.Time
	containerSeed       int64
}

// NewRateLimiter creates a new RateLimiter with the given configuration.
// Configuration values are used as-is without applying defaults.
func NewRateLimiter(config RateLimiterConfig) *RateLimiter {
	return &RateLimiter{
		maxRetries:              config.MaxRetries,
		baseDelay:               config.BaseDelay,
		maxDelay:                config.MaxDelay,
		jitterFactor:            config.JitterFactor,
		circuitBreakerThreshold: config.CircuitBreakerThreshold,
		circuitBreakerTimeout:   config.CircuitBreakerTimeout,
		onCircuitOpen:           config.OnCircuitOpen,
		onCircuitClose:          config.OnCircuitClose,
		containerSeed:           getContainerSeed(),
	}
}

// RateLimiterConfig holds configuration for the rate limiter.
type RateLimiterConfig struct {
	MaxRetries              int           // Maximum number of retry attempts
	BaseDelay               time.Duration // Base delay for exponential backoff
	MaxDelay                time.Duration // Maximum delay between retries
	JitterFactor            float64       // Jitter factor (0.0-1.0) to prevent thundering herd
	CircuitBreakerThreshold int           // Number of consecutive failures before opening circuit
	CircuitBreakerTimeout   time.Duration // Time to wait before attempting to close circuit
	OnCircuitOpen           func()        // Callback when circuit opens (optional)
	OnCircuitClose          func()        // Callback when circuit closes (optional)
}

// getContainerSeed generates a container-specific seed for consistent jitter distribution.
func getContainerSeed() int64 {
	hostname := os.Getenv("HOSTNAME")
	if hostname == "" {
		hostname = "localhost"
	}
	pid := os.Getpid()
	identifier := fmt.Sprintf("%s-%d", hostname, pid)

	hash := md5.Sum([]byte(identifier))
	hexStr := hex.EncodeToString(hash[:])
	seed, _ := strconv.ParseInt(hexStr[:8], 16, 64)
	return seed
}

// isRateLimitError checks if an error is a rate limit error.
func isRateLimitError(err error) bool {
	if err == nil {
		return false
	}

	errMsg := strings.ToLower(err.Error())

	// Check for common rate limit keywords
	keywords := []string{
		"rate limit",
		"rate-limit",
		"rate_limit",
		"too many requests",
		"quota exceeded",
		"temporarily rate-limited",
		"rate limited",
		"requests per",
		"rpm exceeded",
		"tpm exceeded",
		"usage limit",
		"throttled",
		"throttling",
		"429", // HTTP 429 status
		"503", // HTTP 503 status (service unavailable, often due to rate limits)
	}

	for _, keyword := range keywords {
		if strings.Contains(errMsg, keyword) {
			return true
		}
	}

	return false
}

// calculateBackoffDelay calculates the delay with exponential backoff and jitter.
func (rl *RateLimiter) calculateBackoffDelay(attempt int) time.Duration {
	// Exponential backoff: baseDelay * (2 ^ attempt)
	exponent := math.Pow(2, float64(attempt))
	backoffDelay := time.Duration(float64(rl.baseDelay) * exponent)

	// Cap at max delay
	if backoffDelay > rl.maxDelay {
		backoffDelay = rl.maxDelay
	}

	// Add jitter to distribute load
	// Use time-based randomness combined with container seed for true randomness
	// while maintaining some distribution across containers
	rng := rand.New(rand.NewSource(rl.containerSeed + int64(attempt) + time.Now().UnixNano()))
	jitterRange := float64(backoffDelay) * rl.jitterFactor
	jitter := (rng.Float64()*2 - 1) * jitterRange // Random value between -jitterRange and +jitterRange

	delay := time.Duration(float64(backoffDelay) + jitter)

	// Ensure minimum delay
	if delay < 100*time.Millisecond {
		delay = 100 * time.Millisecond
	}

	return delay
}

// checkCircuitBreaker checks if the circuit breaker is open.
// Must be called with mu held.
func (rl *RateLimiter) checkCircuitBreaker() CircuitState {
	if rl.circuitOpenTime == nil {
		return CircuitClosed
	}

	// Check if circuit breaker timeout has passed
	if time.Since(*rl.circuitOpenTime) > rl.circuitBreakerTimeout {
		// Timeout passed - enter half-open state
		return CircuitHalfOpen
	}

	return CircuitOpen
}

// updateCircuitBreaker updates the circuit breaker state based on operation result.
func (rl *RateLimiter) updateCircuitBreaker(success bool) {
	rl.mu.Lock()
	wasOpen := rl.circuitOpenTime != nil

	if success {
		// Reset on success
		rl.consecutiveFailures = 0
		if rl.circuitOpenTime != nil {
			rl.circuitOpenTime = nil
			rl.mu.Unlock()
			// Trigger callback outside the lock
			if rl.onCircuitClose != nil {
				rl.onCircuitClose()
			}
			return
		}
	} else {
		// Increment failures
		rl.consecutiveFailures++

		// Open circuit if threshold reached
		if rl.consecutiveFailures >= rl.circuitBreakerThreshold && !wasOpen {
			now := time.Now()
			rl.circuitOpenTime = &now
			rl.mu.Unlock()
			// Trigger callback outside the lock
			if rl.onCircuitOpen != nil {
				rl.onCircuitOpen()
			}
			return
		}
	}

	rl.mu.Unlock()
}

// GetCircuitState returns the current state of the circuit breaker.
func (rl *RateLimiter) GetCircuitState() CircuitState {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	return rl.checkCircuitBreaker()
}

// GetConsecutiveFailures returns the current count of consecutive failures.
func (rl *RateLimiter) GetConsecutiveFailures() int {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	return rl.consecutiveFailures
}

// ExecuteWithRetry executes a function with rate limit retry logic.
func (rl *RateLimiter) ExecuteWithRetry(ctx context.Context, fn func() (*Response, error)) (*Response, error) {
	// Check circuit breaker
	rl.mu.Lock()
	circuitState := rl.checkCircuitBreaker()
	rl.mu.Unlock()
	
	if circuitState == CircuitOpen {
		return nil, fmt.Errorf("%w: too many consecutive rate limit failures, will retry after %v",
			ErrCircuitOpen, rl.circuitBreakerTimeout)
	}

	var lastErr error

	for attempt := 0; attempt <= rl.maxRetries; attempt++ {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Execute the function
		result, err := fn()

		if err == nil {
			// Success - update circuit breaker and return
			rl.updateCircuitBreaker(true)
			return result, nil
		}

		lastErr = err

		// Check if this is a rate limit error
		if !isRateLimitError(err) {
			// Not a rate limit error - return immediately
			return nil, err
		}

		// Update circuit breaker for rate limit failure
		rl.updateCircuitBreaker(false)

		// Check if we've exceeded max retries
		if attempt >= rl.maxRetries {
			break
		}

		// Calculate backoff delay
		delay := rl.calculateBackoffDelay(attempt)

		// Wait before retry
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(delay):
			// Continue to next attempt
		}
	}

	// All retries exhausted
	return nil, fmt.Errorf("%w: after %d attempts: %v", ErrRateLimitExceeded, rl.maxRetries+1, lastErr)
}

// ExecuteStreamWithRetry executes a streaming function with rate limit retry logic.
func (rl *RateLimiter) ExecuteStreamWithRetry(ctx context.Context, fn func() (<-chan StreamChunk, <-chan error)) (<-chan StreamChunk, <-chan error) {
	chunkCh := make(chan StreamChunk)
	errCh := make(chan error, 1)

	go func() {
		defer close(chunkCh)
		defer close(errCh)

		// Check circuit breaker
		rl.mu.Lock()
		circuitState := rl.checkCircuitBreaker()
		rl.mu.Unlock()
		
		if circuitState == CircuitOpen {
			errCh <- fmt.Errorf("%w: too many consecutive rate limit failures, will retry after %v",
				ErrCircuitOpen, rl.circuitBreakerTimeout)
			return
		}

		var lastErr error

		for attempt := 0; attempt <= rl.maxRetries; attempt++ {
			// Check context cancellation
			select {
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			default:
			}

		// Execute the streaming function
		resultChunkCh, resultErrCh := fn()

		// Forward chunks - prioritize reading all chunks before checking errors
		streamErr := error(nil)
		chunksDone := false
		
		for !chunksDone {
			select {
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			case chunk, ok := <-resultChunkCh:
				if !ok {
					// Chunk channel closed, now check for any errors
					chunksDone = true
					break
				}
				chunkCh <- chunk
			case err, ok := <-resultErrCh:
				if ok && err != nil {
					// Store error but continue reading chunks
					streamErr = err
				}
			}
		}

		// Now check if there was an error after all chunks are read
		if streamErr != nil {
			lastErr = streamErr

			// Check if this is a rate limit error
			if !isRateLimitError(streamErr) {
				// Not a rate limit error - forward and return
				errCh <- streamErr
				return
			}

			// Update circuit breaker for rate limit failure
			rl.updateCircuitBreaker(false)

			// Break inner loop to retry
			goto retry
		}

		// Stream completed successfully
		rl.updateCircuitBreaker(true)
		return

		retry:
			// Check if we've exceeded max retries
			if attempt >= rl.maxRetries {
				break
			}

			// Calculate backoff delay
			delay := rl.calculateBackoffDelay(attempt)

			// Wait before retry
			select {
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			case <-time.After(delay):
				// Continue to next attempt
			}
		}

		// All retries exhausted
		errCh <- fmt.Errorf("%w: after %d attempts: %v", ErrRateLimitExceeded, rl.maxRetries+1, lastErr)
	}()

	return chunkCh, errCh
}
