package ai

import (
	"context"
	"crypto/md5"
	"encoding/binary"
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

// ErrCircuitOpen is returned when the circuit breaker is open and the request
// is rejected without being attempted.
var ErrCircuitOpen = errors.New("circuit breaker is open")

// ErrMaxRetriesExceeded is returned when all retry attempts have been exhausted
// due to repeated rate-limit responses.
var ErrMaxRetriesExceeded = errors.New("max retries exceeded")

// CircuitState represents the state of the circuit breaker.
type CircuitState int

const (
	// CircuitClosed allows requests through normally.
	CircuitClosed CircuitState = iota
	// CircuitOpen rejects requests immediately without attempting them.
	CircuitOpen
	// CircuitHalfOpen allows a single test request to probe recovery.
	CircuitHalfOpen
)

// String returns a human-readable name for the circuit state.
func (s CircuitState) String() string {
	switch s {
	case CircuitClosed:
		return "closed"
	case CircuitOpen:
		return "open"
	case CircuitHalfOpen:
		return "half-open"
	default:
		return "unknown"
	}
}

// RateLimiterConfig configures a RateLimiter. Zero-value fields fall back to
// defaults via NewRateLimiter.
type RateLimiterConfig struct {
	// MaxRetries is the number of retry attempts after the initial call.
	MaxRetries int
	// BaseDelay is the starting backoff delay, doubled each attempt.
	BaseDelay time.Duration
	// MaxDelay caps the backoff delay.
	MaxDelay time.Duration
	// JitterFactor is the fraction (0..1) of random variation applied to each delay.
	JitterFactor float64
	// CircuitBreakerThreshold is the number of consecutive failures before the
	// circuit opens. Zero or negative disables the circuit breaker.
	CircuitBreakerThreshold int
	// CircuitBreakerTimeout is how long the circuit stays open before probing.
	CircuitBreakerTimeout time.Duration
}

// Default rate-limiter parameters. These mirror the Python and TypeScript SDKs.
const (
	defaultRateLimitMaxRetries     = 5
	defaultRateLimitBaseDelay      = 500 * time.Millisecond
	defaultRateLimitMaxDelay       = 30 * time.Second
	defaultRateLimitJitterFactor   = 0.25
	defaultCircuitBreakerThreshold = 5
	defaultCircuitBreakerTimeout   = 30 * time.Second
	rateLimiterMinDelay            = 100 * time.Millisecond
)

// RateLimiter provides adaptive exponential backoff with jitter and a circuit
// breaker for AI provider calls. It is stateless across processes: jitter is
// seeded per-container so many instances naturally distribute retry load.
//
// A zero RateLimiter is not usable; construct one with NewRateLimiter.
type RateLimiter struct {
	maxRetries              int
	baseDelay               time.Duration
	maxDelay                time.Duration
	jitterFactor            float64
	circuitBreakerThreshold int
	circuitBreakerTimeout   time.Duration

	containerSeed int64

	mu                  sync.Mutex
	consecutiveFailures int
	// openedAt is the time the circuit last opened. Zero when the circuit has
	// never opened or has since closed.
	openedAt time.Time
	// probing is true while a single half-open probe request is in flight.
	// It gates concurrent callers to exactly one probe after the open timeout.
	probing bool
	// open is true while the circuit is open or half-open (i.e. not closed).
	open bool

	// now and sleep are overridable for testing.
	now   func() time.Time
	sleep func(ctx context.Context, d time.Duration) error
}

// NewRateLimiter creates a RateLimiter, applying defaults for any zero-value
// config fields.
func NewRateLimiter(cfg RateLimiterConfig) *RateLimiter {
	if cfg.MaxRetries <= 0 {
		cfg.MaxRetries = defaultRateLimitMaxRetries
	}
	if cfg.BaseDelay <= 0 {
		cfg.BaseDelay = defaultRateLimitBaseDelay
	}
	if cfg.MaxDelay <= 0 {
		cfg.MaxDelay = defaultRateLimitMaxDelay
	}
	if cfg.JitterFactor < 0 {
		cfg.JitterFactor = 0
	}
	if cfg.JitterFactor == 0 {
		cfg.JitterFactor = defaultRateLimitJitterFactor
	}
	if cfg.CircuitBreakerThreshold == 0 {
		cfg.CircuitBreakerThreshold = defaultCircuitBreakerThreshold
	}
	if cfg.CircuitBreakerTimeout <= 0 {
		cfg.CircuitBreakerTimeout = defaultCircuitBreakerTimeout
	}

	return &RateLimiter{
		maxRetries:              cfg.MaxRetries,
		baseDelay:               cfg.BaseDelay,
		maxDelay:                cfg.MaxDelay,
		jitterFactor:            cfg.JitterFactor,
		circuitBreakerThreshold: cfg.CircuitBreakerThreshold,
		circuitBreakerTimeout:   cfg.CircuitBreakerTimeout,
		containerSeed:           containerSeed(),
		now:                     time.Now,
		sleep:                   sleepWithContext,
	}
}

// Execute runs fn with rate-limit retry logic. It retries on rate-limit errors
// (HTTP 429/503 or rate-limit keywords), applying exponential backoff with
// jitter. Non-rate-limit errors are returned immediately.
//
// It returns ErrCircuitOpen (wrapped) if the circuit breaker is open, and
// ErrMaxRetriesExceeded (wrapped, joined with the last error) when all retries
// are exhausted.
func (r *RateLimiter) Execute(ctx context.Context, fn func() error) error {
	if r == nil {
		return fn()
	}

	// Consult the circuit breaker. admit reports whether this call may proceed
	// and whether it was admitted as the single half-open probe.
	admitted, isProbe := r.admit()
	if !admitted {
		return fmt.Errorf("%w: too many consecutive rate-limit failures; retry after %s",
			ErrCircuitOpen, r.circuitBreakerTimeout)
	}

	var lastErr error
	for attempt := 0; attempt <= r.maxRetries; attempt++ {
		err := fn()
		if err == nil {
			r.onResult(isProbe, true)
			return nil
		}

		lastErr = err

		if !isRateLimitError(err) {
			// A non-rate-limit error does not reflect on provider health, so it
			// neither trips the breaker nor consumes the probe. Release the probe
			// slot so a subsequent caller can retry the half-open transition.
			r.releaseProbe(isProbe)
			return err
		}

		if attempt >= r.maxRetries {
			r.onResult(isProbe, false)
			break
		}

		delay := r.backoffDelay(attempt, retryAfterFromError(err))
		if sleepErr := r.sleep(ctx, delay); sleepErr != nil {
			// The caller went away before we could finish retrying. Release the
			// probe slot without a verdict so the breaker is not stuck half-open.
			r.releaseProbe(isProbe)
			return sleepErr
		}
	}

	return fmt.Errorf("%w after %d attempts: %w", ErrMaxRetriesExceeded, r.maxRetries, lastErr)
}

// State returns the current circuit breaker state. It is advisory only:
// concurrent callers may change the state immediately after this returns.
func (r *RateLimiter) State() CircuitState {
	if r == nil || r.circuitBreakerThreshold <= 0 {
		return CircuitClosed
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.stateLocked()
}

// stateLocked computes the circuit state; the caller must hold r.mu.
func (r *RateLimiter) stateLocked() CircuitState {
	if !r.open {
		return CircuitClosed
	}
	if r.now().Sub(r.openedAt) > r.circuitBreakerTimeout {
		return CircuitHalfOpen
	}
	return CircuitOpen
}

// admit decides whether a call may proceed and whether it is the single
// half-open probe. When the breaker is disabled it always admits (not a probe).
//
// Semantics:
//   - closed: admit, not a probe.
//   - open (within timeout): reject.
//   - half-open (timeout elapsed): admit exactly one probe; concurrent callers
//     are rejected until that probe reports back via onResult/releaseProbe.
func (r *RateLimiter) admit() (admitted bool, isProbe bool) {
	if r.circuitBreakerThreshold <= 0 {
		return true, false
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	switch r.stateLocked() {
	case CircuitClosed:
		return true, false
	case CircuitHalfOpen:
		if r.probing {
			// A probe is already in flight; fast-fail everyone else.
			return false, false
		}
		r.probing = true
		return true, true
	default: // CircuitOpen
		return false, false
	}
}

// onResult records the verdict of an admitted call and advances the breaker.
// success closes the circuit; a rate-limit failure (re)opens it. isProbe marks
// whether this call held the half-open probe slot, which is always released.
func (r *RateLimiter) onResult(isProbe, success bool) {
	if r.circuitBreakerThreshold <= 0 {
		return
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if isProbe {
		r.probing = false
	}

	if success {
		// Recovery: close the circuit and reset the failure count.
		r.consecutiveFailures = 0
		r.open = false
		r.openedAt = time.Time{}
		return
	}

	// Failure. A failed probe re-opens the circuit immediately (restart the
	// timeout); otherwise trip the breaker once the threshold is reached.
	r.consecutiveFailures++
	if isProbe || r.consecutiveFailures >= r.circuitBreakerThreshold {
		r.open = true
		r.openedAt = r.now()
	}
}

// releaseProbe clears the half-open probe slot without recording a verdict,
// used when an admitted probe neither succeeded nor produced a rate-limit
// failure (non-rate-limit error or context cancellation). This keeps the
// breaker from being wedged half-open with no probe in flight.
func (r *RateLimiter) releaseProbe(isProbe bool) {
	if r.circuitBreakerThreshold <= 0 || !isProbe {
		return
	}
	r.mu.Lock()
	r.probing = false
	r.mu.Unlock()
}

// backoffDelay computes the delay for a given 0-based attempt, honoring a
// server-suggested retryAfter when present and reasonable, and adding
// deterministic per-container jitter.
func (r *RateLimiter) backoffDelay(attempt int, retryAfter time.Duration) time.Duration {
	var base time.Duration
	if retryAfter > 0 && retryAfter <= r.maxDelay {
		base = retryAfter
	} else {
		scaled := float64(r.baseDelay) * math.Pow(2, float64(attempt))
		if scaled > float64(r.maxDelay) {
			base = r.maxDelay
		} else {
			base = time.Duration(scaled)
		}
	}

	// Deterministic jitter seeded per-container plus attempt, so instances
	// spread their retries without coordination.
	rng := rand.New(rand.NewSource(r.containerSeed + int64(attempt)))
	jitterRange := float64(base) * r.jitterFactor
	jitter := (rng.Float64()*2 - 1) * jitterRange

	delay := time.Duration(float64(base) + jitter)
	if delay < rateLimiterMinDelay {
		delay = rateLimiterMinDelay
	}
	return delay
}

// isRateLimitError reports whether err represents a provider rate-limit or
// transient overload condition worth retrying.
func isRateLimitError(err error) bool {
	if err == nil {
		return false
	}

	var apiErr *APIError
	if errors.As(err, &apiErr) {
		if apiErr.StatusCode == 429 || apiErr.StatusCode == 503 {
			return true
		}
	}

	msg := strings.ToLower(err.Error())
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
	}
	for _, kw := range keywords {
		if strings.Contains(msg, kw) {
			return true
		}
	}
	return false
}

// retryAfterFromError extracts a Retry-After hint from an APIError body when the
// provider surfaces one. It returns 0 when no usable value is present.
func retryAfterFromError(err error) time.Duration {
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		return 0
	}
	if apiErr.Code == "" {
		return 0
	}
	// Some providers surface the retry hint in the error Code as a bare number
	// of seconds. Parse conservatively; ignore anything non-numeric.
	if secs, parseErr := strconv.ParseFloat(strings.TrimSpace(apiErr.Code), 64); parseErr == nil && secs > 0 {
		return time.Duration(secs * float64(time.Second))
	}
	return 0
}

// containerSeed derives a stable per-process seed for jitter distribution.
func containerSeed() int64 {
	host, _ := os.Hostname()
	identifier := fmt.Sprintf("%s-%d", host, os.Getpid())
	sum := md5.Sum([]byte(identifier))
	return int64(binary.BigEndian.Uint32(sum[:4]))
}

// sleepWithContext sleeps for d or returns early if ctx is cancelled.
func sleepWithContext(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return ctx.Err()
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
