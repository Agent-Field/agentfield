# Rate Limiter Usage Guide

The Go SDK now includes built-in rate limiting with exponential backoff and circuit breaker patterns for production-grade AI API resilience.

## Quick Start

Rate limiting is **enabled by default** when you create an AI-enabled agent:

```go
agent, err := agentfield.New(agentfield.Config{
    NodeID: "my-agent",
    AIConfig: &ai.Config{
        Model:  "gpt-4o",
        APIKey: os.Getenv("OPENAI_API_KEY"),
        // Rate limiting is automatically enabled with sensible defaults
    },
})
```

## Default Behavior

When enabled, the rate limiter automatically:
- **Retries** up to 5 times on rate limit errors (429, 503)
- **Exponential backoff**: 1s → 2s → 4s → 8s → 16s → 30s (capped)
- **Jitter**: ±10% randomization to prevent thundering herd
- **Circuit breaker**: Opens after 5 consecutive failures, stays open for 60s

## Custom Configuration

Configure rate limiting behavior through `AIConfig`:

```go
agent, err := agentfield.New(agentfield.Config{
    NodeID: "my-agent",
    AIConfig: &ai.Config{
        Model:  "gpt-4o",
        APIKey: os.Getenv("OPENAI_API_KEY"),
        
        // Rate Limiting Configuration
        RateLimitMaxRetries:         10,              // More retries for critical operations
        RateLimitBaseDelay:          500 * time.Millisecond, // Faster initial retry
        RateLimitMaxDelay:           60 * time.Second,       // Higher max delay
        RateLimitJitterFactor:       0.2,              // More jitter (20%)
        
        // Circuit Breaker Configuration
        CircuitBreakerThreshold:     3,               // Open after 3 consecutive failures
        CircuitBreakerTimeout:       30 * time.Second, // Try again after 30s
    },
})
```

## Disabling Rate Limiting

For testing or special cases, you can disable rate limiting:

```go
agent, err := agentfield.New(agentfield.Config{
    NodeID: "my-agent",
    AIConfig: &ai.Config{
        Model:              "gpt-4o",
        APIKey:             os.Getenv("OPENAI_API_KEY"),
        DisableRateLimiter: true, // Disable rate limiting
    },
})
```

## How It Works

### Exponential Backoff

Each retry waits longer than the previous one:
- **Retry 1**: baseDelay × 2^0 = 1s
- **Retry 2**: baseDelay × 2^1 = 2s
- **Retry 3**: baseDelay × 2^2 = 4s
- **Retry 4**: baseDelay × 2^3 = 8s
- ...capped at `maxDelay`

### Jitter

Random variation (±10% by default) prevents all containers from retrying at the exact same time:
- Without jitter: 100 containers retry at exactly 2.0s → thundering herd
- With jitter: 100 containers retry between 1.8s-2.2s → distributed load

The jitter seed is container-specific (based on hostname + PID), ensuring consistent but distributed behavior.

### Circuit Breaker

Protects the system from repeatedly calling a failing service:

1. **Closed** (normal): All requests are attempted
2. **Open** (protecting): Requests are immediately rejected with `ErrCircuitOpen`
3. **Half-Open** (testing): After timeout, allows one test request

States:
- Opens after N consecutive rate limit failures
- Stays open for the configured timeout
- Closes on first successful request

## Error Handling

The rate limiter automatically detects rate limit errors by checking for:
- HTTP status codes: 429 (Too Many Requests), 503 (Service Unavailable)
- Keywords: "rate limit", "quota exceeded", "throttled", "rpm exceeded", etc.

### Error Types

```go
response, err := agent.AI(ctx, "Analyze this data")
if err != nil {
    if errors.Is(err, ai.ErrRateLimitExceeded) {
        // All retries exhausted - back off for longer or try later
        fmt.Println("Rate limit retries exhausted")
    } else if errors.Is(err, ai.ErrCircuitOpen) {
        // Circuit breaker is open - service is down/overloaded
        fmt.Println("Circuit breaker open, try again later")
    } else {
        // Other error (non-rate-limit)
        fmt.Println("Other error:", err)
    }
}
```

## Examples

### High-Priority Operations

For critical operations, increase retry attempts:

```go
agent, err := agentfield.New(agentfield.Config{
    NodeID: "critical-agent",
    AIConfig: &ai.Config{
        Model:  "gpt-4o",
        APIKey: os.Getenv("OPENAI_API_KEY"),
        
        RateLimitMaxRetries: 20,              // Very persistent
        RateLimitMaxDelay:   300 * time.Second, // Wait up to 5 minutes
    },
})
```

### Development/Testing

For dev environments, use faster retries or disable completely:

```go
agent, err := agentfield.New(agentfield.Config{
    NodeID: "dev-agent",
    AIConfig: &ai.Config{
        Model:              "gpt-4o",
        APIKey:             os.Getenv("OPENAI_API_KEY"),
        RateLimitBaseDelay: 100 * time.Millisecond, // Faster retries
        RateLimitMaxRetries: 2,                      // Fail fast
        // Or: DisableRateLimiter: true,
    },
})
```

### Distributed Systems

For systems with many containers, increase jitter:

```go
agent, err := agentfield.New(agentfield.Config{
    NodeID: "worker-node",
    AIConfig: &ai.Config{
        Model:  "gpt-4o",
        APIKey: os.Getenv("OPENAI_API_KEY"),
        
        RateLimitJitterFactor: 0.25, // 25% jitter for better distribution
    },
})
```

## Production Best Practices

1. **Monitor Circuit Breaker State**: Log when circuit opens/closes to detect service issues
2. **Tune for Your Provider**: Different LLM providers have different rate limits
3. **Set Reasonable Timeouts**: Balance between persistence and failing fast
4. **Use Context Cancellation**: Always pass context with timeout for request cancellation

```go
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
defer cancel()

response, err := agent.AI(ctx, "Long-running analysis")
if err != nil {
    if errors.Is(err, context.DeadlineExceeded) {
        fmt.Println("Operation timed out (including all retries)")
    }
}
```

## Comparison with Python SDK

The Go SDK rate limiter provides the same functionality as the Python SDK's `StatelessRateLimiter`:

| Feature | Python SDK | Go SDK |
|---------|-----------|--------|
| Exponential Backoff | ✅ | ✅ |
| Jitter | ✅ | ✅ |
| Circuit Breaker | ✅ | ✅ |
| Container-specific Seed | ✅ | ✅ |
| Rate Limit Detection | ✅ | ✅ |
| Configurable Thresholds | ✅ | ✅ |

## Implementation Notes

- **Thread-safe**: Safe for concurrent use across goroutines
- **Per-client state**: Each AI client has its own rate limiter instance
- **Stateless design**: No coordination needed between containers
- **Automatic**: Works for both `Complete()` and `StreamComplete()` calls
