# Go SDK AI Package

This package provides AI/LLM capabilities for the AgentField Go SDK, supporting both OpenAI and OpenRouter APIs with structured output support.

## Features

- ✅ **OpenAI & OpenRouter Support**: Works with both OpenAI API and OpenRouter for multi-model routing
- ✅ **Structured Outputs**: JSON schema validation with Go struct support
- ✅ **Streaming**: Support for streaming responses
- ✅ **Type-Safe**: Automatic conversion from Go structs to JSON schemas
- ✅ **Functional Options**: Clean, idiomatic Go API with functional options pattern
- ✅ **Rate Limiting**: Automatic exponential backoff with jitter and a circuit breaker
- ✅ **Automatic Configuration**: Reads from environment variables by default

## Quick Start

### Basic Usage

```go
import (
    "context"
    "github.com/Agent-Field/agentfield/sdk/go/agent"
    "github.com/Agent-Field/agentfield/sdk/go/ai"
)

// Create agent with AI configured
aiConfig := ai.DefaultConfig() // Reads from env vars
agent, err := agent.New(agent.Config{
    NodeID:   "my-agent",
    Version:  "1.0.0",
    AgentFieldURL: "http://localhost:8080",
    AIConfig: aiConfig,
})

// Make a simple AI call
response, err := agent.AI(ctx, "What is 2+2?")
fmt.Println(response.Text())
```

### Structured Outputs

```go
// Define your response schema
type WeatherResponse struct {
    Location    string  `json:"location" description:"City name"`
    Temperature float64 `json:"temperature" description:"Temperature in Celsius"`
    Conditions  string  `json:"conditions" description:"Weather conditions"`
}

// Call AI with schema
response, err := agent.AI(ctx, "What's the weather in Paris?",
    ai.WithSystem("You are a weather assistant"),
    ai.WithSchema(WeatherResponse{}))

// Parse response into struct
var weather WeatherResponse
response.Into(&weather)
fmt.Printf("%s: %.1f°C, %s\n", weather.Location, weather.Temperature, weather.Conditions)
```

### Streaming Responses

```go
chunks, errs := agent.AIStream(ctx, "Tell me a story")
for chunk := range chunks {
    if len(chunk.Choices) > 0 {
        fmt.Print(chunk.Choices[0].Delta.Content)
    }
}
if err := <-errs; err != nil {
    log.Fatal(err)
}
```

## Configuration

### Environment Variables

The SDK automatically reads from environment variables:

```bash
# For OpenAI (default)
export OPENAI_API_KEY="sk-..."
export AI_MODEL="gpt-4o"  # Optional, defaults to gpt-4o

# For OpenRouter
export OPENROUTER_API_KEY="sk-..."
export AI_MODEL="openai/gpt-4o"  # Use OpenRouter model format

# Custom base URL
export AI_BASE_URL="https://api.openai.com/v1"
```

### Manual Configuration

```go
aiConfig := &ai.Config{
    APIKey:      "sk-...",
    BaseURL:     "https://api.openai.com/v1",
    Model:       "gpt-4o",
    Temperature: 0.7,
    MaxTokens:   4096,
}
```

### OpenRouter Configuration

```go
aiConfig := &ai.Config{
    APIKey:   os.Getenv("OPENROUTER_API_KEY"),
    BaseURL:  "https://openrouter.ai/api/v1",
    Model:    "anthropic/claude-3.5-sonnet", // OpenRouter format: provider/model
    SiteURL:  "https://myapp.com",  // For OpenRouter rankings
    SiteName: "My AI App",
}
```

### Infron Configuration

Infron is an OpenAI-compatible gateway that serves the standard
`<provider>/<model>` ids, so wiring it up is a base URL and a model prefix:

```go
aiConfig := &ai.Config{
    APIKey:   os.Getenv("INFRON_API_KEY"),
    BaseURL:  "https://llm.onerouter.pro/v1",
    Model:    "moonshotai/kimi-k2.6", // the id as published by the model vendor
    SiteURL:  "https://myapp.com",    // app attribution
    SiteName: "My AI App",
}
```

`ai.DefaultConfig()` picks this up from `INFRON_API_KEY` automatically. A
gateway key already present in the environment keeps precedence.

An `infron/` model prefix is accepted as a routing marker for callers that
select a gateway by model string, and is stripped before the request goes out
(the gateway serves the bare id):

```go
Model: "infron/moonshotai/kimi-k2.6"  // sent as moonshotai/kimi-k2.6
```

Note that Infron reports native cost at the top level of the body (and of the
final stream chunk) rather than nested under `usage.cost`. The SDK normalizes
both shapes into `Usage.Cost`, so cost tracking reads the same either way.

### Rate Limiting & Circuit Breaker

The client can automatically retry rate-limited AI calls (HTTP 429/503, or
provider errors whose message contains a rate-limit keyword) using exponential
backoff with jitter, and stop issuing requests once a provider is clearly down.

Rate limiting is opt-in: it is enabled only when `RateLimitMaxRetries > 0`.

```go
aiConfig := &ai.Config{
    APIKey: os.Getenv("OPENAI_API_KEY"),
    Model:  "gpt-4o",

    // Rate limiting
    RateLimitMaxRetries:   5,               // retries after the initial call
    RateLimitBaseDelay:    time.Second,     // starting delay, doubled per attempt
    RateLimitMaxDelay:     30 * time.Second, // delay ceiling
    RateLimitJitterFactor: 0.25,            // +/-25% random variation

    // Circuit breaker
    CircuitBreakerThreshold: 5,                // open after 5 consecutive failures
    CircuitBreakerTimeout:   60 * time.Second, // probe recovery after 60s
}

resp, err := agent.AI(ctx, "Hello, world!")
if err != nil {
    switch {
    case errors.Is(err, ai.ErrCircuitOpen):
        // Too many consecutive failures; back off and try again later.
    case errors.Is(err, ai.ErrMaxRetriesExceeded):
        // All retries were exhausted. The last provider error is wrapped.
    }
}
```

**Behavior**

1. **Exponential backoff**: attempt `n` waits `BaseDelay * 2^n`, capped at
   `MaxDelay`. A server-provided retry hint is honored when present and within
   `MaxDelay`.
2. **Jitter**: each delay is varied by `+/- JitterFactor`, seeded per process so
   many instances spread their retries without coordination.
3. **Circuit breaker**: opens after `CircuitBreakerThreshold` consecutive
   rate-limit failures. While open, calls fail fast with `ai.ErrCircuitOpen`.
   After `CircuitBreakerTimeout` the circuit becomes half-open and allows one
   probe; a success closes it. Set `CircuitBreakerThreshold` to a negative value
   to disable the breaker while keeping retries.

Any zero-value field falls back to a sensible default (5 retries, 500ms base,
30s max, 0.25 jitter, threshold 5, 30s timeout).

## API Reference

### AI Client

#### `ai.NewClient(config *Config, opts ...ClientOption) (*Client, error)`
Creates a new AI client with the given configuration.

Use `ai.WithHTTPClient` to provide a custom `*http.Client`, or
`ai.WithTransport` to provide a custom `http.RoundTripper` while retaining the
default client's timeout.

#### `client.Complete(ctx context.Context, prompt string, opts ...Option) (*Response, error)`
Makes a chat completion request.

#### `client.StreamComplete(ctx context.Context, prompt string, opts ...Option) (<-chan StreamChunk, <-chan error)`
Makes a streaming chat completion request.

### Agent Methods

#### `agent.AI(ctx context.Context, prompt string, opts ...Option) (*Response, error)`
Makes an AI call using the agent's configured AI client.

#### `agent.AIStream(ctx context.Context, prompt string, opts ...Option) (<-chan StreamChunk, <-chan error)`
Makes a streaming AI call.

### Options

Functional options for customizing AI requests:

- `ai.WithSystem(content string)` - Add a system prompt
- `ai.WithModel(model string)` - Override the default model
- `ai.WithTemperature(temp float64)` - Set temperature (0.0-2.0)
- `ai.WithMaxTokens(tokens int)` - Set max tokens
- `ai.WithStream()` - Enable streaming
- `ai.WithJSONMode()` - Enable JSON object mode
- `ai.WithSchema(schema interface{})` - Enable structured outputs with schema
##### Multimodal
- `ai.WithImageFile(path string)` - Attach an image from a local file
- `ai.WithImageURL(url string)` - Attach an image from a remote URL
- `ai.WithImageBytes(data []byte, mimeType string)` - Add an image from raw bytes (SDK encodes automatically)

### Multimodal Inputs (Images)

You can attach images files to AI requests.

```go
// Image from file
response, _ := agent.AI(ctx, "Describe this image",
    ai.WithImageFile("./photo.jpg"),
)

// Image from URL
response, _ = agent.AI(ctx, "Describe this image",
    ai.WithImageURL("https://example.com/image.jpg"),
)

// Image from bytes
data, _ := os.ReadFile("image.png")
response, _ = agent.AI(ctx, "What's in this image?",
    ai.WithImageBytes(data, "image/png"),
)
```

### Response Methods

- `response.Text()` - Get the text content
- `response.JSON(dest interface{})` - Parse response as JSON
- `response.Into(dest interface{})` - Alias for JSON()

## Structured Output Schema

Define schemas using Go structs with JSON tags:

```go
type MyResponse struct {
    // Required field
    Name string `json:"name" description:"The person's name"`

    // Optional field (use omitempty)
    Age int `json:"age,omitempty" description:"The person's age"`

    // Array
    Hobbies []string `json:"hobbies" description:"List of hobbies"`

    // Nested object
    Address struct {
        Street string `json:"street"`
        City   string `json:"city"`
    } `json:"address"`
}
```

The SDK automatically converts Go structs to JSON schemas compatible with OpenAI's structured output format.

## Using AI in Reasoners

You can use AI within your reasoners to create intelligent workflows:

```go
agent.RegisterReasoner("smart_reasoner", func(ctx context.Context, input map[string]any) (any, error) {
    question := input["question"].(string)

    // Call AI within the reasoner
    response, err := agent.AI(ctx, question,
        ai.WithSystem("You are a helpful assistant"),
        ai.WithTemperature(0.7))
    if err != nil {
        return nil, err
    }

    return map[string]any{
        "answer": response.Text(),
        "model":  response.Model,
    }, nil
})
```

## Comparison with Python SDK

The Go SDK provides similar functionality to the Python SDK's `agent.ai()` method:

| Feature            | Python SDK              | Go SDK                           |
| ------------------ | ----------------------- | -------------------------------- |
| Simple text calls  | `agent.ai("prompt")`    | `agent.AI(ctx, "prompt")`        |
| System prompts     | `system="..."` kwarg    | `ai.WithSystem("...")` option    |
| Structured outputs | `schema=Model` kwarg    | `ai.WithSchema(Model{})` option  |
| Streaming          | `stream=True` kwarg     | `agent.AIStream()` method        |
| Model override     | `model="..."` kwarg     | `ai.WithModel("...")` option     |
| Temperature        | `temperature=0.7` kwarg | `ai.WithTemperature(0.7)` option |

## Error Handling

```go
response, err := agent.AI(ctx, "prompt")
if err != nil {
    // Handle API errors
    log.Printf("AI call failed: %v", err)
    return
}

// Parse structured response
var result MyStruct
if err := response.Into(&result); err != nil {
    // Handle parsing errors
    log.Printf("Failed to parse response: %v", err)
    return
}
```

## Performance Considerations

1. **Connection Pooling**: The HTTP client uses connection pooling for efficient requests
2. **Context Cancellation**: Always use contexts with timeouts for AI calls
3. **Streaming**: Use streaming for long responses to improve perceived latency
4. **Model Selection**: Choose appropriate models for your use case (faster models = lower latency)

## Examples

See the [examples/ai-agent](../examples/ai-agent/) directory for complete examples including:
- Simple text responses
- Structured outputs
- Sentiment analysis
- Agent reasoners with AI
- Streaming responses
- OpenRouter usage

## License

Proprietary - Authorized users only
