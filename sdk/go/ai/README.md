# Go SDK AI Package

This package provides AI/LLM capabilities for the Haxen Go SDK, supporting both OpenAI and OpenRouter APIs with structured output support.

## Features

- ✅ **OpenAI & OpenRouter Support**: Works with both OpenAI API and OpenRouter for multi-model routing
- ✅ **Structured Outputs**: JSON schema validation with Go struct support
- ✅ **Streaming**: Support for streaming responses
- ✅ **Type-Safe**: Automatic conversion from Go structs to JSON schemas
- ✅ **Functional Options**: Clean, idiomatic Go API with functional options pattern
- ✅ **Automatic Configuration**: Reads from environment variables by default

## Quick Start

### Basic Usage

```go
import (
    "context"
    "github.com/agentfield/haxen/sdk/go/agent"
    "github.com/agentfield/haxen/sdk/go/ai"
)

// Create agent with AI configured
aiConfig := ai.DefaultConfig() // Reads from env vars
agent, err := agent.New(agent.Config{
    NodeID:   "my-agent",
    Version:  "1.0.0",
    HaxenURL: "http://localhost:8080",
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

## API Reference

### AI Client

#### `ai.NewClient(config *Config) (*Client, error)`
Creates a new AI client with the given configuration.

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

| Feature | Python SDK | Go SDK |
|---------|-----------|--------|
| Simple text calls | `agent.ai("prompt")` | `agent.AI(ctx, "prompt")` |
| System prompts | `system="..."` kwarg | `ai.WithSystem("...")` option |
| Structured outputs | `schema=Model` kwarg | `ai.WithSchema(Model{})` option |
| Streaming | `stream=True` kwarg | `agent.AIStream()` method |
| Model override | `model="..."` kwarg | `ai.WithModel("...")` option |
| Temperature | `temperature=0.7` kwarg | `ai.WithTemperature(0.7)` option |

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
