package ai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
)

// Message represents a chat message.

// Request represents an AI completion request.
type Request struct {
	// Messages for the chat completion
	Messages []Message `json:"messages"`

	// APIKeyOverride overrides the client's configured API key for this request only.
	// This is used to support per-call api_key overrides for parity with the Python SDK.
	APIKeyOverride string `json:"-"`

	// Model to use (overrides default)
	Model string `json:"model,omitempty"`

	// Temperature (0.0 to 2.0)
	Temperature *float64 `json:"temperature,omitempty"`

	// Maximum tokens to generate
	MaxTokens *int `json:"max_tokens,omitempty"`

	// Enable streaming
	Stream bool `json:"stream,omitempty"`

	// Response format for structured outputs
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`

	// Tools available for the model to call
	Tools []ToolDefinition `json:"tools,omitempty"`

	// ToolChoice controls how the model selects tools ("auto", "none", or specific)
	ToolChoice interface{} `json:"tool_choice,omitempty"`
}

// ToolDefinition describes a tool available to the model.
type ToolDefinition struct {
	Type     string       `json:"type"` // "function"
	Function ToolFunction `json:"function"`
}

// ToolFunction describes the function within a tool definition.
type ToolFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// ToolCall represents a tool call made by the model.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"` // "function"
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction contains the function name and arguments from a tool call.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type Message struct {
	Role       string        `json:"role"`
	Content    []ContentPart `json:"content"`
	ToolCalls  []ToolCall    `json:"tool_calls,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
}

type ContentPart struct {
	Type     string        `json:"type"` // "text" or "image_url"
	Text     string        `json:"text,omitempty"`
	ImageURL *ImageURLData `json:"image_url,omitempty"`
}

// ImageURLData holds the URL and optional detail level for image content parts.
type ImageURLData struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// MarshalJSON serializes a Message. If the content is a single text part,
// it serializes content as a plain string for maximum API compatibility.
func (m Message) MarshalJSON() ([]byte, error) {
	// Tool result messages have a simple string content + tool_call_id
	if m.Role == "tool" && m.ToolCallID != "" {
		content := ""
		if len(m.Content) == 1 && m.Content[0].Type == "text" {
			content = m.Content[0].Text
		}
		return json.Marshal(struct {
			Role       string `json:"role"`
			Content    string `json:"content"`
			ToolCallID string `json:"tool_call_id"`
		}{Role: m.Role, Content: content, ToolCallID: m.ToolCallID})
	}
	// Assistant messages with tool calls
	if m.Role == "assistant" && len(m.ToolCalls) > 0 {
		content := ""
		if len(m.Content) == 1 && m.Content[0].Type == "text" {
			content = m.Content[0].Text
		}
		return json.Marshal(struct {
			Role      string     `json:"role"`
			Content   *string    `json:"content"`
			ToolCalls []ToolCall `json:"tool_calls"`
		}{Role: m.Role, Content: &content, ToolCalls: m.ToolCalls})
	}
	if len(m.Content) == 1 && m.Content[0].Type == "text" && m.Content[0].ImageURL == nil {
		return json.Marshal(struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		}{Role: m.Role, Content: m.Content[0].Text})
	}
	type Alias Message
	return json.Marshal((Alias)(m))
}

func (m *Message) UnmarshalJSON(data []byte) error {
	// First, extract non-content fields
	aux := &struct {
		Role       string          `json:"role"`
		Content    json.RawMessage `json:"content"`
		ToolCalls  []ToolCall      `json:"tool_calls,omitempty"`
		ToolCallID string          `json:"tool_call_id,omitempty"`
	}{}

	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}

	m.Role = aux.Role
	m.ToolCalls = aux.ToolCalls
	m.ToolCallID = aux.ToolCallID

	// Handle null content (common in tool-call assistant messages)
	if len(aux.Content) == 0 || string(aux.Content) == "null" {
		return nil
	}

	var s string
	if err := json.Unmarshal(aux.Content, &s); err == nil {
		m.Content = []ContentPart{{Type: "text", Text: s}}
		return nil
	}

	var arr []ContentPart
	if err := json.Unmarshal(aux.Content, &arr); err != nil {
		return err
	}
	m.Content = arr
	return nil
}

// ResponseFormat specifies the desired output format.
type ResponseFormat struct {
	Type       string      `json:"type"` // "json_object" or "json_schema"
	JSONSchema *JSONSchema `json:"json_schema,omitempty"`
}

// JSONSchema defines the structure for structured outputs.
type JSONSchema struct {
	Name   string          `json:"name"`
	Strict bool            `json:"strict"`
	Schema json.RawMessage `json:"schema"`
}

// Option is a functional option for configuring an AI request.
type Option func(*Request) error

// WithSystem adds a system message to the request.
func WithSystem(content string) Option {
	return func(r *Request) error {
		r.Messages = append([]Message{
			{
				Role: "system",
				Content: []ContentPart{
					{Type: "text", Text: content},
				},
			},
		}, r.Messages...)
		return nil
	}
}

// WithModel overrides the default model.
func WithModel(model string) Option {
	return func(r *Request) error {
		r.Model = model
		return nil
	}
}

// WithAPIKey overrides the client's configured API key for this request only.
func WithAPIKey(apiKey string) Option {
	return func(r *Request) error {
		r.APIKeyOverride = apiKey
		return nil
	}
}

// WithTemperature sets the temperature.
func WithTemperature(temp float64) Option {
	return func(r *Request) error {
		r.Temperature = &temp
		return nil
	}
}

// WithMaxTokens sets the maximum tokens.
func WithMaxTokens(tokens int) Option {
	return func(r *Request) error {
		r.MaxTokens = &tokens
		return nil
	}
}

// WithStream enables streaming responses.
func WithStream() Option {
	return func(r *Request) error {
		r.Stream = true
		return nil
	}
}

// WithJSONMode enables JSON object mode (non-strict).
func WithJSONMode() Option {
	return func(r *Request) error {
		r.ResponseFormat = &ResponseFormat{
			Type: "json_object",
		}
		return nil
	}
}

// WithSchema enables structured output with a JSON schema.
// Accepts either a Go struct (will be converted to JSON schema) or json.RawMessage.
func WithSchema(schema interface{}) Option {
	return func(r *Request) error {
		var schemaBytes json.RawMessage
		var schemaName string

		switch v := schema.(type) {
		case json.RawMessage:
			schemaBytes = v
			schemaName = "response"
		case []byte:
			schemaBytes = json.RawMessage(v)
			schemaName = "response"
		case string:
			schemaBytes = json.RawMessage(v)
			schemaName = "response"
		default:
			// Convert Go struct to JSON schema
			schemaMap, name, err := structToJSONSchema(v)
			if err != nil {
				return fmt.Errorf("convert schema: %w", err)
			}
			schemaBytes, err = json.Marshal(schemaMap)
			if err != nil {
				return fmt.Errorf("marshal schema: %w", err)
			}
			schemaName = name
		}

		r.ResponseFormat = &ResponseFormat{
			Type: "json_schema",
			JSONSchema: &JSONSchema{
				Name:   schemaName,
				Strict: true,
				Schema: schemaBytes,
			},
		}
		return nil
	}
}

// WithTools sets tool definitions for the request.
func WithTools(tools []ToolDefinition) Option {
	return func(r *Request) error {
		r.Tools = tools
		r.ToolChoice = "auto"
		return nil
	}
}

// Image options
func WithImageFile(path string) Option {
	return func(r *Request) error {
		data, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("read image file: %w", err)
		}

		mimeType := detectMIMEType(path)
		encoded := base64.StdEncoding.EncodeToString(data)

		if len(r.Messages) == 0 {
			r.Messages = append(r.Messages, Message{
				Role:    "user",
				Content: []ContentPart{},
			})
		}

		last := &r.Messages[len(r.Messages)-1]
		last.Content = append(last.Content, ContentPart{
			Type: "image_url",
			ImageURL: &ImageURLData{
				URL: "data:" + mimeType + ";base64," + encoded,
			},
		})

		return nil
	}
}

// WithImageURL attaches an image from a remote URL.
func WithImageURL(url string) Option {
	return func(r *Request) error {
		if len(r.Messages) == 0 {
			r.Messages = append(r.Messages, Message{
				Role:    "user",
				Content: []ContentPart{},
			})
		}

		last := &r.Messages[len(r.Messages)-1]
		last.Content = append(last.Content, ContentPart{
			Type: "image_url",
			ImageURL: &ImageURLData{
				URL: url,
			},
		})

		return nil
	}
}

// WithImageBytes attaches an image from raw bytes (SDK encodes automatically).
func WithImageBytes(data []byte, mimeType string) Option {
	return func(r *Request) error {
		if len(data) == 0 {
			return nil
		}

		encoded := base64.StdEncoding.EncodeToString(data)

		if len(r.Messages) == 0 {
			r.Messages = append(r.Messages, Message{
				Role:    "user",
				Content: []ContentPart{},
			})
		}

		last := &r.Messages[len(r.Messages)-1]
		last.Content = append(last.Content, ContentPart{
			Type: "image_url",
			ImageURL: &ImageURLData{
				URL: "data:" + mimeType + ";base64," + encoded,
			},
		})

		return nil
	}
}

// structToJSONSchema converts a Go struct to a JSON schema.
// This is a simplified version - you may want to use a library like
// github.com/invopop/jsonschema for production.
func structToJSONSchema(v interface{}) (map[string]interface{}, string, error) {
	t := reflect.TypeOf(v)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		return nil, "", fmt.Errorf("schema must be a struct, got %v", t.Kind())
	}

	schemaName := t.Name()
	if schemaName == "" {
		schemaName = "response"
	}

	properties := make(map[string]interface{})
	required := []string{}

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		jsonTag := field.Tag.Get("json")
		if jsonTag == "" || jsonTag == "-" {
			continue
		}

		// Parse json tag (e.g., "name,omitempty")
		fieldName := jsonTag
		isRequired := true
		if idx := len(jsonTag); idx > 0 {
			for j, c := range jsonTag {
				if c == ',' {
					fieldName = jsonTag[:j]
					if len(jsonTag) > j+1 && jsonTag[j+1:] == "omitempty" {
						isRequired = false
					}
					break
				}
			}
		}

		// Build property schema
		prop := make(map[string]interface{})
		prop["type"] = goTypeToJSONType(field.Type)

		// Add description from struct tag if present
		if desc := field.Tag.Get("description"); desc != "" {
			prop["description"] = desc
		}

		properties[fieldName] = prop

		if isRequired {
			required = append(required, fieldName)
		}
	}

	schema := map[string]interface{}{
		"type":                 "object",
		"properties":           properties,
		"required":             required,
		"additionalProperties": false,
	}

	return schema, schemaName, nil
}

// goTypeToJSONType converts Go types to JSON schema types.
func goTypeToJSONType(t reflect.Type) string {
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	switch t.Kind() {
	case reflect.String:
		return "string"
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return "integer"
	case reflect.Float32, reflect.Float64:
		return "number"
	case reflect.Bool:
		return "boolean"
	case reflect.Slice, reflect.Array:
		return "array"
	case reflect.Map, reflect.Struct:
		return "object"
	default:
		return "string"
	}
}
