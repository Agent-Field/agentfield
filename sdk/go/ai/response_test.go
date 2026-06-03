package ai

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestResponse_Text(t *testing.T) {
	tests := []struct {
		name     string
		response *Response
		expected string
	}{
		{
			name: "single choice with content",
			response: &Response{
				Choices: []Choice{
					{
						Message: Message{
							Role: "assistant",
							Content: []ContentPart{
								{Type: "text", Text: "Hello, world!"},
							},
						},
					},
				},
			},
			expected: "Hello, world!",
		},
		{
			name: "empty choices",
			response: &Response{
				Choices: []Choice{},
			},
			expected: "",
		},
		{
			name: "nil choices",
			response: &Response{
				Choices: nil,
			},
			expected: "",
		},
		{
			name: "multiple choices returns first",
			response: &Response{
				Choices: []Choice{
					{
						Message: Message{
							Content: []ContentPart{
								{Type: "text", Text: "First"},
							},
						},
					},
					{
						Message: Message{
							Content: []ContentPart{
								{Type: "text", Text: "Second"},
							},
						},
					},
				},
			},
			expected: "First",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.response.Text()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestResponse_JSON(t *testing.T) {
	tests := []struct {
		name        string
		response    *Response
		dest        interface{}
		wantErr     bool
		checkResult func(t *testing.T, dest interface{})
	}{
		{
			name: "valid JSON content",
			response: &Response{
				Choices: []Choice{
					{
						Message: Message{
							Content: []ContentPart{
								{Type: "text", Text: `{"name":"John","age":30}`},
							},
						},
					},
				},
			},
			dest: &struct {
				Name string `json:"name"`
				Age  int    `json:"age"`
			}{},
			wantErr: false,
			checkResult: func(t *testing.T, dest interface{}) {
				obj := dest.(*struct {
					Name string `json:"name"`
					Age  int    `json:"age"`
				})
				assert.Equal(t, "John", obj.Name)
				assert.Equal(t, 30, obj.Age)
			},
		},
		{
			name: "empty content",
			response: &Response{
				Choices: []Choice{
					{
						Message: Message{
							Content: []ContentPart{
								{Type: "text", Text: ""},
							},
						},
					},
				},
			},
			dest:    &map[string]interface{}{},
			wantErr: true,
		},
		{
			name: "invalid JSON",
			response: &Response{
				Choices: []Choice{
					{
						Message: Message{
							Content: []ContentPart{
								{Type: "text", Text: "not json"},
							},
						},
					},
				},
			},
			dest:    &map[string]interface{}{},
			wantErr: true,
		},
		{
			name: "no choices",
			response: &Response{
				Choices: []Choice{},
			},
			dest:    &map[string]interface{}{},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.response.JSON(tt.dest)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				if tt.checkResult != nil {
					tt.checkResult(t, tt.dest)
				}
			}
		})
	}
}

func TestResponse_Into(t *testing.T) {
	response := &Response{
		Choices: []Choice{
			{
				Message: Message{
					Content: []ContentPart{
						{Type: "text", Text: `{"value":42}`},
					},
				},
			},
		},
	}

	var dest struct {
		Value int `json:"value"`
	}

	err := response.Into(&dest)
	assert.NoError(t, err)
	assert.Equal(t, 42, dest.Value)
}

func TestErrorResponse(t *testing.T) {
	errResp := ErrorResponse{
		Error: ErrorDetail{
			Message: "Invalid API key",
			Type:    "invalid_request_error",
			Code:    "invalid_api_key",
		},
	}

	// Verify structure
	assert.Equal(t, "Invalid API key", errResp.Error.Message)
	assert.Equal(t, "invalid_request_error", errResp.Error.Type)
	assert.Equal(t, "invalid_api_key", errResp.Error.Code)
}

func TestStreamChunk(t *testing.T) {
	chunk := StreamChunk{
		ID:      "chatcmpl-123",
		Object:  "chat.completion.chunk",
		Created: 1234567890,
		Model:   "gpt-4o",
		Choices: []StreamDelta{
			{
				Index: 0,
				Delta: MessageDelta{
					Content: "Hello",
				},
			},
		},
	}

	assert.Equal(t, "chatcmpl-123", chunk.ID)
	assert.Equal(t, "chat.completion.chunk", chunk.Object)
	assert.Len(t, chunk.Choices, 1)
	assert.Equal(t, "Hello", chunk.Choices[0].Delta.Content)
}

func TestStreamDelta_WithFinishReason(t *testing.T) {
	reason := "stop"
	delta := StreamDelta{
		Index: 0,
		Delta: MessageDelta{
			Content: "done",
		},
		FinishReason: &reason,
	}

	assert.NotNil(t, delta.FinishReason)
	assert.Equal(t, "stop", *delta.FinishReason)
}

func TestMessageDelta(t *testing.T) {
	delta := MessageDelta{
		Role:    "assistant",
		Content: "partial content",
	}

	assert.Equal(t, "assistant", delta.Role)
	assert.Equal(t, "partial content", delta.Content)
}

func TestUsage(t *testing.T) {
	usage := &Usage{
		PromptTokens:     10,
		CompletionTokens: 20,
		TotalTokens:      30,
	}

	assert.Equal(t, 10, usage.PromptTokens)
	assert.Equal(t, 20, usage.CompletionTokens)
	assert.Equal(t, 30, usage.TotalTokens)
}

func TestResponse_MarshalUnmarshal(t *testing.T) {
	original := &Response{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "gpt-4o",
		Choices: []Choice{
			{
				Index: 0,
				Message: Message{
					Role: "assistant",
					Content: []ContentPart{
						{Type: "text", Text: "Hello"},
					},
				},
				FinishReason: "stop",
			},
		},
		Usage: &Usage{
			PromptTokens:     5,
			CompletionTokens: 10,
			TotalTokens:      15,
		},
	}

	// Marshal to JSON
	data, err := json.Marshal(original)
	assert.NoError(t, err)

	// Unmarshal back
	var unmarshaled Response
	err = json.Unmarshal(data, &unmarshaled)
	assert.NoError(t, err)

	// Verify fields
	assert.Equal(t, original.ID, unmarshaled.ID)
	assert.Equal(t, original.Model, unmarshaled.Model)
	assert.Len(t, unmarshaled.Choices, 1)
	assert.Equal(t, original.Choices[0].Message.Content, unmarshaled.Choices[0].Message.Content)
	assert.NotNil(t, unmarshaled.Usage)
	assert.Equal(t, original.Usage.TotalTokens, unmarshaled.Usage.TotalTokens)
}
