package ai

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMessageUnmarshalJSON_AdditionalCases(t *testing.T) {
	t.Run("string content", func(t *testing.T) {
		var msg Message
		err := json.Unmarshal([]byte(`{"role":"assistant","content":"hello"}`), &msg)
		require.NoError(t, err)
		require.Len(t, msg.Content, 1)
		assert.Equal(t, "text", msg.Content[0].Type)
		assert.Equal(t, "hello", msg.Content[0].Text)
	})

	t.Run("array content", func(t *testing.T) {
		var msg Message
		err := json.Unmarshal([]byte(`{"role":"user","content":[{"type":"text","text":"hello"},{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]}`), &msg)
		require.NoError(t, err)
		require.Len(t, msg.Content, 2)
		assert.Equal(t, "hello", msg.Content[0].Text)
		require.NotNil(t, msg.Content[1].ImageURL)
		assert.Equal(t, "https://example.com/image.png", msg.Content[1].ImageURL.URL)
	})

	t.Run("null content with tool fields", func(t *testing.T) {
		var msg Message
		err := json.Unmarshal([]byte(`{"role":"assistant","content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}}],"tool_call_id":"tool-1"}`), &msg)
		require.NoError(t, err)
		assert.Equal(t, "assistant", msg.Role)
		assert.Len(t, msg.ToolCalls, 1)
		assert.Equal(t, "tool-1", msg.ToolCallID)
		assert.Nil(t, msg.Content)
	})

	t.Run("invalid content shape", func(t *testing.T) {
		var msg Message
		err := json.Unmarshal([]byte(`{"role":"assistant","content":123}`), &msg)
		assert.Error(t, err)
	})
}
