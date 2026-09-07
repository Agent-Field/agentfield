package harness

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResultTextReturnsRawResult(t *testing.T) {
	tests := []struct {
		name   string
		result Result
		want   string
	}{
		{name: "zero value", result: Result{}, want: ""},
		{name: "plain text", result: Result{Result: "hello"}, want: "hello"},
		{name: "whitespace and unicode", result: Result{Result: " \t你好\nworld\r\n "}, want: " \t你好\nworld\r\n "},
		{
			name: "raw JSON rather than parsed value",
			result: Result{
				Result: ` { "message": "hello" } `,
				Parsed: map[string]any{"message": "hello"},
			},
			want: ` { "message": "hello" } `,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, tt.result.Text())
		})
	}
}

func TestFailureTypeJSONSerialization(t *testing.T) {
	tests := []struct {
		name  string
		value FailureType
		want  string
	}{
		{name: "zero value", value: "", want: `""`},
		{name: "none", value: FailureNone, want: `"none"`},
		{name: "crash", value: FailureCrash, want: `"crash"`},
		{name: "timeout", value: FailureTimeout, want: `"timeout"`},
		{name: "API error", value: FailureAPIError, want: `"api_error"`},
		{name: "no output", value: FailureNoOutput, want: `"no_output"`},
		{name: "schema", value: FailureSchema, want: `"schema"`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.value)
			require.NoError(t, err)
			assert.Equal(t, tt.want, string(data))

			var decoded FailureType
			require.NoError(t, json.Unmarshal(data, &decoded))
			assert.Equal(t, tt.value, decoded)
		})
	}
}
