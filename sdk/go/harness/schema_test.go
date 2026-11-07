package harness

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOutputPath(t *testing.T) {
	assert.Equal(t, filepath.Join("/tmp", outputFilename), OutputPath("/tmp"))
	assert.Equal(t, filepath.Join("foo", outputFilename), OutputPath("foo"))
}

func TestSchemaPath(t *testing.T) {
	assert.Equal(t, filepath.Join("/tmp", schemaFilename), SchemaPath("/tmp"))
}

func TestBuildPromptSuffix_SmallSchema(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
	}
	suffix := BuildPromptSuffix(schema, "/tmp/test")
	assert.Contains(t, suffix, "CRITICAL OUTPUT REQUIREMENTS")
	assert.Contains(t, suffix, OutputPath("/tmp/test"))
	assert.Contains(t, suffix, `"name"`)
}

func TestBuildPromptSuffix_LargeSchema(t *testing.T) {
	// Build a schema that exceeds 4000 tokens (~16000 chars)
	props := make(map[string]any)
	for i := 0; i < 500; i++ {
		key := "field_" + string(rune('a'+i%26)) + string(rune('0'+i/26))
		props[key] = map[string]any{
			"type":        "string",
			"description": "A very long description that helps pad the schema size to exceed the threshold for large schemas.",
		}
	}
	schema := map[string]any{
		"type":       "object",
		"properties": props,
	}

	dir := t.TempDir()
	suffix := BuildPromptSuffix(schema, dir)
	assert.Contains(t, suffix, "Read the JSON Schema at")
	assert.Contains(t, suffix, SchemaPath(dir))

	// Schema file should exist
	_, err := os.Stat(SchemaPath(dir))
	assert.NoError(t, err)
}

func TestCosmeticRepair(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "markdown fences",
			input:    "```json\n{\"a\": 1}\n```",
			expected: `{"a": 1}`,
		},
		{
			name:     "leading text",
			input:    "Here is the JSON: {\"a\": 1}",
			expected: `{"a": 1}`,
		},
		{
			name:     "trailing comma",
			input:    `{"a": 1, "b": 2,}`,
			expected: `{"a": 1, "b": 2}`,
		},
		{
			name:     "unclosed brace",
			input:    `{"a": 1`,
			expected: `{"a": 1}`,
		},
		{
			name:     "valid json unchanged",
			input:    `{"a": 1}`,
			expected: `{"a": 1}`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := cosmeticRepair(tc.input)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestReadAndParse(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.json")

	t.Run("valid json", func(t *testing.T) {
		err := os.WriteFile(path, []byte(`{"key": "value"}`), 0o644)
		require.NoError(t, err)
		data, err := ReadAndParse(path)
		assert.NoError(t, err)
		assert.Equal(t, "value", data["key"])
	})

	t.Run("empty file", func(t *testing.T) {
		err := os.WriteFile(path, []byte(""), 0o644)
		require.NoError(t, err)
		_, err = ReadAndParse(path)
		assert.Error(t, err)
	})

	t.Run("missing file", func(t *testing.T) {
		_, err := ReadAndParse(filepath.Join(dir, "nonexistent.json"))
		assert.Error(t, err)
	})

	t.Run("invalid json", func(t *testing.T) {
		err := os.WriteFile(path, []byte("not json"), 0o644)
		require.NoError(t, err)
		_, err = ReadAndParse(path)
		assert.Error(t, err)
	})
}

func TestReadRepairAndParse(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.json")

	// Markdown-fenced JSON
	err := os.WriteFile(path, []byte("```json\n{\"a\": 1}\n```"), 0o644)
	require.NoError(t, err)
	data, err := ReadRepairAndParse(path)
	assert.NoError(t, err)
	assert.Equal(t, float64(1), data["a"])
}

func TestParseAndValidate(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.json")

	type TestStruct struct {
		Name     string `json:"name"`
		Severity string `json:"severity"`
	}

	t.Run("valid file", func(t *testing.T) {
		err := os.WriteFile(path, []byte(`{"name": "test", "severity": "high"}`), 0o644)
		require.NoError(t, err)
		var dest TestStruct
		data, err := ParseAndValidate(path, &dest)
		assert.NoError(t, err)
		assert.Equal(t, "test", data["name"])
		assert.Equal(t, "test", dest.Name)
		assert.Equal(t, "high", dest.Severity)
	})

	t.Run("repairable file", func(t *testing.T) {
		err := os.WriteFile(path, []byte("```json\n{\"name\": \"test\", \"severity\": \"low\"}\n```"), 0o644)
		require.NoError(t, err)
		var dest TestStruct
		data, err := ParseAndValidate(path, &dest)
		assert.NoError(t, err)
		assert.NotNil(t, data)
		assert.Equal(t, "test", dest.Name)
	})
}

func TestTryParseFromText(t *testing.T) {
	type TestStruct struct {
		Result string `json:"result"`
	}

	t.Run("fenced code block", func(t *testing.T) {
		text := "Here is the output:\n```json\n{\"result\": \"ok\"}\n```\nDone."
		var dest TestStruct
		data, err := TryParseFromText(text, &dest)
		assert.NoError(t, err)
		assert.Equal(t, "ok", data["result"])
		assert.Equal(t, "ok", dest.Result)
	})

	t.Run("bare json block", func(t *testing.T) {
		text := "Some preamble text\n{\"result\": \"found\"}\nMore text"
		var dest TestStruct
		data, err := TryParseFromText(text, &dest)
		assert.NoError(t, err)
		assert.Equal(t, "found", data["result"])
	})

	t.Run("empty text", func(t *testing.T) {
		var dest TestStruct
		_, err := TryParseFromText("", &dest)
		assert.Error(t, err)
	})
}

func TestDiagnoseOutputFailure(t *testing.T) {
	dir := t.TempDir()
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
	}

	t.Run("missing file", func(t *testing.T) {
		result := DiagnoseOutputFailure(filepath.Join(dir, "missing.json"), schema)
		assert.Contains(t, result, "NOT created")
	})

	t.Run("empty file", func(t *testing.T) {
		path := filepath.Join(dir, "empty.json")
		os.WriteFile(path, []byte(""), 0o644)
		result := DiagnoseOutputFailure(path, schema)
		assert.Contains(t, result, "empty")
	})

	t.Run("invalid json", func(t *testing.T) {
		path := filepath.Join(dir, "bad.json")
		os.WriteFile(path, []byte("not json"), 0o644)
		result := DiagnoseOutputFailure(path, schema)
		assert.Contains(t, result, "invalid JSON")
	})

	t.Run("valid json wrong schema", func(t *testing.T) {
		path := filepath.Join(dir, "wrong.json")
		os.WriteFile(path, []byte(`{"other": "field"}`), 0o644)
		result := DiagnoseOutputFailure(path, schema)
		assert.Contains(t, result, "Missing top-level keys: [name]")
	})
}

// THE DIAGNOSIS NAMES THE REAL FAILURE. Right keys, wrong value type: the old
// message listed two equal key sets in different orders and nothing to fix.
func TestTheDiagnosisNamesTheValueThatIsWrongNotTheKeys(t *testing.T) {
	type finding struct {
		Driver   string            `json:"driver"`
		Stars    int64             `json:"stars"`
		Features map[string]string `json:"features"`
	}
	schema, err := StructToJSONSchema(finding{})
	require.NoError(t, err)
	path := filepath.Join(t.TempDir(), "out.json")
	require.NoError(t, os.WriteFile(path, []byte(`{"features":{"cgo":false},"stars":"4k","driver":"ncruces"}`), 0o644))

	got := DiagnoseOutputFailureFor(path, schema, &finding{})
	assert.NotContains(t, got, "top-level keys")
	assert.Contains(t, got, "stars", got)
	assert.Contains(t, got, "features", got)

	var untouched finding
	_ = DiagnoseOutputFailureFor(path, schema, &untouched)
	assert.Equal(t, finding{}, untouched, "the diagnosis never writes to dest")
}

func TestBuildFollowupPrompt(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
	}
	dir := t.TempDir()
	prompt := BuildFollowupPrompt("File was empty", dir, schema)
	assert.Contains(t, prompt, "PREVIOUS ATTEMPT FAILED")
	assert.Contains(t, prompt, "File was empty")
	assert.Contains(t, prompt, OutputPath(dir))
	assert.Contains(t, prompt, `"name"`)
}

func TestCleanupTempFiles(t *testing.T) {
	dir := t.TempDir()
	outPath := OutputPath(dir)
	schemaPath := SchemaPath(dir)
	os.WriteFile(outPath, []byte("{}"), 0o644)
	os.WriteFile(schemaPath, []byte("{}"), 0o644)

	CleanupTempFiles(dir)

	_, err := os.Stat(outPath)
	assert.True(t, os.IsNotExist(err))
	_, err = os.Stat(schemaPath)
	assert.True(t, os.IsNotExist(err))
}

func TestExtractJSONBlocks(t *testing.T) {
	text := `prefix {"small": 1} middle {"larger": {"nested": true}} end`
	blocks := extractJSONBlocks(text)
	assert.Len(t, blocks, 2)
	// Largest first
	assert.Contains(t, blocks[0], "nested")
}

func TestStructToJSONSchema(t *testing.T) {
	type Example struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}
	schema, err := StructToJSONSchema(Example{Name: "test", Age: 0})
	require.NoError(t, err)
	assert.Equal(t, "object", schema["type"])
	props, ok := schema["properties"].(map[string]any)
	require.True(t, ok)
	assert.Contains(t, props, "name")
	assert.Contains(t, props, "age")

	// Verify it's valid JSON
	_, err = json.Marshal(schema)
	assert.NoError(t, err)
}

// THE SCHEMA NAMES WHAT GOES INSIDE. `files: array` let the model write
// objects where the result type wanted strings.
func TestTheSchemaNamesTheElementTypes(t *testing.T) {
	type file struct {
		Path string `json:"path"`
	}
	type node struct {
		Kids []node `json:"kids,omitempty"`
	}
	type findings struct {
		Files  []string          `json:"files"`
		Counts map[string]int64  `json:"counts"`
		Where  []file            `json:"where"`
		Tree   node              `json:"tree"`
		Extra  map[string]string `json:"extra,omitempty"`
	}
	schema, err := StructToJSONSchema(findings{})
	require.NoError(t, err)
	props := schema["properties"].(map[string]any)
	assert.Equal(t, map[string]any{"type": "array", "items": map[string]any{"type": "string"}}, props["files"])
	assert.Equal(t, map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "integer"}}, props["counts"])
	where := props["where"].(map[string]any)["items"].(map[string]any)
	assert.Equal(t, map[string]any{"path": map[string]any{"type": "string"}}, where["properties"])
	kids := props["tree"].(map[string]any)["properties"].(map[string]any)["kids"].(map[string]any)
	assert.Equal(t, map[string]any{"type": "object"}, kids["items"], "a type that holds itself stops")
	assert.ElementsMatch(t, []string{"files", "counts", "where", "tree"}, schema["required"])

	path := filepath.Join(t.TempDir(), "out.json")
	require.NoError(t, os.WriteFile(path, []byte(`{"files":[{"path":"a.go"}],"counts":{},"where":[],"tree":{}}`), 0o644))
	assert.Contains(t, DiagnoseOutputFailureFor(path, schema, &findings{}), "/files/0")
}
