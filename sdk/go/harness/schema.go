package harness

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
)

const (
	outputFilename            = ".agentfield_output.json"
	schemaFilename            = ".agentfield_schema.json"
	largeSchemaTokenThreshold = 4000
)

// OutputPath returns the deterministic output file path.
func OutputPath(dir string) string {
	return filepath.Join(dir, outputFilename)
}

// SchemaPath returns the schema file path for large schemas.
func SchemaPath(dir string) string {
	return filepath.Join(dir, schemaFilename)
}

// estimateTokens gives a rough token count (~4 chars per token).
func estimateTokens(text string) int {
	return len(text) / 4
}

// BuildPromptSuffix constructs the OUTPUT REQUIREMENTS instruction that tells
// the coding agent to write JSON to a deterministic file path.
func BuildPromptSuffix(jsonSchema map[string]any, dir string) string {
	outputPath := OutputPath(dir)
	schemaJSON, err := json.MarshalIndent(jsonSchema, "", "  ")
	if err != nil {
		return fmt.Sprintf(
			"\n\n---\n"+
				"CRITICAL OUTPUT REQUIREMENTS:\n"+
				"You MUST use your Write tool to create this file: %s\n"+
				"The file MUST contain ONLY valid JSON.\n"+
				"Do NOT output the JSON in your response text — write it to the file.",
			outputPath,
		)
	}

	if estimateTokens(string(schemaJSON)) > largeSchemaTokenThreshold {
		schemaPath := SchemaPath(dir)
		_ = writeSchemaFile(string(schemaJSON), dir)
		return fmt.Sprintf(
			"\n\n---\n"+
				"CRITICAL OUTPUT REQUIREMENTS:\n"+
				"Read the JSON Schema at: %s\n"+
				"You MUST use your Write tool to create this file: %s\n"+
				"The file MUST contain ONLY valid JSON conforming to that schema.\n"+
				"Do NOT output the JSON in your response text — write it to the file.",
			schemaPath, outputPath,
		)
	}

	return fmt.Sprintf(
		"\n\n---\n"+
			"CRITICAL OUTPUT REQUIREMENTS:\n"+
			"You MUST use your Write tool to create this file: %s\n"+
			"The file MUST contain ONLY valid JSON matching the schema below.\n"+
			"Do NOT output the JSON in your response text — write it to the file.\n\n"+
			"Required JSON Schema:\n%s\n\n"+
			"Write ONLY valid JSON to the file. No markdown fences, no comments, no extra text.",
		outputPath, string(schemaJSON),
	)
}

// writeSchemaFile writes the schema JSON to the schema file.
func writeSchemaFile(schemaJSON string, dir string) error {
	path := SchemaPath(dir)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(schemaJSON), 0o600)
}

// ReadAndParse reads a JSON file and unmarshals it. Returns nil on any failure.
func ReadAndParse(filePath string) (map[string]any, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	data = []byte(strings.TrimSpace(string(data)))
	if len(data) == 0 {
		return nil, fmt.Errorf("empty file")
	}
	var result map[string]any
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, err
	}
	return result, nil
}

// cosmeticRepair attempts to fix common JSON formatting issues.
//
// Limitations: brace/bracket balancing is naive and does not understand JSON
// strings, so braces inside quoted strings can be miscounted.
func cosmeticRepair(raw string) string {
	text := strings.TrimSpace(raw)

	// Remove markdown fences
	fenceRe := regexp.MustCompile("(?s)^```(?:json)?\\s*\n(.*?)```\\s*$")
	if m := fenceRe.FindStringSubmatch(text); len(m) > 1 {
		text = strings.TrimSpace(m[1])
	}

	// Skip leading non-JSON text
	if len(text) > 0 && text[0] != '{' && text[0] != '[' {
		for i, ch := range text {
			if ch == '{' || ch == '[' {
				text = text[i:]
				break
			}
		}
	}

	// Remove trailing commas before closing brackets
	trailingComma := regexp.MustCompile(`,\s*([}\]])`)
	text = trailingComma.ReplaceAllString(text, "$1")

	// Close unclosed braces/brackets
	openBraces := strings.Count(text, "{") - strings.Count(text, "}")
	openBrackets := strings.Count(text, "[") - strings.Count(text, "]")
	if openBrackets > 0 {
		text += strings.Repeat("]", openBrackets)
	}
	if openBraces > 0 {
		text += strings.Repeat("}", openBraces)
	}

	return text
}

// ReadRepairAndParse reads, cosmetically repairs, and parses a JSON file.
func ReadRepairAndParse(filePath string) (map[string]any, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	content := strings.TrimSpace(string(data))
	if content == "" {
		return nil, fmt.Errorf("empty file")
	}
	repaired := cosmeticRepair(content)
	var result map[string]any
	if err := json.Unmarshal([]byte(repaired), &result); err != nil {
		return nil, err
	}
	return result, nil
}

// ParseAndValidate runs the full parse pipeline: read → parse → validate,
// then cosmetic repair → parse → validate.
//
// The dest parameter must be a pointer to a struct. On success the struct
// is populated via JSON round-trip and a map representation is returned.
func ParseAndValidate(filePath string, dest any) (map[string]any, error) {
	// Layer 1: direct parse
	data, err := ReadAndParse(filePath)
	if err == nil {
		if e := unmarshalInto(data, dest); e == nil {
			return data, nil
		}
	}

	// Layer 2: cosmetic repair
	data, err = ReadRepairAndParse(filePath)
	if err == nil {
		if e := unmarshalInto(data, dest); e == nil {
			return data, nil
		}
	}

	return nil, fmt.Errorf("parse and validate failed for %s", filePath)
}

// TryParseFromText extracts JSON from LLM conversation text as a fallback
// when the agent outputs JSON in its response instead of writing to the file.
func TryParseFromText(text string, dest any) (map[string]any, error) {
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("empty text")
	}

	// Strategy 1: fenced code blocks
	fenceRe := regexp.MustCompile("(?s)```(?:json)?\\s*\n(.*?)```")
	for _, m := range fenceRe.FindAllStringSubmatch(text, -1) {
		if len(m) > 1 {
			var data map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(m[1])), &data); err == nil {
				if err := unmarshalInto(data, dest); err == nil {
					return data, nil
				}
			}
		}
	}

	// Strategy 2: largest top-level { ... } block
	candidates := extractJSONBlocks(text)
	for _, candidate := range candidates {
		var data map[string]any
		if err := json.Unmarshal([]byte(candidate), &data); err == nil {
			if err := unmarshalInto(data, dest); err == nil {
				return data, nil
			}
		}
	}

	// Strategy 3: cosmetic repair on entire text
	repaired := cosmeticRepair(text)
	var data map[string]any
	if err := json.Unmarshal([]byte(repaired), &data); err == nil {
		if err := unmarshalInto(data, dest); err == nil {
			return data, nil
		}
	}

	return nil, fmt.Errorf("could not extract valid JSON from text")
}

// extractJSONBlocks finds top-level { ... } blocks, sorted largest first.
func extractJSONBlocks(text string) []string {
	var candidates []string
	depth := 0
	start := -1
	for i, ch := range text {
		switch ch {
		case '{':
			if depth == 0 {
				start = i
			}
			depth++
		case '}':
			depth--
			if depth == 0 && start >= 0 {
				candidates = append(candidates, text[start:i+1])
				start = -1
			}
		}
	}
	// Sort by length descending (largest first)
	sort.Slice(candidates, func(i, j int) bool {
		return len(candidates[i]) > len(candidates[j])
	})
	return candidates
}

// unmarshalInto validates data against the dest struct via JSON round-trip.
func unmarshalInto(data map[string]any, dest any) error {
	b, err := json.Marshal(data)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, dest)
}

// CleanupTempFiles removes harness temp files.
//
// For safety, this is a no-op when dir is empty or ".".
func CleanupTempFiles(dir string) {
	if dir == "" || dir == "." {
		return
	}
	for _, name := range []string{outputFilename, schemaFilename} {
		os.Remove(filepath.Join(dir, name))
	}
}

// DiagnoseOutputFailure returns a human-readable error describing why the
// output file failed validation.
func DiagnoseOutputFailure(filePath string, jsonSchema map[string]any) string {
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return "The output file was NOT created."
	}

	data, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Sprintf("Could not read output file: %v", err)
	}

	content := strings.TrimSpace(string(data))
	if content == "" {
		return "The output file exists but is empty."
	}

	var parsed map[string]any
	if err := json.Unmarshal([]byte(content), &parsed); err != nil {
		snippet := content
		if len(snippet) > 500 {
			snippet = snippet[:500]
		}
		return fmt.Sprintf(
			"The file contains invalid JSON. Parse error: %v\nFile content (first 500 chars):\n%s",
			err, snippet,
		)
	}

	// JSON parses but may not match schema
	props, _ := jsonSchema["properties"].(map[string]any)
	expectedKeys := make([]string, 0, len(props))
	for k := range props {
		expectedKeys = append(expectedKeys, k)
	}
	actualKeys := make([]string, 0, len(parsed))
	for k := range parsed {
		actualKeys = append(actualKeys, k)
	}
	return fmt.Sprintf(
		"JSON parses but may not match expected schema.\nExpected top-level keys: %v\nActual top-level keys: %v",
		expectedKeys, actualKeys,
	)
}

// BuildFollowupPrompt constructs a retry prompt after schema validation failure.
func BuildFollowupPrompt(errorMessage string, dir string, jsonSchema map[string]any) string {
	outputPath := OutputPath(dir)
	schemaPath := SchemaPath(dir)

	var b strings.Builder
	fmt.Fprintf(&b, "PREVIOUS ATTEMPT FAILED. The JSON output at %s failed validation.\n", outputPath)
	fmt.Fprintf(&b, "Error: %s\n\n", errorMessage)

	if jsonSchema != nil {
		schemaJSON, err := json.MarshalIndent(jsonSchema, "", "  ")
		if err != nil {
			fmt.Fprintf(&b, "The schema could not be serialized (%v).\n", err)
			fmt.Fprintf(&b, "Write valid JSON to %s and include all expected top-level fields.\n\n", outputPath)
		} else if estimateTokens(string(schemaJSON)) > largeSchemaTokenThreshold {
			if _, err := os.Stat(schemaPath); err == nil {
				fmt.Fprintf(&b, "The required JSON Schema is at: %s\nRe-read the schema file carefully.\n", schemaPath)
			} else {
				_ = writeSchemaFile(string(schemaJSON), dir)
				fmt.Fprintf(&b, "The required JSON Schema has been written to: %s\nRead that file for the exact expected structure.\n", schemaPath)
			}
		} else {
			fmt.Fprintf(&b, "The JSON MUST conform to this schema:\n%s\n\n", string(schemaJSON))
		}
	} else if _, err := os.Stat(schemaPath); err == nil {
		fmt.Fprintf(&b, "The required JSON Schema is at: %s\nRe-read the schema file carefully.\n", schemaPath)
	}

	fmt.Fprintf(&b, "Use your Write tool to create or overwrite the file: %s\n", outputPath)
	b.WriteString("The file must contain ONLY valid JSON matching the schema. No markdown fences, no extra text, no comments.\n")
	b.WriteString("Each field defined in the schema must be present as a top-level key in your JSON object.")

	return b.String()
}

// topLevelField describes one top-level schema property for the incremental
// build: its name and whether the schema marks it required.
type topLevelField struct {
	Name     string
	Required bool
}

// getTopLevelFields returns the schema's top-level properties and whether each
// is required. Mirrors the Python _schema.get_top_level_fields.
//
// Note on ordering: Go's JSON schema is a map[string]any, so the original
// property order from the source schema is not preserved (Python dicts keep
// insertion order). Field names are sorted for deterministic prompt output;
// this affects only the order of the field list, not correctness — the agent
// is asked to add every field regardless of order.
func getTopLevelFields(jsonSchema map[string]any) []topLevelField {
	props, _ := jsonSchema["properties"].(map[string]any)
	required := requiredSet(jsonSchema)

	names := make([]string, 0, len(props))
	for name := range props {
		names = append(names, name)
	}
	sort.Strings(names)

	fields := make([]topLevelField, 0, len(names))
	for _, name := range names {
		fields = append(fields, topLevelField{Name: name, Required: required[name]})
	}
	return fields
}

// requiredSet extracts the schema's "required" list as a set.
func requiredSet(jsonSchema map[string]any) map[string]bool {
	set := make(map[string]bool)
	if req, ok := jsonSchema["required"].([]any); ok {
		for _, r := range req {
			if s, ok := r.(string); ok {
				set[s] = true
			}
		}
	}
	// Also accept a pre-typed []string (callers who build schemas in Go).
	if req, ok := jsonSchema["required"].([]string); ok {
		for _, s := range req {
			set[s] = true
		}
	}
	return set
}

// BuildIncrementalPromptSuffix builds the OUTPUT REQUIREMENTS suffix that
// instructs a field-by-field build. Byte-for-byte parity with the Python
// _schema.build_incremental_prompt_suffix.
func BuildIncrementalPromptSuffix(jsonSchema map[string]any, dir string) string {
	outputPath := OutputPath(dir)
	schemaJSON, err := json.MarshalIndent(jsonSchema, "", "  ")
	if err != nil {
		// Fall back to the single-shot suffix if the schema cannot be
		// serialized — matches the defensive posture of BuildPromptSuffix.
		return BuildPromptSuffix(jsonSchema, dir)
	}

	fields := getTopLevelFields(jsonSchema)
	fieldLineParts := make([]string, 0, len(fields))
	for _, f := range fields {
		req := "optional"
		if f.Required {
			req = "required"
		}
		fieldLineParts = append(fieldLineParts, fmt.Sprintf("  - %s (%s)", f.Name, req))
	}
	fieldLines := strings.Join(fieldLineParts, "\n")

	var schemaRef string
	if estimateTokens(string(schemaJSON)) > largeSchemaTokenThreshold {
		schemaPath := SchemaPath(dir)
		_ = writeSchemaFile(string(schemaJSON), dir)
		schemaRef = fmt.Sprintf(
			"The full JSON Schema is at: %s\nRead it for each field's exact shape.\n",
			schemaPath,
		)
	} else {
		schemaRef = fmt.Sprintf("Full JSON Schema:\n%s\n", string(schemaJSON))
	}

	return "\n\n---\n" +
		"CRITICAL OUTPUT REQUIREMENTS (incremental build):\n" +
		fmt.Sprintf("Produce a single JSON object in this file using your Write/Edit tools: %s\n", outputPath) +
		"Build it ONE FIELD AT A TIME so nothing gets truncated:\n" +
		"  1. First create the file with an empty object: {}\n" +
		"  2. Then add each field listed below one at a time using Edit, and after\n" +
		"     each edit re-read the file to confirm it is still valid JSON.\n" +
		"  3. Each field's value MUST conform to its shape in the schema.\n" +
		"  4. Do not finish until every required field is present.\n\n" +
		fmt.Sprintf("Top-level fields to add:\n%s\n\n", fieldLines) +
		fmt.Sprintf("%s\n", schemaRef) +
		fmt.Sprintf("The final file at %s MUST contain ONLY the complete valid JSON "+
			"object — no markdown fences, no commentary, no extra text.", outputPath)
}

// DiagnoseFieldFailures maps each missing/invalid top-level field to a short
// reason. Returns an empty map when the file validates cleanly. Mirrors the
// Python _schema.diagnose_field_failures, adapted to the Go dest-struct
// contract.
//
// Go has no per-field pydantic error list, so where Python enumerates
// validation errors by field location, Go can only detect (a) missing required
// fields — via the schema's "required" list against the parsed object — and
// (b) a whole-document type mismatch when the object fails to unmarshal into
// dest, reported once under the "_root" key (matching Python's fallback).
func DiagnoseFieldFailures(filePath string, jsonSchema map[string]any, dest any) map[string]string {
	props, _ := jsonSchema["properties"].(map[string]any)
	propNames := make([]string, 0, len(props))
	for name := range props {
		propNames = append(propNames, name)
	}
	sort.Strings(propNames)
	required := requiredSet(jsonSchema)

	data, err := ReadAndParse(filePath)
	if err != nil {
		data, err = ReadRepairAndParse(filePath)
	}

	failures := make(map[string]string)

	if err != nil || data == nil {
		// Whole file unusable — report every required field (or all fields if
		// none are required) as needing to be written.
		targets := make([]string, 0)
		for _, name := range propNames {
			if required[name] {
				targets = append(targets, name)
			}
		}
		if len(targets) == 0 {
			targets = propNames
		}
		for _, name := range targets {
			failures[name] = "output file missing or not a JSON object"
		}
		return failures
	}

	// Required-field presence check (sorted for deterministic prompt output).
	reqNames := make([]string, 0, len(required))
	for name := range required {
		reqNames = append(reqNames, name)
	}
	sort.Strings(reqNames)
	for _, name := range reqNames {
		if _, present := data[name]; !present {
			failures[name] = "missing required field"
		}
	}

	// Validation against the destination struct. Use a throwaway instance so
	// the caller's dest is not mutated during diagnosis.
	if dest != nil {
		if fresh := freshDest(dest); fresh != nil {
			if e := unmarshalInto(data, fresh); e != nil {
				if _, exists := failures["_root"]; !exists {
					failures["_root"] = truncate(e.Error(), 200)
				}
			}
		}
	}

	return failures
}

// freshDest returns a new zero-valued instance of the same type dest points to
// (a pointer), so validation can run without mutating the caller's value.
// Returns nil when dest is not a non-nil pointer.
func freshDest(dest any) any {
	rv := reflect.ValueOf(dest)
	if rv.Kind() != reflect.Pointer || rv.IsNil() {
		return nil
	}
	return reflect.New(rv.Elem().Type()).Interface()
}

// BuildIncrementalFollowup builds the follow-up prompt that asks the agent to
// patch only the failing fields. Byte-for-byte parity with the Python
// _schema.build_incremental_followup.
func BuildIncrementalFollowup(fieldErrors map[string]string, dir string, jsonSchema map[string]any) string {
	outputPath := OutputPath(dir)

	// Deterministic order (Go maps randomize; Python preserves dict order).
	names := make([]string, 0, len(fieldErrors))
	for name := range fieldErrors {
		names = append(names, name)
	}
	sort.Strings(names)
	lineParts := make([]string, 0, len(names))
	for _, name := range names {
		lineParts = append(lineParts, fmt.Sprintf("  - %s: %s", name, fieldErrors[name]))
	}
	fieldLines := strings.Join(lineParts, "\n")

	var schemaRef string
	schemaJSON, err := json.MarshalIndent(jsonSchema, "", "  ")
	if err == nil && estimateTokens(string(schemaJSON)) > largeSchemaTokenThreshold {
		schemaPath := SchemaPath(dir)
		if _, statErr := os.Stat(schemaPath); statErr != nil {
			_ = writeSchemaFile(string(schemaJSON), dir)
		}
		schemaRef = fmt.Sprintf("Full schema is at: %s\n", schemaPath)
	} else if err == nil {
		schemaRef = fmt.Sprintf("Full schema:\n%s\n", string(schemaJSON))
	}

	return fmt.Sprintf(
		"PARTIAL OUTPUT NEEDS FIXES. The JSON at %s is incomplete or invalid.\n", outputPath) +
		"Patch ONLY these fields, one at a time, using Edit, keeping the file valid JSON after each change:\n" +
		fmt.Sprintf("%s\n\n", fieldLines) +
		schemaRef +
		"Leave every already-correct field unchanged. Do NOT rewrite the whole file."
}
