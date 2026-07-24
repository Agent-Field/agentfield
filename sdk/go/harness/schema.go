package harness

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"

	"github.com/santhosh-tekuri/jsonschema/v5"
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

// codexStrictJSONSchema rewrites a JSON schema into the strict form codex's
// --output-schema (OpenAI structured output) requires. On every object node it
// drops each property's "default", marks ALL properties required, and sets
// additionalProperties:false; it recurses into properties, array items,
// allOf/anyOf/oneOf branches, and $defs/definitions. Port of
// _codex_strict_json_schema (codex_harness_patch.py:23-65).
//
// The input schema is not mutated — every level is copied before edits.
func codexStrictJSONSchema(schema map[string]any) map[string]any {
	strict := make(map[string]any, len(schema))
	for k, v := range schema {
		strict[k] = v
	}

	schemaType, _ := strict["type"].(string)

	if schemaType == "object" {
		if props, ok := strict["properties"].(map[string]any); ok {
			cleaned := make(map[string]any, len(props))
			keys := make([]string, 0, len(props))
			for key, value := range props {
				keys = append(keys, key)
				if child, ok := value.(map[string]any); ok {
					childCopy := make(map[string]any, len(child))
					for ck, cv := range child {
						if ck == "default" {
							continue
						}
						childCopy[ck] = cv
					}
					cleaned[key] = codexStrictJSONSchema(childCopy)
				} else {
					cleaned[key] = value
				}
			}
			// Sort for deterministic output (Go maps randomize iteration order;
			// Python preserves dict insertion order). codex validation is
			// order-independent, so this only stabilizes the emitted file.
			sort.Strings(keys)
			strict["properties"] = cleaned
			strict["required"] = keys
			strict["additionalProperties"] = false
		}
	}

	if schemaType == "array" {
		if items, ok := strict["items"].(map[string]any); ok {
			strict["items"] = codexStrictJSONSchema(items)
		}
	}

	for _, key := range []string{"allOf", "anyOf", "oneOf"} {
		if branch, ok := strict[key].([]any); ok {
			newBranch := make([]any, len(branch))
			for i, item := range branch {
				if m, ok := item.(map[string]any); ok {
					newBranch[i] = codexStrictJSONSchema(m)
				} else {
					newBranch[i] = item
				}
			}
			strict[key] = newBranch
		}
	}

	for _, defKey := range []string{"$defs", "definitions"} {
		if defs, ok := strict[defKey].(map[string]any); ok {
			newDefs := make(map[string]any, len(defs))
			for key, value := range defs {
				if m, ok := value.(map[string]any); ok {
					newDefs[key] = codexStrictJSONSchema(m)
				} else {
					newDefs[key] = value
				}
			}
			strict[defKey] = newDefs
		}
	}

	return strict
}

// codexSchemaStrictExpressible reports whether a strict-rewritten schema can be
// enforced server-side via codex's --output-schema. OpenAI's strict-mode
// validator (probed live against codex-cli 0.144.1) requires, on EVERY node:
//
//   - a "type" key, unless the node is a $ref or an anyOf/oneOf/allOf
//     combinator — so bare {} nodes (Go `any` / Python `Any` fields) are out;
//   - for objects: "additionalProperties" supplied and false, plus "required"
//     listing every property key. Free-form maps (map[string]any /
//     dict[str, Any] — an object node with no "properties") cannot satisfy
//     this without forcing the model to emit an empty object, so they are out;
//   - typed maps (additionalProperties as a subschema) are rejected outright.
//
// format / minItems / default / anyOf-with-null / $defs+$ref all pass the
// validator and stay expressible. When this returns false the runner keeps the
// codex-native prompt + --output-last-message but drops --output-schema,
// leaving enforcement to local validation — the server would 400 the request
// with invalid_json_schema otherwise.
func codexSchemaStrictExpressible(schema map[string]any) bool {
	// $ref targets live in $defs/definitions; every definition must itself be
	// expressible for any reference to it to be.
	for _, defKey := range []string{"$defs", "definitions"} {
		if defs, ok := schema[defKey].(map[string]any); ok {
			for _, value := range defs {
				m, ok := value.(map[string]any)
				if !ok || !codexSchemaStrictExpressible(m) {
					return false
				}
			}
		}
	}

	if _, ok := schema["$ref"].(string); ok {
		return true // target checked via the $defs walk above
	}

	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if branch, ok := schema[key].([]any); ok {
			for _, item := range branch {
				m, ok := item.(map[string]any)
				if !ok || !codexSchemaStrictExpressible(m) {
					return false
				}
			}
			return true
		}
	}

	switch typeValue := schema["type"].(type) {
	case string:
		switch typeValue {
		case "object":
			props, ok := schema["properties"].(map[string]any)
			if !ok {
				return false // free-form map — strict mode would force {}
			}
			if ap, ok := schema["additionalProperties"].(bool); !ok || ap {
				return false
			}
			for _, value := range props {
				m, ok := value.(map[string]any)
				if !ok || !codexSchemaStrictExpressible(m) {
					return false
				}
			}
			return true
		case "array":
			items, ok := schema["items"].(map[string]any)
			if !ok {
				return false // tuple/itemless arrays — not expressible
			}
			return codexSchemaStrictExpressible(items)
		default:
			return true // primitive leaf
		}
	case []any:
		// e.g. ["string","null"]; only primitive members are safely strict.
		for _, tv := range typeValue {
			s, ok := tv.(string)
			if !ok || s == "object" || s == "array" {
				return false
			}
		}
		return len(typeValue) > 0
	default:
		return false // no type, no $ref, no combinator — e.g. {} for Any
	}
}

// BuildCodexNativeSuffix constructs the prompt suffix for codex's native
// structured output. Unlike BuildPromptSuffix (which asks the model to Write a
// file), this tells the model to emit its final JSON answer directly — the
// codex CLI persists it to outputPath via --output-last-message and validates
// it against schemaPath via --output-schema. Port of the codex-native suffix in
// codex_harness_patch.py:141-148 + 179-186.
func BuildCodexNativeSuffix(schemaPath, outputPath string) string {
	return "\n\n---\n" +
		"CRITICAL CODEX STRUCTURED OUTPUT REQUIREMENTS:\n" +
		fmt.Sprintf("Return a single final JSON object conforming to the schema at: %s\n", schemaPath) +
		fmt.Sprintf("The Codex CLI will persist your final response to: %s\n", outputPath) +
		"Return the JSON object as your final answer. Do not use markdown fences, comments, or surrounding prose.\n" +
		"Do not try to create .agentfield_output.json yourself; the Codex CLI will persist your final JSON response for AgentField."
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

// validateAgainstSchema validates a parsed JSON object against a JSON Schema
// map and returns a concise, prompt-friendly error when it does not conform.
//
// A JSON round-trip into a Go struct (unmarshalInto) accepts missing required
// fields, invalid enum values, and extra fields silently — so on its own it
// lets malformed output pass the harness. This compiles the schema and runs
// real JSON Schema validation so the runner's schema-retry loop fires for those
// cases. When the schema cannot be serialized or compiled, it returns nil (no
// regression versus the previous unmarshal-only behavior — validation is simply
// skipped rather than blocking).
func validateAgainstSchema(data map[string]any, schema map[string]any) error {
	schemaBytes, err := json.Marshal(schema)
	if err != nil {
		return nil
	}
	compiler := jsonschema.NewCompiler()
	if err := compiler.AddResource("mem://agentfield/schema.json", bytes.NewReader(schemaBytes)); err != nil {
		return nil
	}
	compiled, err := compiler.Compile("mem://agentfield/schema.json")
	if err != nil {
		return nil
	}
	// Normalize the data through JSON so the validator sees canonical types
	// (e.g. float64 for numbers, []any for arrays) regardless of how it was
	// produced upstream.
	dataBytes, err := json.Marshal(data)
	if err != nil {
		return nil
	}
	var normalized any
	if err := json.Unmarshal(dataBytes, &normalized); err != nil {
		return nil
	}
	if verr := compiled.Validate(normalized); verr != nil {
		return fmt.Errorf("schema validation failed: %s", conciseSchemaError(verr))
	}
	return nil
}

// conciseSchemaError flattens a jsonschema ValidationError tree into a short,
// prompt-friendly string listing the most specific (leaf) failures with their
// instance locations.
func conciseSchemaError(err error) string {
	ve, ok := err.(*jsonschema.ValidationError)
	if !ok {
		return truncate(err.Error(), 300)
	}

	var leaves []string
	var walk func(e *jsonschema.ValidationError)
	walk = func(e *jsonschema.ValidationError) {
		if len(e.Causes) == 0 {
			loc := e.InstanceLocation
			if loc == "" {
				loc = "/"
			}
			leaves = append(leaves, fmt.Sprintf("%s: %s", loc, e.Message))
			return
		}
		for _, c := range e.Causes {
			walk(c)
		}
	}
	walk(ve)

	if len(leaves) == 0 {
		return truncate(ve.Message, 300)
	}
	sort.Strings(leaves)
	return truncate(strings.Join(leaves, "; "), 400)
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
