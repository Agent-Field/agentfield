// Package inputs provides typed accessors for the untyped map[string]any
// payloads that capability handlers receive from RegisterReasoner. Use these
// helpers instead of inline type assertions so all first-party nodes parse
// input shapes consistently.
package inputs

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// RequiredString returns the trimmed string value for key, or an error if the
// key is missing or empty.
func RequiredString(input map[string]any, key string) (string, error) {
	value := String(input, key)
	if value == "" {
		return "", fmt.Errorf("%s is required", key)
	}
	return value, nil
}

// String returns the trimmed string value for key, or "" if absent or not a
// string.
func String(input map[string]any, key string) string {
	if input == nil {
		return ""
	}
	value, ok := input[key].(string)
	if !ok {
		return ""
	}
	return strings.TrimSpace(value)
}

// Int returns the int value for key. It accepts native int, float64
// (JSON-decoded numbers), and json.Number. Returns 0 when missing or
// unconvertible.
func Int(input map[string]any, key string) int {
	if input == nil {
		return 0
	}
	switch value := input[key].(type) {
	case int:
		return value
	case float64:
		return int(value)
	case json.Number:
		n, _ := value.Int64()
		return int(n)
	default:
		return 0
	}
}

// Object returns the map value for key. Returns an error when missing or
// empty so callers can fail fast on malformed reasoner inputs.
func Object(input map[string]any, key string) (map[string]any, error) {
	if input == nil {
		return nil, errors.New(key + " is required")
	}
	value, ok := input[key].(map[string]any)
	if !ok || len(value) == 0 {
		return nil, errors.New(key + " is required")
	}
	return value, nil
}

// FirstNonBlank returns the first trimmed-non-empty value in values, or "".
func FirstNonBlank(values ...string) string {
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}
