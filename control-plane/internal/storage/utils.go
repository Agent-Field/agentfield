package storage

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"
)

// safeJSONRawMessage safely creates a json.RawMessage from a potentially corrupted string
// It validates the JSON and provides a safe fallback if the data is corrupted
func safeJSONRawMessage(data string, fallback string, context string) json.RawMessage {
	if data == "" {
		return json.RawMessage(fallback)
	}

	// Validate JSON before creating RawMessage
	if json.Valid([]byte(data)) {
		return json.RawMessage(data)
	}

	// Log corruption warning with context
	log.Printf("WARNING: Corrupted JSON data detected in %s, using fallback. Data preview: %.100s", context, data)

	// Return safe fallback with error indication
	errorFallback := fmt.Sprintf(`{"error": "corrupted_json_data", "context": "%s", "preview": "%s"}`,
		context, strings.ReplaceAll(data[:min(50, len(data))], `"`, `\"`))
	return json.RawMessage(errorFallback)
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
