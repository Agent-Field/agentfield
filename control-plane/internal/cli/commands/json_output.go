package commands

import (
	"encoding/json"
	"fmt"
	"os"
)

// jsonEnvelope mirrors the af agent-mode AgentResponse envelope shape
// ({ok, data, error:{code,message,hint}}) for framework-based lifecycle
// commands (install/run) running with --json.
type jsonEnvelope struct {
	OK    bool               `json:"ok"`
	Data  interface{}        `json:"data,omitempty"`
	Error *jsonEnvelopeError `json:"error,omitempty"`
}

type jsonEnvelopeError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Hint    string `json:"hint,omitempty"`
}

// printJSONEnvelope writes the envelope to the given writer (the real stdout,
// which callers capture before redirecting service diagnostics to stderr).
func printJSONEnvelope(out *os.File, env jsonEnvelope) error {
	b, err := json.MarshalIndent(env, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal output: %w", err)
	}
	if _, err := fmt.Fprintln(out, string(b)); err != nil {
		return fmt.Errorf("write output: %w", err)
	}
	return nil
}

func printJSONSuccess(out *os.File, data interface{}) error {
	return printJSONEnvelope(out, jsonEnvelope{OK: true, Data: data})
}

func printJSONError(out *os.File, code, message, hint string) {
	_ = printJSONEnvelope(out, jsonEnvelope{
		OK:    false,
		Error: &jsonEnvelopeError{Code: code, Message: message, Hint: hint},
	})
}

// redirectStdoutToStderr points os.Stdout at os.Stderr so service-layer
// progress prints don't pollute the JSON envelope on the real stdout. It
// returns the original stdout and a restore func.
func redirectStdoutToStderr() (*os.File, func()) {
	original := os.Stdout
	os.Stdout = os.Stderr
	return original, func() { os.Stdout = original }
}
