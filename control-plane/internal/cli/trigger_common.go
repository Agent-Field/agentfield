package cli

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"golang.org/x/term"
	"gopkg.in/yaml.v3"
)

type cliExitError struct {
	Code int
	Err  error
}

func (e cliExitError) Error() string {
	if e.Err == nil {
		return fmt.Sprintf("exit %d", e.Code)
	}
	return e.Err.Error()
}

// ExitCode returns the process exit code for command errors with explicit CLI semantics.
func ExitCode(err error) int {
	var exitErr cliExitError
	if errors.As(err, &exitErr) && exitErr.Code > 0 {
		return exitErr.Code
	}
	return 1
}

// IsCLIExitError reports whether an error intentionally carries CLI process semantics.
func IsCLIExitError(err error) bool {
	var exitErr cliExitError
	return errors.As(err, &exitErr)
}

func commandContext() (context.Context, context.CancelFunc) {
	return signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
}

func isInputTerminal() bool {
	return term.IsTerminal(int(os.Stdin.Fd()))
}

func isOutputTerminal() bool {
	return term.IsTerminal(int(os.Stdout.Fd()))
}

func autoOutputFormat(format string, stdoutTTY bool) string {
	format = strings.ToLower(strings.TrimSpace(format))
	if format != "" {
		return format
	}
	if stdoutTTY {
		return "pretty"
	}
	return "json"
}

func writeValue(w io.Writer, value interface{}, format string) error {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "yaml":
		data, err := yaml.Marshal(value)
		if err != nil {
			return fmt.Errorf("marshal yaml: %w", err)
		}
		_, err = w.Write(data)
		return err
	case "pretty":
		data, err := json.MarshalIndent(value, "", "  ")
		if err != nil {
			return fmt.Errorf("marshal json: %w", err)
		}
		_, err = fmt.Fprintln(w, string(data))
		return err
	default:
		data, err := json.Marshal(value)
		if err != nil {
			return fmt.Errorf("marshal json: %w", err)
		}
		_, err = fmt.Fprintln(w, string(data))
		return err
	}
}

func parseStructuredInput(data []byte, source string) (map[string]interface{}, error) {
	data = bytes.TrimSpace(data)
	if len(data) == 0 {
		return nil, fmt.Errorf("%s is empty", source)
	}

	var decoded interface{}
	ext := strings.ToLower(filepath.Ext(source))
	if ext == ".yaml" || ext == ".yml" {
		if err := yaml.Unmarshal(data, &decoded); err != nil {
			return nil, fmt.Errorf("parse yaml %s: %w", source, err)
		}
	} else {
		if err := json.Unmarshal(data, &decoded); err != nil {
			return nil, fmt.Errorf("parse json %s: %w", source, err)
		}
	}

	normalized := normalizeYAML(decoded)
	obj, ok := normalized.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("%s must decode to a JSON object", source)
	}
	return obj, nil
}

func parseInputSource(value string) (map[string]interface{}, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil, errors.New("--in cannot be empty")
	}
	if strings.HasPrefix(value, "@") {
		path := strings.TrimPrefix(value, "@")
		if strings.TrimSpace(path) == "" {
			return nil, errors.New("@ input path cannot be empty")
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read input file %q: %w", path, err)
		}
		return parseStructuredInput(data, path)
	}
	return parseStructuredInput([]byte(value), "inline input")
}

func normalizeYAML(value interface{}) interface{} {
	switch v := value.(type) {
	case map[string]interface{}:
		out := make(map[string]interface{}, len(v))
		for key, item := range v {
			out[key] = normalizeYAML(item)
		}
		return out
	case map[interface{}]interface{}:
		out := make(map[string]interface{}, len(v))
		for key, item := range v {
			out[fmt.Sprint(key)] = normalizeYAML(item)
		}
		return out
	case []interface{}:
		out := make([]interface{}, len(v))
		for i, item := range v {
			out[i] = normalizeYAML(item)
		}
		return out
	default:
		return v
	}
}

func splitReasonerTarget(target string) (string, string, error) {
	parts := strings.SplitN(strings.TrimSpace(target), ".", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return "", "", fmt.Errorf("target must be in <node>.<reasoner> format")
	}
	return parts[0], parts[1], nil
}

func httpExitCode(statusCode int) int {
	switch statusCode {
	case http.StatusUnauthorized, http.StatusForbidden:
		return 3
	default:
		if statusCode >= 500 {
			return 3
		}
		if statusCode >= 400 {
			return 2
		}
		return 0
	}
}

func terminalStatus(status string) bool {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "succeeded", "success", "completed", "failed", "error", "cancelled", "canceled", "timeout", "timed_out":
		return true
	default:
		return false
	}
}

func failedStatus(status string) bool {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "failed", "error", "cancelled", "canceled", "timeout", "timed_out":
		return true
	default:
		return false
	}
}

func extractField(value interface{}, path string) (interface{}, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return value, nil
	}
	if !strings.HasPrefix(path, ".") {
		return nil, fmt.Errorf("field path must start with '.'")
	}

	current := value
	tokens, err := parseFieldPath(path[1:])
	if err != nil {
		return nil, err
	}
	for _, token := range tokens {
		switch token.kind {
		case "key":
			obj, ok := current.(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf("field %q is not an object", token.value)
			}
			next, ok := obj[token.value]
			if !ok {
				return nil, fmt.Errorf("field %q not found", token.value)
			}
			current = next
		case "index":
			arr, ok := current.([]interface{})
			if !ok {
				return nil, fmt.Errorf("field is not an array at index %d", token.index)
			}
			if token.index < 0 || token.index >= len(arr) {
				return nil, fmt.Errorf("array index %d out of range", token.index)
			}
			current = arr[token.index]
		}
	}
	return current, nil
}

type fieldToken struct {
	kind  string
	value string
	index int
}

func parseFieldPath(path string) ([]fieldToken, error) {
	var tokens []fieldToken
	for path != "" {
		keyEnd := strings.IndexAny(path, ".[")
		if keyEnd == -1 {
			keyEnd = len(path)
		}
		if keyEnd > 0 {
			tokens = append(tokens, fieldToken{kind: "key", value: path[:keyEnd]})
			path = path[keyEnd:]
		}
		if strings.HasPrefix(path, ".") {
			path = path[1:]
			continue
		}
		if strings.HasPrefix(path, "[") {
			closeIdx := strings.Index(path, "]")
			if closeIdx < 0 {
				return nil, fmt.Errorf("missing closing ] in field path")
			}
			index, err := strconv.Atoi(path[1:closeIdx])
			if err != nil {
				return nil, fmt.Errorf("invalid array index %q", path[1:closeIdx])
			}
			tokens = append(tokens, fieldToken{kind: "index", index: index})
			path = path[closeIdx+1:]
			continue
		}
		if path != "" {
			return nil, fmt.Errorf("invalid field path near %q", path)
		}
	}
	return tokens, nil
}

func makeRequest(ctx context.Context, method, path string, body interface{}, accept string) (*http.Response, error) {
	server := strings.TrimRight(GetServerURL(), "/")
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}

	var requestBody io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("encode request body: %w", err)
		}
		requestBody = bytes.NewReader(encoded)
	}

	req, err := http.NewRequestWithContext(ctx, method, server+path, requestBody)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	if accept == "" {
		accept = "application/json"
	}
	req.Header.Set("Accept", accept)
	req.Header.Set("User-Agent", "af-cli/triggers")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if key := strings.TrimSpace(GetAPIKey()); key != "" {
		req.Header.Set("X-API-Key", key)
	}

	client := triggerHTTPClient(accept)
	resp, err := client.Do(req)
	if err != nil {
		// A cancelled context (Ctrl-C / timeout the caller already handles)
		// should surface as-is; only a genuine transport failure gets the
		// "start the control plane" guidance.
		if ctx.Err() != nil {
			return nil, err
		}
		return nil, controlPlaneUnreachableError(err)
	}
	return resp, nil
}

// controlPlaneUnreachableError wraps a transport-level failure to reach the
// control plane with a consistent, actionable hint. Every CLI command that
// talks to the control plane (call/ls/tail/wait) routes through makeRequest,
// so wrapping here surfaces the same guidance everywhere instead of leaking a
// bare Go dial error to a harness driving the CLI.
func controlPlaneUnreachableError(err error) error {
	// The trailing period and embedded newline are deliberate: this is a
	// top-level, user-facing CLI hint printed to a harness, not an error meant
	// to be wrapped mid-sentence, so ST1005 does not apply here.
	//nolint:staticcheck // deliberate multi-line user-facing hint
	return fmt.Errorf("%w\nControl plane not reachable at %s. Start it with `af server` or launch the AgentField desktop app.",
		err, strings.TrimRight(GetServerURL(), "/"))
}

func triggerHTTPClient(accept string) *http.Client {
	if strings.EqualFold(strings.TrimSpace(accept), "text/event-stream") {
		return &http.Client{}
	}
	timeout := GetRequestTimeout()
	if timeout <= 0 {
		timeout = 30
	}
	return &http.Client{Timeout: time.Duration(timeout) * time.Second}
}

func readJSONResponse(resp *http.Response, out interface{}) ([]byte, error) {
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	if len(bytes.TrimSpace(body)) == 0 {
		return body, nil
	}
	if out != nil {
		if err := json.Unmarshal(body, out); err != nil {
			return body, fmt.Errorf("decode response: %w", err)
		}
	}
	return body, nil
}

func appendQuery(path string, values url.Values) string {
	if encoded := values.Encode(); encoded != "" {
		return path + "?" + encoded
	}
	return path
}

func scanSSE(r io.Reader, onEvent func(map[string]interface{}) bool) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	var data strings.Builder
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if data.Len() > 0 {
				var event map[string]interface{}
				if err := json.Unmarshal([]byte(data.String()), &event); err != nil {
					return fmt.Errorf("decode sse data: %w", err)
				}
				data.Reset()
				if !onEvent(event) {
					return nil
				}
			}
			continue
		}
		if strings.HasPrefix(line, "data:") {
			data.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return nil
}
