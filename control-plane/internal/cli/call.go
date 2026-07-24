package cli

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
)

type callOptions struct {
	inputSource    string
	printSchema    bool
	async          bool
	noInteractive  bool
	interactive    bool
	outputFormat   string
	fieldPath      string
	stdinTTY       bool
	stdoutTTY      bool
	stdin          io.Reader
	stderr         io.Writer
	stdout         io.Writer
	promptAnswered bool
}

type callResponse struct {
	ExecutionID  string      `json:"execution_id"`
	RunID        string      `json:"run_id"`
	WorkflowID   string      `json:"workflow_id"`
	Status       string      `json:"status"`
	Result       interface{} `json:"result,omitempty"`
	ErrorMessage *string     `json:"error_message,omitempty"`
	Error        *string     `json:"error,omitempty"`
}

func NewCallCommand() *cobra.Command {
	opts := &callOptions{}
	cmd := &cobra.Command{
		Use:   "call <node>.<reasoner>",
		Short: "Trigger a reasoner",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			opts.stdinTTY = isInputTerminal()
			opts.stdoutTTY = isOutputTerminal()
			opts.stdin = os.Stdin
			opts.stdout = os.Stdout
			opts.stderr = os.Stderr
			ctx, cancel := commandContext()
			defer cancel()
			return runCall(ctx, args[0], opts)
		},
		SilenceUsage: true,
	}
	cmd.Flags().StringVar(&opts.inputSource, "in", "", "Input payload as inline JSON or @path/to/file.{json,yaml}")
	cmd.Flags().BoolVar(&opts.printSchema, "schema", false, "Print the reasoner's input schema and exit")
	cmd.Flags().BoolVar(&opts.async, "async", false, "Return run_id immediately without streaming")
	cmd.Flags().BoolVar(&opts.noInteractive, "no-interactive", false, "Never prompt for input")
	cmd.Flags().BoolVar(&opts.interactive, "interactive", false, "Force interactive prompts")
	cmd.Flags().StringVarP(&opts.outputFormat, "output", "o", "", "Output format: pretty, json, yaml")
	cmd.Flags().StringVar(&opts.fieldPath, "field", "", "Extract a single field from the result, e.g. .findings[0].score")
	return cmd
}

func runCall(ctx context.Context, target string, opts *callOptions) error {
	if opts == nil {
		opts = &callOptions{stdin: os.Stdin, stdout: os.Stdout, stderr: os.Stderr, stdinTTY: isInputTerminal(), stdoutTTY: isOutputTerminal()}
	}
	if opts.stdout == nil {
		opts.stdout = os.Stdout
	}
	if opts.stderr == nil {
		opts.stderr = os.Stderr
	}
	if opts.stdin == nil {
		opts.stdin = os.Stdin
	}
	format := autoOutputFormat(opts.outputFormat, opts.stdoutTTY)
	if format != "pretty" && format != "json" && format != "yaml" {
		return cliExitError{Code: 2, Err: fmt.Errorf("output format must be pretty, json, or yaml")}
	}
	if opts.interactive && opts.noInteractive {
		return cliExitError{Code: 2, Err: fmt.Errorf("--interactive and --no-interactive cannot be used together")}
	}
	node, reasoner, err := splitReasonerTarget(target)
	if err != nil {
		return cliExitError{Code: 2, Err: err}
	}

	schema, schemaErr := fetchReasonerSchema(ctx, node, reasoner)
	if opts.printSchema {
		if schemaErr != nil {
			return cliExitError{Code: 3, Err: schemaErr}
		}
		if schema == nil {
			schema = map[string]interface{}{}
		}
		if err := writeValue(opts.stdout, schema, format); err != nil {
			return cliExitError{Code: 2, Err: err}
		}
		return nil
	}

	input, err := resolveCallInput(opts, schema, schemaErr)
	if err != nil {
		return err
	}
	if schemaErr == nil {
		if err := validateInputAgainstSchema(input, schema); err != nil {
			return cliExitError{Code: 2, Err: err}
		}
	}

	if opts.async {
		resp, err := executeReasoner(ctx, target, input, true)
		if err != nil {
			return err
		}
		runID := firstNonEmptyString(resp.RunID, resp.WorkflowID, resp.ExecutionID)
		if runID == "" {
			return cliExitError{Code: 3, Err: fmt.Errorf("server accepted async execution without run_id")}
		}
		// When a machine format is explicitly requested (-o json/-o yaml), emit a
		// structured envelope so a harness parsing stdout gets valid JSON instead
		// of a bare token. The default/pretty path keeps the bare run-id line that
		// shell scripts capture via `RUN_ID=$(af call node.reasoner --async)`.
		if requested := strings.ToLower(strings.TrimSpace(opts.outputFormat)); requested == "json" || requested == "yaml" {
			return writeValue(opts.stdout, map[string]interface{}{"run_id": runID, "status": "accepted"}, requested)
		}
		_, _ = fmt.Fprintln(opts.stdout, runID)
		return nil
	}

	if opts.stdoutTTY {
		resp, err := executeReasoner(ctx, target, input, true)
		if err != nil {
			return err
		}
		runID := firstNonEmptyString(resp.RunID, resp.WorkflowID, resp.ExecutionID)
		executionID := firstNonEmptyString(resp.ExecutionID, runID)
		if runID == "" {
			return cliExitError{Code: 3, Err: fmt.Errorf("server accepted execution without run_id")}
		}
		tailErr := streamExecutionEvents(ctx, runID, 0, "pretty", opts.stderr)
		if tailErr != nil {
			if ctx.Err() != nil {
				fmt.Fprintf(opts.stderr, "\nDetached. Resume with: af tail %s\n", runID)
				return nil
			}
			fmt.Fprintf(opts.stderr, "warning: progress stream ended: %v\n", tailErr)
		}
		status, err := fetchExecutionStatus(ctx, executionID)
		if err != nil {
			return err
		}
		return printCallResult(opts.stdout, opts.stderr, status, format, opts.fieldPath)
	}

	resp, err := executeReasoner(ctx, target, input, false)
	if err != nil {
		return err
	}
	return printCallResult(opts.stdout, opts.stderr, resp, format, opts.fieldPath)
}

func resolveCallInput(opts *callOptions, schema map[string]interface{}, schemaErr error) (map[string]interface{}, error) {
	if strings.TrimSpace(opts.inputSource) != "" {
		input, err := parseInputSource(opts.inputSource)
		if err != nil {
			return nil, cliExitError{Code: 2, Err: err}
		}
		return input, nil
	}

	interactive := opts.interactive || (opts.stdinTTY && !opts.noInteractive)
	if !opts.stdinTTY && !opts.interactive {
		data, err := io.ReadAll(opts.stdin)
		if err != nil {
			return nil, cliExitError{Code: 2, Err: fmt.Errorf("read stdin: %w", err)}
		}
		if len(bytes.TrimSpace(data)) > 0 {
			input, err := parseStructuredInput(data, "stdin")
			if err != nil {
				return nil, cliExitError{Code: 2, Err: err}
			}
			return input, nil
		}
		if schemaErr != nil {
			return nil, cliExitError{Code: 3, Err: schemaErr}
		}
		if schemaRequiresInput(schema) {
			return nil, cliExitError{Code: 2, Err: fmt.Errorf("input required; pass --in, pipe JSON to stdin, or run interactively")}
		}
		return map[string]interface{}{}, nil
	}

	if schemaErr != nil {
		return nil, cliExitError{Code: 3, Err: schemaErr}
	}
	if !schemaRequiresInput(schema) {
		return defaultsFromSchema(schema), nil
	}
	if !interactive {
		return nil, cliExitError{Code: 2, Err: fmt.Errorf("input required and prompting is disabled; pass --in or stdin")}
	}
	input, err := promptForSchema(opts.stdin, opts.stderr, schema)
	if err != nil {
		return nil, cliExitError{Code: 2, Err: err}
	}
	return input, nil
}

func executeReasoner(ctx context.Context, target string, input map[string]interface{}, async bool) (*callResponse, error) {
	path := "/api/v1/execute/" + url.PathEscape(target)
	if async {
		path = "/api/v1/execute/async/" + url.PathEscape(target)
	}
	resp, err := makeRequest(ctx, http.MethodPost, path, map[string]interface{}{"input": input}, "application/json")
	if err != nil {
		return nil, cliExitError{Code: 3, Err: err}
	}
	var decoded callResponse
	body, err := readJSONResponse(resp, &decoded)
	if err != nil {
		return nil, cliExitError{Code: 3, Err: err}
	}
	if resp.StatusCode >= http.StatusBadRequest {
		if failedStatus(decoded.Status) {
			return &decoded, cliExitError{Code: 1, Err: fmt.Errorf("execution failed: %s", strings.TrimSpace(string(body)))}
		}
		return nil, cliExitError{Code: httpExitCode(resp.StatusCode), Err: fmt.Errorf("request failed with status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))}
	}
	if decoded.Status != "" && failedStatus(decoded.Status) {
		return &decoded, cliExitError{Code: 1, Err: fmt.Errorf("execution failed: %s", decoded.Status)}
	}
	return &decoded, nil
}

func fetchExecutionStatus(ctx context.Context, executionID string) (*callResponse, error) {
	resp, err := makeRequest(ctx, http.MethodGet, "/api/v1/executions/"+url.PathEscape(executionID), nil, "application/json")
	if err != nil {
		return nil, cliExitError{Code: 3, Err: err}
	}
	var decoded callResponse
	body, err := readJSONResponse(resp, &decoded)
	if err != nil {
		return nil, cliExitError{Code: 3, Err: err}
	}
	if resp.StatusCode >= http.StatusBadRequest {
		return nil, cliExitError{Code: httpExitCode(resp.StatusCode), Err: fmt.Errorf("status request failed with status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))}
	}
	return &decoded, nil
}

func printCallResult(stdout, stderr io.Writer, resp *callResponse, format, fieldPath string) error {
	if resp == nil {
		return cliExitError{Code: 3, Err: fmt.Errorf("empty execution response")}
	}
	if failedStatus(resp.Status) {
		msg := firstNonEmptyString(pointerValue(resp.ErrorMessage), pointerValue(resp.Error), "execution failed")
		fmt.Fprintln(stderr, msg)
		return cliExitError{Code: 1, Err: fmt.Errorf("%s", msg)}
	}
	result := resp.Result
	if fieldPath != "" {
		extracted, err := extractField(result, fieldPath)
		if err != nil {
			return cliExitError{Code: 2, Err: err}
		}
		result = extracted
	}
	if err := writeValue(stdout, result, format); err != nil {
		return cliExitError{Code: 2, Err: err}
	}
	return nil
}

func fetchReasonerSchema(ctx context.Context, node, reasoner string) (map[string]interface{}, error) {
	values := url.Values{}
	values.Set("agent", node)
	values.Set("reasoner", reasoner)
	values.Set("include_input_schema", "true")
	values.Set("include_descriptions", "false")
	values.Set("limit", "1")
	resp, err := makeRequest(ctx, http.MethodGet, appendQuery("/api/v1/discovery/capabilities", values), nil, "application/json")
	if err != nil {
		return nil, err
	}
	var decoded struct {
		Capabilities []struct {
			Reasoners []struct {
				ID          string                 `json:"id"`
				InputSchema map[string]interface{} `json:"input_schema"`
			} `json:"reasoners"`
		} `json:"capabilities"`
	}
	body, err := readJSONResponse(resp, &decoded)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= http.StatusBadRequest {
		return nil, fmt.Errorf("schema request failed with status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	for _, capability := range decoded.Capabilities {
		for _, item := range capability.Reasoners {
			if item.ID == reasoner {
				return item.InputSchema, nil
			}
		}
	}
	return nil, fmt.Errorf("reasoner %s.%s not found", node, reasoner)
}

func schemaRequiresInput(schema map[string]interface{}) bool {
	required, _ := schema["required"].([]interface{})
	return len(required) > 0
}

func defaultsFromSchema(schema map[string]interface{}) map[string]interface{} {
	out := map[string]interface{}{}
	props, _ := schema["properties"].(map[string]interface{})
	for name, raw := range props {
		prop, _ := raw.(map[string]interface{})
		if value, ok := prop["default"]; ok {
			out[name] = value
		}
	}
	return out
}

func promptForSchema(stdin io.Reader, stderr io.Writer, schema map[string]interface{}) (map[string]interface{}, error) {
	reader := bufio.NewReader(stdin)
	input := defaultsFromSchema(schema)
	required := requiredFields(schema)
	props, _ := schema["properties"].(map[string]interface{})
	for _, name := range required {
		prop, _ := props[name].(map[string]interface{})
		for {
			fmt.Fprintf(stderr, "%s: ", name)
			line, err := reader.ReadString('\n')
			if err != nil && err != io.EOF {
				return nil, fmt.Errorf("read prompt: %w", err)
			}
			value, err := parsePromptValue(strings.TrimSpace(line), prop)
			if err != nil {
				fmt.Fprintf(stderr, "invalid %s: %v\n", name, err)
				continue
			}
			input[name] = value
			break
		}
	}
	fmt.Fprint(stderr, "Submit? [y/N]: ")
	line, err := reader.ReadString('\n')
	if err != nil && err != io.EOF {
		return nil, fmt.Errorf("read confirmation: %w", err)
	}
	switch strings.ToLower(strings.TrimSpace(line)) {
	case "y", "yes":
		return input, nil
	default:
		return nil, fmt.Errorf("cancelled")
	}
}

func requiredFields(schema map[string]interface{}) []string {
	raw, _ := schema["required"].([]interface{})
	fields := make([]string, 0, len(raw))
	for _, item := range raw {
		if s, ok := item.(string); ok && strings.TrimSpace(s) != "" {
			fields = append(fields, s)
		}
	}
	return fields
}

func parsePromptValue(value string, property map[string]interface{}) (interface{}, error) {
	if value == "" {
		if def, ok := property["default"]; ok {
			return def, nil
		}
		return nil, fmt.Errorf("required")
	}
	switch strings.ToLower(fmt.Sprint(property["type"])) {
	case "integer":
		return strconv.Atoi(value)
	case "number":
		return strconv.ParseFloat(value, 64)
	case "boolean":
		return strconv.ParseBool(value)
	case "object", "array":
		var decoded interface{}
		if err := json.Unmarshal([]byte(value), &decoded); err != nil {
			return nil, err
		}
		return decoded, nil
	default:
		return value, nil
	}
}

func validateInputAgainstSchema(input map[string]interface{}, schema map[string]interface{}) error {
	if schema == nil {
		return nil
	}
	for _, name := range requiredFields(schema) {
		if _, ok := input[name]; !ok {
			return fmt.Errorf("missing required field %q", name)
		}
	}
	props, _ := schema["properties"].(map[string]interface{})
	for name, raw := range props {
		value, exists := input[name]
		if !exists || value == nil {
			continue
		}
		prop, _ := raw.(map[string]interface{})
		if err := validateSchemaType(name, value, strings.ToLower(fmt.Sprint(prop["type"]))); err != nil {
			return err
		}
	}
	return nil
}

func validateSchemaType(name string, value interface{}, schemaType string) error {
	switch schemaType {
	case "", "string":
		if schemaType == "string" {
			if _, ok := value.(string); !ok {
				return fmt.Errorf("field %q must be a string", name)
			}
		}
	case "integer":
		switch value.(type) {
		case int, int64, float64:
			if f, ok := value.(float64); ok && f != float64(int64(f)) {
				return fmt.Errorf("field %q must be an integer", name)
			}
		default:
			return fmt.Errorf("field %q must be an integer", name)
		}
	case "number":
		switch value.(type) {
		case int, int64, float64:
		default:
			return fmt.Errorf("field %q must be a number", name)
		}
	case "boolean":
		if _, ok := value.(bool); !ok {
			return fmt.Errorf("field %q must be a boolean", name)
		}
	case "object", "array":
		// The Python SDK serializes every Optional[...] parameter as
		// {"type": "object"} (e.g. `pr_url: str | None` becomes an object), so
		// enforcing the declared type here would wrongly reject a concrete scalar
		// passed to an optional field. The control plane validates structured
		// input itself and is the source of truth, so skip the client-side check
		// for object/array rather than reject input the server would accept.
	}
	return nil
}

func pointerValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}
