package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/spf13/cobra"
)

type sessionStartOptions struct {
	provider     string
	transport    string
	model        string
	voice        string
	outputFormat string
	stdout       io.Writer
}

type sessionToolOptions struct {
	target       string
	inputSource  string
	outputFormat string
	stdin        io.Reader
	stdout       io.Writer
}

type sessionOfferOptions struct {
	provider     string
	transport    string
	sdpSource    string
	outputFormat string
	stdin        io.Reader
	stdout       io.Writer
}

func NewSessionCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "session",
		Short: "Start and interact with AgentField realtime sessions",
	}
	cmd.AddCommand(newSessionStartCommand())
	cmd.AddCommand(newSessionOfferCommand())
	cmd.AddCommand(newSessionToolCommand())
	cmd.AddCommand(newSessionWorkflowsCommand())
	return cmd
}

func newSessionStartCommand() *cobra.Command {
	opts := &sessionStartOptions{}
	cmd := &cobra.Command{
		Use:   "start <node>.<session>",
		Short: "Start a provider-backed AgentField session",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx, cancel := commandContext()
			defer cancel()
			opts.stdout = os.Stdout
			return runSessionStart(ctx, args[0], opts)
		},
	}
	cmd.Flags().StringVar(&opts.provider, "provider", "", "Explicit session provider, e.g. openai")
	cmd.Flags().StringVar(&opts.transport, "transport", "", "Explicit session transport, e.g. webrtc")
	cmd.Flags().StringVar(&opts.model, "model", "", "Provider model")
	cmd.Flags().StringVar(&opts.voice, "voice", "", "Provider voice")
	cmd.Flags().StringVarP(&opts.outputFormat, "output", "o", "json", "Output format: json, pretty, yaml")
	return cmd
}

func runSessionStart(ctx context.Context, target string, opts *sessionStartOptions) error {
	if opts.stdout == nil {
		opts.stdout = os.Stdout
	}
	payload := map[string]interface{}{
		"provider":  opts.provider,
		"transport": opts.transport,
		"model":     opts.model,
		"voice":     opts.voice,
	}
	resp, err := makeRequest(ctx, http.MethodPost, "/api/v1/session-targets/"+target+"/start", payload, "application/json")
	if err != nil {
		return cliExitError{Code: 3, Err: err}
	}
	var decoded map[string]interface{}
	body, err := readJSONResponse(resp, &decoded)
	if err != nil {
		return cliExitError{Code: 3, Err: err}
	}
	if resp.StatusCode >= http.StatusBadRequest {
		return cliExitError{Code: httpExitCode(resp.StatusCode), Err: fmt.Errorf("session start failed with status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))}
	}
	return writeValue(opts.stdout, decoded, autoOutputFormat(opts.outputFormat, false))
}

func newSessionOfferCommand() *cobra.Command {
	opts := &sessionOfferOptions{}
	cmd := &cobra.Command{
		Use:   "offer <session_id>",
		Short: "Create a realtime WebRTC offer through the control plane",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx, cancel := commandContext()
			defer cancel()
			opts.stdin = os.Stdin
			opts.stdout = os.Stdout
			return runSessionOffer(ctx, args[0], opts)
		},
	}
	cmd.Flags().StringVar(&opts.provider, "provider", "", "Explicit session provider")
	cmd.Flags().StringVar(&opts.transport, "transport", "", "Explicit session transport")
	cmd.Flags().StringVar(&opts.sdpSource, "sdp", "", "SDP offer as inline text, @path, or - for stdin; defaults to stdin")
	cmd.Flags().StringVarP(&opts.outputFormat, "output", "o", "raw", "Output format: raw, json, pretty, yaml")
	return cmd
}

func runSessionOffer(ctx context.Context, sessionID string, opts *sessionOfferOptions) error {
	if opts.stdout == nil {
		opts.stdout = os.Stdout
	}
	sdp, err := readSessionSDP(opts.sdpSource, opts.stdin)
	if err != nil {
		return cliExitError{Code: 2, Err: err}
	}
	values := url.Values{}
	if strings.TrimSpace(opts.provider) != "" {
		values.Set("provider", opts.provider)
	}
	if strings.TrimSpace(opts.transport) != "" {
		values.Set("transport", opts.transport)
	}
	path := "/api/v1/session-instances/" + url.PathEscape(sessionID) + "/realtime-offer"
	if encoded := values.Encode(); encoded != "" {
		path += "?" + encoded
	}
	resp, err := makeRawRequest(ctx, http.MethodPost, path, strings.NewReader(sdp), "application/sdp", "application/sdp")
	if err != nil {
		return cliExitError{Code: 3, Err: err}
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return cliExitError{Code: 3, Err: fmt.Errorf("read response: %w", err)}
	}
	if resp.StatusCode >= http.StatusBadRequest {
		return cliExitError{Code: httpExitCode(resp.StatusCode), Err: fmt.Errorf("session offer failed with status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))}
	}

	format := strings.ToLower(strings.TrimSpace(opts.outputFormat))
	if format == "" || format == "raw" {
		_, err := opts.stdout.Write(body)
		return err
	}
	return writeValue(opts.stdout, map[string]interface{}{"answer_sdp": string(body)}, autoOutputFormat(format, false))
}

func readSessionSDP(source string, stdin io.Reader) (string, error) {
	sourceToken := strings.TrimSpace(source)
	switch {
	case sourceToken == "" || sourceToken == "-":
		if stdin == nil {
			return "", fmt.Errorf("SDP offer required; pass --sdp, --sdp @path, or pipe SDP on stdin")
		}
		data, err := io.ReadAll(stdin)
		if err != nil {
			return "", fmt.Errorf("read SDP from stdin: %w", err)
		}
		if strings.TrimSpace(string(data)) == "" {
			return "", fmt.Errorf("SDP offer required; pass --sdp, --sdp @path, or pipe SDP on stdin")
		}
		return string(data), nil
	case strings.HasPrefix(sourceToken, "@"):
		path := strings.TrimSpace(strings.TrimPrefix(sourceToken, "@"))
		if path == "" {
			return "", fmt.Errorf("SDP file path is required after @")
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return "", fmt.Errorf("read SDP file %s: %w", path, err)
		}
		if strings.TrimSpace(string(data)) == "" {
			return "", fmt.Errorf("SDP file %s is empty", path)
		}
		return string(data), nil
	default:
		if strings.TrimSpace(source) == "" {
			return "", fmt.Errorf("SDP offer required; pass --sdp, --sdp @path, or pipe SDP on stdin")
		}
		return source, nil
	}
}

func makeRawRequest(ctx context.Context, method, path string, body io.Reader, contentType string, accept string) (*http.Response, error) {
	server := strings.TrimRight(GetServerURL(), "/")
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	req, err := http.NewRequestWithContext(ctx, method, server+path, body)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	if accept == "" {
		accept = "application/json"
	}
	req.Header.Set("Accept", accept)
	req.Header.Set("User-Agent", "af-cli/session")
	if strings.TrimSpace(contentType) != "" {
		req.Header.Set("Content-Type", contentType)
	}
	if key := strings.TrimSpace(GetAPIKey()); key != "" {
		req.Header.Set("X-API-Key", key)
	}
	client := triggerHTTPClient(accept)
	return client.Do(req)
}

func newSessionToolCommand() *cobra.Command {
	opts := &sessionToolOptions{}
	cmd := &cobra.Command{
		Use:   "tool <session_id> <tool>",
		Short: "Invoke a session tool through AgentField execute/async",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx, cancel := commandContext()
			defer cancel()
			opts.stdin = os.Stdin
			opts.stdout = os.Stdout
			return runSessionTool(ctx, args[0], args[1], opts)
		},
	}
	cmd.Flags().StringVar(&opts.target, "target", "", "Explicit <node>.<reasoner> target")
	cmd.Flags().StringVar(&opts.inputSource, "in", "", "Input payload as inline JSON or @path")
	cmd.Flags().StringVarP(&opts.outputFormat, "output", "o", "json", "Output format: json, pretty, yaml")
	return cmd
}

func runSessionTool(ctx context.Context, sessionID string, tool string, opts *sessionToolOptions) error {
	input := map[string]interface{}{}
	if strings.TrimSpace(opts.inputSource) != "" {
		parsed, err := parseInputSource(opts.inputSource)
		if err != nil {
			return cliExitError{Code: 2, Err: err}
		}
		input = parsed
	} else if opts.stdin != nil {
		data, _ := io.ReadAll(opts.stdin)
		if len(strings.TrimSpace(string(data))) > 0 {
			if err := json.Unmarshal(data, &input); err != nil {
				return cliExitError{Code: 2, Err: fmt.Errorf("parse stdin JSON: %w", err)}
			}
		}
	}
	payload := map[string]interface{}{"target": opts.target, "input": input}
	resp, err := makeRequest(ctx, http.MethodPost, "/api/v1/session-instances/"+sessionID+"/tools/"+tool, payload, "application/json")
	if err != nil {
		return cliExitError{Code: 3, Err: err}
	}
	var decoded map[string]interface{}
	body, err := readJSONResponse(resp, &decoded)
	if err != nil {
		return cliExitError{Code: 3, Err: err}
	}
	if resp.StatusCode >= http.StatusBadRequest {
		return cliExitError{Code: httpExitCode(resp.StatusCode), Err: fmt.Errorf("session tool failed with status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))}
	}
	return writeValue(opts.stdout, decoded, autoOutputFormat(opts.outputFormat, false))
}

func newSessionWorkflowsCommand() *cobra.Command {
	var output string
	cmd := &cobra.Command{
		Use:   "workflows <session_id>",
		Short: "List workflows associated with a session",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx, cancel := commandContext()
			defer cancel()
			resp, err := makeRequest(ctx, http.MethodPost, "/api/v1/agentic/query", map[string]interface{}{
				"resource": "workflows",
				"filters":  map[string]interface{}{"session_id": args[0]},
			}, "application/json")
			if err != nil {
				return cliExitError{Code: 3, Err: err}
			}
			var decoded map[string]interface{}
			body, err := readJSONResponse(resp, &decoded)
			if err != nil {
				return cliExitError{Code: 3, Err: err}
			}
			if resp.StatusCode >= http.StatusBadRequest {
				return cliExitError{Code: httpExitCode(resp.StatusCode), Err: fmt.Errorf("session workflows failed with status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))}
			}
			return writeValue(os.Stdout, decoded, autoOutputFormat(output, false))
		},
	}
	cmd.Flags().StringVarP(&output, "output", "o", "json", "Output format: json, pretty, yaml")
	return cmd
}
