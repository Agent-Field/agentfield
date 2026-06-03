package cli

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

type AgentResponse struct {
	OK    bool        `json:"ok"`
	Data  interface{} `json:"data,omitempty"`
	Error *AgentError `json:"error,omitempty"`
	Meta  *AgentMeta  `json:"meta,omitempty"`
}

type AgentError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Hint    string `json:"hint,omitempty"`
}

type AgentMeta struct {
	Server     string `json:"server"`
	Latency    string `json:"latency,omitempty"`
	StatusCode int    `json:"status_code,omitempty"`
}

func NewAgentCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "agent",
		Short: "Agent-mode JSON interface for agentic APIs",
		Long:  "Machine-friendly CLI wrapper around /api/v1/agentic endpoints.",
		Args:  cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) == 0 || args[0] == "help" {
				agentOutput(agentHelpData())
				return nil
			}
			available := []string{}
			for _, sub := range cmd.Commands() {
				if !sub.Hidden && sub.Name() != "help" && sub.Name() != "completion" {
					available = append(available, sub.Name())
				}
			}
			agentError(
				"unknown_command",
				fmt.Sprintf("unknown agent subcommand: %s", args[0]),
				fmt.Sprintf("Available commands: %s. Run 'af agent help' for full structured reference.", strings.Join(available, ", ")),
			)
			return nil
		},
		SilenceErrors: true,
		SilenceUsage:  true,
	}

	cmd.PersistentFlags().StringVarP(&outputFormat, "output", "o", "json", "Output format: json, compact")
	cmd.PersistentFlags().IntVarP(&requestTimeout, "timeout", "t", 30, "Request timeout in seconds")

	cmd.AddCommand(newAgentStatusCmd())
	cmd.AddCommand(newAgentDiscoverCmd())
	cmd.AddCommand(newAgentQueryCmd())
	cmd.AddCommand(newAgentRunCmd())
	cmd.AddCommand(newAgentAgentSummaryCmd())
	cmd.AddCommand(newAgentKBCmd())
	cmd.AddCommand(newAgentBatchCmd())
	cmd.SetHelpCommand(newAgentHelpCmd())

	return cmd
}

func newAgentStatusCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "status",
		Short: "Get system status summary",
		Run: func(cmd *cobra.Command, args []string) {
			proxyToServer(http.MethodGet, "/api/v1/agentic/status", nil)
		},
	}
}

func newAgentDiscoverCmd() *cobra.Command {
	var query string
	var group string
	var method string
	var limit int

	cmd := &cobra.Command{
		Use:   "discover",
		Short: "Search agentic API endpoint catalog",
		Run: func(cmd *cobra.Command, args []string) {
			params := url.Values{}
			if strings.TrimSpace(query) != "" {
				params.Set("q", strings.TrimSpace(query))
			}
			if strings.TrimSpace(group) != "" {
				params.Set("group", strings.TrimSpace(group))
			}
			if strings.TrimSpace(method) != "" {
				params.Set("method", strings.ToUpper(strings.TrimSpace(method)))
			}
			if limit > 0 {
				params.Set("limit", strconv.Itoa(limit))
			}

			path := "/api/v1/agentic/discover"
			if encoded := params.Encode(); encoded != "" {
				path += "?" + encoded
			}
			proxyToServer(http.MethodGet, path, nil)
		},
	}

	cmd.Flags().StringVarP(&query, "query", "q", "", "Search query")
	cmd.Flags().StringVarP(&group, "group", "g", "", "Endpoint group filter")
	cmd.Flags().StringVar(&method, "method", "", "HTTP method filter")
	cmd.Flags().IntVar(&limit, "limit", 20, "Result limit")

	return cmd
}

func newAgentQueryCmd() *cobra.Command {
	var resource string
	var status string
	var agentID string
	var runID string
	var since string
	var until string
	var limit int
	var offset int
	var include string

	cmd := &cobra.Command{
		Use:   "query",
		Short: "Run unified resource query",
		Run: func(cmd *cobra.Command, args []string) {
			resource = strings.TrimSpace(resource)
			if resource == "" {
				agentError("missing_required_flag", "--resource is required", "Set --resource to one of runs, executions, agents, workflows, sessions.")
			}

			filters := map[string]string{}
			if v := strings.TrimSpace(status); v != "" {
				filters["status"] = v
			}
			if v := strings.TrimSpace(agentID); v != "" {
				filters["agent_id"] = v
			}
			if v := strings.TrimSpace(runID); v != "" {
				filters["run_id"] = v
			}
			if v := strings.TrimSpace(since); v != "" {
				filters["since"] = v
			}
			if v := strings.TrimSpace(until); v != "" {
				filters["until"] = v
			}

			payload := map[string]interface{}{
				"resource": resource,
				"filters":  filters,
				"limit":    limit,
				"offset":   offset,
			}

			if strings.TrimSpace(include) != "" {
				parts := strings.Split(include, ",")
				includes := make([]string, 0, len(parts))
				for _, p := range parts {
					if trimmed := strings.TrimSpace(p); trimmed != "" {
						includes = append(includes, trimmed)
					}
				}
				if len(includes) > 0 {
					payload["include"] = includes
				}
			}

			proxyToServer(http.MethodPost, "/api/v1/agentic/query", payload)
		},
	}

	cmd.Flags().StringVarP(&resource, "resource", "r", "", "Resource type: runs, executions, agents, workflows, sessions")
	cmd.Flags().StringVar(&status, "status", "", "Status filter")
	cmd.Flags().StringVar(&agentID, "agent-id", "", "Agent ID filter")
	cmd.Flags().StringVar(&runID, "run-id", "", "Run ID filter")
	cmd.Flags().StringVar(&since, "since", "", "RFC3339 lower time bound")
	cmd.Flags().StringVar(&until, "until", "", "RFC3339 upper time bound")
	cmd.Flags().IntVar(&limit, "limit", 20, "Result limit")
	cmd.Flags().IntVar(&offset, "offset", 0, "Result offset")
	cmd.Flags().StringVar(&include, "include", "", "Comma-separated related resources")

	return cmd
}

func newAgentRunCmd() *cobra.Command {
	var runID string

	cmd := &cobra.Command{
		Use:   "run",
		Short: "Fetch run overview by ID",
		Run: func(cmd *cobra.Command, args []string) {
			runID = strings.TrimSpace(runID)
			if runID == "" {
				agentError("missing_required_flag", "--id is required", "Provide a run ID, for example: af agent run --id run_123")
			}
			proxyToServer(http.MethodGet, "/api/v1/agentic/run/"+url.PathEscape(runID), nil)
		},
	}

	cmd.Flags().StringVar(&runID, "id", "", "Run ID")
	return cmd
}

func newAgentAgentSummaryCmd() *cobra.Command {
	var agentID string

	cmd := &cobra.Command{
		Use:   "agent-summary",
		Short: "Fetch agent summary by ID",
		Run: func(cmd *cobra.Command, args []string) {
			agentID = strings.TrimSpace(agentID)
			if agentID == "" {
				agentError("missing_required_flag", "--id is required", "Provide an agent ID, for example: af agent agent-summary --id analyst")
			}
			proxyToServer(http.MethodGet, "/api/v1/agentic/agent/"+url.PathEscape(agentID)+"/summary", nil)
		},
	}

	cmd.Flags().StringVar(&agentID, "id", "", "Agent ID")
	return cmd
}

func newAgentKBCmd() *cobra.Command {
	kbCmd := &cobra.Command{
		Use:   "kb",
		Short: "Knowledge base commands",
		Run: func(cmd *cobra.Command, args []string) {
			agentOutput(map[string]interface{}{
				"message": "Use a kb subcommand",
				"available": []string{
					"topics",
					"search",
					"read",
					"guide",
				},
			})
		},
	}

	kbCmd.AddCommand(newAgentKBTopicsCmd())
	kbCmd.AddCommand(newAgentKBSearchCmd())
	kbCmd.AddCommand(newAgentKBReadCmd())
	kbCmd.AddCommand(newAgentKBGuideCmd())

	return kbCmd
}

func newAgentKBTopicsCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "topics",
		Short: "List KB topics",
		Run: func(cmd *cobra.Command, args []string) {
			proxyToServer(http.MethodGet, "/api/v1/agentic/kb/topics", nil)
		},
	}
}

func newAgentKBSearchCmd() *cobra.Command {
	var query string
	var topic string
	var sdk string
	var difficulty string
	var limit int

	cmd := &cobra.Command{
		Use:   "search",
		Short: "Search KB articles",
		Run: func(cmd *cobra.Command, args []string) {
			params := url.Values{}
			if v := strings.TrimSpace(query); v != "" {
				params.Set("q", v)
			}
			if v := strings.TrimSpace(topic); v != "" {
				params.Set("topic", v)
			}
			if v := strings.TrimSpace(sdk); v != "" {
				params.Set("sdk", v)
			}
			if v := strings.TrimSpace(difficulty); v != "" {
				params.Set("difficulty", v)
			}
			if limit > 0 {
				params.Set("limit", strconv.Itoa(limit))
			}

			path := "/api/v1/agentic/kb/articles"
			if encoded := params.Encode(); encoded != "" {
				path += "?" + encoded
			}
			proxyToServer(http.MethodGet, path, nil)
		},
	}

	cmd.Flags().StringVarP(&query, "query", "q", "", "Full-text query")
	cmd.Flags().StringVar(&topic, "topic", "", "Topic filter")
	cmd.Flags().StringVar(&sdk, "sdk", "", "SDK filter")
	cmd.Flags().StringVar(&difficulty, "difficulty", "", "Difficulty filter")
	cmd.Flags().IntVar(&limit, "limit", 50, "Result limit")

	return cmd
}

func newAgentKBReadCmd() *cobra.Command {
	var articleID string

	cmd := &cobra.Command{
		Use:   "read",
		Short: "Read a KB article by ID",
		Run: func(cmd *cobra.Command, args []string) {
			articleID = strings.TrimSpace(articleID)
			if articleID == "" {
				agentError("missing_required_flag", "--id is required", "Provide an article ID, for example: af agent kb read --id patterns/hunt-prove")
			}
			proxyToServer(http.MethodGet, "/api/v1/agentic/kb/articles/"+escapePathSegments(articleID), nil)
		},
	}

	cmd.Flags().StringVar(&articleID, "id", "", "Article ID")
	return cmd
}

func newAgentKBGuideCmd() *cobra.Command {
	var goal string

	cmd := &cobra.Command{
		Use:   "guide",
		Short: "Get goal-oriented KB guide",
		Run: func(cmd *cobra.Command, args []string) {
			goal = strings.TrimSpace(goal)
			if goal == "" {
				agentError("missing_required_flag", "--goal is required", "Describe what you want to build, for example: af agent kb guide --goal 'build security auditor'")
			}

			params := url.Values{}
			params.Set("goal", goal)
			proxyToServer(http.MethodGet, "/api/v1/agentic/kb/guide?"+params.Encode(), nil)
		},
	}

	cmd.Flags().StringVar(&goal, "goal", "", "Learning goal description")
	return cmd
}

func newAgentBatchCmd() *cobra.Command {
	var file string

	cmd := &cobra.Command{
		Use:   "batch",
		Short: "Execute batch API operations",
		Run: func(cmd *cobra.Command, args []string) {
			input, err := readBatchInput(file)
			if err != nil {
				agentError("batch_input_error", err.Error(), "Use --file <path> or pipe valid JSON to stdin.")
			}

			var payload interface{}
			if err := json.Unmarshal(input, &payload); err != nil {
				agentError("invalid_batch_json", "Batch payload must be valid JSON", "Expected shape: {\"operations\":[{\"id\":\"op1\",\"method\":\"GET\",\"path\":\"/api/v1/agentic/status\"}]}")
			}

			proxyToServer(http.MethodPost, "/api/v1/agentic/batch", payload)
		},
	}

	cmd.Flags().StringVarP(&file, "file", "f", "-", "Path to batch JSON file (or - for stdin)")
	return cmd
}

func newAgentHelpCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "help",
		Short: "Show structured JSON help for agent mode",
		Run: func(cmd *cobra.Command, args []string) {
			agentOutput(agentHelpData())
		},
	}
}

func agentOutput(data interface{}) {
	resp := AgentResponse{
		OK:   true,
		Data: data,
		Meta: &AgentMeta{Server: GetServerURL()},
	}

	if err := outputAgentJSON(resp); err != nil {
		fmt.Fprintf(os.Stderr, "{\"ok\":false,\"error\":{\"code\":\"output_failed\",\"message\":%q}}\n", err.Error())
		os.Exit(1)
	}
}

func agentError(code, message, hint string) {
	resp := AgentResponse{
		OK: false,
		Error: &AgentError{
			Code:    code,
			Message: message,
			Hint:    hint,
		},
		Meta: &AgentMeta{Server: GetServerURL()},
	}

	if err := outputAgentJSON(resp); err != nil {
		fmt.Fprintf(os.Stderr, "{\"ok\":false,\"error\":{\"code\":\"output_failed\",\"message\":%q}}\n", err.Error())
		os.Exit(1)
	}
	os.Exit(1)
}

func outputAgentJSON(v interface{}) error {
	format := strings.TrimSpace(strings.ToLower(GetOutputFormat()))
	if format == "" {
		format = "json"
	}

	var (
		b   []byte
		err error
	)

	switch format {
	case "compact":
		b, err = json.Marshal(v)
	default:
		b, err = json.MarshalIndent(v, "", "  ")
	}
	if err != nil {
		return fmt.Errorf("marshal output: %w", err)
	}

	if _, err := fmt.Fprintln(os.Stdout, string(b)); err != nil {
		return fmt.Errorf("write output: %w", err)
	}
	return nil
}

func agentHTTP(method, path string, body interface{}) ([]byte, int, error) {
	server := strings.TrimRight(GetServerURL(), "/")
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}

	timeout := GetRequestTimeout()
	if timeout <= 0 {
		timeout = 30
	}

	var requestBody io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return nil, 0, fmt.Errorf("encode request body: %w", err)
		}
		requestBody = bytes.NewReader(encoded)
	}

	client := &http.Client{Timeout: time.Duration(timeout) * time.Second}
	req, err := http.NewRequest(method, server+path, requestBody)
	if err != nil {
		return nil, 0, fmt.Errorf("build request: %w", err)
	}

	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "af-cli/agent-mode")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if key := strings.TrimSpace(GetAPIKey()); key != "" {
		req.Header.Set("X-API-Key", key)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, resp.StatusCode, fmt.Errorf("read response: %w", err)
	}

	return respBody, resp.StatusCode, nil
}

func proxyToServer(method, path string, body interface{}) {
	start := time.Now()
	respBody, statusCode, err := agentHTTP(method, path, body)
	latency := time.Since(start)
	if err != nil {
		agentError("request_failed", err.Error(), "Verify --server, --api-key, and --timeout. Ensure the control plane is reachable.")
	}

	var decoded interface{}
	if err := json.Unmarshal(respBody, &decoded); err != nil {
		message := "server returned non-JSON response"
		if statusCode >= 400 {
			message = "server returned non-JSON error response"
		}
		hint := "Check server logs or retry with the correct endpoint and credentials."
		agentError("invalid_response", message, hint)
	}

	if payload, ok := decoded.(map[string]interface{}); ok {
		meta := map[string]interface{}{}
		if existing, exists := payload["meta"]; exists {
			if existingMap, ok := existing.(map[string]interface{}); ok {
				for k, v := range existingMap {
					meta[k] = v
				}
			}
		}

		meta["server"] = GetServerURL()
		meta["latency"] = latency.String()
		meta["status_code"] = statusCode
		payload["meta"] = meta

		if _, hasOK := payload["ok"]; !hasOK {
			payload["ok"] = statusCode < http.StatusBadRequest
		}

		if statusCode >= http.StatusBadRequest {
			if _, hasError := payload["error"]; !hasError {
				payload["error"] = map[string]interface{}{
					"code":    "request_failed",
					"message": fmt.Sprintf("request failed with status %d", statusCode),
					"hint":    defaultHintForStatus(statusCode),
				}
			} else if errMap, ok := payload["error"].(map[string]interface{}); ok {
				if _, hasHint := errMap["hint"]; !hasHint {
					errMap["hint"] = defaultHintForStatus(statusCode)
				}
			}
		}

		if err := outputAgentJSON(payload); err != nil {
			agentError("output_failed", err.Error(), "Retry with --output json and check stdout permissions.")
		}
		if statusCode >= http.StatusBadRequest {
			os.Exit(1)
		}
		return
	}

	if statusCode >= http.StatusBadRequest {
		agentError(
			"request_failed",
			fmt.Sprintf("request failed with status %d", statusCode),
			defaultHintForStatus(statusCode),
		)
	}

	agentOutput(decoded)
}

func readBatchInput(file string) ([]byte, error) {
	source := strings.TrimSpace(file)
	if source == "" || source == "-" {
		data, err := io.ReadAll(os.Stdin)
		if err != nil {
			return nil, fmt.Errorf("read stdin: %w", err)
		}
		if len(bytes.TrimSpace(data)) == 0 {
			return nil, fmt.Errorf("stdin is empty")
		}
		return data, nil
	}

	data, err := os.ReadFile(source)
	if err != nil {
		return nil, fmt.Errorf("read file %q: %w", source, err)
	}
	if len(bytes.TrimSpace(data)) == 0 {
		return nil, fmt.Errorf("file %q is empty", source)
	}
	return data, nil
}

func defaultHintForStatus(statusCode int) string {
	switch statusCode {
	case http.StatusUnauthorized:
		return "Provide a valid API key with --api-key or AGENTFIELD_API_KEY."
	case http.StatusForbidden:
		return "API key is valid but lacks required permissions for this endpoint."
	case http.StatusNotFound:
		return "Check the endpoint path and identifier values (run ID, agent ID, or article ID)."
	case http.StatusBadRequest:
		return "Review command flags and payload shape, then retry."
	default:
		if statusCode >= 500 {
			return "Server error. Retry shortly or inspect control plane logs."
		}
		return "Request failed. Verify flags, credentials, and endpoint parameters."
	}
}

func escapePathSegments(id string) string {
	parts := strings.Split(id, "/")
	escaped := make([]string, 0, len(parts))
	for _, part := range parts {
		trimmed := strings.TrimSpace(part)
		if trimmed == "" {
			continue
		}
		escaped = append(escaped, url.PathEscape(trimmed))
	}
	return strings.Join(escaped, "/")
}

func agentHelpData() map[string]interface{} {
	return map[string]interface{}{
		"command":     "af agent",
		"description": "Machine-friendly CLI wrapper for AgentField agentic APIs",
		"version":     "v1",
		"global_flags": []map[string]interface{}{
			{"name": "server", "short": "s", "type": "string", "default": "http://localhost:8080", "description": "Control plane URL (env: AGENTFIELD_SERVER)"},
			{"name": "api-key", "short": "k", "type": "string", "default": "", "description": "API key for authenticated endpoints (env: AGENTFIELD_API_KEY)"},
			{"name": "output", "short": "o", "type": "string", "default": "json", "description": "Output format: json or compact"},
			{"name": "timeout", "short": "t", "type": "int", "default": 30, "description": "Request timeout in seconds"},
		},
		"subcommands": []map[string]interface{}{
			{
				"name":        "status",
				"description": "Get system status overview",
				"usage":       "af agent status",
				"flags":       []interface{}{},
				"example":     "af agent status -s http://localhost:8080",
			},
			{
				"name":        "discover",
				"description": "Discover agentic API endpoints",
				"usage":       "af agent discover [--query] [--group] [--method] [--limit]",
				"flags": []map[string]string{
					{"name": "query", "short": "q", "type": "string"},
					{"name": "group", "short": "g", "type": "string"},
					{"name": "method", "short": "", "type": "string"},
					{"name": "limit", "short": "", "type": "int"},
				},
				"example": "af agent discover -q execute --group agentic --method GET --limit 10",
			},
			{
				"name":        "query",
				"description": "Query runs/executions/agents/workflows/sessions",
				"usage":       "af agent query --resource runs [--status] [--agent-id] [--run-id] [--since] [--until] [--limit] [--offset] [--include]",
				"flags": []map[string]string{
					{"name": "resource", "short": "r", "type": "string (required)"},
					{"name": "status", "short": "", "type": "string"},
					{"name": "agent-id", "short": "", "type": "string"},
					{"name": "run-id", "short": "", "type": "string"},
					{"name": "since", "short": "", "type": "string(RFC3339)"},
					{"name": "until", "short": "", "type": "string(RFC3339)"},
					{"name": "limit", "short": "", "type": "int"},
					{"name": "offset", "short": "", "type": "int"},
					{"name": "include", "short": "", "type": "string(csv)"},
				},
				"example": "af agent query -r executions --agent-id analyst --status completed --limit 25",
			},
			{
				"name":        "run",
				"description": "Get run overview by run ID",
				"usage":       "af agent run --id <run_id>",
				"flags":       []map[string]string{{"name": "id", "short": "", "type": "string (required)"}},
				"example":     "af agent run --id run_20260318_001",
			},
			{
				"name":        "agent-summary",
				"description": "Get agent summary by agent ID",
				"usage":       "af agent agent-summary --id <agent_id>",
				"flags":       []map[string]string{{"name": "id", "short": "", "type": "string (required)"}},
				"example":     "af agent agent-summary --id sec-analyst",
			},
			{
				"name":        "kb topics",
				"description": "List knowledge base topics",
				"usage":       "af agent kb topics",
				"flags":       []interface{}{},
				"example":     "af agent kb topics",
			},
			{
				"name":        "kb search",
				"description": "Search knowledge base articles",
				"usage":       "af agent kb search [--query] [--topic] [--sdk] [--difficulty] [--limit]",
				"flags": []map[string]string{
					{"name": "query", "short": "q", "type": "string"},
					{"name": "topic", "short": "", "type": "string"},
					{"name": "sdk", "short": "", "type": "string"},
					{"name": "difficulty", "short": "", "type": "string"},
					{"name": "limit", "short": "", "type": "int"},
				},
				"example": "af agent kb search -q harness --sdk python --limit 5",
			},
			{
				"name":        "kb read",
				"description": "Read a knowledge base article",
				"usage":       "af agent kb read --id <article_id>",
				"flags":       []map[string]string{{"name": "id", "short": "", "type": "string (required)"}},
				"example":     "af agent kb read --id patterns/hunt-prove",
			},
			{
				"name":        "kb guide",
				"description": "Get goal-oriented KB guide",
				"usage":       "af agent kb guide --goal <text>",
				"flags":       []map[string]string{{"name": "goal", "short": "", "type": "string (required)"}},
				"example":     "af agent kb guide --goal \"build compliance workflow\"",
			},
			{
				"name":        "batch",
				"description": "Execute batched API operations",
				"usage":       "af agent batch [--file <path>|stdin]",
				"flags":       []map[string]string{{"name": "file", "short": "f", "type": "string"}},
				"example":     "af agent batch -f operations.json",
			},
			{
				"name":        "help",
				"description": "Show this structured help payload",
				"usage":       "af agent help",
				"flags":       []interface{}{},
				"example":     "af agent help",
			},
		},
		"quick_start": []string{
			"af agent status",
			"af agent discover -q run",
			"af agent query -r runs --limit 10",
			"af agent run --id <run_id>",
			"af agent kb topics",
			"af agent kb guide --goal 'build swe agent'",
		},
		"auth": map[string]interface{}{
			"method":           "Set X-API-Key header via --api-key or AGENTFIELD_API_KEY",
			"public_endpoints": []string{"GET /api/v1/agentic/kb/topics", "GET /api/v1/agentic/kb/articles", "GET /api/v1/agentic/kb/articles/:article_id", "GET /api/v1/agentic/kb/guide"},
			"requires_auth":    []string{"GET /api/v1/agentic/status", "GET /api/v1/agentic/discover", "POST /api/v1/agentic/query", "GET /api/v1/agentic/run/:run_id", "GET /api/v1/agentic/agent/:agent_id/summary", "POST /api/v1/agentic/batch"},
		},
		"response_schemas": map[string]interface{}{
			"success": map[string]interface{}{
				"ok":   true,
				"data": "<endpoint_payload>",
				"meta": map[string]string{"server": "string", "latency": "duration", "status_code": "int"},
			},
			"error": map[string]interface{}{
				"ok": false,
				"error": map[string]string{
					"code":    "string",
					"message": "string",
					"hint":    "string",
				},
				"meta": map[string]string{"server": "string", "latency": "duration", "status_code": "int"},
			},
		},
	}
}
