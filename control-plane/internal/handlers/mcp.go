package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// MCPStore is the storage surface the embedded MCP server needs: discovery
// (ListAgents) plus the execution store used to start async runs and inspect
// their status. storage.StorageProvider satisfies it, so the route wiring can
// pass the same handle the REST discovery/execute handlers use — MCP tools call
// the service layer directly rather than looping back through HTTP.
type MCPStore interface {
	AgentLister
	ExecutionStore
}

// MCPAuthorizer applies the control plane's target-aware authorization policy
// before an MCP execution is created. It returns the resolved target DID for
// forwarding to the target agent.
type MCPAuthorizer func(ctx context.Context, callerDID, target string, input map[string]interface{}) (string, error)

// JSON-RPC 2.0 error codes (subset used by the MCP transport).
const (
	mcpErrParse          = -32700
	mcpErrInvalidRequest = -32600
	mcpErrMethodNotFound = -32601
	mcpErrInvalidParams  = -32602
	mcpErrInternal       = -32603
)

const (
	// defaultProtocolVersion is echoed when the client requests a version we
	// don't recognize. Kept in sync with the newest spec revision we implement.
	defaultProtocolVersion = "2025-06-18"
	// mcpDiscoveryLimit is an effectively-unbounded page size — the MCP tools
	// surface every matching agent in one call, unlike the paginated REST API.
	mcpDiscoveryLimit = 100000
	// wait_run bounds: a tool call must never hang a harness, so the server-side
	// poll loop is hard-capped regardless of the requested timeout.
	mcpWaitDefaultSeconds = 60
	mcpWaitMaxSeconds     = 120
)

// mcpWaitPollInterval is how often wait_run re-reads run state. A var (not a
// const) so tests can shrink it.
var mcpWaitPollInterval = 500 * time.Millisecond

// knownProtocolVersions are the MCP protocol revisions this server understands.
// When a client asks for one of these we echo it back; otherwise we answer with
// defaultProtocolVersion.
var knownProtocolVersions = map[string]bool{
	"2025-06-18": true,
	"2025-03-26": true,
	"2024-11-05": true,
}

// jsonrpcRequest is a single JSON-RPC 2.0 request. A request with no ID is a
// notification and receives no response body.
type jsonrpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type jsonrpcError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

type jsonrpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  interface{}     `json:"result,omitempty"`
	Error   *jsonrpcError   `json:"error,omitempty"`
}

// mcpTool is the tools/list entry for a single tool.
type mcpTool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"inputSchema"`
}

// mcpServer holds the dependencies the MCP tools dispatch against.
type mcpServer struct {
	store         MCPStore
	payloads      services.PayloadStore
	webhooks      services.WebhookDispatcher
	timeout       time.Duration
	internalToken string
	version       string
	authorize     MCPAuthorizer
}

// MCPHandler builds the streamable-HTTP MCP endpoint handler. It speaks JSON-RPC
// 2.0 over a single POST /mcp, stateless (no session requirement), and exposes
// AgentField discovery + execution as MCP tools.
func MCPHandler(store MCPStore, payloads services.PayloadStore, webhooks services.WebhookDispatcher, timeout time.Duration, internalToken, version string, authorize MCPAuthorizer) gin.HandlerFunc {
	srv := &mcpServer{
		store:         store,
		payloads:      payloads,
		webhooks:      webhooks,
		timeout:       timeout,
		internalToken: internalToken,
		version:       strings.TrimSpace(version),
		authorize:     authorize,
	}
	if srv.version == "" {
		srv.version = "dev"
	}
	return srv.handle
}

// handle is the POST /mcp entry point.
func (s *mcpServer) handle(c *gin.Context) {
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		s.writeError(c, nullID(), mcpErrParse, "failed to read request body: "+err.Error(), nil)
		return
	}

	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		s.writeError(c, nullID(), mcpErrInvalidRequest, "empty request body", nil)
		return
	}
	// Batch requests (a top-level JSON array) are intentionally unsupported —
	// harnesses send a single message per POST. Reject clearly rather than
	// silently processing only the first element.
	if trimmed[0] == '[' {
		s.writeError(c, nullID(), mcpErrInvalidRequest, "batch requests are not supported; send a single JSON-RPC message", nil)
		return
	}

	var req jsonrpcRequest
	if err := json.Unmarshal(trimmed, &req); err != nil {
		s.writeError(c, nullID(), mcpErrParse, "parse error: "+err.Error(), nil)
		return
	}

	// Notifications (no id) and any notifications/* method carry no response
	// body — acknowledge with 202 per the streamable-HTTP transport.
	if len(req.ID) == 0 || strings.HasPrefix(req.Method, "notifications/") {
		c.Status(http.StatusAccepted)
		return
	}

	switch req.Method {
	case "initialize":
		s.handleInitialize(c, req)
	case "ping":
		s.writeResult(c, req.ID, map[string]interface{}{})
	case "tools/list":
		s.writeResult(c, req.ID, map[string]interface{}{"tools": mcpToolCatalog()})
	case "tools/call":
		s.handleToolsCall(c, req)
	default:
		s.writeError(c, req.ID, mcpErrMethodNotFound, "method not found: "+req.Method, nil)
	}
}

func (s *mcpServer) handleInitialize(c *gin.Context, req jsonrpcRequest) {
	var params struct {
		ProtocolVersion string `json:"protocolVersion"`
	}
	// Tolerate absent/invalid params — initialize should still succeed.
	_ = json.Unmarshal(req.Params, &params)

	s.writeResult(c, req.ID, map[string]interface{}{
		"protocolVersion": negotiateProtocolVersion(params.ProtocolVersion),
		"capabilities": map[string]interface{}{
			"tools": map[string]interface{}{},
		},
		"serverInfo": map[string]interface{}{
			"name":    "agentfield",
			"version": s.version,
		},
	})
}

func (s *mcpServer) handleToolsCall(c *gin.Context, req jsonrpcRequest) {
	var params struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal(req.Params, &params); err != nil {
		s.writeError(c, req.ID, mcpErrInvalidParams, "invalid params: "+err.Error(), nil)
		return
	}

	ctx := c.Request.Context()
	var (
		result map[string]interface{}
		err    error
	)
	switch params.Name {
	case "discover_agents":
		result, err = s.toolDiscoverAgents(ctx, params.Arguments)
	case "get_reasoner_schema":
		result, err = s.toolGetReasonerSchema(ctx, params.Arguments)
	case "execute_reasoner":
		result, err = s.toolExecuteReasoner(c, params.Arguments)
	case "get_run":
		result, err = s.toolGetRun(c, params.Arguments)
	case "wait_run":
		result, err = s.toolWaitRun(c, params.Arguments)
	default:
		s.writeError(c, req.ID, mcpErrInvalidParams, "unknown tool: "+params.Name, nil)
		return
	}

	if err != nil {
		// Genuine internal failure (e.g. storage error). Business/validation
		// failures are returned as isError tool results, not JSON-RPC errors.
		logger.Logger.Warn().Err(err).Str("tool", params.Name).Msg("mcp tool call failed")
		s.writeError(c, req.ID, mcpErrInternal, err.Error(), nil)
		return
	}
	s.writeResult(c, req.ID, result)
}

// ---- tools -----------------------------------------------------------------

func (s *mcpServer) toolDiscoverAgents(ctx context.Context, rawArgs json.RawMessage) (map[string]interface{}, error) {
	var args struct {
		Health string `json:"health"`
	}
	if len(bytes.TrimSpace(rawArgs)) > 0 {
		if err := json.Unmarshal(rawArgs, &args); err != nil {
			return mcpToolError("invalid arguments: " + err.Error()), nil
		}
	}
	health := strings.ToLower(strings.TrimSpace(args.Health))
	if health == "" {
		health = "active"
	}
	if health != "active" && health != "all" {
		return mcpToolError(`invalid "health": must be "active" or "all"`), nil
	}

	agents, err := s.store.ListAgents(ctx, types.AgentFilters{})
	if err != nil {
		return nil, fmt.Errorf("list agents: %w", err)
	}

	filters := DiscoveryFilters{
		IncludeDescriptions: true,
		Format:              "json",
		Limit:               mcpDiscoveryLimit,
	}
	if health == "active" {
		hs := types.HealthStatusActive
		filters.HealthStatus = &hs
	}

	resp := buildDiscoveryResponse(agents, filters)
	out := make([]map[string]interface{}, 0, len(resp.Capabilities))
	for _, capp := range resp.Capabilities {
		reasoners := make([]map[string]interface{}, 0, len(capp.Reasoners))
		for _, r := range capp.Reasoners {
			rm := map[string]interface{}{
				"id":     r.ID,
				"target": capp.AgentID + "." + r.ID,
			}
			if r.Description != nil {
				rm["description"] = *r.Description
			}
			if len(r.Tags) > 0 {
				rm["tags"] = r.Tags
			}
			reasoners = append(reasoners, rm)
		}
		out = append(out, map[string]interface{}{
			"id":             capp.AgentID,
			"health_status":  capp.HealthStatus,
			"last_heartbeat": capp.LastHeartbeat.UTC().Format(time.RFC3339),
			"reasoners":      reasoners,
		})
	}

	return mcpToolText(map[string]interface{}{
		"health": health,
		"count":  len(out),
		"agents": out,
	}), nil
}

func (s *mcpServer) toolGetReasonerSchema(ctx context.Context, rawArgs json.RawMessage) (map[string]interface{}, error) {
	var args struct {
		Node     string `json:"node"`
		Reasoner string `json:"reasoner"`
	}
	if err := json.Unmarshal(rawArgs, &args); err != nil {
		return mcpToolError("invalid arguments: " + err.Error()), nil
	}
	node := strings.TrimSpace(args.Node)
	reasoner := strings.TrimSpace(args.Reasoner)
	if node == "" || reasoner == "" {
		return mcpToolError(`"node" and "reasoner" are required`), nil
	}

	agents, err := s.store.ListAgents(ctx, types.AgentFilters{})
	if err != nil {
		return nil, fmt.Errorf("list agents: %w", err)
	}

	filters := DiscoveryFilters{
		AgentIDs:            []string{node},
		IncludeDescriptions: true,
		IncludeInputSchema:  true,
		IncludeOutputSchema: true,
		Format:              "json",
		Limit:               mcpDiscoveryLimit,
	}
	resp := buildDiscoveryResponse(agents, filters)
	for _, capp := range resp.Capabilities {
		if capp.AgentID != node {
			continue
		}
		for _, r := range capp.Reasoners {
			if r.ID != reasoner {
				continue
			}
			result := map[string]interface{}{
				"node":     node,
				"reasoner": r.ID,
				"target":   node + "." + r.ID,
			}
			if r.Description != nil {
				result["description"] = *r.Description
			}
			if len(r.Tags) > 0 {
				result["tags"] = r.Tags
			}
			if r.InputSchema != nil {
				result["input_schema"] = r.InputSchema
			}
			if r.OutputSchema != nil {
				result["output_schema"] = r.OutputSchema
			}
			return mcpToolText(result), nil
		}
	}

	return mcpToolError(fmt.Sprintf(
		"reasoner %q not found on node %q. Call discover_agents to list available agents and reasoners.",
		reasoner, node)), nil
}

func (s *mcpServer) toolExecuteReasoner(c *gin.Context, rawArgs json.RawMessage) (map[string]interface{}, error) {
	var args struct {
		Target string                 `json:"target"`
		Input  map[string]interface{} `json:"input"`
	}
	if err := json.Unmarshal(rawArgs, &args); err != nil {
		return mcpToolError("invalid arguments: " + err.Error()), nil
	}
	target := strings.TrimSpace(args.Target)
	if target == "" {
		return mcpToolError(`"target" is required in the form "node.reasoner"`), nil
	}
	parsed, err := parseTarget(target)
	if err != nil {
		return mcpToolError(fmt.Sprintf(
			"invalid target %q: %v. Expected \"node.reasoner\". Call discover_agents to see valid targets.",
			target, err)), nil
	}

	ctx := c.Request.Context()
	agents, err := s.store.ListAgents(ctx, types.AgentFilters{})
	if err != nil {
		return nil, fmt.Errorf("list agents: %w", err)
	}
	if !targetExists(agents, parsed.NodeID, parsed.TargetName) {
		return mcpToolError(fmt.Sprintf(
			"target %q not found. Call discover_agents to list available agents and reasoners.",
			target)), nil
	}

	input := args.Input
	if input == nil {
		input = map[string]interface{}{}
	}

	callerDID := middleware.GetVerifiedCallerDID(c)
	targetDID := ""
	if s.authorize != nil {
		targetDID, err = s.authorize(ctx, callerDID, target, input)
		if err != nil {
			return mcpToolError("access denied: " + err.Error()), nil
		}
	}

	headers := readExecutionHeaders(c)
	// A verified DID is the authoritative MCP run owner. This both forwards the
	// caller identity and prevents a client-supplied actor header from granting
	// access to another principal's MCP runs.
	if callerDID != "" {
		headers.actorID = &callerDID
	}
	runID, execID, err := s.startAsyncRun(ctx, target, input, headers, callerDID, targetDID)
	if err != nil {
		return mcpToolError("failed to start execution: " + err.Error()), nil
	}

	return mcpToolText(map[string]interface{}{
		"run_id":       runID,
		"execution_id": execID,
		"status":       "accepted",
		"target":       target,
	}), nil
}

func (s *mcpServer) toolGetRun(c *gin.Context, rawArgs json.RawMessage) (map[string]interface{}, error) {
	ctx := c.Request.Context()
	var args struct {
		RunID string `json:"run_id"`
	}
	if err := json.Unmarshal(rawArgs, &args); err != nil {
		return mcpToolError("invalid arguments: " + err.Error()), nil
	}
	runID := strings.TrimSpace(args.RunID)
	if runID == "" {
		return mcpToolError(`"run_id" is required`), nil
	}

	view, _, found, err := s.buildRunViewForCaller(ctx, runID, middleware.GetVerifiedCallerDID(c))
	if err != nil {
		return nil, err
	}
	if !found {
		return mcpToolError(fmt.Sprintf("run %q not found", runID)), nil
	}
	return mcpToolText(view), nil
}

func (s *mcpServer) toolWaitRun(c *gin.Context, rawArgs json.RawMessage) (map[string]interface{}, error) {
	ctx := c.Request.Context()
	var args struct {
		RunID          string `json:"run_id"`
		TimeoutSeconds *int   `json:"timeout_seconds"`
	}
	if err := json.Unmarshal(rawArgs, &args); err != nil {
		return mcpToolError("invalid arguments: " + err.Error()), nil
	}
	runID := strings.TrimSpace(args.RunID)
	if runID == "" {
		return mcpToolError(`"run_id" is required`), nil
	}

	timeout := mcpWaitDefaultSeconds
	if args.TimeoutSeconds != nil {
		timeout = *args.TimeoutSeconds
	}
	if timeout < 1 {
		timeout = 1
	}
	if timeout > mcpWaitMaxSeconds {
		timeout = mcpWaitMaxSeconds
	}

	deadline := time.Now().Add(time.Duration(timeout) * time.Second)
	ticker := time.NewTicker(mcpWaitPollInterval)
	defer ticker.Stop()

	for {
		view, terminal, found, err := s.buildRunViewForCaller(ctx, runID, middleware.GetVerifiedCallerDID(c))
		if err != nil {
			return nil, err
		}
		if !found {
			return mcpToolError(fmt.Sprintf("run %q not found", runID)), nil
		}
		if terminal {
			view["timed_out"] = false
			return mcpToolText(view), nil
		}
		if !time.Now().Before(deadline) {
			view["timed_out"] = true
			return mcpToolText(view), nil
		}
		select {
		case <-ctx.Done():
			view["timed_out"] = true
			return mcpToolText(view), nil
		case <-ticker.C:
		}
	}
}

// ---- shared helpers --------------------------------------------------------

// startAsyncRun starts an asynchronous execution of target ("node.reasoner")
// with the given input, mirroring the /execute/async handler but driven from
// the MCP tool. It returns as soon as the job is enqueued.
func (s *mcpServer) startAsyncRun(ctx context.Context, target string, input map[string]interface{}, headers executionHeaders, callerDID, targetDID string) (runID, execID string, err error) {
	controller := newExecutionController(s.store, s.payloads, s.webhooks, s.timeout, s.internalToken)
	plan, err := controller.prepareExecutionForTarget(ctx, target, ExecuteRequest{Input: input}, headers, callerDID, targetDID)
	if err != nil {
		return "", "", err
	}

	if err := CheckExecutionPreconditions(plan.target.NodeID, plan.llmEndpoint); err != nil {
		_ = controller.failExecution(ctx, plan, err, 0, nil)
		return "", "", err
	}

	controller.publishExecutionStartedEvent(plan)

	pool := getAsyncWorkerPool()
	job := asyncExecutionJob{controller: controller, plan: *plan}
	if ok := pool.submit(job); !ok {
		ReleaseExecutionSlot(plan.target.NodeID)
		queueErr := errors.New("async execution queue is full; retry later")
		_ = controller.failExecution(ctx, plan, queueErr, 0, nil)
		return "", "", queueErr
	}

	return plan.exec.RunID, plan.exec.ExecutionID, nil
}

// buildRunViewForCaller limits DID-authenticated callers to runs they own. MCP
// executions record that ownership in ActorID when started; old or non-MCP runs
// without the matching owner are intentionally indistinguishable from missing.
func (s *mcpServer) buildRunViewForCaller(ctx context.Context, runID, callerDID string) (view map[string]interface{}, terminal, found bool, err error) {
	execs, err := s.store.QueryExecutionRecords(ctx, types.ExecutionFilter{RunID: &runID})
	if err != nil {
		return nil, false, false, fmt.Errorf("query executions: %w", err)
	}
	if len(execs) == 0 {
		return nil, false, false, nil
	}
	if callerDID != "" {
		root := rootExecution(execs)
		if root == nil || root.ActorID == nil || *root.ActorID != callerDID {
			return nil, false, false, nil
		}
	}

	terminal = true
	summaries := make([]map[string]interface{}, 0, len(execs))
	for _, e := range execs {
		if !types.IsTerminalExecutionStatus(e.Status) {
			terminal = false
		}
		summaries = append(summaries, map[string]interface{}{
			"execution_id": e.ExecutionID,
			"reasoner_id":  e.ReasonerID,
			"node_id":      e.NodeID,
			"status":       e.Status,
		})
	}

	view = map[string]interface{}{
		"run_id":     runID,
		"status":     deriveRunStatus(execs),
		"executions": summaries,
	}

	if root := rootExecution(execs); root != nil {
		if result := decodeJSON(root.ResultPayload); result != nil {
			view["result"] = result
		}
		if root.ErrorMessage != nil && strings.TrimSpace(*root.ErrorMessage) != "" {
			view["error"] = *root.ErrorMessage
		}
	}
	if _, hasErr := view["error"]; !hasErr {
		for _, e := range execs {
			if e.ErrorMessage != nil && strings.TrimSpace(*e.ErrorMessage) != "" {
				view["error"] = *e.ErrorMessage
				break
			}
		}
	}

	return view, terminal, true, nil
}

// rootExecution returns the run's root execution (no parent), falling back to
// the first record if none is unambiguously the root.
func rootExecution(execs []*types.Execution) *types.Execution {
	for _, e := range execs {
		if e.ParentExecutionID == nil || strings.TrimSpace(*e.ParentExecutionID) == "" {
			return e
		}
	}
	if len(execs) > 0 {
		return execs[0]
	}
	return nil
}

// deriveRunStatus collapses per-execution statuses into a single run status:
// any terminal failure wins; otherwise all-terminal is "succeeded" and anything
// still in flight is "running".
func deriveRunStatus(execs []*types.Execution) string {
	allTerminal := true
	for _, e := range execs {
		switch e.Status {
		case types.ExecutionStatusFailed, types.ExecutionStatusTimeout, types.ExecutionStatusCancelled:
			return e.Status
		}
		if !types.IsTerminalExecutionStatus(e.Status) {
			allTerminal = false
		}
	}
	if allTerminal {
		return types.ExecutionStatusSucceeded
	}
	return types.ExecutionStatusRunning
}

// targetExists reports whether node.name resolves to a registered reasoner or
// skill in the discovery snapshot.
func targetExists(agents []*types.AgentNode, node, name string) bool {
	for _, a := range agents {
		if a == nil || a.ID != node {
			continue
		}
		for _, r := range a.Reasoners {
			if r.ID == name {
				return true
			}
		}
		for _, sk := range a.Skills {
			if sk.ID == name {
				return true
			}
		}
	}
	return false
}

func negotiateProtocolVersion(requested string) string {
	requested = strings.TrimSpace(requested)
	if knownProtocolVersions[requested] {
		return requested
	}
	return defaultProtocolVersion
}

func (s *mcpServer) writeResult(c *gin.Context, id json.RawMessage, result interface{}) {
	c.JSON(http.StatusOK, jsonrpcResponse{JSONRPC: "2.0", ID: idOrNull(id), Result: result})
}

func (s *mcpServer) writeError(c *gin.Context, id json.RawMessage, code int, message string, data interface{}) {
	c.JSON(http.StatusOK, jsonrpcResponse{
		JSONRPC: "2.0",
		ID:      idOrNull(id),
		Error:   &jsonrpcError{Code: code, Message: message, Data: data},
	})
}

func idOrNull(id json.RawMessage) json.RawMessage {
	if len(id) == 0 {
		return nullID()
	}
	return id
}

func nullID() json.RawMessage {
	return json.RawMessage("null")
}

// mcpToolText wraps a value as a successful MCP tool result: a single text
// content block holding compact JSON, which is what harnesses parse best.
func mcpToolText(v interface{}) map[string]interface{} {
	b, err := json.Marshal(v)
	if err != nil {
		return mcpToolError("failed to encode result: " + err.Error())
	}
	return map[string]interface{}{
		"content": []map[string]interface{}{
			{"type": "text", "text": string(b)},
		},
		"isError": false,
	}
}

// mcpToolError wraps a message as an MCP tool error result (isError: true). Used
// for validation/business failures the harness should surface to the model.
func mcpToolError(msg string) map[string]interface{} {
	return map[string]interface{}{
		"content": []map[string]interface{}{
			{"type": "text", "text": msg},
		},
		"isError": true,
	}
}

// mcpToolCatalog returns the tools/list payload: the five AgentField tools with
// their JSON Schemas.
func mcpToolCatalog() []mcpTool {
	return []mcpTool{
		{
			Name:        "discover_agents",
			Description: "List agent nodes on the control plane and their reasoners (id, description, entrypoint tags). Defaults to only active (healthy) agents; pass health:\"all\" to include unhealthy ones.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"health": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"all", "active"},
						"default":     "active",
						"description": "Which agents to include. \"active\" (default) returns only healthy agents; \"all\" includes unhealthy ones with their health_status.",
					},
				},
				"additionalProperties": false,
			},
		},
		{
			Name:        "get_reasoner_schema",
			Description: "Return the input and output JSON Schema for a specific reasoner, as served by the discovery surface.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"node":     map[string]interface{}{"type": "string", "description": "Agent node id."},
					"reasoner": map[string]interface{}{"type": "string", "description": "Reasoner id on that node."},
				},
				"required":             []string{"node", "reasoner"},
				"additionalProperties": false,
			},
		},
		{
			Name:        "execute_reasoner",
			Description: "Start an asynchronous execution of a reasoner. Returns a run_id immediately; poll with get_run or wait_run.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"target": map[string]interface{}{
						"type":        "string",
						"description": "Target in the form \"node.reasoner\" (from discover_agents).",
					},
					"input": map[string]interface{}{
						"type":                 "object",
						"description":          "Input object for the reasoner. Use get_reasoner_schema for its shape.",
						"additionalProperties": true,
					},
				},
				"required":             []string{"target"},
				"additionalProperties": false,
			},
		},
		{
			Name:        "get_run",
			Description: "Fetch the current status, result/error, and per-execution summaries for a run.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"run_id": map[string]interface{}{"type": "string", "description": "Run id returned by execute_reasoner."},
				},
				"required":             []string{"run_id"},
				"additionalProperties": false,
			},
		},
		{
			Name:        "wait_run",
			Description: "Poll a run server-side until it reaches a terminal state or the timeout elapses. Returns the same shape as get_run plus a timed_out flag.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"run_id": map[string]interface{}{"type": "string", "description": "Run id returned by execute_reasoner."},
					"timeout_seconds": map[string]interface{}{
						"type":        "integer",
						"minimum":     1,
						"maximum":     mcpWaitMaxSeconds,
						"default":     mcpWaitDefaultSeconds,
						"description": "Max seconds to wait (capped at 120).",
					},
				},
				"required":             []string{"run_id"},
				"additionalProperties": false,
			},
		},
	}
}
