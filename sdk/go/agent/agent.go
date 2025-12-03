package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/ai"
	"github.com/Agent-Field/agentfield/sdk/go/client"
	"github.com/Agent-Field/agentfield/sdk/go/types"
)

type executionContextKey struct{}

// ExecutionContext captures the headers AgentField sends with each execution request.
type ExecutionContext struct {
	RunID             string
	ExecutionID       string
	ParentExecutionID string
	SessionID         string
	ActorID           string
	WorkflowID        string
	ParentWorkflowID  string
	RootWorkflowID    string
	Depth             int
	AgentNodeID       string
	ReasonerName      string
	StartedAt         time.Time
}

func init() {
	rand.Seed(time.Now().UnixNano())
}

// HandlerFunc processes a reasoner invocation.
type HandlerFunc func(ctx context.Context, input map[string]any) (any, error)

// ReasonerOption applies metadata to a reasoner registration.
type ReasonerOption func(*Reasoner)

// WithInputSchema overrides the auto-generated input schema.
func WithInputSchema(raw json.RawMessage) ReasonerOption {
	return func(r *Reasoner) {
		if len(raw) > 0 {
			r.InputSchema = raw
		}
	}
}

// WithOutputSchema overrides the default output schema.
func WithOutputSchema(raw json.RawMessage) ReasonerOption {
	return func(r *Reasoner) {
		if len(raw) > 0 {
			r.OutputSchema = raw
		}
	}
}

// WithCLI marks this reasoner as CLI-accessible.
func WithCLI() ReasonerOption {
	return func(r *Reasoner) {
		r.CLIEnabled = true
	}
}

// WithDefaultCLI marks the reasoner as the default CLI handler.
func WithDefaultCLI() ReasonerOption {
	return func(r *Reasoner) {
		r.CLIEnabled = true
		r.DefaultCLI = true
	}
}

// WithCLIFormatter registers a custom formatter for CLI output.
func WithCLIFormatter(formatter func(context.Context, any, error)) ReasonerOption {
	return func(r *Reasoner) {
		r.CLIFormatter = formatter
	}
}

// WithDescription adds a human-readable description for help/list commands.
func WithDescription(desc string) ReasonerOption {
	return func(r *Reasoner) {
		r.Description = desc
	}
}

// Reasoner represents a single handler exposed by the agent.
type Reasoner struct {
	Name         string
	Handler      HandlerFunc
	InputSchema  json.RawMessage
	OutputSchema json.RawMessage

	CLIEnabled   bool
	DefaultCLI   bool
	CLIFormatter func(context.Context, any, error)
	Description  string
}

// Config drives Agent behaviour.
type Config struct {
	NodeID         string
	Version        string
	TeamID         string
	AgentFieldURL  string
	ListenAddress  string
	PublicURL      string
	Token          string
	DeploymentType string

	LeaseRefreshInterval time.Duration
	DisableLeaseLoop     bool
	Logger               *log.Logger

	// AIConfig configures LLM/AI capabilities
	// If nil, AI features will be disabled
	AIConfig *ai.Config

	// CLIConfig controls CLI-specific behaviour and help text.
	CLIConfig *CLIConfig
}

// CLIConfig controls CLI behaviour and presentation.
type CLIConfig struct {
	AppName        string
	AppDescription string
	DisableColors  bool

	DefaultOutputFormat string
	HelpPreamble        string
	HelpEpilog          string
	EnvironmentVars     []string
}

// Agent manages registration, lease renewal, and HTTP routing.
type Agent struct {
	cfg        Config
	client     *client.Client
	httpClient *http.Client
	reasoners  map[string]*Reasoner
	aiClient   *ai.Client // AI/LLM client

	serverMu sync.RWMutex
	server   *http.Server

	stopLease chan struct{}
	logger    *log.Logger

	router      http.Handler
	handlerOnce sync.Once

	initMu        sync.Mutex
	initialized   bool
	leaseLoopOnce sync.Once

	defaultCLIReasoner string
}

// New constructs an Agent.
func New(cfg Config) (*Agent, error) {
	if cfg.NodeID == "" {
		return nil, errors.New("config.NodeID is required")
	}
	if cfg.Version == "" {
		return nil, errors.New("config.Version is required")
	}
	if cfg.TeamID == "" {
		cfg.TeamID = "default"
	}
	if cfg.ListenAddress == "" {
		cfg.ListenAddress = ":8001"
	}
	if cfg.PublicURL == "" {
		cfg.PublicURL = "http://localhost" + cfg.ListenAddress
	}
	if strings.TrimSpace(cfg.DeploymentType) == "" {
		cfg.DeploymentType = "long_running"
	}
	if cfg.LeaseRefreshInterval <= 0 {
		cfg.LeaseRefreshInterval = 2 * time.Minute
	}
	if cfg.Logger == nil {
		cfg.Logger = log.New(os.Stdout, "[agent] ", log.LstdFlags)
	}

	httpClient := &http.Client{
		Timeout: 15 * time.Second,
	}

	// Initialize AI client if config provided
	var aiClient *ai.Client
	var err error
	if cfg.AIConfig != nil {
		aiClient, err = ai.NewClient(cfg.AIConfig)
		if err != nil {
			return nil, fmt.Errorf("initialize AI client: %w", err)
		}
	}

	a := &Agent{
		cfg:        cfg,
		httpClient: httpClient,
		reasoners:  make(map[string]*Reasoner),
		aiClient:   aiClient,
		stopLease:  make(chan struct{}),
		logger:     cfg.Logger,
	}

	if strings.TrimSpace(cfg.AgentFieldURL) != "" {
		c, err := client.New(cfg.AgentFieldURL, client.WithHTTPClient(httpClient), client.WithBearerToken(cfg.Token))
		if err != nil {
			return nil, err
		}
		a.client = c
	}

	return a, nil
}

func contextWithExecution(ctx context.Context, exec ExecutionContext) context.Context {
	return context.WithValue(ctx, executionContextKey{}, exec)
}

func executionContextFrom(ctx context.Context) ExecutionContext {
	if ctx == nil {
		return ExecutionContext{}
	}
	if val, ok := ctx.Value(executionContextKey{}).(ExecutionContext); ok {
		return val
	}
	return ExecutionContext{}
}

// ChildContext creates a new execution context for a nested local call.
func (ec ExecutionContext) ChildContext(agentNodeID, reasonerName string) ExecutionContext {
	runID := ec.RunID
	if runID == "" {
		runID = ec.WorkflowID
	}
	if runID == "" {
		runID = generateRunID()
	}

	workflowID := ec.WorkflowID
	if workflowID == "" {
		workflowID = runID
	}
	rootWorkflowID := ec.RootWorkflowID
	if rootWorkflowID == "" {
		rootWorkflowID = workflowID
	}

	return ExecutionContext{
		RunID:             runID,
		ExecutionID:       generateExecutionID(),
		ParentExecutionID: ec.ExecutionID,
		SessionID:         ec.SessionID,
		ActorID:           ec.ActorID,
		WorkflowID:        workflowID,
		ParentWorkflowID:  workflowID,
		RootWorkflowID:    rootWorkflowID,
		Depth:             ec.Depth + 1,
		AgentNodeID:       agentNodeID,
		ReasonerName:      reasonerName,
		StartedAt:         time.Now(),
	}
}

func generateRunID() string {
	return fmt.Sprintf("run_%d_%06d", time.Now().UnixNano(), rand.Intn(1_000_000))
}

func generateExecutionID() string {
	return fmt.Sprintf("exec_%d_%06d", time.Now().UnixNano(), rand.Intn(1_000_000))
}

func cloneInputMap(input map[string]any) map[string]any {
	if input == nil {
		return nil
	}
	copied := make(map[string]any, len(input))
	for k, v := range input {
		copied[k] = v
	}
	return copied
}

func stringFromMap(m map[string]any, keys ...string) string {
	for _, key := range keys {
		if val, ok := m[key]; ok {
			if str, ok := val.(string); ok && strings.TrimSpace(str) != "" {
				return strings.TrimSpace(str)
			}
		}
	}
	return ""
}

func rawToMap(raw json.RawMessage) map[string]any {
	if len(raw) == 0 {
		return map[string]any{}
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return map[string]any{}
	}
	return out
}

// RegisterReasoner makes a handler available at /reasoners/{name}.
func (a *Agent) RegisterReasoner(name string, handler HandlerFunc, opts ...ReasonerOption) {
	if handler == nil {
		panic("nil handler supplied")
	}

	meta := &Reasoner{
		Name:         name,
		Handler:      handler,
		InputSchema:  json.RawMessage(`{"type":"object","additionalProperties":true}`),
		OutputSchema: json.RawMessage(`{"type":"object","additionalProperties":true}`),
	}
	for _, opt := range opts {
		opt(meta)
	}

	if meta.DefaultCLI {
		if a.defaultCLIReasoner != "" && a.defaultCLIReasoner != name {
			a.logger.Printf("warn: default CLI reasoner already set to %s, ignoring default flag on %s", a.defaultCLIReasoner, name)
			meta.DefaultCLI = false
		} else {
			a.defaultCLIReasoner = name
		}
	}

	a.reasoners[name] = meta
}

// Initialize registers the agent with the AgentField control plane without starting a listener.
func (a *Agent) Initialize(ctx context.Context) error {
	a.initMu.Lock()
	defer a.initMu.Unlock()

	if a.initialized {
		return nil
	}

	if a.client == nil {
		return errors.New("AgentFieldURL is required when running in server mode")
	}

	if len(a.reasoners) == 0 {
		return errors.New("no reasoners registered")
	}

	if err := a.registerNode(ctx); err != nil {
		return fmt.Errorf("register node: %w", err)
	}

	if err := a.markReady(ctx); err != nil {
		a.logger.Printf("warn: initial status update failed: %v", err)
	}

	a.startLeaseLoop()
	a.initialized = true
	return nil
}

// Run intelligently routes between CLI and server modes.
func (a *Agent) Run(ctx context.Context) error {
	args := os.Args[1:]
	if len(args) == 0 && !a.hasCLIReasoners() {
		return a.Serve(ctx)
	}

	if len(args) > 0 && args[0] == "serve" {
		return a.Serve(ctx)
	}

	return a.runCLI(ctx, args)
}

// Serve starts the agent HTTP server, registers with the control plane, and blocks until ctx is cancelled.
func (a *Agent) Serve(ctx context.Context) error {
	if err := a.Initialize(ctx); err != nil {
		return err
	}

	if err := a.startServer(); err != nil {
		return fmt.Errorf("start server: %w", err)
	}

	// listen for shutdown.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)

	select {
	case <-ctx.Done():
		return a.shutdown(context.Background())
	case sig := <-sigCh:
		a.logger.Printf("received signal %s, shutting down", sig)
		return a.shutdown(context.Background())
	}
}

func (a *Agent) registerNode(ctx context.Context) error {
	now := time.Now().UTC()

	reasoners := make([]types.ReasonerDefinition, 0, len(a.reasoners))
	for _, reasoner := range a.reasoners {
		reasoners = append(reasoners, types.ReasonerDefinition{
			ID:           reasoner.Name,
			InputSchema:  reasoner.InputSchema,
			OutputSchema: reasoner.OutputSchema,
		})
	}

	payload := types.NodeRegistrationRequest{
		ID:        a.cfg.NodeID,
		TeamID:    a.cfg.TeamID,
		BaseURL:   strings.TrimSuffix(a.cfg.PublicURL, "/"),
		Version:   a.cfg.Version,
		Reasoners: reasoners,
		Skills:    []types.SkillDefinition{},
		CommunicationConfig: types.CommunicationConfig{
			Protocols:         []string{"http"},
			HeartbeatInterval: "0s",
		},
		HealthStatus:  "healthy",
		LastHeartbeat: now,
		RegisteredAt:  now,
		Metadata: map[string]any{
			"deployment": map[string]any{
				"environment": "development",
				"platform":    "go",
			},
			"sdk": map[string]any{
				"language": "go",
			},
		},
		Features:       map[string]any{},
		DeploymentType: a.cfg.DeploymentType,
	}

	_, err := a.client.RegisterNode(ctx, payload)
	if err != nil {
		return err
	}

	a.logger.Printf("node %s registered with AgentField", a.cfg.NodeID)
	return nil
}

func (a *Agent) markReady(ctx context.Context) error {
	score := 100
	_, err := a.client.UpdateStatus(ctx, a.cfg.NodeID, types.NodeStatusUpdate{
		Phase:       "ready",
		HealthScore: &score,
	})
	return err
}

func (a *Agent) startServer() error {
	server := &http.Server{
		Addr:    a.cfg.ListenAddress,
		Handler: a.Handler(),
	}
	a.serverMu.Lock()
	a.server = server
	a.serverMu.Unlock()

	go func() {
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			a.logger.Printf("server error: %v", err)
		}
	}()

	a.logger.Printf("listening on %s", a.cfg.ListenAddress)
	return nil
}

// Handler exposes the agent as an http.Handler for serverless or custom hosting scenarios.
func (a *Agent) Handler() http.Handler {
	return a.handler()
}

// ServeHTTP implements http.Handler directly.
func (a *Agent) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	a.Handler().ServeHTTP(w, r)
}

// Execute runs a specific reasoner by name.
func (a *Agent) Execute(ctx context.Context, reasonerName string, input map[string]any) (any, error) {
	reasoner, ok := a.reasoners[reasonerName]
	if !ok {
		return nil, fmt.Errorf("unknown reasoner %q", reasonerName)
	}
	if input == nil {
		input = make(map[string]any)
	}
	return reasoner.Handler(ctx, input)
}

// HandleServerlessEvent allows custom serverless entrypoints to normalize arbitrary
// platform events (Lambda, Vercel, Supabase, etc.) before delegating to the agent.
// The adapter can rewrite the incoming event into the generic payload that
// handleExecute expects: keys like path, target/reasoner, input, execution_context.
func (a *Agent) HandleServerlessEvent(ctx context.Context, event map[string]any, adapter func(map[string]any) map[string]any) (map[string]any, int, error) {
	if adapter != nil {
		event = adapter(event)
	}

	path := stringFromMap(event, "path", "rawPath")
	reasoner := stringFromMap(event, "reasoner", "target", "skill")
	if reasoner == "" && path != "" {
		cleaned := strings.Trim(path, "/")
		parts := strings.Split(cleaned, "/")
		if len(parts) >= 2 && (parts[0] == "execute" || parts[0] == "reasoners" || parts[0] == "skills") {
			reasoner = parts[1]
		} else if len(parts) == 1 {
			reasoner = parts[0]
		}
	}
	if reasoner == "" {
		return map[string]any{"error": "missing target or reasoner"}, http.StatusBadRequest, nil
	}

	input := extractInputFromServerless(event)
	execCtx := a.buildExecutionContextFromServerless(&http.Request{Header: http.Header{}}, event, reasoner)
	ctx = contextWithExecution(ctx, execCtx)

	handler, ok := a.reasoners[reasoner]
	if !ok {
		return map[string]any{"error": "reasoner not found"}, http.StatusNotFound, nil
	}

	result, err := handler.Handler(ctx, input)
	if err != nil {
		return map[string]any{"error": err.Error()}, http.StatusInternalServerError, nil
	}

	// Normalize to map for consistent JSON responses.
	if payload, ok := result.(map[string]any); ok {
		return payload, http.StatusOK, nil
	}
	return map[string]any{"result": result}, http.StatusOK, nil
}

func (a *Agent) handler() http.Handler {
	a.handlerOnce.Do(func() {
		mux := http.NewServeMux()
		mux.HandleFunc("/health", a.healthHandler)
		mux.HandleFunc("/discover", a.handleDiscover)
		mux.HandleFunc("/execute", a.handleExecute)
		mux.HandleFunc("/execute/", a.handleExecute)
		mux.HandleFunc("/reasoners/", a.handleReasoner)
		a.router = mux
	})
	return a.router
}

func (a *Agent) healthHandler(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}

func (a *Agent) handleDiscover(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, http.StatusOK, a.discoveryPayload())
}

func (a *Agent) discoveryPayload() map[string]any {
	reasoners := make([]map[string]any, 0, len(a.reasoners))
	for _, reasoner := range a.reasoners {
		reasoners = append(reasoners, map[string]any{
			"id":            reasoner.Name,
			"input_schema":  rawToMap(reasoner.InputSchema),
			"output_schema": rawToMap(reasoner.OutputSchema),
			"tags":          []string{},
		})
	}

	deployment := strings.TrimSpace(a.cfg.DeploymentType)
	if deployment == "" {
		deployment = "long_running"
	}

	return map[string]any{
		"node_id":         a.cfg.NodeID,
		"version":         a.cfg.Version,
		"deployment_type": deployment,
		"reasoners":       reasoners,
		"skills":          []map[string]any{},
	}
}

func (a *Agent) handleExecute(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	targetName := strings.TrimPrefix(r.URL.Path, "/execute")
	targetName = strings.TrimPrefix(targetName, "/")

	var payload map[string]any
	if r.Body != nil {
		defer r.Body.Close()
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil && !errors.Is(err, io.EOF) {
			http.Error(w, fmt.Sprintf("invalid JSON: %v", err), http.StatusBadRequest)
			return
		}
	}
	if payload == nil {
		payload = make(map[string]any)
	}

	reasonerName := strings.TrimSpace(targetName)
	if reasonerName == "" {
		reasonerName = stringFromMap(payload, "reasoner", "target", "skill")
	}

	if reasonerName == "" {
		http.Error(w, "missing target or reasoner", http.StatusBadRequest)
		return
	}

	reasoner, ok := a.reasoners[reasonerName]
	if !ok {
		http.NotFound(w, r)
		return
	}

	input := extractInputFromServerless(payload)
	execCtx := a.buildExecutionContextFromServerless(r, payload, reasonerName)
	ctx := contextWithExecution(r.Context(), execCtx)

	result, err := reasoner.Handler(ctx, input)
	if err != nil {
		a.logger.Printf("reasoner %s failed: %v", reasonerName, err)
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, result)
}

func extractInputFromServerless(payload map[string]any) map[string]any {
	if payload == nil {
		return map[string]any{}
	}

	if raw, ok := payload["input"]; ok {
		if m, ok := raw.(map[string]any); ok {
			return m
		}
		return map[string]any{"value": raw}
	}

	filtered := make(map[string]any)
	for k, v := range payload {
		switch strings.ToLower(k) {
		case "target", "reasoner", "skill", "type", "target_type", "path", "execution_context", "executioncontext", "context":
			continue
		default:
			filtered[k] = v
		}
	}
	return filtered
}

func (a *Agent) buildExecutionContextFromServerless(r *http.Request, payload map[string]any, reasonerName string) ExecutionContext {
	execCtx := ExecutionContext{
		RunID:             strings.TrimSpace(r.Header.Get("X-Run-ID")),
		ExecutionID:       strings.TrimSpace(r.Header.Get("X-Execution-ID")),
		ParentExecutionID: strings.TrimSpace(r.Header.Get("X-Parent-Execution-ID")),
		SessionID:         strings.TrimSpace(r.Header.Get("X-Session-ID")),
		ActorID:           strings.TrimSpace(r.Header.Get("X-Actor-ID")),
		WorkflowID:        strings.TrimSpace(r.Header.Get("X-Workflow-ID")),
		AgentNodeID:       a.cfg.NodeID,
		ReasonerName:      reasonerName,
		StartedAt:         time.Now(),
	}

	if ctxMap, ok := payload["execution_context"].(map[string]any); ok {
		if execCtx.ExecutionID == "" {
			execCtx.ExecutionID = stringFromMap(ctxMap, "execution_id", "executionId")
		}
		if execCtx.RunID == "" {
			execCtx.RunID = stringFromMap(ctxMap, "run_id", "runId")
		}
		if execCtx.WorkflowID == "" {
			execCtx.WorkflowID = stringFromMap(ctxMap, "workflow_id", "workflowId")
		}
		if execCtx.ParentExecutionID == "" {
			execCtx.ParentExecutionID = stringFromMap(ctxMap, "parent_execution_id", "parentExecutionId")
		}
		if execCtx.SessionID == "" {
			execCtx.SessionID = stringFromMap(ctxMap, "session_id", "sessionId")
		}
		if execCtx.ActorID == "" {
			execCtx.ActorID = stringFromMap(ctxMap, "actor_id", "actorId")
		}
	}

	if execCtx.RunID == "" {
		execCtx.RunID = generateRunID()
	}
	if execCtx.ExecutionID == "" {
		execCtx.ExecutionID = generateExecutionID()
	}
	if execCtx.WorkflowID == "" {
		execCtx.WorkflowID = execCtx.RunID
	}
	if execCtx.RootWorkflowID == "" {
		execCtx.RootWorkflowID = execCtx.WorkflowID
	}
	if execCtx.ParentWorkflowID == "" && execCtx.ParentExecutionID != "" {
		execCtx.ParentWorkflowID = execCtx.RootWorkflowID
	}

	return execCtx
}

func (a *Agent) handleReasoner(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	name := strings.TrimPrefix(r.URL.Path, "/reasoners/")
	if name == "" {
		http.NotFound(w, r)
		return
	}

	reasoner, ok := a.reasoners[name]
	if !ok {
		http.NotFound(w, r)
		return
	}

	defer r.Body.Close()
	var input map[string]any
	if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
		http.Error(w, fmt.Sprintf("invalid JSON: %v", err), http.StatusBadRequest)
		return
	}

	execCtx := ExecutionContext{
		RunID:             r.Header.Get("X-Run-ID"),
		ExecutionID:       r.Header.Get("X-Execution-ID"),
		ParentExecutionID: r.Header.Get("X-Parent-Execution-ID"),
		SessionID:         r.Header.Get("X-Session-ID"),
		ActorID:           r.Header.Get("X-Actor-ID"),
		WorkflowID:        r.Header.Get("X-Workflow-ID"),
		AgentNodeID:       a.cfg.NodeID,
		ReasonerName:      name,
		StartedAt:         time.Now(),
	}
	if execCtx.WorkflowID == "" {
		execCtx.WorkflowID = execCtx.RunID
	}
	if execCtx.RootWorkflowID == "" {
		execCtx.RootWorkflowID = execCtx.WorkflowID
	}

	ctx := contextWithExecution(r.Context(), execCtx)

	// In serverless mode we want a synchronous execution so the control plane can return
	// the result immediately; skip the async path even if an execution ID is present.
	if a.cfg.DeploymentType != "serverless" && execCtx.ExecutionID != "" && strings.TrimSpace(a.cfg.AgentFieldURL) != "" {
		go a.executeReasonerAsync(reasoner, cloneInputMap(input), execCtx)
		writeJSON(w, http.StatusAccepted, map[string]any{
			"status":        "processing",
			"execution_id":  execCtx.ExecutionID,
			"run_id":        execCtx.RunID,
			"reasoner_name": name,
		})
		return
	}

	result, err := reasoner.Handler(ctx, input)
	if err != nil {
		a.logger.Printf("reasoner %s failed: %v", name, err)
		response := map[string]any{
			"error": err.Error(),
		}
		writeJSON(w, http.StatusInternalServerError, response)
		return
	}

	writeJSON(w, http.StatusOK, result)
}

func (a *Agent) executeReasonerAsync(reasoner *Reasoner, input map[string]any, execCtx ExecutionContext) {
	ctx := contextWithExecution(context.Background(), execCtx)
	start := time.Now()

	defer func() {
		if rec := recover(); rec != nil {
			errMsg := fmt.Sprintf("panic: %v", rec)
			payload := map[string]any{
				"status":        "failed",
				"error":         errMsg,
				"execution_id":  execCtx.ExecutionID,
				"run_id":        execCtx.RunID,
				"completed_at":  time.Now().UTC().Format(time.RFC3339),
				"duration_ms":   time.Since(start).Milliseconds(),
				"reasoner_name": reasoner.Name,
			}
			if err := a.sendExecutionStatus(execCtx.ExecutionID, payload); err != nil {
				a.logger.Printf("failed to send panic status: %v", err)
			}
		}
	}()

	result, err := reasoner.Handler(ctx, input)
	payload := map[string]any{
		"execution_id":  execCtx.ExecutionID,
		"run_id":        execCtx.RunID,
		"completed_at":  time.Now().UTC().Format(time.RFC3339),
		"duration_ms":   time.Since(start).Milliseconds(),
		"reasoner_name": reasoner.Name,
	}

	if err != nil {
		payload["status"] = "failed"
		payload["error"] = err.Error()
	} else {
		payload["status"] = "succeeded"
		payload["result"] = result
	}

	if err := a.sendExecutionStatus(execCtx.ExecutionID, payload); err != nil {
		a.logger.Printf("async status update failed: %v", err)
	}
}

func (a *Agent) sendExecutionStatus(executionID string, payload map[string]any) error {
	base := strings.TrimSpace(a.cfg.AgentFieldURL)
	if executionID == "" || base == "" {
		return fmt.Errorf("missing execution id or AgentField URL")
	}
	callbackURL := strings.TrimSuffix(base, "/") + "/api/v1/executions/" + url.PathEscape(executionID) + "/status"
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("encode status payload: %w", err)
	}
	return a.postExecutionStatus(context.Background(), callbackURL, payloadBytes)
}

func (a *Agent) postExecutionStatus(ctx context.Context, callbackURL string, payload []byte) error {
	var lastErr error
	for attempt := 0; attempt < 5; attempt++ {
		attemptCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
		req, err := http.NewRequestWithContext(attemptCtx, http.MethodPost, callbackURL, bytes.NewReader(payload))
		if err != nil {
			cancel()
			return fmt.Errorf("create status request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := a.httpClient.Do(req)
		if err != nil {
			lastErr = err
		} else {
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				cancel()
				return nil
			}
			lastErr = fmt.Errorf("status update returned %d", resp.StatusCode)
		}
		cancel()
		if attempt < 4 {
			time.Sleep(time.Second << attempt)
		}
	}
	return lastErr
}

// Call invokes another reasoner via the AgentField control plane, preserving execution context.
func (a *Agent) Call(ctx context.Context, target string, input map[string]any) (map[string]any, error) {
	if strings.TrimSpace(a.cfg.AgentFieldURL) == "" {
		return nil, errors.New("AgentFieldURL is required to call other reasoners")
	}

	if !strings.Contains(target, ".") {
		target = fmt.Sprintf("%s.%s", a.cfg.NodeID, strings.TrimPrefix(target, "."))
	}

	execCtx := executionContextFrom(ctx)
	runID := execCtx.RunID
	if runID == "" {
		runID = generateRunID()
	}

	payload := map[string]any{"input": input}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal call payload: %w", err)
	}

	url := fmt.Sprintf("%s/api/v1/execute/%s", strings.TrimSuffix(a.cfg.AgentFieldURL, "/"), strings.TrimPrefix(target, "/"))
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-Run-ID", runID)
	if execCtx.ExecutionID != "" {
		req.Header.Set("X-Parent-Execution-ID", execCtx.ExecutionID)
	}
	if execCtx.WorkflowID != "" {
		req.Header.Set("X-Workflow-ID", execCtx.WorkflowID)
	}
	if execCtx.SessionID != "" {
		req.Header.Set("X-Session-ID", execCtx.SessionID)
	}
	if execCtx.ActorID != "" {
		req.Header.Set("X-Actor-ID", execCtx.ActorID)
	}
	if a.cfg.Token != "" {
		req.Header.Set("Authorization", "Bearer "+a.cfg.Token)
	}

	resp, err := a.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("perform execute call: %w", err)
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read execute response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("execute failed: %s", strings.TrimSpace(string(bodyBytes)))
	}

	var execResp struct {
		ExecutionID  string         `json:"execution_id"`
		RunID        string         `json:"run_id"`
		Status       string         `json:"status"`
		Result       map[string]any `json:"result"`
		ErrorMessage *string        `json:"error_message"`
	}
	if err := json.Unmarshal(bodyBytes, &execResp); err != nil {
		return nil, fmt.Errorf("decode execute response: %w", err)
	}

	if execResp.ErrorMessage != nil && *execResp.ErrorMessage != "" {
		return nil, fmt.Errorf("execute error: %s", *execResp.ErrorMessage)
	}
	if !strings.EqualFold(execResp.Status, "succeeded") {
		return nil, fmt.Errorf("execute status %s", execResp.Status)
	}

	return execResp.Result, nil
}

// emitWorkflowEvent sends a workflow event to the control plane asynchronously.
// Failures are logged but do not impact the caller.
func (a *Agent) emitWorkflowEvent(
	execCtx ExecutionContext,
	status string,
	input map[string]any,
	result any,
	err error,
	durationMS int64,
) {
	if strings.TrimSpace(a.cfg.AgentFieldURL) == "" {
		return
	}

	event := types.WorkflowExecutionEvent{
		ExecutionID: execCtx.ExecutionID,
		WorkflowID:  execCtx.WorkflowID,
		RunID:       execCtx.RunID,
		ReasonerID:  execCtx.ReasonerName,
		Type:        execCtx.ReasonerName,
		AgentNodeID: a.cfg.NodeID,
		Status:      status,
	}

	if execCtx.ParentExecutionID != "" {
		event.ParentExecutionID = &execCtx.ParentExecutionID
	}
	if execCtx.ParentWorkflowID != "" {
		event.ParentWorkflowID = &execCtx.ParentWorkflowID
	}
	if input != nil {
		event.InputData = input
	}
	if result != nil {
		event.Result = result
	}
	if err != nil {
		event.Error = err.Error()
	}
	if durationMS > 0 {
		event.DurationMS = &durationMS
	}

	if sendErr := a.sendWorkflowEvent(event); sendErr != nil {
		a.logger.Printf("workflow event send failed: %v", sendErr)
	}
}

func (a *Agent) sendWorkflowEvent(event types.WorkflowExecutionEvent) error {
	url := strings.TrimSuffix(a.cfg.AgentFieldURL, "/") + "/api/v1/workflow/executions/events"

	body, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("marshal event: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if a.cfg.Token != "" {
		req.Header.Set("Authorization", "Bearer "+a.cfg.Token)
	}

	resp, err := a.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()

	bodyBytes, _ := io.ReadAll(resp.Body)

	if resp.StatusCode >= 400 {
		return fmt.Errorf("server returned %d: %s", resp.StatusCode, strings.TrimSpace(string(bodyBytes)))
	}

	return nil
}

// CallLocal invokes a registered reasoner directly within this agent process,
// maintaining execution lineage and emitting workflow events to the control plane.
// It should be used for same-node composition; use Call for cross-node calls.
func (a *Agent) CallLocal(ctx context.Context, reasonerName string, input map[string]any) (any, error) {
	reasoner, ok := a.reasoners[reasonerName]
	if !ok {
		return nil, fmt.Errorf("unknown reasoner %q", reasonerName)
	}

	parentCtx := executionContextFrom(ctx)

	childCtx := a.buildChildContext(parentCtx, reasonerName)
	ctx = contextWithExecution(ctx, childCtx)

	a.emitWorkflowEvent(childCtx, "running", input, nil, nil, 0)

	start := time.Now()
	result, err := reasoner.Handler(ctx, input)
	durationMS := time.Since(start).Milliseconds()

	if err != nil {
		a.emitWorkflowEvent(childCtx, "failed", input, nil, err, durationMS)
	} else {
		a.emitWorkflowEvent(childCtx, "succeeded", input, result, nil, durationMS)
	}

	return result, err
}

func (a *Agent) buildChildContext(parent ExecutionContext, reasonerName string) ExecutionContext {
	if parent.RunID == "" && parent.ExecutionID == "" {
		runID := generateRunID()
		return ExecutionContext{
			RunID:          runID,
			ExecutionID:    generateExecutionID(),
			SessionID:      parent.SessionID,
			ActorID:        parent.ActorID,
			WorkflowID:     runID,
			RootWorkflowID: runID,
			Depth:          0,
			AgentNodeID:    a.cfg.NodeID,
			ReasonerName:   reasonerName,
			StartedAt:      time.Now(),
		}
	}

	return parent.ChildContext(a.cfg.NodeID, reasonerName)
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if payload == nil {
		return
	}
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		// best-effort fallback
		_, _ = w.Write([]byte(`{}`))
	}
}

func (a *Agent) startLeaseLoop() {
	if a.cfg.DisableLeaseLoop || a.cfg.LeaseRefreshInterval <= 0 {
		return
	}

	a.leaseLoopOnce.Do(func() {
		ticker := time.NewTicker(a.cfg.LeaseRefreshInterval)
		go func() {
			for {
				select {
				case <-ticker.C:
					ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
					if err := a.markReady(ctx); err != nil {
						a.logger.Printf("lease refresh failed: %v", err)
					}
					cancel()
				case <-a.stopLease:
					ticker.Stop()
					return
				}
			}
		}()
	})
}

func (a *Agent) shutdown(ctx context.Context) error {
	close(a.stopLease)

	if _, err := a.client.Shutdown(ctx, a.cfg.NodeID, types.ShutdownRequest{Reason: "shutdown"}); err != nil {
		a.logger.Printf("failed to notify shutdown: %v", err)
	}

	a.serverMu.RLock()
	server := a.server
	a.serverMu.RUnlock()

	if server != nil {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			return err
		}
	}
	return nil
}

// AI makes an AI/LLM call with the given prompt and options.
// Returns an error if AI is not configured for this agent.
//
// Example usage:
//
//	response, err := agent.AI(ctx, "What is the weather?",
//	    ai.WithSystem("You are a weather assistant"),
//	    ai.WithTemperature(0.7))
func (a *Agent) AI(ctx context.Context, prompt string, opts ...ai.Option) (*ai.Response, error) {
	if a.aiClient == nil {
		return nil, errors.New("AI not configured for this agent; set AIConfig in agent Config")
	}
	return a.aiClient.Complete(ctx, prompt, opts...)
}

// AIStream makes a streaming AI/LLM call.
// Returns channels for streaming chunks and errors.
//
// Example usage:
//
//	chunks, errs := agent.AIStream(ctx, "Tell me a story")
//	for chunk := range chunks {
//	    fmt.Print(chunk.Choices[0].Delta.Content)
//	}
//	if err := <-errs; err != nil {
//	    log.Fatal(err)
//	}
func (a *Agent) AIStream(ctx context.Context, prompt string, opts ...ai.Option) (<-chan ai.StreamChunk, <-chan error) {
	if a.aiClient == nil {
		errCh := make(chan error, 1)
		errCh <- errors.New("AI not configured for this agent; set AIConfig in agent Config")
		close(errCh)
		chunkCh := make(chan ai.StreamChunk)
		close(chunkCh)
		return chunkCh, errCh
	}
	return a.aiClient.StreamComplete(ctx, prompt, opts...)
}

// ExecutionContextFrom returns the execution context embedded in the provided context, if any.
func ExecutionContextFrom(ctx context.Context) ExecutionContext {
	return executionContextFrom(ctx)
}
