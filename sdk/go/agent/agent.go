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
	"strings"
	"runtime"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/ai"
	"github.com/Agent-Field/agentfield/sdk/go/client"
	"github.com/Agent-Field/agentfield/sdk/go/did"
	"github.com/Agent-Field/agentfield/sdk/go/harness"
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

	// DID fields — populated when DID authentication is enabled.
	CallerDID    string
	TargetDID    string
	AgentNodeDID string
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

// WithAcceptsWebhook sets the accepts_webhook flag for UI-managed trigger guardrailing.
// Pass "true" to explicitly opt in, "false" to refuse, or omit for auto-detect (True if
// any triggers declared, else "warn").
func WithAcceptsWebhook(flag string) ReasonerOption {
	return func(r *Reasoner) {
		if flag != "" {
			r.AcceptsWebhook = &flag
		}
	}
}

// Reasoner represents a single handler exposed by the agent.
type Reasoner struct {
	Name         string
	Handler      HandlerFunc
	InputSchema  json.RawMessage
	OutputSchema json.RawMessage
	Tags         []string

	CLIEnabled   bool
	DefaultCLI   bool
	CLIFormatter func(context.Context, any, error)
	Description  string

	// VCEnabled overrides the agent-level VCEnabled setting for this reasoner.
	// nil = inherit agent setting, true/false = override.
	VCEnabled *bool

	// RequireRealtimeValidation forces control-plane verification for this
	// reasoner, skipping local verification even when enabled.
	RequireRealtimeValidation bool

	// Triggers declares external Sources whose events should invoke this
	// reasoner. Bindings are sent at registration time; the control plane
	// upserts a code-managed Trigger per binding. The canonical form is
	// WithTriggers(...); the WithEventTrigger / WithScheduleTrigger helpers
	// are sugar that append to this slice.
	Triggers []types.TriggerBinding

	// AcceptsWebhook controls whether UI-managed triggers can invoke this reasoner.
	// nil/empty = inherit default ("warn"), "true" = explicitly opt in, "false" = explicitly refuse.
	// Auto-set to "true" if any Triggers are declared.
	AcceptsWebhook *string
}

// EventTrigger describes an external event source binding for a reasoner.
// Use it as a typed argument to WithTriggers — the control plane registers
// the binding and minting of the public ingest URL happens server-side.
type EventTrigger struct {
	Source       string
	Types        []string
	SecretEnv    string
	Config       map[string]any
}

// ScheduleTrigger describes a cron-style schedule binding for a reasoner.
// Use it as a typed argument to WithTriggers; the control plane runs the
// schedule and dispatches a synthetic "tick" event when it fires.
type ScheduleTrigger struct {
	Cron     string
	Timezone string
}

// captureCodeOrigin captures the file and line number of the caller using runtime.Caller.
// skip=1 walks past captureCodeOrigin to the immediate caller (e.g., WithEventTrigger).
// skip=2 walks past captureCodeOrigin and WithEventTrigger to the user's call site.
func captureCodeOrigin(skip int) string {
	_, file, line, ok := runtime.Caller(skip)
	if !ok {
		return ""
	}
	return fmt.Sprintf("%s:%d", file, line)
}

// WithTriggers is the canonical decorator-equivalent for declaring trigger
// bindings on a Go reasoner. Pass a mix of EventTrigger and ScheduleTrigger
// values; unknown types are silently ignored so adding new kinds later is
// backward compatible.
func WithTriggers(triggers ...any) ReasonerOption {
	codeOrigin := captureCodeOrigin(2)
	return func(r *Reasoner) {
		for _, t := range triggers {
			binding, ok := triggerToBinding(t)
			if !ok {
				continue
			}
			if binding.CodeOrigin == "" {
				binding.CodeOrigin = codeOrigin
			}
			r.Triggers = append(r.Triggers, binding)
		}
	}
}

// WithEventTrigger is sugar that appends a single EventTrigger to the
// reasoner's bindings. It is equivalent to
// WithTriggers(EventTrigger{Source: source, Types: types}).
func WithEventTrigger(source string, eventTypes ...string) ReasonerOption {
	codeOrigin := captureCodeOrigin(2)
	return func(r *Reasoner) {
		b, _ := triggerToBinding(EventTrigger{Source: source, Types: eventTypes})
		b.CodeOrigin = codeOrigin
		r.Triggers = append(r.Triggers, b)
	}
}

// WithScheduleTrigger is sugar for declaring a single cron schedule. The
// expression follows the standard 5-field cron format.
func WithScheduleTrigger(expression string) ReasonerOption {
	codeOrigin := captureCodeOrigin(2)
	return func(r *Reasoner) {
		b, _ := triggerToBinding(ScheduleTrigger{Cron: expression})
		b.CodeOrigin = codeOrigin
		r.Triggers = append(r.Triggers, b)
	}
}

// WithTriggerSecretEnv attaches a secret env var name to the most recently
// declared trigger binding on the reasoner. Use it after WithEventTrigger
// when the source requires a secret (Stripe, GitHub, Slack, generic_hmac).
func WithTriggerSecretEnv(envVar string) ReasonerOption {
	return func(r *Reasoner) {
		if len(r.Triggers) == 0 {
			return
		}
		r.Triggers[len(r.Triggers)-1].SecretEnvVar = envVar
	}
}

// WithTriggerConfig attaches a config blob to the most recently declared
// trigger binding. Source impls validate the config server-side.
func WithTriggerConfig(cfg map[string]any) ReasonerOption {
	return func(r *Reasoner) {
		if len(r.Triggers) == 0 {
			return
		}
		raw, err := json.Marshal(cfg)
		if err != nil {
			return
		}
		r.Triggers[len(r.Triggers)-1].Config = raw
	}
}

// triggerToBinding normalizes typed trigger values into the wire payload.
func triggerToBinding(t any) (types.TriggerBinding, bool) {
	switch v := t.(type) {
	case EventTrigger:
		var cfg json.RawMessage
		if len(v.Config) > 0 {
			if raw, err := json.Marshal(v.Config); err == nil {
				cfg = raw
			}
		}
		return types.TriggerBinding{
			Source:       v.Source,
			EventTypes:   v.Types,
			Config:       cfg,
			SecretEnvVar: v.SecretEnv,
		}, true
	case ScheduleTrigger:
		tz := v.Timezone
		if tz == "" {
			tz = "UTC"
		}
		cfg, _ := json.Marshal(map[string]any{
			"expression": v.Cron,
			"timezone":   tz,
		})
		return types.TriggerBinding{Source: "cron", Config: cfg}, true
	default:
		return types.TriggerBinding{}, false
	}
}

// WithVCEnabled overrides VC generation for this specific reasoner.
func WithVCEnabled(enabled bool) ReasonerOption {
	return func(r *Reasoner) {
		r.VCEnabled = &enabled
	}
}

// WithReasonerTags sets tags for this reasoner (used for tag-based authorization).
func WithReasonerTags(tags ...string) ReasonerOption {
	return func(r *Reasoner) {
		r.Tags = tags
	}
}

// WithRequireRealtimeValidation forces control-plane verification for this
// reasoner instead of local verification, even when LocalVerification is enabled.
func WithRequireRealtimeValidation() ReasonerOption {
	return func(r *Reasoner) {
		r.RequireRealtimeValidation = true
	}
}

// ExecuteError is a structured error from agent-to-agent calls via the control
// plane. It preserves the HTTP status code and any structured error details
// (e.g., permission_denied response fields) so callers can inspect them.
type ExecuteError struct {
	StatusCode   int
	Message      string
	ErrorDetails interface{}
}

func (e *ExecuteError) Error() string {
	return e.Message
}

// Config drives Agent behaviour.
type Config struct {
	// NodeID is the unique identifier for this agent node. Required.
	// Must be a non-empty identifier suitable for registration (alphanumeric
	// characters, hyphens are recommended). Example: "my-agent-1".
	NodeID string

	// Version identifies the agent implementation version. Required.
	// Typically in semver or short string form (e.g. "v1.2.3" or "1.0.0").
	Version string

	// TeamID groups related agents together for organization. Optional.
	// Default: "default" (if empty, New() sets it to "default").
	TeamID string

	// AgentFieldURL is the base URL of the AgentField control plane server.
	// Optional for local-only or serverless usage, required when registering
	// with a control plane or making cross-node calls. Default: empty.
	// Format: a valid HTTP/HTTPS URL, e.g. "https://agentfield.example.com".
	AgentFieldURL string

	// ListenAddress is the network address the agent HTTP server binds to.
	// Optional. Default: ":8001" (if empty, New() sets it to ":8001").
	// Format: "host:port" or ":port" (e.g. ":8001" or "0.0.0.0:8001").
	ListenAddress string

	// PublicURL is the public-facing base URL reported to the control plane.
	// Optional. Default: "http://localhost" + ListenAddress (if empty,
	// New() constructs a default using ListenAddress).
	// Format: a valid HTTP/HTTPS URL.
	PublicURL string

	// Token is the bearer token used for authenticating to the control plane.
	// Optional. Default: empty (no auth). When set, the token is sent as
	// an Authorization: Bearer <token> header on control-plane requests.
	Token string

	// DeploymentType describes how the agent runs (affects execution behavior).
	// Optional. Default: "long_running". Common values: "long_running",
	// "serverless". Use a descriptive string for custom modes.
	DeploymentType string

	// LeaseRefreshInterval controls how frequently the agent refreshes its
	// lease/heartbeat with the control plane. Optional.
	// Default: 2m (2 minutes). Valid: any positive time.Duration. This value
	// is also sent to the control plane as the HeartbeatInterval during node
	// registration; when zero, the registration falls back to 30s. When
	// DisableLeaseLoop is true, "0s" is registered instead so the control
	// plane does not expect heartbeats.
	LeaseRefreshInterval time.Duration

	// DisableLeaseLoop disables automatic periodic lease refreshes.
	// Optional. Default: false. When true, node registration reports
	// HeartbeatInterval as "0s" to signal that the agent does not heartbeat.
	DisableLeaseLoop bool

	// Logger is used for agent logging output. Optional.
	// Default: a standard logger writing to stdout with the "[agent] " prefix
	// (if nil, New() creates a default logger).
	Logger *log.Logger

	// AIConfig configures LLM/AI capabilities for the agent.
	// Optional. If nil, AI features are disabled. Provide a valid
	// *ai.Config to enable AI-related APIs.
	AIConfig *ai.Config

	// CLIConfig controls CLI-specific behaviour and help text.
	// Optional. If nil, CLI behavior uses sensible defaults.
	CLIConfig *CLIConfig

	// MemoryBackend allows plugging in a custom memory storage backend.
	// Optional. If nil, an in-memory backend is used (data lost on restart).
	MemoryBackend MemoryBackend

	// DID is the agent's decentralized identifier for DID authentication.
	// Optional. If set along with PrivateKeyJWK, enables DID auth on
	// all control plane requests without auto-registration.
	DID string

	// PrivateKeyJWK is the JWK-formatted Ed25519 private key for signing
	// DID-authenticated requests. Optional. Must be set together with DID.
	PrivateKeyJWK string

	// EnableDID enables automatic DID registration during Initialize().
	// The agent registers with the control plane's DID service to obtain
	// a cryptographic identity (Ed25519 keys and DID). DID authentication
	// is then applied to all subsequent control plane requests.
	// If DID and PrivateKeyJWK are already set, registration is skipped.
	// Optional. Default: false.
	EnableDID bool

	// VCEnabled enables Verifiable Credential generation after each execution.
	// Requires DID authentication (either EnableDID or DID/PrivateKeyJWK).
	// When enabled, the agent generates a W3C Verifiable Credential for each
	// reasoner execution and stores it on the control plane for audit trails.
	// Optional. Default: false.
	VCEnabled bool

	// Tags are metadata labels attached to the agent during registration.
	// Used by the control plane for protection rules (e.g., agents tagged
	// "sensitive" require permission for cross-agent calls).
	// Optional. Default: nil.
	Tags []string

	// InternalToken is validated on incoming requests when RequireOriginAuth
	// is true. The control plane sends this token as Authorization: Bearer
	// when forwarding execution requests. If empty, Token is used instead.
	// Optional. Default: "" (falls back to Token).
	InternalToken string

	// RequireOriginAuth when true, validates that incoming execution
	// requests include an Authorization header matching InternalToken
	// (or Token if InternalToken is empty). This ensures only the
	// control plane can invoke reasoners, blocking direct access to the
	// agent's HTTP port. /health and /discover endpoints remain open.
	// Optional. Default: false.
	RequireOriginAuth bool

	// LocalVerification enables decentralized verification of incoming
	// requests using cached policies, revocations, and the admin's public key.
	// When enabled, the agent verifies DID signatures locally without
	// hitting the control plane for every call.
	// Optional. Default: false.
	LocalVerification bool

	// VerificationRefreshInterval controls how often the local verifier
	// refreshes its caches from the control plane.
	// Optional. Default: 5 minutes.
	VerificationRefreshInterval time.Duration

	// HarnessConfig configures the default harness runner for dispatching
	// tasks to external coding agents (opencode, claude-code).
	// Optional. If nil, Harness() calls require per-call provider options.
	HarnessConfig *HarnessConfig
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
	memory     *Memory    // Memory system for state management

	// DID/VC subsystem
	didManager  *did.Manager
	vcGenerator *did.VCGenerator

	// Local verification (decentralized mode)
	localVerifier               *LocalVerifier
	realtimeValidationFunctions map[string]struct{}

	harnessRunner *harness.Runner

	serverMu sync.RWMutex
	server   *http.Server

	stopLease chan struct{}
	logger    *log.Logger

	router      http.Handler
	handlerOnce sync.Once

	procLogRing *processLogRing
	procLogOnce sync.Once

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
		cfg:                         cfg,
		httpClient:                  httpClient,
		reasoners:                   make(map[string]*Reasoner),
		aiClient:                    aiClient,
		memory:                      NewMemory(cfg.MemoryBackend),
		stopLease:                   make(chan struct{}),
		logger:                      cfg.Logger,
		realtimeValidationFunctions: make(map[string]struct{}),
	}

	// Initialize local verifier if enabled
	if cfg.LocalVerification && cfg.AgentFieldURL != "" {
		refreshInterval := cfg.VerificationRefreshInterval
		if refreshInterval <= 0 {
			refreshInterval = 5 * time.Minute
		}
		a.localVerifier = NewLocalVerifier(cfg.AgentFieldURL, refreshInterval, cfg.Token)
		cfg.Logger.Printf("Local verification enabled (refresh every %s)", refreshInterval)
	}

	if strings.TrimSpace(cfg.AgentFieldURL) != "" {
		opts := []client.Option{client.WithHTTPClient(httpClient), client.WithBearerToken(cfg.Token)}
		if cfg.DID != "" && cfg.PrivateKeyJWK != "" {
			opts = append(opts, client.WithDIDAuth(cfg.DID, cfg.PrivateKeyJWK))
		}
		c, err := client.New(cfg.AgentFieldURL, opts...)
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
		CallerDID:         ec.CallerDID,
		TargetDID:         ec.TargetDID,
		AgentNodeDID:      ec.AgentNodeDID,
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
	a.logExecutionInfo(ctx, "reasoner.invoke.start", "starting local reasoner execution", map[string]any{
		"reasoner_id": reasonerName,
		"mode":        "direct",
	})
	start := time.Now()
	result, err := reasoner.Handler(ctx, input)
	durationMS := time.Since(start).Milliseconds()
	if err != nil {
		a.logExecutionError(ctx, "reasoner.invoke.failed", "reasoner execution failed", map[string]any{
			"reasoner_id": reasonerName,
			"mode":        "direct",
			"duration_ms": durationMS,
			"error":       err.Error(),
		})
		return nil, err
	}
	a.logExecutionInfo(ctx, "reasoner.invoke.complete", "reasoner execution completed", map[string]any{
		"reasoner_id": reasonerName,
		"mode":        "direct",
		"duration_ms": durationMS,
	})
	return result, nil
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

	a.logExecutionInfo(ctx, "reasoner.invoke.start", "starting serverless reasoner execution", map[string]any{
		"reasoner_id": reasoner,
		"mode":        "serverless",
	})
	start := time.Now()
	result, err := handler.Handler(ctx, input)
	durationMS := time.Since(start).Milliseconds()
	if err != nil {
		a.logExecutionError(ctx, "reasoner.invoke.failed", "reasoner execution failed", map[string]any{
			"reasoner_id": reasoner,
			"mode":        "serverless",
			"duration_ms": durationMS,
			"error":       err.Error(),
		})
		return map[string]any{"error": err.Error()}, http.StatusInternalServerError, nil
	}

	a.logExecutionInfo(ctx, "reasoner.invoke.complete", "reasoner execution completed", map[string]any{
		"reasoner_id": reasoner,
		"mode":        "serverless",
		"duration_ms": durationMS,
	})

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
		mux.HandleFunc("/agentfield/v1/logs", a.handleAgentfieldLogs)
		mux.HandleFunc("/execute", a.handleExecute)
		mux.HandleFunc("/execute/", a.handleExecute)
		mux.HandleFunc("/reasoners/", a.handleReasoner)

		var handler http.Handler = mux

		// Apply local verification middleware if enabled
		if a.localVerifier != nil {
			handler = a.localVerificationMiddleware(handler)
		}

		originToken := a.cfg.InternalToken
		if originToken == "" {
			originToken = a.cfg.Token
		}
		if a.cfg.RequireOriginAuth && originToken != "" {
			a.router = a.originAuthMiddleware(handler, originToken)
		} else {
			a.router = handler
		}
	})
	return a.router
}

// originAuthMiddleware validates that incoming requests to execute/reasoner
// endpoints include an Authorization header matching the expected token.
// Health and discovery endpoints are exempt.
func (a *Agent) originAuthMiddleware(next http.Handler, token string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path
		if path == "/health" || path == "/discover" || path == "/agentfield/v1/logs" {
			next.ServeHTTP(w, r)
			return
		}

		expected := "Bearer " + token
		if r.Header.Get("Authorization") != expected {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			w.Write([]byte(`{"error":"unauthorized","message":"valid Authorization header required"}`))
			return
		}

		next.ServeHTTP(w, r)
	})
}

// localVerificationMiddleware verifies incoming DID signatures locally
// using cached admin public key and checks revocation lists.
func (a *Agent) localVerificationMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path

		// Only verify execution endpoints
		if path == "/health" || path == "/discover" || path == "/agentfield/v1/logs" {
			next.ServeHTTP(w, r)
			return
		}

		// Extract function name to check realtime validation requirement
		funcName := ""
		if strings.HasPrefix(path, "/execute/") {
			funcName = strings.TrimPrefix(path, "/execute/")
		} else if strings.HasPrefix(path, "/reasoners/") {
			funcName = strings.TrimPrefix(path, "/reasoners/")
		}
		funcName = strings.TrimSuffix(funcName, "/")

		// Skip local verification for realtime-validated functions
		if _, skip := a.realtimeValidationFunctions[funcName]; skip {
			next.ServeHTTP(w, r)
			return
		}

		// Refresh cache if stale — block until refresh completes so that
		// registration and revocation checks use up-to-date data.
		if a.localVerifier.NeedsRefresh() {
			if err := a.localVerifier.Refresh(); err != nil {
				a.logger.Printf("warn: local verification cache refresh failed: %v", err)
			}
		}

		// Allow trusted control-plane requests to bypass DID verification.
		// The control plane sends Authorization: Bearer <internal_token> when
		// forwarding execution requests on behalf of callers.
		internalToken := a.cfg.InternalToken
		if internalToken == "" {
			internalToken = a.cfg.Token
		}
		if internalToken != "" {
			if r.Header.Get("Authorization") == "Bearer "+internalToken {
				next.ServeHTTP(w, r)
				return
			}
		}

		// Extract DID auth headers
		callerDID := r.Header.Get("X-Caller-DID")
		signature := r.Header.Get("X-DID-Signature")
		timestamp := r.Header.Get("X-DID-Timestamp")
		nonce := r.Header.Get("X-DID-Nonce")

		// Require DID authentication — fail closed when no caller DID provided.
		if callerDID == "" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			json.NewEncoder(w).Encode(map[string]string{
				"error":   "did_auth_required",
				"message": "DID authentication required",
			})
			return
		}

		// Require signature when caller DID is present.
		if signature == "" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			json.NewEncoder(w).Encode(map[string]string{
				"error":   "signature_required",
				"message": "DID signature required",
			})
			return
		}

		// Check revocation
		if a.localVerifier.CheckRevocation(callerDID) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusForbidden)
			json.NewEncoder(w).Encode(map[string]string{
				"error":   "did_revoked",
				"message": "Caller DID " + callerDID + " has been revoked",
			})
			return
		}

		// Check registration — reject DIDs not registered with the control plane
		if !a.localVerifier.CheckRegistration(callerDID) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusForbidden)
			json.NewEncoder(w).Encode(map[string]string{
				"error":   "did_not_registered",
				"message": "Caller DID " + callerDID + " is not registered with the control plane",
			})
			return
		}

		// Verify signature — need to read and buffer the body
		body, err := io.ReadAll(r.Body)
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte(`{"error":"body_read_error","message":"Failed to read request body"}`))
			return
		}
		// Restore body for downstream handlers
		r.Body = io.NopCloser(bytes.NewReader(body))

		if !a.localVerifier.VerifySignature(callerDID, signature, timestamp, body, nonce) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			w.Write([]byte(`{"error":"signature_invalid","message":"DID signature verification failed"}`))
			return
		}

		// Evaluate access policies after successful signature verification.
		if !a.localVerifier.EvaluatePolicy(nil, a.cfg.Tags, funcName, nil) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusForbidden)
			json.NewEncoder(w).Encode(map[string]string{
				"error":   "policy_denied",
				"message": "Access denied by policy",
			})
			return
		}

		next.ServeHTTP(w, r)
	})
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
	a.fillDIDContext(&execCtx)
	ctx := contextWithExecution(r.Context(), execCtx)

	start := time.Now()
	a.logExecutionInfo(ctx, "reasoner.invoke.start", "starting reasoner execution", map[string]any{
		"reasoner_id": reasonerName,
		"mode":        "http",
	})
	result, err := reasoner.Handler(ctx, input)
	durationMS := time.Since(start).Milliseconds()

	if err != nil {
		a.logger.Printf("reasoner %s failed: %v", reasonerName, err)
		a.logExecutionError(ctx, "reasoner.invoke.failed", "reasoner execution failed", map[string]any{
			"reasoner_id": reasonerName,
			"mode":        "http",
			"duration_ms": durationMS,
			"error":       err.Error(),
		})
		a.maybeGenerateVC(execCtx, input, nil, "failed", err.Error(), durationMS, reasoner)
		// Propagate structured error details (e.g. from a failed inner Call)
		// so the control plane can expose them to the original caller.
		var execErr *ExecuteError
		if errors.As(err, &execErr) {
			response := map[string]any{"error": execErr.Message}
			if execErr.ErrorDetails != nil {
				response["error_details"] = execErr.ErrorDetails
			}
			// Propagate the upstream HTTP status code (e.g. 403 from permission
			// middleware) so the control plane can forward it to the original caller.
			statusCode := execErr.StatusCode
			if statusCode < 400 {
				statusCode = http.StatusInternalServerError
			}
			writeJSON(w, statusCode, response)
			return
		}
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}

	a.maybeGenerateVC(execCtx, input, result, "succeeded", "", durationMS, reasoner)
	a.logExecutionInfo(ctx, "reasoner.invoke.complete", "reasoner execution completed", map[string]any{
		"reasoner_id": reasonerName,
		"mode":        "http",
		"duration_ms": durationMS,
	})
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
		CallerDID:         strings.TrimSpace(r.Header.Get("X-Caller-DID")),
		TargetDID:         strings.TrimSpace(r.Header.Get("X-Target-DID")),
		AgentNodeDID:      strings.TrimSpace(r.Header.Get("X-Agent-Node-DID")),
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
		CallerDID:         r.Header.Get("X-Caller-DID"),
		TargetDID:         r.Header.Get("X-Target-DID"),
		AgentNodeDID:      r.Header.Get("X-Agent-Node-DID"),
	}
	if execCtx.WorkflowID == "" {
		execCtx.WorkflowID = execCtx.RunID
	}
	if execCtx.RootWorkflowID == "" {
		execCtx.RootWorkflowID = execCtx.WorkflowID
	}
	a.fillDIDContext(&execCtx)

	ctx := contextWithExecution(r.Context(), execCtx)

	// In serverless mode we want a synchronous execution so the control plane can return
	// the result immediately; skip the async path even if an execution ID is present.
	if a.cfg.DeploymentType != "serverless" && execCtx.ExecutionID != "" && strings.TrimSpace(a.cfg.AgentFieldURL) != "" {
		a.logExecutionInfo(ctx, "reasoner.invoke.accepted", "accepted asynchronous execution request", map[string]any{
			"reasoner_id": name,
			"mode":        "async",
		})
		go a.executeReasonerAsync(reasoner, cloneInputMap(input), execCtx)
		writeJSON(w, http.StatusAccepted, map[string]any{
			"status":        "processing",
			"execution_id":  execCtx.ExecutionID,
			"run_id":        execCtx.RunID,
			"reasoner_name": name,
		})
		return
	}

	start := time.Now()
	a.logExecutionInfo(ctx, "reasoner.invoke.start", "starting reasoner execution", map[string]any{
		"reasoner_id": name,
		"mode":        "http",
	})
	result, err := reasoner.Handler(ctx, input)
	durationMS := time.Since(start).Milliseconds()

	if err != nil {
		a.logger.Printf("reasoner %s failed: %v", name, err)
		a.logExecutionError(ctx, "reasoner.invoke.failed", "reasoner execution failed", map[string]any{
			"reasoner_id": name,
			"mode":        "http",
			"duration_ms": durationMS,
			"error":       err.Error(),
		})
		a.maybeGenerateVC(execCtx, input, nil, "failed", err.Error(), durationMS, reasoner)
		// Preserve structured downstream errors (e.g. policy denies from inner
		// agent calls) so local endpoint callers receive the correct status code.
		var execErr *ExecuteError
		if errors.As(err, &execErr) {
			response := map[string]any{"error": execErr.Message}
			if execErr.ErrorDetails != nil {
				response["error_details"] = execErr.ErrorDetails
			}
			statusCode := execErr.StatusCode
			if statusCode < 400 {
				statusCode = http.StatusInternalServerError
			}
			writeJSON(w, statusCode, response)
			return
		}

		writeJSON(w, http.StatusInternalServerError, map[string]any{
			"error": err.Error(),
		})
		return
	}

	a.maybeGenerateVC(execCtx, input, result, "succeeded", "", durationMS, reasoner)
	a.logExecutionInfo(ctx, "reasoner.invoke.complete", "reasoner execution completed", map[string]any{
		"reasoner_id": name,
		"mode":        "http",
		"duration_ms": durationMS,
	})
	writeJSON(w, http.StatusOK, result)
}

func (a *Agent) executeReasonerAsync(reasoner *Reasoner, input map[string]any, execCtx ExecutionContext) {
	ctx := contextWithExecution(context.Background(), execCtx)
	start := time.Now()
	a.logExecutionInfo(ctx, "reasoner.invoke.start", "starting asynchronous reasoner execution", map[string]any{
		"reasoner_id": reasoner.Name,
		"mode":        "async",
	})

	defer func() {
		if rec := recover(); rec != nil {
			errMsg := fmt.Sprintf("panic: %v", rec)
			durationMS := time.Since(start).Milliseconds()
			payload := map[string]any{
				"status":        "failed",
				"error":         errMsg,
				"execution_id":  execCtx.ExecutionID,
				"run_id":        execCtx.RunID,
				"completed_at":  time.Now().UTC().Format(time.RFC3339),
				"duration_ms":   durationMS,
				"reasoner_name": reasoner.Name,
			}
			a.logExecutionError(ctx, "reasoner.invoke.failed", "reasoner execution panicked", map[string]any{
				"reasoner_id": reasoner.Name,
				"mode":        "async",
				"duration_ms": durationMS,
				"error":       errMsg,
			})
			a.maybeGenerateVC(execCtx, input, nil, "failed", errMsg, durationMS, reasoner)
			if err := a.sendExecutionStatus(execCtx.ExecutionID, payload); err != nil {
				a.logger.Printf("failed to send panic status: %v", err)
				a.logExecutionWarn(ctx, "execution.status.failed", "failed to send panic status update", map[string]any{
					"reasoner_id": reasoner.Name,
					"error":       err.Error(),
				})
			}
		}
	}()

	result, err := reasoner.Handler(ctx, input)
	durationMS := time.Since(start).Milliseconds()
	payload := map[string]any{
		"execution_id":  execCtx.ExecutionID,
		"run_id":        execCtx.RunID,
		"completed_at":  time.Now().UTC().Format(time.RFC3339),
		"duration_ms":   durationMS,
		"reasoner_name": reasoner.Name,
	}

	if err != nil {
		payload["status"] = "failed"
		payload["error"] = err.Error()
		a.logExecutionError(ctx, "reasoner.invoke.failed", "reasoner execution failed", map[string]any{
			"reasoner_id": reasoner.Name,
			"mode":        "async",
			"duration_ms": durationMS,
			"error":       err.Error(),
		})
		a.maybeGenerateVC(execCtx, input, nil, "failed", err.Error(), durationMS, reasoner)
	} else {
		payload["status"] = "succeeded"
		payload["result"] = result
		a.logExecutionInfo(ctx, "reasoner.invoke.complete", "reasoner execution completed", map[string]any{
			"reasoner_id": reasoner.Name,
			"mode":        "async",
			"duration_ms": durationMS,
		})
		a.maybeGenerateVC(execCtx, input, result, "succeeded", "", durationMS, reasoner)
	}

	if err := a.sendExecutionStatus(execCtx.ExecutionID, payload); err != nil {
		a.logger.Printf("async status update failed: %v", err)
		a.logExecutionWarn(ctx, "execution.status.failed", "failed to send execution status update", map[string]any{
			"reasoner_id": reasoner.Name,
			"error":       err.Error(),
		})
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

		// Include API auth headers (Bearer token / API key)
		if a.cfg.Token != "" {
			req.Header.Set("Authorization", "Bearer "+a.cfg.Token)
		}

		// Sign request with DID auth headers if configured
		if a.client != nil {
			a.client.SignHTTPRequest(req, payload)
		}

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

	a.logExecutionInfo(ctx, "call.outbound.start", "starting cross-node call", map[string]any{
		"target":      target,
		"reasoner_id": strings.TrimPrefix(target, a.cfg.NodeID+"."),
	})
	url := fmt.Sprintf("%s/api/v1/execute/%s", strings.TrimSuffix(a.cfg.AgentFieldURL, "/"), strings.TrimPrefix(target, "/"))
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		a.logExecutionError(ctx, "call.outbound.failed", "failed to build cross-node call request", map[string]any{
			"target": target,
			"error":  err.Error(),
		})
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
	// DID metadata headers for execution context propagation.
	if a.didManager != nil && a.didManager.IsRegistered() {
		req.Header.Set("X-Agent-Node-DID", a.didManager.GetAgentDID())
	}
	if execCtx.AgentNodeDID != "" {
		req.Header.Set("X-Agent-Node-DID", execCtx.AgentNodeDID)
	}
	// Include caller agent identity for permission middleware
	req.Header.Set("X-Caller-Agent-ID", a.cfg.NodeID)

	if a.cfg.Token != "" {
		req.Header.Set("Authorization", "Bearer "+a.cfg.Token)
	}

	// Sign request with DID auth headers if configured
	if a.client != nil {
		a.client.SignHTTPRequest(req, body)
	}

	resp, err := a.httpClient.Do(req)
	if err != nil {
		a.logExecutionError(ctx, "call.outbound.failed", "cross-node call failed", map[string]any{
			"target": target,
			"error":  err.Error(),
		})
		return nil, fmt.Errorf("perform execute call: %w", err)
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read execute response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		// Try to parse structured error from control plane response.
		var errResp struct {
			Error        string      `json:"error"`
			ErrorDetails interface{} `json:"error_details"`
		}
		if json.Unmarshal(bodyBytes, &errResp) == nil && errResp.Error != "" {
			a.logExecutionError(ctx, "call.outbound.failed", "cross-node call rejected", map[string]any{
				"target": target,
				"status": resp.StatusCode,
				"error":  errResp.Error,
			})
			return nil, &ExecuteError{
				StatusCode:   resp.StatusCode,
				Message:      errResp.Error,
				ErrorDetails: errResp.ErrorDetails,
			}
		}
		a.logExecutionError(ctx, "call.outbound.failed", "cross-node call returned error status", map[string]any{
			"target": target,
			"status": resp.StatusCode,
		})
		return nil, &ExecuteError{
			StatusCode: resp.StatusCode,
			Message:    fmt.Sprintf("execute failed (%d): %s", resp.StatusCode, strings.TrimSpace(string(bodyBytes))),
		}
	}

	var execResp struct {
		ExecutionID  string         `json:"execution_id"`
		RunID        string         `json:"run_id"`
		Status       string         `json:"status"`
		Result       map[string]any `json:"result"`
		ErrorMessage *string        `json:"error_message"`
		ErrorDetails interface{}    `json:"error_details"`
	}
	if err := json.Unmarshal(bodyBytes, &execResp); err != nil {
		a.logExecutionError(ctx, "call.outbound.failed", "failed to decode cross-node call response", map[string]any{
			"target": target,
			"error":  err.Error(),
		})
		return nil, fmt.Errorf("decode execute response: %w", err)
	}

	if execResp.ErrorMessage != nil && *execResp.ErrorMessage != "" {
		a.logExecutionError(ctx, "call.outbound.failed", "cross-node call returned execution error", map[string]any{
			"target": target,
			"error":  *execResp.ErrorMessage,
		})
		return nil, &ExecuteError{
			StatusCode:   resp.StatusCode,
			Message:      *execResp.ErrorMessage,
			ErrorDetails: execResp.ErrorDetails,
		}
	}
	if !strings.EqualFold(execResp.Status, "succeeded") {
		a.logExecutionError(ctx, "call.outbound.failed", "cross-node call did not succeed", map[string]any{
			"target": target,
			"status": execResp.Status,
		})
		return nil, &ExecuteError{
			StatusCode:   resp.StatusCode,
			Message:      fmt.Sprintf("execute status %s", execResp.Status),
			ErrorDetails: execResp.ErrorDetails,
		}
	}

	a.logExecutionInfo(ctx, "call.outbound.complete", "cross-node call completed", map[string]any{
		"target":       target,
		"execution_id": execResp.ExecutionID,
		"run_id":       execResp.RunID,
	})
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

	// Sign request with DID auth headers if configured
	if a.client != nil {
		a.client.SignHTTPRequest(req, body)
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

	a.logExecutionInfo(ctx, "call.local.start", "starting local reasoner call", map[string]any{
		"reasoner_id": reasonerName,
		"depth":       childCtx.Depth,
	})
	a.emitWorkflowEvent(childCtx, "running", input, nil, nil, 0)

	start := time.Now()
	result, err := reasoner.Handler(ctx, input)
	durationMS := time.Since(start).Milliseconds()

	if err != nil {
		a.logExecutionError(ctx, "call.local.failed", "local reasoner call failed", map[string]any{
			"reasoner_id": reasonerName,
			"duration_ms": durationMS,
			"error":       err.Error(),
		})
		a.emitWorkflowEvent(childCtx, "failed", input, nil, err, durationMS)
	} else {
		a.logExecutionInfo(ctx, "call.local.complete", "local reasoner call completed", map[string]any{
			"reasoner_id": reasonerName,
			"duration_ms": durationMS,
		})
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

// AIWithTools makes an AI call with tool-calling support.
// It discovers available capabilities, converts them to tool schemas,
// and runs a tool-call loop that dispatches calls via agent.Call().
//
// Example:
//
//	resp, trace, err := agent.AIWithTools(ctx, "Help the user with their ticket",
//	    ai.DefaultToolCallConfig(),
//	    agent.WithDiscoveryInputSchema(true),
//	)
func (a *Agent) AIWithTools(ctx context.Context, prompt string, config ai.ToolCallConfig, discoveryOpts ...DiscoveryOption) (*ai.Response, *ai.ToolCallTrace, error) {
	if a.aiClient == nil {
		return nil, nil, errors.New("AI not configured for this agent; set AIConfig in agent Config")
	}

	// Discover available tools
	discoveryOpts = append(discoveryOpts, WithDiscoveryInputSchema(true))
	result, err := a.Discover(ctx, discoveryOpts...)
	if err != nil {
		return nil, nil, fmt.Errorf("discover tools: %w", err)
	}
	if result.JSON == nil {
		return nil, nil, errors.New("discovery returned no JSON result")
	}

	tools := ai.CapabilitiesToToolDefinitions(result.JSON.Capabilities)
	if len(tools) == 0 {
		// No tools available, fall back to regular AI call
		resp, err := a.AI(ctx, prompt)
		return resp, &ai.ToolCallTrace{TotalTurns: 1}, err
	}

	messages := []ai.Message{
		{
			Role:    "user",
			Content: []ai.ContentPart{{Type: "text", Text: prompt}},
		},
	}

	callFn := func(ctx context.Context, target string, input map[string]interface{}) (map[string]interface{}, error) {
		if strings.Contains(target, ":skill:") {
			parts := strings.SplitN(target, ":skill:", 2)
			target = parts[0] + "." + parts[1]
		} else if strings.Contains(target, ":") {
			parts := strings.SplitN(target, ":", 2)
			target = parts[0] + "." + parts[1]
		}
		return a.Call(ctx, target, input)
	}

	return a.aiClient.ExecuteToolCallLoop(ctx, messages, tools, config, callFn)
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

// Memory returns the agent's memory system for state management.
// Memory provides hierarchical scoped storage (workflow, session, user, global).
//
// Example usage:
//
//	// Store in default (session) scope
//	agent.Memory().Set(ctx, "key", "value")
//
//	// Retrieve from session scope
//	val, _ := agent.Memory().Get(ctx, "key")
//
//	// Use global scope for cross-session data
//	agent.Memory().GlobalScope().Set(ctx, "shared_key", data)
func (a *Agent) Memory() *Memory {
	return a.memory
}

// formatHeartbeatInterval formats a LeaseRefreshInterval as a Go duration
// string for the control plane's HeartbeatInterval field. If the interval
// is zero (which should not happen after NewAgent's default, but guards
// against direct field construction), it falls back to the documented
// default so the control plane does not receive "0s".
func formatHeartbeatInterval(d time.Duration) string {
	if d <= 0 {
		return defaultHeartbeatInterval.String()
	}
	return d.String()
}

// defaultHeartbeatInterval is the fallback sent to the control plane when
// LeaseRefreshInterval is zero. Kept in sync with the LeaseRefreshInterval
// default applied in NewAgent.
var defaultHeartbeatInterval = 30 * time.Second
