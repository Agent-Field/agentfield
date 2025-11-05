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
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/agentfield/haxen/sdk/go/ai"
	"github.com/agentfield/haxen/sdk/go/client"
	"github.com/agentfield/haxen/sdk/go/types"
)

type executionContextKey struct{}

// ExecutionContext captures the headers Haxen sends with each execution request.
type ExecutionContext struct {
	RunID             string
	ExecutionID       string
	ParentExecutionID string
	SessionID         string
	ActorID           string
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

// Reasoner represents a single handler exposed by the agent.
type Reasoner struct {
	Name         string
	Handler      HandlerFunc
	InputSchema  json.RawMessage
	OutputSchema json.RawMessage
}

// Config drives Agent behaviour.
type Config struct {
	NodeID        string
	Version       string
	TeamID        string
	HaxenURL      string
	ListenAddress string
	PublicURL     string
	Token         string

	LeaseRefreshInterval time.Duration
	DisableLeaseLoop     bool
	Logger               *log.Logger

	// AIConfig configures LLM/AI capabilities
	// If nil, AI features will be disabled
	AIConfig *ai.Config
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
	if cfg.HaxenURL == "" {
		return nil, errors.New("config.HaxenURL is required")
	}
	if cfg.ListenAddress == "" {
		cfg.ListenAddress = ":8001"
	}
	if cfg.PublicURL == "" {
		cfg.PublicURL = "http://localhost" + cfg.ListenAddress
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

	c, err := client.New(cfg.HaxenURL, client.WithHTTPClient(httpClient), client.WithBearerToken(cfg.Token))
	if err != nil {
		return nil, err
	}

	// Initialize AI client if config provided
	var aiClient *ai.Client
	if cfg.AIConfig != nil {
		aiClient, err = ai.NewClient(cfg.AIConfig)
		if err != nil {
			return nil, fmt.Errorf("initialize AI client: %w", err)
		}
	}

	return &Agent{
		cfg:        cfg,
		client:     c,
		httpClient: httpClient,
		reasoners:  make(map[string]*Reasoner),
		aiClient:   aiClient,
		stopLease:  make(chan struct{}),
		logger:     cfg.Logger,
	}, nil
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

func generateRunID() string {
	return fmt.Sprintf("run_%d_%06d", time.Now().UnixNano(), rand.Intn(1_000_000))
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

	a.reasoners[name] = meta
}

// Initialize registers the agent with the Haxen control plane without starting a listener.
func (a *Agent) Initialize(ctx context.Context) error {
	a.initMu.Lock()
	defer a.initMu.Unlock()

	if a.initialized {
		return nil
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

// Run starts the agent HTTP server, registers with the control plane, and blocks until ctx is cancelled.
func (a *Agent) Run(ctx context.Context) error {
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
		Features: map[string]any{},
	}

	_, err := a.client.RegisterNode(ctx, payload)
	if err != nil {
		return err
	}

	a.logger.Printf("node %s registered with Haxen", a.cfg.NodeID)
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

func (a *Agent) handler() http.Handler {
	a.handlerOnce.Do(func() {
		mux := http.NewServeMux()
		mux.HandleFunc("/health", a.healthHandler)
		mux.HandleFunc("/reasoners/", a.handleReasoner)
		a.router = mux
	})
	return a.router
}

func (a *Agent) healthHandler(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
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
	}

	ctx := contextWithExecution(r.Context(), execCtx)

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

// Call invokes another reasoner via the Haxen control plane, preserving execution context.
func (a *Agent) Call(ctx context.Context, target string, input map[string]any) (map[string]any, error) {
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

	url := fmt.Sprintf("%s/api/v1/execute/%s", strings.TrimSuffix(a.cfg.HaxenURL, "/"), strings.TrimPrefix(target, "/"))
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
