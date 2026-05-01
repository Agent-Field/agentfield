package agent

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/types"
)

// Initialize registers the agent with the AgentField control plane without starting a listener.
func (a *Agent) Initialize(ctx context.Context) error {
	a.initMu.Lock()
	defer a.initMu.Unlock()

	if a.initialized {
		return nil
	}

	a.logExecutionInfo(ctx, "agent.initialize.start", "initializing agent", map[string]any{
		"node_id":         a.cfg.NodeID,
		"deployment_type": a.cfg.DeploymentType,
	})

	a.ensureProcessLogRing()

	if a.client == nil {
		a.logExecutionError(ctx, "agent.initialize.failed", "agent field URL is required for server mode", map[string]any{
			"node_id": a.cfg.NodeID,
		})
		return errors.New("AgentFieldURL is required when running in server mode")
	}

	if len(a.reasoners) == 0 {
		a.logExecutionError(ctx, "agent.initialize.failed", "no reasoners registered", map[string]any{
			"node_id": a.cfg.NodeID,
		})
		return errors.New("no reasoners registered")
	}

	if err := a.registerNode(ctx); err != nil {
		a.logExecutionError(ctx, "agent.initialize.failed", "node registration failed", map[string]any{
			"node_id": a.cfg.NodeID,
			"error":   err.Error(),
		})
		return fmt.Errorf("register node: %w", err)
	}

	// Auto-register DIDs if enabled and not already configured.
	if a.cfg.EnableDID || a.cfg.VCEnabled {
		if err := a.initializeDIDSystem(ctx); err != nil {
			a.logger.Printf("warn: DID initialization failed: %v (continuing without DID)", err)
		}
	}

	// Mark agent as ready. The control plane protects pending_approval state
	// (returns 409 if still pending), so this is safe to call unconditionally.
	// For agents that went through tag approval, the admin process transitions
	// them to "starting" first, so markReady correctly advances to "ready".
	if err := a.markReady(ctx); err != nil {
		a.logger.Printf("warn: initial status update failed: %v", err)
	}

	a.startLeaseLoop()
	a.initialized = true
	a.logExecutionInfo(ctx, "agent.initialize.complete", "agent initialized", map[string]any{
		"node_id": a.cfg.NodeID,
	})
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
	a.logExecutionInfo(ctx, "agent.serve.start", "starting agent server", map[string]any{
		"node_id": a.cfg.NodeID,
	})
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
		a.logExecutionInfo(context.Background(), "agent.serve.stop", "context cancelled, shutting down", map[string]any{
			"node_id": a.cfg.NodeID,
		})
		return a.shutdown(context.Background())
	case sig := <-sigCh:
		a.logger.Printf("received signal %s, shutting down", sig)
		a.logExecutionInfo(context.Background(), "agent.serve.stop", "received termination signal", map[string]any{
			"node_id": a.cfg.NodeID,
			"signal":  sig.String(),
		})
		return a.shutdown(context.Background())
	}
}

func (a *Agent) registerNode(ctx context.Context) error {
	now := time.Now().UTC()
	a.logExecutionInfo(ctx, "node.register.start", "registering node with control plane", map[string]any{
		"node_id":         a.cfg.NodeID,
		"deployment_type": a.cfg.DeploymentType,
	})

	reasoners := make([]types.ReasonerDefinition, 0, len(a.reasoners))
	for _, reasoner := range a.reasoners {
		reasoners = append(reasoners, types.ReasonerDefinition{
			ID:             reasoner.Name,
			InputSchema:    reasoner.InputSchema,
			OutputSchema:   reasoner.OutputSchema,
			Tags:           reasoner.Tags,
			ProposedTags:   reasoner.Tags,
			Triggers:       reasoner.Triggers,
			AcceptsWebhook: reasoner.AcceptsWebhook,
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
			HeartbeatInterval: a.registeredHeartbeatInterval(),
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
			"tags": a.cfg.Tags,
		},
		Features:       map[string]any{},
		DeploymentType: a.cfg.DeploymentType,
	}

	resp, err := a.client.RegisterNode(ctx, payload)
	if err != nil {
		a.logExecutionError(ctx, "node.register.failed", "node registration failed", map[string]any{
			"node_id": a.cfg.NodeID,
			"error":   err.Error(),
		})
		return err
	}

	// Handle pending approval state: poll until approved
	if resp != nil && resp.Status == "pending_approval" {
		a.logger.Printf("node %s registered but awaiting tag approval (pending tags: %v)", a.cfg.NodeID, resp.PendingTags)
		a.logExecutionWarn(ctx, "node.register.pending_approval", "node registered but awaiting tag approval", map[string]any{
			"node_id":      a.cfg.NodeID,
			"pending_tags": resp.PendingTags,
		})
		if err := a.waitForApproval(ctx); err != nil {
			return fmt.Errorf("tag approval wait failed: %w", err)
		}
		a.logger.Printf("node %s tag approval granted", a.cfg.NodeID)
		a.logExecutionInfo(ctx, "node.register.approved", "tag approval granted", map[string]any{
			"node_id": a.cfg.NodeID,
		})
		return nil
	}

	a.logger.Printf("node %s registered with AgentField", a.cfg.NodeID)
	a.logExecutionInfo(ctx, "node.register.complete", "node registered with control plane", map[string]any{
		"node_id": a.cfg.NodeID,
	})
	return nil
}

func (a *Agent) waitForApproval(ctx context.Context) error {
	const approvalTimeout = 5 * time.Minute
	ctx, cancel := context.WithTimeout(ctx, approvalTimeout)
	defer cancel()

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	a.logExecutionInfo(ctx, "node.approval.wait.start", "waiting for tag approval", map[string]any{
		"node_id": a.cfg.NodeID,
	})

	for {
		select {
		case <-ctx.Done():
			if ctx.Err() == context.DeadlineExceeded {
				a.logExecutionWarn(ctx, "node.approval.wait.timeout", "tag approval timed out", map[string]any{
					"node_id": a.cfg.NodeID,
				})
				return fmt.Errorf("tag approval timed out after %s", approvalTimeout)
			}
			a.logExecutionWarn(ctx, "node.approval.wait.cancelled", "tag approval wait cancelled", map[string]any{
				"node_id": a.cfg.NodeID,
				"error":   ctx.Err().Error(),
			})
			return ctx.Err()
		case <-ticker.C:
			node, err := a.client.GetNode(ctx, a.cfg.NodeID)
			if err != nil {
				a.logger.Printf("polling for approval status failed: %v", err)
				continue
			}
			status, _ := node["lifecycle_status"].(string)
			if status != "" && status != "pending_approval" {
				a.logExecutionInfo(ctx, "node.approval.wait.complete", "tag approval granted", map[string]any{
					"node_id": a.cfg.NodeID,
					"status":  status,
				})
				return nil
			}
			a.logger.Printf("node %s still pending approval...", a.cfg.NodeID)
			a.logExecutionInfo(ctx, "node.approval.wait.pending", "tag approval still pending", map[string]any{
				"node_id": a.cfg.NodeID,
			})
		}
	}
}

func (a *Agent) markReady(ctx context.Context) error {
	score := 100
	_, err := a.client.UpdateStatus(ctx, a.cfg.NodeID, types.NodeStatusUpdate{
		Phase:       "ready",
		Version:     a.cfg.Version,
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
			a.logExecutionError(context.Background(), "server.failed", "server listener exited with error", map[string]any{
				"node_id": a.cfg.NodeID,
				"error":   err.Error(),
			})
		}
	}()

	a.logger.Printf("listening on %s", a.cfg.ListenAddress)
	a.logExecutionInfo(context.Background(), "server.start", "agent HTTP server listening", map[string]any{
		"node_id": a.cfg.NodeID,
		"listen":  a.cfg.ListenAddress,
	})
	return nil
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
	a.logExecutionInfo(ctx, "agent.shutdown.start", "shutting down agent", map[string]any{
		"node_id": a.cfg.NodeID,
	})
	close(a.stopLease)

	if a.client != nil {
		if _, err := a.client.Shutdown(ctx, a.cfg.NodeID, types.ShutdownRequest{Reason: "shutdown", Version: a.cfg.Version}); err != nil {
			a.logger.Printf("failed to notify shutdown: %v", err)
			a.logExecutionWarn(ctx, "agent.shutdown.status_failed", "failed to notify control plane about shutdown", map[string]any{
				"node_id": a.cfg.NodeID,
				"error":   err.Error(),
			})
		}
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
	a.logExecutionInfo(ctx, "agent.shutdown.complete", "agent shutdown complete", map[string]any{
		"node_id": a.cfg.NodeID,
	})
	return nil
}

// registeredHeartbeatInterval returns the HeartbeatInterval value sent to
// the control plane during node registration. When the lease loop is
// disabled, the agent does not send periodic heartbeats, so we advertise
// "0s" to signal that behavior accurately rather than registering a
// cadence the agent does not honor.
func (a *Agent) registeredHeartbeatInterval() string {
	if a.cfg.DisableLeaseLoop {
		return "0s"
	}
	return formatHeartbeatInterval(a.cfg.LeaseRefreshInterval)
}
