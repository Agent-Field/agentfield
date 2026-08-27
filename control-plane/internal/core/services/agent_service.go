// agentfield/internal/core/services/agent_service.go
package services

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
)

// DefaultAgentService implements the AgentService interface
type DefaultAgentService struct {
	processManager  interfaces.ProcessManager
	portManager     interfaces.PortManager
	registryStorage interfaces.RegistryStorage
	agentClient     interfaces.AgentClient
	agentfieldHome  string
	confirmation    packages.ProcessConfirmationPolicy
}

// NewAgentService creates a new agent service instance
func NewAgentService(
	processManager interfaces.ProcessManager,
	portManager interfaces.PortManager,
	registryStorage interfaces.RegistryStorage,
	agentClient interfaces.AgentClient,
	agentfieldHome string,
) interfaces.AgentService {
	return &DefaultAgentService{
		processManager:  processManager,
		portManager:     portManager,
		registryStorage: registryStorage,
		agentClient:     agentClient,
		agentfieldHome:  agentfieldHome,
	}
}

// RunAgent starts an installed agent
func (as *DefaultAgentService) RunAgent(name string, options domain.RunOptions) (*domain.RunningAgent, error) {
	return as.runAgentGuarded(name, options, map[string]bool{})
}

// runAgentGuarded starts a node; inProgress tracks nodes already being started
// in this dependency chain to break cycles.
func (as *DefaultAgentService) runAgentGuarded(name string, options domain.RunOptions, inProgress map[string]bool) (*domain.RunningAgent, error) {
	fmt.Printf("🚀 Launching agent node: %s\n", name)
	inProgress[name] = true

	// 1. Check if agent node is installed
	registry, err := as.loadRegistryDirect()
	if err != nil {
		return nil, fmt.Errorf("failed to load registry: %w", err)
	}

	// Try to find the agent with exact name first, then try normalized versions
	agentNode, actualName, exists := as.findAgentInRegistry(registry, name)
	if !exists {
		return nil, fmt.Errorf("agent node %s not installed", name)
	}

	// Use the actual name from registry for all subsequent operations
	name = actualName
	legacyDesiredState := agentNode.DesiredState == ""
	agentNode.EnsureDesiredState()

	// 2. Check current state and reconcile if needed
	actuallyRunning, wasReconciled := as.reconcileLifecycleProcessState(&agentNode, name)
	if wasReconciled || legacyDesiredState {
		// Save reconciled state
		if err := as.updateRegistryEntry(name, agentNode); err != nil {
			fmt.Printf("Warning: failed to save reconciled registry state: %v\n", err)
		}
	}

	// If actually running after reconciliation, return appropriate message
	if actuallyRunning {
		if agentNode.Runtime.Port == nil {
			agentNode.Status = "stopped"
			agentNode.Runtime.PID = nil
			agentNode.Runtime.StartedAt = nil
			agentNode.Runtime.BootID = ""
			agentNode.Runtime.StartTime = ""
			if err := as.updateRegistryEntry(name, agentNode); err != nil {
				return nil, fmt.Errorf("failed to reconcile running agent without a port: %w", err)
			}
		} else {
			return nil, fmt.Errorf("agent node %s is already running on port %d", name, *agentNode.Runtime.Port)
		}
	}

	// An explicit start is the user's intent to run. Record it before the
	// launch so a container replacement during or after this call restores
	// the node; a stop that lands while the node is still starting is written
	// after this point and wins, because updateRuntimeInfo never overrides
	// desired_state.
	if agentNode.DesiredState != packages.DesiredStateRunning {
		agentNode.DesiredState = packages.DesiredStateRunning
		if err := as.updateRegistryEntry(name, agentNode); err != nil {
			// Not fatal here: the runtime write after readiness reports an
			// unwritable registry with the error callers already handle.
			fmt.Printf("Warning: failed to record running intent for %s: %v\n", name, err)
		}
	}

	// Reserve a requested/preferred port before dependencies start. Otherwise
	// a dependency's FindFreePort can claim the node's restore/update port.
	var releasePort func()
	requestedPort := options.Port
	options.Port, releasePort, err = as.reserveRequestedPort(options.Port, options.PortIsPreference)
	if err != nil {
		return nil, err
	}
	defer releasePort()

	// 2b. Start declared node dependencies first while the requested node port
	// is reserved.
	as.startNodeDependencies(agentNode, inProgress, options)

	// 3. Allocate port
	fmt.Printf("🔍 Searching for available port...\n")
	port := options.Port
	if port == 0 {
		port, err = as.portManager.FindFreePort(8001)
		if err != nil {
			return nil, fmt.Errorf("failed to allocate port: %w", err)
		}
	}

	// 4-5. Start the process and wait for readiness. Automatically allocated and
	// internal preferred ports retry exactly once on a fresh port after a strict
	// bind conflict. Explicit user ports fail on the requested port.
	retryOnConflict := requestedPort <= 0 || options.PortIsPreference
	pid, port, startErr := as.startWithPortRetry(port, retryOnConflict, func(p int) (int, error, bool) {
		return as.attemptStart(agentNode, name, p)
	})
	if startErr != nil {
		// Surface the node's own log inline so the real traceback / exit reason
		// is visible without a separate `af logs` round-trip.
		as.printStartupFailureDiagnostics(agentNode, name)
		return nil, startErr
	}

	fmt.Printf("🧠 Agent node registered with AgentField Server\n")

	// 6. Update registry with runtime info. A node the registry cannot record
	// must not be left running: the next restore would start another copy
	// beside it (a full volume, for instance, fails every write).
	if err := as.updateRuntimeInfo(name, port, pid); err != nil {
		if stopErr := as.processManager.Stop(pid); stopErr != nil {
			fmt.Printf("Warning: could not stop unrecorded node %s (pid %d): %v\n", name, pid, stopErr)
		}
		return nil, fmt.Errorf("failed to update runtime info: %w", err)
	}

	// 7. Display agent node capabilities
	if err := as.displayCapabilities(agentNode, port); err != nil {
		fmt.Printf("⚠️  Could not fetch capabilities: %v\n", err)
	}

	fmt.Printf("\n💡 Agent node running in background (PID: %d)\n", pid)
	fmt.Printf("💡 View logs: af logs %s\n", name)
	fmt.Printf("💡 Stop agent node: af stop %s\n", name)

	// Convert to domain model and return
	runningAgent := as.convertToRunningAgent(agentNode)
	runningAgent.PID = pid
	runningAgent.Port = port
	runningAgent.StartedAt = time.Now()

	return &runningAgent, nil
}

func (as *DefaultAgentService) reserveRequestedPort(port int, allowFallback bool) (int, func(), error) {
	if port <= 0 {
		return port, func() {}, nil
	}
	if err := as.portManager.ReservePort(port); err != nil {
		if allowFallback {
			return 0, func() {}, nil
		}
		return 0, func() {}, fmt.Errorf("requested port %d is unavailable: %w", port, err)
	}
	return port, func() { _ = as.portManager.ReleasePort(port) }, nil
}

// attemptStart builds the process config, starts the node on the given port,
// and waits for it to answer its health check. On failure it stops the process
// and reports whether the failure was a strict-port bind conflict (detected
// from the node's own log), which the caller uses to decide on a fresh-port
// retry.
func (as *DefaultAgentService) attemptStart(agentNode packages.InstalledPackage, name string, port int) (pid int, err error, portConflict bool) {
	fmt.Printf("✅ Assigned port: %d\n", port)

	fmt.Printf("📡 Starting agent node process...\n")
	processConfig, err := as.buildProcessConfig(agentNode, port)
	if err != nil {
		return 0, err, false
	}
	pid, err = as.processManager.Start(processConfig)
	if err != nil {
		return 0, fmt.Errorf("failed to start agent node: %w", err), false
	}

	healthPath := "/health"
	expectedNodeID := name
	if metadata, err := packages.ParsePackageMetadata(agentNode.Path); err == nil {
		healthPath = metadata.HealthcheckPath()
		if metadata.AgentNode.NodeID != "" {
			expectedNodeID = metadata.AgentNode.NodeID
		}
	}

	if waitErr := as.waitForAgentNode(port, healthPath, expectedNodeID, nodeReadyTimeout()); waitErr != nil {
		// Read the log before killing so a strict-port exit is still visible.
		conflict := logIndicatesPortConflict(readLogTailLines(agentNode.Runtime.LogFile, 40))
		if stopErr := as.processManager.Stop(pid); stopErr != nil {
			return 0, fmt.Errorf("agent node failed to start: %w (additionally failed to stop process: %v)", waitErr, stopErr), conflict
		}
		return 0, fmt.Errorf("agent node failed to start: %w", waitErr), conflict
	}
	return pid, nil, false
}

// startWithPortRetry runs attemptFn on the initial port. When retryOnConflict
// is true (for an automatically allocated port), a strict-port conflict is
// retried exactly once on a fresh port. It returns the final pid, the port
// actually used, and any error.
func (as *DefaultAgentService) startWithPortRetry(initialPort int, retryOnConflict bool, attemptFn func(port int) (pid int, err error, portConflict bool)) (int, int, error) {
	port := initialPort
	pid, err, conflict := attemptFn(port)
	if err == nil || !conflict || !retryOnConflict {
		return pid, port, err
	}

	retryPort, rerr := as.freshRetryPort(port)
	if rerr != nil || retryPort == port {
		// No distinct fresh port to try — keep the original failure.
		return pid, port, err
	}

	fmt.Printf("⚠️  Port %d unavailable, retrying on a fresh port\n", port)
	pid, err, _ = attemptFn(retryPort)
	return pid, retryPort, err
}

// freshRetryPort excludes the port that just failed to bind, then asks the port
// manager for a new one, so the retry never reuses the conflicting port.
func (as *DefaultAgentService) freshRetryPort(failedPort int) (int, error) {
	_ = as.portManager.ReservePort(failedPort)
	return as.portManager.FindFreePort(8001)
}

// printStartupFailureDiagnostics prints the tail of the node's log so the real
// exit reason (traceback, bind error, missing dependency, …) is visible inline,
// plus a pointer to the full logs.
func (as *DefaultAgentService) printStartupFailureDiagnostics(agentNode packages.InstalledPackage, name string) {
	lines := readLogTailLines(agentNode.Runtime.LogFile, 15)
	if len(lines) > 0 {
		fmt.Printf("\n📄 Last %d line(s) of %s log:\n", len(lines), name)
		for _, line := range lines {
			fmt.Printf("    %s\n", line)
		}
	}
	fmt.Printf("💡 Full logs: af logs %s\n", name)
}

// readLogTailLines returns the last n lines of the file at path. A missing or
// unreadable file yields nil so callers treat "no log" and "no diagnostic info"
// the same way.
func readLogTailLines(path string, n int) []string {
	if path == "" {
		return nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	text := strings.TrimRight(string(data), "\n")
	if text == "" {
		return nil
	}
	lines := strings.Split(text, "\n")
	if n > 0 && len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return lines
}

// logIndicatesPortConflict reports whether the node's log tail shows the SDK's
// strict-port bind failure — the node was assigned a port that turned out to be
// unavailable at bind time and exited so the control plane can reallocate. The
// SDK logs "AGENTFIELD_STRICT_PORT set but the assigned port N is unavailable"
// and raises "assigned port N is unavailable"; both contain "assigned port" and
// "unavailable".
func logIndicatesPortConflict(lines []string) bool {
	for _, line := range lines {
		if strings.Contains(line, "assigned port") && strings.Contains(line, "unavailable") {
			return true
		}
	}
	return false
}

// StopAgent stops a running agent with robust error handling
func (as *DefaultAgentService) StopAgent(name string) error {
	return as.stopAgent(name, false)
}

// StopAgentForUpdate stops the observed process without changing the user's
// persisted intent that the package should be running. Package update jobs use
// this path; API/CLI stops continue to use StopAgent.
func (as *DefaultAgentService) StopAgentForUpdate(name string) error {
	return as.stopAgent(name, true)
}

func (as *DefaultAgentService) stopAgent(name string, preserveDesiredState bool) error {
	// Load registry to get agent info
	registry, err := as.loadRegistryDirect()
	if err != nil {
		return fmt.Errorf("failed to load registry: %w", err)
	}

	// Try to find the agent with exact name first, then try normalized versions
	pkg, actualName, exists := as.findAgentInRegistry(registry, name)
	if !exists {
		return fmt.Errorf("agent %s is not installed", name)
	}

	// Use the actual name from registry for all subsequent operations
	name = actualName
	pkg.EnsureDesiredState()
	desiredStateChanged := false
	if !preserveDesiredState && pkg.DesiredState != packages.DesiredStateStopped {
		pkg.DesiredState = packages.DesiredStateStopped
		desiredStateChanged = true
	}

	// Check current state and reconcile if needed. An existing PID whose
	// identity cannot be established is deliberately not mutated: an explicit
	// stop must tell the operator to kill it manually instead of erasing the
	// only PID that can recover it.
	assessment := as.assessLifecycleProcess(pkg, name)
	actuallyRunning, wasReconciled := reconcileRecordedAssessment(&pkg, name, assessment)
	if assessment.Ownership == packages.RecordedProcessUnknown {
		return fmt.Errorf("could not verify that process %d is %s; stop it manually", *pkg.Runtime.PID, name)
	}
	if wasReconciled || desiredStateChanged {
		// Save reconciled state
		if err := as.updateRegistryEntry(name, pkg); err != nil {
			return fmt.Errorf("failed to save reconciled registry state: %w", err)
		}
	}

	// An explicit stop is idempotent. Reconciliation may already have observed
	// the process exit, but the desired stopped state persisted above still
	// matters because it disarms boot restore.
	if !actuallyRunning {
		return nil
	}

	if assessment.Owned() {
		result, err := packages.StopRecordedProcessWithAssessment(context.Background(), name, pkg, assessment)
		if err != nil {
			return err
		}
		if result.HTTPAttempted {
			fmt.Printf("🛑 Attempting graceful HTTP shutdown for agent %s\n", name)
		}
		if result.InterruptSent || result.ForceKillNeeded {
			fmt.Printf("🔄 Falling back to process signal shutdown for agent %s\n", name)
		}
	}

	// Update registry to mark as stopped
	pkg.Status = "stopped"
	pkg.Runtime.PID = nil
	if !preserveDesiredState {
		pkg.Runtime.Port = nil
	}
	pkg.Runtime.StartedAt = nil
	pkg.Runtime.BootID = ""
	pkg.Runtime.StartTime = ""
	// Save registry
	if err := as.updateRegistryEntry(name, pkg); err != nil {
		return fmt.Errorf("failed to update registry: %w", err)
	}

	return nil
}

// GetAgentStatus returns the status of a specific agent with process reconciliation
func (as *DefaultAgentService) GetAgentStatus(name string) (*domain.AgentStatus, error) {
	registry, err := as.loadRegistryDirect()
	if err != nil {
		return nil, fmt.Errorf("failed to load registry: %w", err)
	}

	// Try to find the agent with exact name first, then try normalized versions
	pkg, actualName, exists := as.findAgentInRegistry(registry, name)
	if !exists {
		return nil, fmt.Errorf("agent %s is not installed", name)
	}

	// Use the actual name from registry for all subsequent operations
	name = actualName

	// Reconcile registry state with actual process state
	actuallyRunning, reconciled := as.reconcileReadProcessState(&pkg, name)
	if reconciled {
		// Save updated registry if reconciliation occurred
		if err := as.updateRegistryEntry(name, pkg); err != nil {
			fmt.Printf("Warning: failed to save reconciled registry state: %v\n", err)
		}
	}

	status := &domain.AgentStatus{
		Name:      pkg.Name,
		IsRunning: actuallyRunning,
	}

	if actuallyRunning && pkg.Runtime.Port != nil {
		status.Port = *pkg.Runtime.Port
	}

	if actuallyRunning && pkg.Runtime.PID != nil {
		status.PID = *pkg.Runtime.PID
	}

	if pkg.Runtime.StartedAt != nil {
		if startedAt, err := time.Parse(time.RFC3339, *pkg.Runtime.StartedAt); err == nil {
			status.LastSeen = startedAt
			// Calculate uptime if running
			if actuallyRunning {
				uptime := time.Since(startedAt)
				status.Uptime = uptime.String()
			}
		}
	}

	return status, nil
}

// reconcileLifecycleProcessState applies the signal-safe shared ownership rule
// before any lifecycle decision. A recycled or unidentified PID is never enough
// by itself to block a start or permit a stop.
func (as *DefaultAgentService) reconcileLifecycleProcessState(pkg *packages.InstalledPackage, name string) (bool, bool) {
	return as.reconcileProcessStateWithProbes(
		pkg, name, runtime.GOOS, true, make(map[int]packages.HealthIdentity),
		packages.RuntimeProcessStatus, packages.ProbeHealthIdentity,
	)
}

func (as *DefaultAgentService) assessLifecycleProcess(pkg packages.InstalledPackage, name string) packages.RecordedProcessAssessment {
	return packages.AssessRecordedProcessWith(
		context.Background(), name, pkg, packages.RuntimeProcessStatus,
		packages.ProbeHealthIdentity, as.lifecycleConfirmationPolicy(),
	)
}

func (as *DefaultAgentService) lifecycleConfirmationPolicy() packages.ProcessConfirmationPolicy {
	if as.confirmation.Attempts > 0 {
		return as.confirmation
	}
	return packages.LifecycleConfirmationPolicy()
}

// reconcileReadProcessState keeps the historical cheap Darwin projection while
// Linux and Windows use the ownership rule. In particular, Darwin dashboard
// polling must not spawn ps for every package.
func (as *DefaultAgentService) reconcileReadProcessState(pkg *packages.InstalledPackage, name string) (bool, bool) {
	return as.reconcileProcessStateWithProbes(
		pkg, name, runtime.GOOS, false, make(map[int]packages.HealthIdentity),
		packages.RuntimeProcessStatus, packages.ProbeHealthIdentity,
	)
}

// reconcileProcessState is retained for focused legacy tests that exercise the
// old cheap PID projection directly. Production lifecycle and status callers
// use the explicit ownership-aware helpers above.
func (as *DefaultAgentService) reconcileProcessState(pkg *packages.InstalledPackage, name string) (bool, bool) {
	return as.reconcileProcessStateWithProbes(
		pkg, name, runtime.GOOS, false, make(map[int]packages.HealthIdentity),
		func(info packages.RuntimeInfo) packages.RuntimeProcessState {
			if packages.RuntimePIDAlive(info) {
				return packages.RuntimeProcessAliveState
			}
			return packages.RuntimeProcessDead
		},
		func(context.Context, int, string) packages.HealthIdentity { return packages.HealthIdentity{} },
	)
}

func (as *DefaultAgentService) reconcileProcessStateWithProbe(
	pkg *packages.InstalledPackage,
	name string,
	goos string,
	memo map[int]packages.HealthIdentity,
	probe func(context.Context, int, string) packages.HealthIdentity,
) (bool, bool) {
	return as.reconcileProcessStateWithProbes(
		pkg, name, goos, false, memo, packages.RuntimeProcessStatus, probe,
	)
}

func (as *DefaultAgentService) reconcileProcessStateWithProbes(
	pkg *packages.InstalledPackage,
	name string,
	goos string,
	lifecycle bool,
	memo map[int]packages.HealthIdentity,
	processStatus func(packages.RuntimeInfo) packages.RuntimeProcessState,
	probe func(context.Context, int, string) packages.HealthIdentity,
) (bool, bool) {
	legacyDesiredState := pkg.DesiredState == ""
	pkg.EnsureDesiredState()
	registryRunning := pkg.Status == "running"

	// If registry says not running, trust it (no process to check)
	if !registryRunning {
		return false, legacyDesiredState
	}

	// Registry says running - verify the process actually exists
	if pkg.Runtime.PID == nil || pkg.Runtime.Port == nil {
		// A running node needs both fields. In particular, a live-looking PID with
		// no port cannot be health-checked and must not reach the already-running
		// projection, which formats the port value.
		fmt.Printf("Warning: Agent %s marked as running without complete runtime information, marking as stopped\n", name)
		pkg.Status = "stopped"
		pkg.Runtime.PID = nil
		pkg.Runtime.StartedAt = nil
		pkg.Runtime.BootID = ""
		pkg.Runtime.StartTime = ""
		return false, true
	}
	if !lifecycle && goos == "darwin" {
		if packages.RuntimePIDAlive(pkg.Runtime) {
			return true, false
		}
		fmt.Printf("Warning: Agent %s process (PID %d) is not alive, marking observed status as stopped\n", name, *pkg.Runtime.PID)
		clearObservedRuntime(pkg)
		return false, true
	}

	memoizedProbe := func(ctx context.Context, port int, path string) packages.HealthIdentity {
		identity, ok := memo[port]
		if !ok {
			identity = probe(ctx, port, path)
			memo[port] = identity
		}
		return identity
	}
	status := processStatus
	// Windows read reconciliation has historically resolved ownership entirely
	// through the health identity because process start-time discovery is not a
	// cheap dashboard operation there.
	if !lifecycle && goos == "windows" {
		status = func(packages.RuntimeInfo) packages.RuntimeProcessState {
			return packages.RuntimeProcessUnknown
		}
	}
	policy := packages.ReadConfirmationPolicy()
	selectedProbe := probe
	if !lifecycle {
		selectedProbe = memoizedProbe
	} else {
		policy = as.lifecycleConfirmationPolicy()
	}
	assessment := packages.AssessRecordedProcessWith(
		context.Background(), name, *pkg, status, selectedProbe, policy,
	)
	return reconcileRecordedAssessment(pkg, name, assessment)

}

func reconcileRecordedAssessment(pkg *packages.InstalledPackage, name string, assessment packages.RecordedProcessAssessment) (bool, bool) {
	if assessment.Ownership == packages.RecordedProcessOursHealthy ||
		assessment.Ownership == packages.RecordedProcessOursUnhealthy ||
		assessment.Ownership == packages.RecordedProcessUnknown {
		return true, false
	}

	fmt.Printf("Warning: Agent %s recorded process is dead or belongs to another node, marking observed status as stopped\n", name)
	clearObservedRuntime(pkg)
	return false, true
}

func clearObservedRuntime(pkg *packages.InstalledPackage) {
	pkg.Status = "stopped"
	pkg.Runtime.PID = nil
	pkg.Runtime.StartedAt = nil
	pkg.Runtime.BootID = ""
	pkg.Runtime.StartTime = ""
}

// ListRunningAgents returns a list of all running agents
func (as *DefaultAgentService) ListRunningAgents() ([]domain.RunningAgent, error) {
	registry, err := as.loadRegistryDirect()
	if err != nil {
		return nil, fmt.Errorf("failed to load registry: %w", err)
	}

	var runningAgents []domain.RunningAgent
	for _, pkg := range registry.Installed {
		if pkg.Status == "running" {
			runningAgents = append(runningAgents, as.convertToRunningAgent(pkg))
		}
	}

	return runningAgents, nil
}

// loadRegistryDirect loads the registry using direct file system access
// TODO: Eventually replace with registryStorage interface usage
func (as *DefaultAgentService) loadRegistryDirect() (*packages.InstallationRegistry, error) {
	registryPath := filepath.Join(as.agentfieldHome, "installed.yaml")
	return packages.LoadInstallationRegistry(registryPath)
}

// saveRegistryDirect saves the registry using direct file system access
// TODO: Eventually replace with registryStorage interface usage
func (as *DefaultAgentService) saveRegistryDirect(registry *packages.InstallationRegistry) error {
	registryPath := filepath.Join(as.agentfieldHome, "installed.yaml")
	return packages.WriteInstallationRegistry(registryPath, registry)
}

func (as *DefaultAgentService) updateRegistryEntry(name string, entry packages.InstalledPackage) error {
	return packages.UpdateInstallationRegistry(filepath.Join(as.agentfieldHome, "installed.yaml"), func(registry *packages.InstallationRegistry) error {
		registry.Installed[name] = entry
		return nil
	})
}

// convertToRunningAgent converts packages.InstalledPackage to domain.RunningAgent
func (as *DefaultAgentService) convertToRunningAgent(pkg packages.InstalledPackage) domain.RunningAgent {
	agent := domain.RunningAgent{
		Name:   pkg.Name,
		Status: pkg.Status,
	}

	if pkg.Runtime.Port != nil {
		agent.Port = *pkg.Runtime.Port
	}

	if pkg.Runtime.PID != nil {
		agent.PID = *pkg.Runtime.PID
	}

	if pkg.Runtime.StartedAt != nil {
		if startedAt, err := time.Parse(time.RFC3339, *pkg.Runtime.StartedAt); err == nil {
			agent.StartedAt = startedAt
		}
	}

	agent.LogFile = pkg.Runtime.LogFile

	return agent
}

// startNodeDependencies starts a node's installed, not-yet-running node
// dependencies before the node itself. inProgress guards against cycles.
func (as *DefaultAgentService) startNodeDependencies(node packages.InstalledPackage, inProgress map[string]bool, options domain.RunOptions) {
	metadata, err := packages.ParsePackageMetadata(node.Path)
	if err != nil {
		return
	}
	for _, ref := range metadata.Dependencies.Nodes {
		depName := packages.NodeDepName(ref)
		if depName == "" || inProgress[depName] {
			continue
		}
		registry, err := as.loadRegistryDirect()
		if err != nil {
			return
		}
		dep, _, exists := as.findAgentInRegistry(registry, depName)
		if !exists {
			fmt.Printf("⚠️  Node dependency %s is declared but not installed (run: af install %s)\n", depName, ref)
			continue
		}
		if running, _ := as.reconcileLifecycleProcessState(&dep, depName); running {
			continue
		}
		fmt.Printf("🔗 Starting node dependency: %s\n", depName)
		// Dependencies get an auto-assigned port, not the parent's --port.
		depOptions := options
		depOptions.Port = 0
		depOptions.PortIsPreference = false
		if _, err := as.runAgentGuarded(depName, depOptions, inProgress); err != nil {
			fmt.Printf("⚠️  Failed to start node dependency %s: %v\n", depName, err)
		}
	}
}

// buildProcessConfig creates a process configuration for starting an agent.
// It reads the manifest entrypoint and resolves declared environment variables
// from the encrypted secret store (prompting for missing required ones).
func (as *DefaultAgentService) buildProcessConfig(agentNode packages.InstalledPackage, port int) (interfaces.ProcessConfig, error) {
	// Read the manifest for the entrypoint, healthcheck and declared env. Fall
	// back to defaults (python main.py) if no manifest is present.
	metadata, err := packages.ParsePackageMetadata(agentNode.Path)
	if err != nil {
		fmt.Printf("⚠️  No usable manifest (%v); falling back to python main.py\n", err)
		metadata = &packages.PackageMetadata{}
	}

	// Prepare environment variables. Export both AGENTFIELD_SERVER (the var the
	// SDK reads) and the legacy AGENTFIELD_SERVER_URL.
	serverURL := resolveServerURL()
	env := os.Environ()
	env = append(env, fmt.Sprintf("PORT=%d", port))
	// Tell the SDK to bind exactly this port and fail fast if it is unavailable,
	// rather than silently moving to another port that the runner is not polling
	// (the readiness check below targets this exact port). Gated by this signal so
	// standalone `python -m <node>.app` keeps its lenient auto-port fallback.
	env = append(env, "AGENTFIELD_STRICT_PORT=1")
	env = append(env, fmt.Sprintf("AGENTFIELD_SERVER=%s", serverURL))
	env = append(env, fmt.Sprintf("AGENTFIELD_SERVER_URL=%s", serverURL))
	// A control plane with an API key configured rejects an unauthenticated
	// registration, so the node needs the same credential the CLI resolved
	// (flag, environment, or `af auth login`). Absent on a default local
	// setup, where the variable is simply not exported.
	if key := packages.ResolveAPIKey(); key != "" {
		env = append(env, fmt.Sprintf("AGENTFIELD_API_KEY=%s", key))
	}
	env = packages.PythonUTF8Env(env)

	// Resolve declared variables from the encrypted secret store. Secrets are
	// injected only into this child process — never written to disk in plaintext.
	resolvedEnv, err := as.resolveNodeEnvironment(agentNode.Name, metadata)
	if err != nil {
		return interfaces.ProcessConfig{}, err
	}
	for key, value := range resolvedEnv {
		env = append(env, fmt.Sprintf("%s=%s", key, value))
	}

	// Launch via the manifest entrypoint (e.g. "python -m pr_af.app"), resolving
	// the launcher for the node's language. A Go node launches its
	// install-time-built binary; a package-relative binary path is resolved
	// against the install dir so exec finds it regardless of cwd. Interpreter
	// resolution lives inside the non-Go branch so a Go node never probes for a
	// venv it does not have, nor warns about a Python it never runs.
	startArgs := metadata.StartCommand()
	command := startArgs[0]
	args := startArgs[1:]

	if metadata.IsGo() {
		command = packages.GoBinaryProgram(agentNode.Path, command)
	} else {
		// Determine Python path - use virtual environment if available
		var pythonPath string
		venvPath := filepath.Join(agentNode.Path, "venv")

		// Check if virtual environment exists (Unix/Linux/macOS)
		if _, err := os.Stat(filepath.Join(venvPath, "bin", "python")); err == nil {
			pythonPath = filepath.Join(venvPath, "bin", "python")
			fmt.Printf("🐍 Using virtual environment: %s\n", venvPath)

			// Complete virtual environment activation for Unix/Linux/macOS
			venvBinPath := filepath.Join(venvPath, "bin")

			// Set VIRTUAL_ENV first (required for proper activation)
			env = append(env, fmt.Sprintf("VIRTUAL_ENV=%s", venvPath))

			// Prepend virtual environment bin to PATH (critical for package resolution)
			currentPath := os.Getenv("PATH")
			env = append(env, fmt.Sprintf("PATH=%s:%s", venvBinPath, currentPath))

			// Unset PYTHONHOME to avoid conflicts with virtual environment
			env = append(env, "PYTHONHOME=")

			// Set PYTHONPATH to ensure proper module resolution
			env = append(env, fmt.Sprintf("PYTHONPATH=%s", filepath.Join(venvPath, "lib")))

			fmt.Printf("✅ Virtual environment fully activated with PATH=%s\n", venvBinPath)

		} else if _, err := os.Stat(filepath.Join(venvPath, "Scripts", "python.exe")); err == nil {
			pythonPath = filepath.Join(venvPath, "Scripts", "python.exe") // Windows
			fmt.Printf("🐍 Using virtual environment: %s\n", venvPath)

			// Complete virtual environment activation for Windows
			venvScriptsPath := filepath.Join(venvPath, "Scripts")

			// Set VIRTUAL_ENV first (required for proper activation)
			env = append(env, fmt.Sprintf("VIRTUAL_ENV=%s", venvPath))

			// Prepend virtual environment Scripts to PATH (critical for package resolution)
			currentPath := os.Getenv("PATH")
			env = append(env, fmt.Sprintf("PATH=%s;%s", venvScriptsPath, currentPath))

			// Unset PYTHONHOME to avoid conflicts with virtual environment
			env = append(env, "PYTHONHOME=")

			// Set PYTHONPATH to ensure proper module resolution
			env = append(env, fmt.Sprintf("PYTHONPATH=%s", filepath.Join(venvPath, "Lib", "site-packages")))

			fmt.Printf("✅ Virtual environment fully activated with PATH=%s\n", venvScriptsPath)

		} else {
			// Try to find python3 or python
			if pythonPath = as.findPythonExecutable(); pythonPath == "" {
				pythonPath = "python" // Final fallback
			}
			fmt.Printf("⚠️  Virtual environment not found at %s, using system Python: %s\n", venvPath, pythonPath)
		}

		if command == "python" || command == "python3" {
			command = pythonPath
		}
	}

	return interfaces.ProcessConfig{
		Command: command,
		Args:    args,
		Env:     env,
		WorkDir: agentNode.Path,
		LogFile: agentNode.Runtime.LogFile,
	}, nil
}

// resolveNodeEnvironment resolves a node's declared variables via the encrypted
// secret store, prompting for missing required ones.
func (as *DefaultAgentService) resolveNodeEnvironment(nodeName string, metadata *packages.PackageMetadata) (map[string]string, error) {
	env := metadata.UserEnvironment
	if len(env.Required) == 0 && len(env.Optional) == 0 && len(env.RequireOneOf) == 0 {
		return map[string]string{}, nil
	}
	store, err := packages.NewSecretStore(as.agentfieldHome)
	if err != nil {
		return nil, fmt.Errorf("failed to open secret store: %w", err)
	}
	resolver := &packages.EnvResolver{Store: store, NodeName: nodeName, Prompter: packages.TTYPrompter{}}
	return resolver.Resolve(env)
}

// nodeReadyTimeout is how long to wait for a freshly started node to answer its
// health check. Import-heavy nodes (large dependency graphs) routinely take more
// than the old hardcoded 10s to boot, which produced spurious "did not become
// ready" failures on nodes that were actually starting fine. Default 30s,
// overridable via AGENTFIELD_NODE_READY_TIMEOUT (whole seconds).
func nodeReadyTimeout() time.Duration {
	if v := os.Getenv("AGENTFIELD_NODE_READY_TIMEOUT"); v != "" {
		if secs, err := strconv.Atoi(v); err == nil && secs > 0 {
			return time.Duration(secs) * time.Second
		}
	}
	return 30 * time.Second
}

// waitForAgentNode waits for the agent node to become ready. A 200 on the
// health endpoint is only trusted when the payload's node_id (if it carries
// one) matches the node just started — on Windows the port probe can miss an
// existing listener (no SO_EXCLUSIVEADDRUSE), and without this check a
// squatter's health response makes a dead agent look started. An empty
// expectedNodeID or a payload without node_id skips the identity check.
func (as *DefaultAgentService) waitForAgentNode(port int, healthPath, expectedNodeID string, timeout time.Duration) error {
	if healthPath == "" {
		healthPath = "/health"
	}
	client := packages.NewNodeHTTPClient(1 * time.Second)
	deadline := time.Now().Add(timeout)

	impostor := ""
	for time.Now().Before(deadline) {
		request, requestErr := http.NewRequest(http.MethodGet, fmt.Sprintf("http://localhost:%d%s", port, healthPath), nil)
		var resp *http.Response
		err := requestErr
		if requestErr == nil {
			resp, err = client.Do(request)
		}
		if err == nil && resp.StatusCode == 200 {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
			resp.Body.Close()
			got := packages.HealthNodeID(body)
			if got == "" || expectedNodeID == "" || packages.NodeIDsEquivalent(got, expectedNodeID) {
				return nil
			}
			impostor = got
		} else if resp != nil {
			resp.Body.Close()
		}
		time.Sleep(500 * time.Millisecond)
	}

	if impostor != "" {
		return fmt.Errorf("port %d is answering health checks as %q, not %q — another process is using the port", port, impostor, expectedNodeID)
	}
	return fmt.Errorf("agent node did not become ready within %v", timeout)
}

// updateRuntimeInfo updates the registry with runtime information
func (as *DefaultAgentService) updateRuntimeInfo(agentNodeName string, port, pid int) error {
	registryPath := filepath.Join(as.agentfieldHome, "installed.yaml")
	// Resolve the process identity before taking the registry lock: on macOS
	// and Windows it shells out (ps / PowerShell) and must not stall every
	// registry read in the process for its bounded duration.
	bootID := packages.CurrentBootID()
	startTime := packages.CurrentProcessStartTime(pid)
	return packages.UpdateInstallationRegistry(registryPath, func(registry *packages.InstallationRegistry) error {
		if agentNode, exists := registry.Installed[agentNodeName]; exists {
			startedAt := time.Now().Format(time.RFC3339)
			agentNode.Status = "running"
			// Establish intent for legacy entries, but never overwrite an
			// explicit stop issued while this start was waiting for readiness.
			if agentNode.DesiredState == "" {
				agentNode.DesiredState = packages.DesiredStateRunning
			}
			agentNode.Runtime.Port = &port
			agentNode.Runtime.PID = &pid
			agentNode.Runtime.StartedAt = &startedAt
			agentNode.Runtime.BootID = bootID
			agentNode.Runtime.StartTime = startTime
			registry.Installed[agentNodeName] = agentNode
		}
		return nil
	})
}

// displayCapabilities fetches and displays agent node capabilities
func (as *DefaultAgentService) displayCapabilities(_ packages.InstalledPackage, port int) error {
	return packages.DisplayCapabilities(port)
}

// findAgentInRegistry finds an agent in the registry by name, handling name normalization
// Returns the agent package, actual name, and whether it was found
func (as *DefaultAgentService) findAgentInRegistry(registry *packages.InstallationRegistry, name string) (packages.InstalledPackage, string, bool) {
	// Try exact match first
	if agentNode, exists := registry.Installed[name]; exists {
		return agentNode, name, true
	}

	// Try with hyphens converted to no hyphens (deepresearchagent -> deep-research-agent)
	for registryName, agentNode := range registry.Installed {
		normalizedRegistryName := strings.ReplaceAll(registryName, "-", "")
		normalizedInputName := strings.ReplaceAll(name, "-", "")

		if normalizedRegistryName == normalizedInputName {
			return agentNode, registryName, true
		}
	}

	// Not found
	return packages.InstalledPackage{}, "", false
}

// findPythonExecutable tries to find a suitable Python executable
func (as *DefaultAgentService) findPythonExecutable() string {
	// Try common Python executable names in order of preference
	candidates := []string{"python3", "python", "python3.11", "python3.10", "python3.9", "python3.8"}

	for _, candidate := range candidates {
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}

		// Also try to find in PATH
		if path, err := exec.LookPath(candidate); err == nil {
			return path
		}
	}

	return "" // Not found
}
