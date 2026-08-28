package packages

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

const (
	shutdownRequestTimeout = 10 * time.Second
	gracefulExitWait       = 3 * time.Second
	processProbeAttempts   = 3
	processProbeInterval   = 3 * time.Second
)

// StopProcessResult describes the externally visible stop path. For an
// identity-unknown PID, Owned is false when the recorded endpoint was silent
// or belonged to another node, in which case the PID was never signalled.
type StopProcessResult struct {
	Owned           bool
	HTTPAttempted   bool
	HTTPAccepted    bool
	HTTPTimedOut    bool
	InterruptSent   bool
	ForceKillNeeded bool
}

// RecordedProcessOwnership is the result of reconciling the recorded process
// identity with the node's health endpoint.
type RecordedProcessOwnership uint8

const (
	RecordedProcessDead RecordedProcessOwnership = iota
	RecordedProcessOursHealthy
	RecordedProcessOursUnhealthy
	RecordedProcessForeign
	RecordedProcessUnknown
)

// RecordedProcessAssessment centralizes the ownership rule used by every
// lifecycle caller. SignalAllowed is false when a healthy endpoint appears to
// be ours but a previously recorded, known identity has already been disproved.
type RecordedProcessAssessment struct {
	Ownership     RecordedProcessOwnership
	ProcessState  RuntimeProcessState
	Health        HealthIdentity
	SignalAllowed bool
}

func (a RecordedProcessAssessment) Owned() bool {
	return a.Ownership == RecordedProcessOursHealthy || a.Ownership == RecordedProcessOursUnhealthy
}

// AssessRecordedProcess applies the shared PID/health ownership rule using
// production probes.
func AssessRecordedProcess(ctx context.Context, name string, entry InstalledPackage) RecordedProcessAssessment {
	return AssessRecordedProcessWith(ctx, name, entry, RuntimeProcessStatus, ProbeHealthIdentity, LifecycleConfirmationPolicy())
}

// ProcessConfirmationPolicy controls whether a silent ownership probe is
// confirmed. Reads use one non-blocking attempt; lifecycle calls retry an
// existing PID before they may stop, restart, or discard it.
type ProcessConfirmationPolicy struct {
	Attempts      int
	Interval      time.Duration
	Sleep         func(context.Context, time.Duration) error
	ProcessExists func(int) bool
}

func LifecycleConfirmationPolicy() ProcessConfirmationPolicy {
	return ProcessConfirmationPolicy{
		Attempts: processProbeAttempts, Interval: processProbeInterval,
		Sleep: sleepWithContext, ProcessExists: processExists,
	}
}

func ReadConfirmationPolicy() ProcessConfirmationPolicy {
	return ProcessConfirmationPolicy{Attempts: 1, ProcessExists: processExists}
}

func sleepWithContext(ctx context.Context, duration time.Duration) error {
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// AssessRecordedProcessWith is the injectable form used by status and
// maintenance callers that memoize or fake their platform probes.
func AssessRecordedProcessWith(
	ctx context.Context,
	name string,
	entry InstalledPackage,
	processStatus func(RuntimeInfo) RuntimeProcessState,
	healthProbe func(context.Context, int, string) HealthIdentity,
	confirmation ProcessConfirmationPolicy,
) RecordedProcessAssessment {
	state := processStatus(entry.Runtime)
	assessment := RecordedProcessAssessment{Ownership: RecordedProcessDead, ProcessState: state}
	attempts := confirmation.Attempts
	if attempts < 1 {
		attempts = 1
	}
	exists := confirmation.ProcessExists
	if exists == nil {
		exists = processExists
	}
	pidExists := entry.Runtime.PID != nil && *entry.Runtime.PID > 0 &&
		(state == RuntimeProcessAliveState || exists(*entry.Runtime.PID))
	if state == RuntimeProcessDead || !pidExists {
		attempts = 1
	}
	for attempt := 0; attempt < attempts; attempt++ {
		if attempt > 0 {
			sleep := confirmation.Sleep
			if sleep == nil {
				sleep = sleepWithContext
			}
			if err := sleep(ctx, confirmation.Interval); err != nil {
				break
			}
		}
		if entry.Runtime.Port != nil && *entry.Runtime.Port > 0 {
			assessment.Health = healthProbe(ctx, *entry.Runtime.Port, PackageHealthcheckPath(entry))
		}
		if assessment.Health.Healthy {
			break
		}
	}

	if assessment.Health.Healthy {
		expectedNodeID := PackageNodeID(entry, name)
		if assessment.Health.NodeID != "" && !NodeIDsEquivalent(assessment.Health.NodeID, expectedNodeID) {
			assessment.Ownership = RecordedProcessForeign
			return assessment
		}
		assessment.Ownership = RecordedProcessOursHealthy
		assessment.SignalAllowed = state != RuntimeProcessDead
		return assessment
	}
	if state == RuntimeProcessAliveState {
		assessment.Ownership = RecordedProcessOursUnhealthy
		assessment.SignalAllowed = true
	} else if state == RuntimeProcessUnknown && pidExists {
		assessment.Ownership = RecordedProcessUnknown
	}
	return assessment
}

// ReinstallState captures whether a forced reinstall must bring the package
// back up and which recorded port it should prefer.
type ReinstallState struct {
	WasRunning    bool
	PreferredPort int
}

// PackageReinstallState reads the restart intent without mutating the registry.
func PackageReinstallState(home, name string) (ReinstallState, error) {
	data, err := os.ReadFile(filepath.Join(home, "installed.yaml"))
	if err != nil {
		return ReinstallState{}, err
	}
	var registry InstallationRegistry
	if err := yaml.Unmarshal(data, &registry); err != nil {
		return ReinstallState{}, err
	}
	entry, ok := registry.Installed[name]
	if !ok || entry.Status != "running" {
		return ReinstallState{}, nil
	}
	state := ReinstallState{WasRunning: true}
	if entry.Runtime.Port != nil {
		state.PreferredPort = *entry.Runtime.Port
	}
	return state, nil
}

// PackageHealthcheckPath resolves the installed manifest's health endpoint,
// falling back to /health for legacy or unreadable packages.
func PackageHealthcheckPath(entry InstalledPackage) string {
	if metadata, err := ParsePackageMetadata(entry.Path); err == nil {
		return metadata.HealthcheckPath()
	}
	return "/health"
}

// PackageNodeID resolves the runtime node ID advertised by the manifest.
func PackageNodeID(entry InstalledPackage, fallback string) string {
	if metadata, err := ParsePackageMetadata(entry.Path); err == nil && strings.TrimSpace(metadata.AgentNode.NodeID) != "" {
		return metadata.AgentNode.NodeID
	}
	return fallback
}

// StopRecordedProcess stops a package process only after the shared ownership
// assessment permits it. Legacy/unsupported identities accept either an
// equivalent or anonymous healthy response as ownership evidence; a foreign
// or silent endpoint is reconciled without signalling the recorded PID.
func StopRecordedProcess(ctx context.Context, name string, entry InstalledPackage) (StopProcessResult, error) {
	result := StopProcessResult{}
	if entry.Runtime.PID == nil || *entry.Runtime.PID <= 0 {
		return result, nil
	}

	assessment := AssessRecordedProcess(ctx, name, entry)
	return StopRecordedProcessWithAssessment(ctx, name, entry, assessment)
}

// StopRecordedProcessWithAssessment performs the stop using an assessment the
// caller already confirmed, avoiding a second multi-probe confirmation window.
func StopRecordedProcessWithAssessment(ctx context.Context, name string, entry InstalledPackage, assessment RecordedProcessAssessment) (StopProcessResult, error) {
	result := StopProcessResult{}
	if entry.Runtime.PID == nil || *entry.Runtime.PID <= 0 {
		return result, nil
	}
	if !assessment.Owned() {
		return result, nil
	}
	result.Owned = true

	httpSupported := !packageUsesGo(entry)
	if httpSupported && entry.Runtime.Port != nil {
		shutdownBudget := agentShutdownBudget(entry)
		result.HTTPAttempted = true
		result.HTTPAccepted, result.HTTPTimedOut = requestHTTPShutdown(ctx, *entry.Runtime.Port, shutdownBudget)
		if result.HTTPAccepted {
			if waitForProcessExit(*entry.Runtime.PID, shutdownBudget) {
				return result, nil
			}
		}
	}

	if !assessment.SignalAllowed {
		// The port still answers as this node but the recorded PID is not the
		// process serving it (restarted by hand, a supervisor, …) and there is
		// no shutdown endpoint to fall back on. Say so instead of reporting a
		// stop that did not happen — the caller keeps the record.
		if assessment.Ownership == RecordedProcessOursHealthy && !result.HTTPAccepted {
			port := 0
			if entry.Runtime.Port != nil {
				port = *entry.Runtime.Port
			}
			return result, fmt.Errorf("agent %s still answers on port %d but its process is not the recorded PID %d; stop it manually", name, port, *entry.Runtime.PID)
		}
		return result, nil
	}
	interruptSent, forceKillNeeded, err := stopProcessWith(
		*entry.Runtime.PID,
		gracefulSignal,
		processExists,
		forceKill,
	)
	result.InterruptSent = interruptSent
	result.ForceKillNeeded = forceKillNeeded
	if err != nil {
		return result, err
	}
	return result, nil
}

const defaultAgentShutdownBudget = 30 * time.Second

func agentShutdownBudget(entry InstalledPackage) time.Duration {
	return agentShutdownBudgetWith(entry, processEnvironment)
}

func agentShutdownBudgetWith(entry InstalledPackage, environment func(int) []string) time.Duration {
	if entry.Runtime.PID == nil || *entry.Runtime.PID <= 0 {
		return defaultAgentShutdownBudget
	}
	for _, item := range environment(*entry.Runtime.PID) {
		key, value, found := strings.Cut(item, "=")
		if found && key == "AGENTFIELD_SHUTDOWN_TIMEOUT" {
			if seconds, err := time.ParseDuration(value); err == nil && seconds >= 0 {
				return seconds
			}
			if seconds, err := time.ParseDuration(value + "s"); err == nil && seconds >= 0 {
				return seconds
			}
			return defaultAgentShutdownBudget
		}
	}
	return defaultAgentShutdownBudget
}

func processEnvironment(pid int) []string {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/environ", pid))
	if err != nil {
		return nil
	}
	return strings.Split(string(data), "\x00")
}

func stopProcessWith(
	pid int,
	graceful func(int) error,
	exists func(int) bool,
	force func(int) error,
) (bool, bool, error) {
	if err := graceful(pid); err != nil {
		if isFinishedProcessError(err) || !exists(pid) {
			return false, false, nil
		}
		if killErr := force(pid); killErr != nil && !isFinishedProcessError(killErr) && exists(pid) {
			return false, true, fmt.Errorf("failed to kill process: %w", killErr)
		}
		return false, true, nil
	}

	if waitForProcessExitWith(pid, gracefulExitWait, exists) {
		return true, false, nil
	}
	if err := force(pid); err != nil && !isFinishedProcessError(err) && exists(pid) {
		return true, true, fmt.Errorf("failed to force kill process: %w", err)
	}
	return true, true, nil
}

func packageUsesGo(entry InstalledPackage) bool {
	metadata, err := ParsePackageMetadata(entry.Path)
	return err == nil && metadata.IsGo()
}

func requestHTTPShutdown(ctx context.Context, port int, shutdownBudget time.Duration) (accepted bool, timedOut bool) {
	requestBody, err := json.Marshal(map[string]interface{}{
		"graceful":        true,
		"timeout_seconds": shutdownBudget.Seconds(),
	})
	if err != nil {
		return false, false
	}
	requestCtx, cancel := context.WithTimeout(ctx, shutdownRequestTimeout)
	defer cancel()
	request, err := http.NewRequestWithContext(requestCtx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/shutdown", port), bytes.NewReader(requestBody))
	if err != nil {
		return false, false
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("User-Agent", "AgentField-CLI/1.0")
	response, err := NewNodeHTTPClient(shutdownRequestTimeout).Do(request)
	if err != nil {
		var netErr net.Error
		return false, errors.Is(err, context.DeadlineExceeded) || errors.As(err, &netErr) && netErr.Timeout()
	}
	defer response.Body.Close()
	return response.StatusCode >= http.StatusOK && response.StatusCode < http.StatusMultipleChoices, false
}

func waitForProcessExit(pid int, timeout time.Duration) bool {
	return waitForProcessExitWith(pid, timeout, processExists)
}

func waitForProcessExitWith(pid int, timeout time.Duration, exists func(int) bool) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if !exists(pid) {
			return true
		}
		time.Sleep(50 * time.Millisecond)
	}
	return !exists(pid)
}

func isFinishedProcessError(err error) bool {
	if errors.Is(err, os.ErrProcessDone) {
		return true
	}
	message := strings.ToLower(err.Error())
	return strings.Contains(message, "process already finished") || strings.Contains(message, "no such process")
}

// StopPackageForReinstall stops and reconciles a running registry entry while
// retaining desired running intent and its preferred port for the replacement.
func StopPackageForReinstall(ctx context.Context, home, name string) (ReinstallState, error) {
	return stopPackageForReinstallWith(ctx, home, name, func(ctx context.Context, name string, entry InstalledPackage) (StopProcessResult, error) {
		assessment := AssessRecordedProcess(ctx, name, entry)
		if assessment.Ownership == RecordedProcessUnknown {
			return StopProcessResult{}, fmt.Errorf("could not verify that process %d is %s; stop it manually", *entry.Runtime.PID, name)
		}
		return StopRecordedProcessWithAssessment(ctx, name, entry, assessment)
	})
}

func stopPackageForReinstallWith(
	ctx context.Context,
	home, name string,
	stop func(context.Context, string, InstalledPackage) (StopProcessResult, error),
) (ReinstallState, error) {
	registryPath := filepath.Join(home, "installed.yaml")
	data, err := os.ReadFile(registryPath)
	if err != nil {
		return ReinstallState{}, err
	}
	var registry InstallationRegistry
	if err := yaml.Unmarshal(data, &registry); err != nil {
		return ReinstallState{}, err
	}
	entry, ok := registry.Installed[name]
	if !ok || entry.Status != "running" {
		return ReinstallState{}, nil
	}
	state := ReinstallState{WasRunning: true}
	if entry.Runtime.Port != nil {
		state.PreferredPort = *entry.Runtime.Port
	}
	entry.EnsureDesiredState()
	if _, err := stop(ctx, name, entry); err != nil {
		return state, err
	}
	entry.Status = "stopped"
	entry.Runtime.PID = nil
	entry.Runtime.StartedAt = nil
	entry.Runtime.BootID = ""
	entry.Runtime.StartTime = ""
	registry.Installed[name] = entry
	return state, UpdateInstallationRegistry(registryPath, func(latest *InstallationRegistry) error {
		latest.Installed[name] = entry
		return nil
	})
}

// RestartPackageAfterReinstall restores a package that was running before its
// replacement landed, preferring the previous port only while it is free.
func RestartPackageAfterReinstall(home, name string, state ReinstallState) error {
	if state.WasRunning {
		if err := os.MkdirAll(filepath.Join(home, "logs"), 0o755); err != nil {
			return err
		}
	}
	return restartPackageAfterReinstall(state, PortAvailable, func(port int) error {
		return (&AgentNodeRunner{AgentFieldHome: home, Port: port, Detach: true}).RunAgentNode(name)
	})
}

func restartPackageAfterReinstall(state ReinstallState, portAvailable func(int) bool, run func(int) error) error {
	if !state.WasRunning {
		return nil
	}
	port := state.PreferredPort
	if port > 0 && !portAvailable(port) {
		port = 0
	}
	return run(port)
}

// PortAvailable reports whether a local TCP port can be reserved.
func PortAvailable(port int) bool {
	listener, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		return false
	}
	_ = listener.Close()
	return true
}
