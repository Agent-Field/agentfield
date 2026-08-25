package packagemaint

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages/updatecheck"
	"github.com/Agent-Field/agentfield/control-plane/internal/services/packagejobs"
	"gopkg.in/yaml.v3"
)

const (
	defaultInterval = 6 * time.Hour
	minimumInterval = 15 * time.Minute
	bootDelay       = 20 * time.Second
	bootSettleDelay = 2 * time.Second
	busyRetryDelay  = 10 * time.Minute
	jobPollInterval = 250 * time.Millisecond
	jobWaitTimeout  = 30 * time.Minute
	restoreTimeout  = 90 * time.Second
	maxBusyChecks   = 3
)

var restoreRetryDelays = []time.Duration{time.Minute, 5 * time.Minute, 15 * time.Minute}

type JobManager interface {
	StartUpdate(name, source string) (*packagejobs.Job, error)
	GetJob(id string) (*packagejobs.Job, bool)
}

type ActiveJobChecker interface {
	ActiveFor(name string) bool
}

type AgentRunner interface {
	RunAgent(name string, options domain.RunOptions) (*domain.RunningAgent, error)
}

type ActiveExecutionChecker interface {
	HasActiveExecutions(ctx context.Context, agentNodeID string) (bool, error)
}

type Skip struct {
	Name   string `json:"name"`
	Reason string `json:"reason"`
}

type Summary struct {
	StartedAt  time.Time `json:"started_at"`
	FinishedAt time.Time `json:"finished_at"`
	Checked    int       `json:"checked"`
	Updated    []string  `json:"updated"`
	Restored   []string  `json:"restored"`
	Skipped    []Skip    `json:"skipped"`
	Errors     []string  `json:"errors"`
	retrySoon  bool
}

type Status struct {
	Enabled           bool      `json:"enabled"`
	Reason            string    `json:"reason"`
	Interval          string    `json:"interval"`
	LastRun           *Summary  `json:"last_run"`
	NextRunAt         time.Time `json:"next_run_at"`
	BootPassCompleted bool      `json:"boot_pass_completed"`
	Hosting           string    `json:"hosting"`
}

type Config struct {
	AgentFieldHome    string
	Checker           *updatecheck.Checker
	Jobs              JobManager
	Agent             AgentRunner
	Executions        ActiveExecutionChecker
	Enabled           func() (bool, string)
	ProcessAlive      func(packages.RuntimeInfo) bool
	HealthProbe       func(context.Context, int, string) packages.HealthIdentity
	PortAvailable     func(int) bool
	Sleep             func(context.Context, time.Duration) error
	Now               func() time.Time
	Interval          time.Duration
	JobWaitTimeout    time.Duration
	RestoreTimeout    time.Duration
	Ready             <-chan struct{}
	OnRestoreState    func(packageName string, active bool)
	HostedInContainer func() bool
	Hosting           func() string
	OnRegistryChange  func()
}

type Service struct {
	home              string
	checker           *updatecheck.Checker
	jobs              JobManager
	agent             AgentRunner
	executions        ActiveExecutionChecker
	enabled           func() (bool, string)
	processAlive      func(packages.RuntimeInfo) bool
	healthProbe       func(context.Context, int, string) packages.HealthIdentity
	portAvailable     func(int) bool
	sleep             func(context.Context, time.Duration) error
	now               func() time.Time
	interval          time.Duration
	jobWaitTimeout    time.Duration
	restoreTimeout    time.Duration
	ready             <-chan struct{}
	onRestoreState    func(string, bool)
	hostedInContainer func() bool
	hosting           func() string
	onRegistryChange  func()

	mu                sync.RWMutex
	running           bool
	stopping          bool
	lastRun           *Summary
	nextRunAt         time.Time
	bootPassCompleted bool
	backoffStep       int
	restoreGrace      map[string]bool
	scheduleChanged   chan struct{}
	registryMu        sync.Mutex
	lifecycleMu       sync.RWMutex
	lifecycleCtx      context.Context
	idle              chan struct{}
}

var (
	ErrPassAlreadyRunning = errors.New("a maintenance pass is already running")
	ErrShuttingDown       = errors.New("package maintenance is shutting down")
)

func New(cfg Config) *Service {
	interval := cfg.Interval
	if interval <= 0 {
		interval = configuredInterval()
	}
	if cfg.Checker == nil {
		cfg.Checker = updatecheck.NewChecker(nil)
	}
	if cfg.Enabled == nil {
		cfg.Enabled = updatesEnabled
	}
	if cfg.ProcessAlive == nil {
		cfg.ProcessAlive = packages.RuntimeProcessAlive
	}
	if cfg.HealthProbe == nil {
		cfg.HealthProbe = packages.ProbeHealthIdentity
	}
	if cfg.PortAvailable == nil {
		cfg.PortAvailable = isPortAvailable
	}
	if cfg.Sleep == nil {
		cfg.Sleep = sleepContext
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	if cfg.JobWaitTimeout <= 0 {
		cfg.JobWaitTimeout = jobWaitTimeout
	}
	if cfg.RestoreTimeout <= 0 {
		cfg.RestoreTimeout = restoreTimeout
	}
	if cfg.HostedInContainer == nil {
		cfg.HostedInContainer = packages.HostedInContainer
	}
	if cfg.Hosting == nil {
		cfg.Hosting = packages.HostingPlatform
	}
	idle := make(chan struct{})
	close(idle)
	return &Service{
		home: cfg.AgentFieldHome, checker: cfg.Checker, jobs: cfg.Jobs,
		agent: cfg.Agent, executions: cfg.Executions, enabled: cfg.Enabled,
		processAlive: cfg.ProcessAlive, healthProbe: cfg.HealthProbe, portAvailable: cfg.PortAvailable, sleep: cfg.Sleep, now: cfg.Now,
		interval: interval, jobWaitTimeout: cfg.JobWaitTimeout, restoreTimeout: cfg.RestoreTimeout,
		ready: cfg.Ready, onRestoreState: cfg.OnRestoreState, hostedInContainer: cfg.HostedInContainer, hosting: cfg.Hosting,
		onRegistryChange: cfg.OnRegistryChange,
		nextRunAt:        cfg.Now().UTC().Add(bootDelay),
		restoreGrace:     make(map[string]bool),
		scheduleChanged:  make(chan struct{}, 1),
		lifecycleCtx:     context.Background(),
		idle:             idle,
	}
}

func configuredInterval() time.Duration {
	raw := strings.TrimSpace(os.Getenv("AGENTFIELD_PACKAGE_UPDATE_INTERVAL"))
	if raw == "" {
		return defaultInterval
	}
	interval, err := time.ParseDuration(raw)
	if err != nil {
		return defaultInterval
	}
	if interval < minimumInterval {
		return minimumInterval
	}
	return interval
}

func updatesEnabled() (bool, string) {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("AGENTFIELD_PACKAGE_AUTO_UPDATE"))) {
	case "0", "false", "off":
		return false, "disabled by AGENTFIELD_PACKAGE_AUTO_UPDATE"
	}
	if _, err := exec.LookPath("git"); err != nil {
		return false, "git is not available on PATH"
	}
	return true, ""
}

func sleepContext(ctx context.Context, duration time.Duration) error {
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func (s *Service) Checker() *updatecheck.Checker { return s.checker }

// SetLifecycleContext supplies the server-owned lifetime used by asynchronous
// run-now passes. Request cancellation must not turn a completed check into a
// cached error, while server shutdown must cancel all maintenance work.
func (s *Service) SetLifecycleContext(ctx context.Context) {
	if ctx == nil {
		ctx = context.Background()
	}
	s.lifecycleMu.Lock()
	s.lifecycleCtx = ctx
	s.lifecycleMu.Unlock()
	s.mu.Lock()
	s.stopping = false
	s.mu.Unlock()
}

func (s *Service) lifecycleContext() context.Context {
	s.lifecycleMu.RLock()
	defer s.lifecycleMu.RUnlock()
	return s.lifecycleCtx
}

// Run owns the readiness-gated startup pass and adaptive maintenance cadence.
func (s *Service) Run(ctx context.Context) {
	s.SetLifecycleContext(ctx)
	s.armBootRestoreGrace()
	ready, err := s.waitForReady(ctx)
	if err != nil {
		s.clearAllRestoreGrace()
		return
	}
	if ready {
		if err := s.sleep(ctx, bootSettleDelay); err != nil {
			s.clearAllRestoreGrace()
			return
		}
	}
	s.RunPass(ctx)
	s.mu.Lock()
	s.bootPassCompleted = true
	s.mu.Unlock()
	s.clearAllRestoreGrace()

	for {
		s.mu.RLock()
		delay := s.nextRunAt.Sub(s.now().UTC())
		s.mu.RUnlock()
		if delay < 0 {
			delay = 0
		}
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-s.scheduleChanged:
			timer.Stop()
			continue
		case <-timer.C:
			s.RunPass(ctx)
		}
	}
}

// waitForReady returns true when readiness won and false when the 20-second
// compatibility fallback elapsed. The fallback already includes the settle
// allowance promised by C14, so only the readiness path sleeps another 2s.
func (s *Service) waitForReady(ctx context.Context) (bool, error) {
	if s.ready == nil {
		return false, s.sleep(ctx, bootDelay)
	}
	select {
	case <-s.ready:
		return true, nil
	default:
	}
	waitCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	fallback := make(chan error, 1)
	go func() { fallback <- s.sleep(waitCtx, bootDelay) }()
	select {
	case <-ctx.Done():
		return false, ctx.Err()
	case <-s.ready:
		return true, nil
	case err := <-fallback:
		return false, err
	}
}

// StartPass starts a pass asynchronously and distinguishes a concurrent pass
// from server shutdown so the HTTP adapter can return the correct contract.
func (s *Service) StartPass() error {
	ctx := s.lifecycleContext()
	if ctx.Err() != nil {
		return ErrShuttingDown
	}
	if err := s.begin(); err != nil {
		return err
	}
	go func() {
		defer s.end()
		s.runPass(ctx)
	}()
	return nil
}

func (s *Service) RunPass(ctx context.Context) Summary {
	if err := s.begin(); err != nil {
		s.mu.RLock()
		defer s.mu.RUnlock()
		if s.lastRun != nil {
			return *s.lastRun
		}
		return Summary{}
	}
	defer s.end()
	return s.runPass(ctx)
}

func (s *Service) begin() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopping {
		return ErrShuttingDown
	}
	if s.running {
		return ErrPassAlreadyRunning
	}
	s.running = true
	s.idle = make(chan struct{})
	return nil
}

// Stop prevents new passes before waiting, so a run-now request cannot start
// while server shutdown is waiting on the current pass's idle channel.
func (s *Service) Stop(ctx context.Context) error {
	s.mu.Lock()
	s.stopping = true
	s.mu.Unlock()
	return s.WaitForIdle(ctx)
}

func (s *Service) end() {
	s.mu.Lock()
	s.running = false
	close(s.idle)
	s.mu.Unlock()
}

// WaitForIdle waits on the channel owned by the current pass. A timed-out
// waiter leaves no goroutine behind, and each later pass gets a fresh channel.
func (s *Service) WaitForIdle(ctx context.Context) error {
	s.mu.RLock()
	idle := s.idle
	s.mu.RUnlock()
	select {
	case <-idle:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *Service) runPass(ctx context.Context) (summary Summary) {
	summary = Summary{StartedAt: s.now().UTC(), Updated: []string{}, Restored: []string{}, Skipped: []Skip{}, Errors: []string{}}
	defer func() {
		if recovered := recover(); recovered != nil {
			message := fmt.Sprintf("maintenance pass panic: %v", recovered)
			logger.Logger.Error().Interface("panic", recovered).Msg("package maintenance pass recovered from panic")
			summary.Errors = append(summary.Errors, message)
			summary = s.finish(summary)
		}
	}()
	registry, err := s.loadRegistryForPass()
	if err != nil {
		summary.Errors = append(summary.Errors, err.Error())
		summary.retrySoon = true
		return s.finish(summary)
	}
	s.restore(ctx, registry, &summary)

	enabled, reason := s.enabled()
	if enabled {
		entries := checkEntries(registry)
		results := s.checker.Check(ctx, entries)
		summary.Checked = len(results)
		for _, result := range results {
			entry := registry.Installed[result.ID]
			if result.Update.Status == updatecheck.StatusFailed {
				summary.Skipped = append(summary.Skipped, Skip{Name: result.ID, Reason: "failed"})
				continue
			}
			if result.Update.Status == updatecheck.StatusPinned {
				summary.Skipped = append(summary.Skipped, Skip{Name: result.ID, Reason: "pinned"})
				continue
			}
			if !entry.AutoUpdateEnabled() {
				summary.Skipped = append(summary.Skipped, Skip{Name: result.ID, Reason: "auto_update_disabled"})
				continue
			}
			if result.Update.Status == updatecheck.StatusError {
				summary.Errors = append(summary.Errors, fmt.Sprintf("%s: %s", result.ID, result.Update.Message))
				continue
			}
			if result.Update.Status != updatecheck.StatusAvailable {
				continue
			}
			ready, busyErr := s.waitUntilIdle(ctx, result.ID)
			if busyErr != nil {
				summary.Errors = append(summary.Errors, fmt.Sprintf("%s: %v", result.ID, busyErr))
				summary.Skipped = append(summary.Skipped, Skip{Name: result.ID, Reason: "busy_check_error"})
				continue
			}
			if !ready {
				deferred := result.Update
				deferred.Status = updatecheck.StatusDeferred
				deferred.Message = "active executions did not finish after three checks"
				s.checker.Set(result.ID, deferred)
				summary.Skipped = append(summary.Skipped, Skip{Name: result.ID, Reason: "deferred"})
				summary.retrySoon = true
				continue
			}
			if err := s.updateOne(ctx, result.ID); err != nil {
				if errors.Is(err, packagejobs.ErrBusy) {
					s.deferUpdate(result.ID, "another package operation remained active after three checks", &summary)
					continue
				}
				failed := result.Update
				failed.Status = updatecheck.StatusFailed
				failed.Message = err.Error()
				s.checker.Set(result.ID, failed)
				summary.Errors = append(summary.Errors, fmt.Sprintf("%s: %v", result.ID, err))
				continue
			}
			s.checker.Clear(result.ID)
			summary.Updated = append(summary.Updated, result.ID)
		}
	} else {
		logger.Logger.Info().Str("reason", reason).Msg("package auto-update maintenance is disabled; boot restore remains active")
	}

	if refreshed, loadErr := s.loadRegistry(); loadErr == nil {
		s.restore(ctx, refreshed, &summary)
	}
	return s.finish(summary)
}

func (s *Service) finish(summary Summary) Summary {
	summary.FinishedAt = s.now().UTC()
	s.mu.Lock()
	copy := summary
	s.lastRun = &copy
	delay := s.interval
	if summary.retrySoon {
		if s.backoffStep < len(restoreRetryDelays) {
			delay = restoreRetryDelays[s.backoffStep]
		}
		s.backoffStep++
	} else {
		s.backoffStep = 0
	}
	s.nextRunAt = summary.FinishedAt.Add(delay)
	s.mu.Unlock()
	select {
	case s.scheduleChanged <- struct{}{}:
	default:
	}
	logger.Logger.Info().Int("checked", summary.Checked).Strs("updated", summary.Updated).Strs("restored", summary.Restored).Int("errors", len(summary.Errors)).Msg("package maintenance pass finished")
	return summary
}

func (s *Service) waitUntilIdle(ctx context.Context, name string) (bool, error) {
	if s.executions == nil {
		return true, nil
	}
	for attempt := 0; attempt < maxBusyChecks; attempt++ {
		busy, err := s.executions.HasActiveExecutions(ctx, name)
		if err != nil {
			return false, err
		}
		if !busy {
			return true, nil
		}
		if attempt+1 < maxBusyChecks {
			if err := s.sleep(ctx, busyRetryDelay); err != nil {
				return false, err
			}
		}
	}
	return false, nil
}

func (s *Service) updateOne(ctx context.Context, name string) error {
	if s.jobs == nil {
		return errors.New("package job manager is unavailable")
	}
	ctx, cancel := context.WithTimeout(ctx, s.jobWaitTimeout)
	defer cancel()
	var job *packagejobs.Job
	var err error
	for attempt := 0; attempt < maxBusyChecks; attempt++ {
		job, err = s.jobs.StartUpdate(name, "")
		if !errors.Is(err, packagejobs.ErrBusy) {
			break
		}
		if attempt+1 < maxBusyChecks {
			if sleepErr := s.sleep(ctx, busyRetryDelay); sleepErr != nil {
				return sleepErr
			}
		}
	}
	if err != nil {
		return err
	}
	for {
		current, ok := s.jobs.GetJob(job.ID)
		if !ok {
			return errors.New("package update job disappeared")
		}
		switch current.Status {
		case packagejobs.StatusSucceeded:
			return nil
		case packagejobs.StatusFailed:
			if current.Error == "" {
				return errors.New("package update failed")
			}
			return errors.New(current.Error)
		}
		if err := s.sleep(ctx, jobPollInterval); err != nil {
			return err
		}
	}
}

func (s *Service) deferUpdate(name, message string, summary *Summary) {
	deferred := s.checker.Cached(name)
	deferred.Status = updatecheck.StatusDeferred
	deferred.Message = message
	s.checker.Set(name, deferred)
	summary.Skipped = append(summary.Skipped, Skip{Name: name, Reason: "deferred"})
	summary.retrySoon = true
}

func (s *Service) restore(ctx context.Context, registry *packages.InstallationRegistry, summary *Summary) {
	if s.agent == nil {
		for _, name := range sortedNames(registry) {
			s.finishRestoreGrace(name)
		}
		return
	}
	names := sortedNames(registry)
	for _, name := range names {
		entry := registry.Installed[name]
		if entry.EffectiveDesiredState() != packages.DesiredStateRunning || contains(summary.Restored, name) {
			s.finishRestoreGrace(name)
			continue
		}
		if active, ok := s.jobs.(ActiveJobChecker); ok && active.ActiveFor(name) {
			if !containsSkip(summary.Skipped, name, "updating") {
				summary.Skipped = append(summary.Skipped, Skip{Name: name, Reason: "updating"})
			}
			s.finishRestoreGrace(name)
			continue
		}
		s.restoreOne(ctx, name, entry, summary)
		s.finishRestoreGrace(name)
	}
}

func (s *Service) restoreOne(ctx context.Context, name string, entry packages.InstalledPackage, summary *Summary) {
	assessment := packages.AssessRecordedProcessWith(
		ctx,
		name,
		entry,
		func(info packages.RuntimeInfo) packages.RuntimeProcessState {
			if strings.TrimSpace(info.StartTime) == "" {
				return packages.RuntimeProcessUnknown
			}
			if s.processAlive(info) {
				return packages.RuntimeProcessAliveState
			}
			return packages.RuntimeProcessDead
		},
		s.healthProbe,
	)
	// An equivalent or anonymous healthy endpoint is already our node, even
	// when the recorded PID is dead or its identity is unavailable. Starting
	// another copy would duplicate a live Go-SDK node.
	if assessment.Ownership == packages.RecordedProcessOursHealthy {
		return
	}
	if assessment.Ownership == packages.RecordedProcessOursUnhealthy && assessment.SignalAllowed {
		stopper, ok := s.agent.(interface{ StopAgentForUpdate(string) error })
		if !ok {
			summary.Errors = append(summary.Errors, fmt.Sprintf("restore %s: cannot safely stop the unhealthy recorded process", name))
			summary.retrySoon = true
			return
		}
		if err := stopper.StopAgentForUpdate(name); err != nil {
			summary.Errors = append(summary.Errors, fmt.Sprintf("restore %s: stop unhealthy process: %v", name, err))
			summary.retrySoon = true
			return
		}
	}
	port := 0
	if entry.Runtime.Port != nil {
		port = *entry.Runtime.Port
		if assessment.Ownership == packages.RecordedProcessForeign || !s.portAvailable(port) {
			port = 0
		}
	}
	if err := s.runAgentWithTimeout(ctx, name, domain.RunOptions{Port: port, Detach: true, PortIsPreference: true}); err != nil {
		summary.Errors = append(summary.Errors, fmt.Sprintf("restore %s: %v", name, err))
		summary.retrySoon = true
		return
	}
	summary.Restored = append(summary.Restored, name)
}

func (s *Service) runAgentWithTimeout(ctx context.Context, name string, options domain.RunOptions) error {
	result := make(chan error, 1)
	go func() {
		_, err := s.agent.RunAgent(name, options)
		result <- err
	}()
	timer := time.NewTimer(s.restoreTimeout)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-result:
		return err
	case <-timer.C:
		return errors.New("timed out")
	}
}

func isPortAvailable(port int) bool {
	listener, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		return false
	}
	_ = listener.Close()
	return true
}

func checkEntries(registry *packages.InstallationRegistry) []updatecheck.Entry {
	names := sortedNames(registry)
	entries := make([]updatecheck.Entry, 0, len(names))
	for _, id := range names {
		entry := registry.Installed[id]
		switch strings.ToLower(entry.Source) {
		case "git", "github", "gitlab", "bitbucket":
		default:
			continue
		}
		if strings.TrimSpace(entry.SourcePath) == "" {
			continue
		}
		entries = append(entries, updatecheck.Entry{ID: id, Name: entry.Name, Source: entry.SourcePath, Ref: entry.Ref, InstalledCommit: entry.Commit})
	}
	return entries
}

func sortedNames(registry *packages.InstallationRegistry) []string {
	names := make([]string, 0, len(registry.Installed))
	for name := range registry.Installed {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func contains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func containsSkip(values []Skip, name, reason string) bool {
	for _, value := range values {
		if value.Name == name && value.Reason == reason {
			return true
		}
	}
	return false
}

func (s *Service) loadRegistry() (*packages.InstallationRegistry, error) {
	s.registryMu.Lock()
	defer s.registryMu.Unlock()
	return loadRegistryFile(filepath.Join(s.home, "installed.yaml"))
}

// loadRegistryForPass performs the legacy desired_state migration as one
// read/modify/write before restore observes any package. Later passes find no
// empty fields, so an explicit stopped intent can never be resurrected.
func (s *Service) loadRegistryForPass() (*packages.InstallationRegistry, error) {
	s.registryMu.Lock()
	path := filepath.Join(s.home, "installed.yaml")
	registry, err := loadRegistryFile(path)
	if err != nil {
		s.registryMu.Unlock()
		return nil, err
	}
	migrated := make([]string, 0)
	hosted := s.hostedInContainer()
	for _, name := range sortedNames(registry) {
		entry := registry.Installed[name]
		if entry.DesiredState != "" {
			continue
		}
		if hosted {
			entry.DesiredState = packages.DesiredStateRunning
		} else {
			entry.DesiredState = entry.EffectiveDesiredState()
		}
		registry.Installed[name] = entry
		migrated = append(migrated, name)
	}
	if len(migrated) > 0 {
		err = writeRegistryFile(path, registry)
	}
	s.registryMu.Unlock()
	if err != nil {
		return nil, err
	}
	if len(migrated) > 0 {
		logger.Logger.Info().Msgf("package maintenance: migrated desired_state for %v", migrated)
		if s.onRegistryChange != nil {
			s.onRegistryChange()
		}
	}
	return registry, nil
}

func (s *Service) armBootRestoreGrace() {
	registry, err := s.loadRegistryForPass()
	if err != nil {
		logger.Logger.Warn().Err(err).Msg("package maintenance: could not prepare boot restore grace")
		return
	}
	for _, name := range sortedNames(registry) {
		if registry.Installed[name].EffectiveDesiredState() != packages.DesiredStateRunning {
			continue
		}
		s.mu.Lock()
		if s.restoreGrace[name] {
			s.mu.Unlock()
			continue
		}
		s.restoreGrace[name] = true
		s.mu.Unlock()
		if s.onRestoreState != nil {
			s.onRestoreState(name, true)
		}
	}
}

func (s *Service) finishRestoreGrace(name string) {
	s.mu.Lock()
	active := s.restoreGrace[name]
	delete(s.restoreGrace, name)
	s.mu.Unlock()
	if active && s.onRestoreState != nil {
		s.onRestoreState(name, false)
	}
}

func (s *Service) clearAllRestoreGrace() {
	s.mu.Lock()
	names := make([]string, 0, len(s.restoreGrace))
	for name := range s.restoreGrace {
		names = append(names, name)
	}
	s.restoreGrace = make(map[string]bool)
	s.mu.Unlock()
	sort.Strings(names)
	if s.onRestoreState != nil {
		for _, name := range names {
			s.onRestoreState(name, false)
		}
	}
}

func loadRegistryFile(path string) (*packages.InstallationRegistry, error) {
	registry := &packages.InstallationRegistry{Installed: make(map[string]packages.InstalledPackage)}
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return registry, nil
	}
	if err != nil {
		return nil, err
	}
	if err := yaml.Unmarshal(data, registry); err != nil {
		return nil, err
	}
	if registry.Installed == nil {
		registry.Installed = make(map[string]packages.InstalledPackage)
	}
	return registry, nil
}

func writeRegistryFile(path string, registry *packages.InstallationRegistry) error {
	data, err := yaml.Marshal(registry)
	if err != nil {
		return err
	}
	mode := os.FileMode(0o644)
	if info, statErr := os.Stat(path); statErr == nil {
		mode = info.Mode().Perm()
	}
	temp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tempPath := temp.Name()
	defer os.Remove(tempPath)
	if _, err := temp.Write(data); err != nil {
		_ = temp.Close()
		return err
	}
	if err := temp.Chmod(mode); err != nil {
		_ = temp.Close()
		return err
	}
	if err := temp.Close(); err != nil {
		return err
	}
	return os.Rename(tempPath, path)
}

func (s *Service) Entries() ([]updatecheck.Entry, error) {
	registry, err := s.loadRegistry()
	if err != nil {
		return nil, err
	}
	return checkEntries(registry), nil
}

func (s *Service) RegistryEntry(id string) (packages.InstalledPackage, bool, error) {
	registry, err := s.loadRegistry()
	if err != nil {
		return packages.InstalledPackage{}, false, err
	}
	entry, ok := registry.Installed[id]
	return entry, ok, nil
}

func (s *Service) RegistryEntries() (map[string]packages.InstalledPackage, error) {
	registry, err := s.loadRegistry()
	if err != nil {
		return nil, err
	}
	entries := make(map[string]packages.InstalledPackage, len(registry.Installed))
	for id, entry := range registry.Installed {
		entries[id] = entry
	}
	return entries, nil
}

func (s *Service) SetAutoUpdate(id string, enabled bool) (packages.InstalledPackage, error) {
	s.registryMu.Lock()
	defer s.registryMu.Unlock()
	path := filepath.Join(s.home, "installed.yaml")
	registry, err := loadRegistryFile(path)
	if err != nil {
		return packages.InstalledPackage{}, err
	}
	entry, ok := registry.Installed[id]
	if !ok {
		return packages.InstalledPackage{}, os.ErrNotExist
	}
	entry.AutoUpdate = &enabled
	registry.Installed[id] = entry
	if err := writeRegistryFile(path, registry); err != nil {
		return packages.InstalledPackage{}, err
	}
	if s.onRegistryChange != nil {
		s.onRegistryChange()
	}
	return entry, nil
}

func (s *Service) Status() Status {
	enabled, reason := s.enabled()
	s.mu.RLock()
	defer s.mu.RUnlock()
	var lastRun *Summary
	if s.lastRun != nil {
		copy := *s.lastRun
		lastRun = &copy
	}
	return Status{
		Enabled: enabled, Reason: reason, Interval: s.interval.String(), LastRun: lastRun,
		NextRunAt: s.nextRunAt, BootPassCompleted: s.bootPassCompleted, Hosting: s.hosting(),
	}
}
