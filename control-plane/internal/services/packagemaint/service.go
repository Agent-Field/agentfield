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
	busyRetryDelay  = 10 * time.Minute
	jobPollInterval = 250 * time.Millisecond
	jobWaitTimeout  = 30 * time.Minute
	maxBusyChecks   = 3
)

type JobManager interface {
	StartUpdate(name, source string) (*packagejobs.Job, error)
	GetJob(id string) (*packagejobs.Job, bool)
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
}

type Status struct {
	Enabled   bool      `json:"enabled"`
	Reason    string    `json:"reason"`
	Interval  string    `json:"interval"`
	LastRun   *Summary  `json:"last_run"`
	NextRunAt time.Time `json:"next_run_at"`
}

type Config struct {
	AgentFieldHome   string
	Checker          *updatecheck.Checker
	Jobs             JobManager
	Agent            AgentRunner
	Executions       ActiveExecutionChecker
	Enabled          func() (bool, string)
	ProcessAlive     func(packages.RuntimeInfo) bool
	HealthProbe      func(context.Context, int, string) packages.HealthIdentity
	PortAvailable    func(int) bool
	Sleep            func(context.Context, time.Duration) error
	Now              func() time.Time
	Interval         time.Duration
	JobWaitTimeout   time.Duration
	OnRegistryChange func()
}

type Service struct {
	home             string
	checker          *updatecheck.Checker
	jobs             JobManager
	agent            AgentRunner
	executions       ActiveExecutionChecker
	enabled          func() (bool, string)
	processAlive     func(packages.RuntimeInfo) bool
	healthProbe      func(context.Context, int, string) packages.HealthIdentity
	portAvailable    func(int) bool
	sleep            func(context.Context, time.Duration) error
	now              func() time.Time
	interval         time.Duration
	jobWaitTimeout   time.Duration
	onRegistryChange func()

	mu           sync.RWMutex
	running      bool
	stopping     bool
	lastRun      *Summary
	nextRunAt    time.Time
	registryMu   sync.Mutex
	lifecycleMu  sync.RWMutex
	lifecycleCtx context.Context
	idle         chan struct{}
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
	idle := make(chan struct{})
	close(idle)
	return &Service{
		home: cfg.AgentFieldHome, checker: cfg.Checker, jobs: cfg.Jobs,
		agent: cfg.Agent, executions: cfg.Executions, enabled: cfg.Enabled,
		processAlive: cfg.ProcessAlive, healthProbe: cfg.HealthProbe, portAvailable: cfg.PortAvailable, sleep: cfg.Sleep, now: cfg.Now,
		interval: interval, jobWaitTimeout: cfg.JobWaitTimeout, onRegistryChange: cfg.OnRegistryChange,
		nextRunAt:    cfg.Now().UTC().Add(bootDelay),
		lifecycleCtx: context.Background(),
		idle:         idle,
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

// Run owns the delayed startup pass and recurring maintenance cadence.
func (s *Service) Run(ctx context.Context) {
	s.SetLifecycleContext(ctx)
	if err := s.sleep(ctx, bootDelay); err != nil {
		return
	}
	s.RunPass(ctx)
	ticker := time.NewTicker(s.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.RunPass(ctx)
		}
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
	registry, err := s.loadRegistry()
	if err != nil {
		summary.Errors = append(summary.Errors, err.Error())
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
				continue
			}
			if err := s.updateOne(ctx, result.ID); err != nil {
				if errors.Is(err, packagejobs.ErrBusy) {
					s.deferUpdate(result.ID, "another package operation remained active after three checks", &summary)
					continue
				}
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
	s.nextRunAt = summary.FinishedAt.Add(s.interval)
	s.mu.Unlock()
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
}

func (s *Service) restore(ctx context.Context, registry *packages.InstallationRegistry, summary *Summary) {
	if s.agent == nil {
		return
	}
	names := sortedNames(registry)
	for _, name := range names {
		entry := registry.Installed[name]
		if entry.EffectiveDesiredState() != packages.DesiredStateRunning || contains(summary.Restored, name) {
			continue
		}
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
			continue
		}
		if assessment.Ownership == packages.RecordedProcessOursUnhealthy && assessment.SignalAllowed {
			stopper, ok := s.agent.(interface{ StopAgentForUpdate(string) error })
			if !ok {
				summary.Errors = append(summary.Errors, fmt.Sprintf("restore %s: cannot safely stop the unhealthy recorded process", name))
				continue
			}
			if err := stopper.StopAgentForUpdate(name); err != nil {
				summary.Errors = append(summary.Errors, fmt.Sprintf("restore %s: stop unhealthy process: %v", name, err))
				continue
			}
		}
		port := 0
		if entry.Runtime.Port != nil {
			port = *entry.Runtime.Port
			if assessment.Ownership == packages.RecordedProcessForeign || !s.portAvailable(port) {
				port = 0
			}
		}
		if _, err := s.agent.RunAgent(name, domain.RunOptions{Port: port, Detach: true, PortIsPreference: true}); err != nil {
			summary.Errors = append(summary.Errors, fmt.Sprintf("restore %s: %v", name, err))
			continue
		}
		summary.Restored = append(summary.Restored, name)
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

func (s *Service) loadRegistry() (*packages.InstallationRegistry, error) {
	s.registryMu.Lock()
	defer s.registryMu.Unlock()
	return loadRegistryFile(filepath.Join(s.home, "installed.yaml"))
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
	data, err := yaml.Marshal(registry)
	if err != nil {
		return packages.InstalledPackage{}, err
	}
	mode := os.FileMode(0o644)
	if info, statErr := os.Stat(path); statErr == nil {
		mode = info.Mode().Perm()
	}
	temp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp-*")
	if err != nil {
		return packages.InstalledPackage{}, err
	}
	tempPath := temp.Name()
	defer os.Remove(tempPath)
	if _, err := temp.Write(data); err != nil {
		_ = temp.Close()
		return packages.InstalledPackage{}, err
	}
	if err := temp.Chmod(mode); err != nil {
		_ = temp.Close()
		return packages.InstalledPackage{}, err
	}
	if err := temp.Close(); err != nil {
		return packages.InstalledPackage{}, err
	}
	if err := os.Rename(tempPath, path); err != nil {
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
	return Status{Enabled: enabled, Reason: reason, Interval: s.interval.String(), LastRun: lastRun, NextRunAt: s.nextRunAt}
}
