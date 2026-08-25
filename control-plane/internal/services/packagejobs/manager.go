package packagejobs

import (
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	coreservices "github.com/Agent-Field/agentfield/control-plane/internal/core/services"
	infrastorage "github.com/Agent-Field/agentfield/control-plane/internal/infrastructure/storage"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/google/uuid"
	"gopkg.in/yaml.v3"
)

const (
	maxJobLines = 500
	maxJobs     = 50
)

var (
	ErrBusy          = errors.New("a package operation is already running")
	ErrInvalidSource = errors.New("invalid package source")
	ErrNotFound      = errors.New("package not found")

	repoPartRE   = regexp.MustCompile(`^[A-Za-z0-9_.-]+$`)
	subdirPartRE = regexp.MustCompile(`^[A-Za-z0-9_./-]+$`)
	safeRefRE    = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._/-]*$`)
	ansiRE       = regexp.MustCompile(`\x1b\[[0-?]*[ -/]*[@-~]`)
)

type ErrExecutionsActive struct {
	Count int
}

func (e *ErrExecutionsActive) Error() string {
	return fmt.Sprintf("%d active execution(s) must finish before updating", e.Count)
}

type JobKind string

const (
	JobInstall   JobKind = "install"
	JobUpdate    JobKind = "update"
	JobUninstall JobKind = "uninstall"
)

type JobStatus string

const (
	StatusPending   JobStatus = "pending"
	StatusRunning   JobStatus = "running"
	StatusSucceeded JobStatus = "succeeded"
	StatusFailed    JobStatus = "failed"
)

type Job struct {
	ID           string     `json:"id"`
	Source       string     `json:"source"`
	Kind         JobKind    `json:"kind"`
	Status       JobStatus  `json:"status"`
	PackageName  string     `json:"package_name,omitempty"`
	Error        string     `json:"error,omitempty"`
	Lines        []string   `json:"lines"`
	StartedAt    *time.Time `json:"started_at,omitempty"`
	FinishedAt   *time.Time `json:"finished_at,omitempty"`
	expectedName string
	previousPort int
}

type installer interface {
	InstallPackage(source string, options domain.InstallOptions) error
	UninstallPackage(name string) error
	ListInstalledPackages() ([]domain.InstalledPackage, error)
	GetPackageInfo(name string) (*domain.InstalledPackage, error)
}

type resultInstaller interface {
	InstallPackageWithResult(source string, options domain.InstallOptions) (string, error)
}

type updateStopper interface {
	StopAgentForUpdate(name string) error
}

type Manager struct {
	mu                sync.RWMutex
	installer         installer
	agentService      interfaces.AgentService
	agentfieldHome    string
	jobs              map[string]*Job
	order             []string
	active            bool
	activePackage     string
	portAvailable     func(int) bool
	clearUpdateCache  func(string)
	onUpdateState     func(string, bool)
	executionActivity func(string) (int, error)
	// onRegistryChange runs synchronously after a mutation lands in
	// installed.yaml so API reads are consistent with API writes (the
	// fsnotify watcher also syncs, but asynchronously).
	onRegistryChange func()
}

// SetOnRegistryChange registers a hook invoked after every successful
// install/update/uninstall. The server wires this to the registry→DB sync.
func (m *Manager) SetOnRegistryChange(fn func()) {
	m.mu.Lock()
	m.onRegistryChange = fn
	m.mu.Unlock()
}

func (m *Manager) notifyRegistryChange() {
	m.mu.RLock()
	fn := m.onRegistryChange
	m.mu.RUnlock()
	if fn != nil {
		fn()
	}
}

// NewManager constructs the package service exactly as the CLI container does.
func NewManager(registryStorage interfaces.RegistryStorage, agentfieldHome string, agentService interfaces.AgentService) *Manager {
	fileSystem := infrastorage.NewFileSystemAdapter()
	return newManager(coreservices.NewPackageService(registryStorage, fileSystem, agentfieldHome), agentService, agentfieldHome)
}

func newManager(inst installer, agentService interfaces.AgentService, agentfieldHome string) *Manager {
	return &Manager{
		installer:      inst,
		agentService:   agentService,
		agentfieldHome: agentfieldHome,
		jobs:           make(map[string]*Job),
		portAvailable:  isPortAvailable,
	}
}

func (m *Manager) SetUpdateCacheClearer(fn func(string)) {
	m.mu.Lock()
	m.clearUpdateCache = fn
	m.mu.Unlock()
}

func (m *Manager) SetUpdateStateHook(fn func(string, bool)) {
	m.mu.Lock()
	m.onUpdateState = fn
	m.mu.Unlock()
}

func (m *Manager) SetExecutionActivity(fn func(packageName string) (int, error)) {
	m.mu.Lock()
	m.executionActivity = fn
	m.mu.Unlock()
}

func ValidateSource(source string) (string, error) {
	source = strings.TrimSpace(source)
	const prefix = "https://github.com/"
	if !strings.HasPrefix(source, prefix) || strings.ContainsAny(source, " \t\r\n?#") {
		return "", ErrInvalidSource
	}
	if i := strings.Index(strings.TrimPrefix(source, prefix), "//"); i >= 0 {
		rawSubdir := strings.TrimPrefix(source, prefix)[i+2:]
		if rawSubdir == "" || strings.HasPrefix(rawSubdir, "/") {
			return "", ErrInvalidSource
		}
		if at := strings.Index(rawSubdir, "@"); at >= 0 && strings.Contains(rawSubdir[at+1:], "/") {
			return "", ErrInvalidSource
		}
	}
	info, err := packages.ParseGitURL(source)
	if err != nil {
		return "", ErrInvalidSource
	}
	repoRaw := strings.TrimSuffix(strings.TrimPrefix(info.CloneURL, prefix), "/")
	parts := strings.Split(repoRaw, "/")
	if len(parts) != 2 || !repoPartRE.MatchString(parts[0]) || !repoPartRE.MatchString(parts[1]) ||
		parts[0] == "." || parts[0] == ".." || parts[1] == "." || parts[1] == ".." ||
		strings.HasPrefix(parts[0], "-") || strings.HasPrefix(parts[1], "-") {
		return "", ErrInvalidSource
	}
	normalized := prefix + strings.Join(parts, "/")
	if info.Subdir != "" {
		subdir := strings.TrimSuffix(info.Subdir, "/")
		if subdir == "" || strings.Contains(subdir, "..") || strings.HasPrefix(subdir, "/") || !subdirPartRE.MatchString(subdir) {
			return "", ErrInvalidSource
		}
		for _, segment := range strings.Split(subdir, "/") {
			if segment == "" || strings.HasPrefix(segment, "-") {
				return "", ErrInvalidSource
			}
		}
		normalized += "//" + subdir
	}
	if info.Ref != "" {
		if !safeRefRE.MatchString(info.Ref) {
			return "", ErrInvalidSource
		}
		normalized += "@" + info.Ref
	}
	return normalized, nil
}

func (m *Manager) StartInstall(source string, force bool) (*Job, error) {
	normalized, err := ValidateSource(source)
	if err != nil {
		return nil, err
	}
	return m.startJob(JobInstall, normalized, "", force, "", 0, nil)
}

func (m *Manager) StartUpdate(packageName, source string, force bool) (*Job, error) {
	return m.startUpdate(packageName, source, false, force)
}

// StartMaintenanceUpdate applies the same recorded-source update contract as
// StartUpdate while constraining superseded redirects to the current name.
func (m *Manager) StartMaintenanceUpdate(packageName, source string) (*Job, error) {
	return m.startUpdate(packageName, source, true, true)
}

func (m *Manager) startUpdate(packageName, source string, unattended, force bool) (*Job, error) {
	entry, err := m.registryEntry(packageName)
	if err != nil {
		return nil, err
	}
	if !unattended && !force {
		m.mu.RLock()
		activity := m.executionActivity
		m.mu.RUnlock()
		if activity != nil {
			count, activityErr := activity(packageName)
			if activityErr != nil {
				return nil, activityErr
			}
			if count > 0 {
				return nil, &ErrExecutionsActive{Count: count}
			}
		}
	}
	if strings.TrimSpace(source) != "" {
		// Desktop catalog updates deliberately replace a stale recorded source;
		// the installer will persist the source that ultimately lands.
		source, err = ValidateSource(source)
	} else {
		source, err = sourceFromRegistry(entry.SourcePath)
	}
	if err != nil {
		return nil, err
	}
	expectedName := ""
	if unattended {
		expectedName = packageName
	}
	previousPort := 0
	if entry.Runtime.Port != nil {
		previousPort = *entry.Runtime.Port
	}
	var beforeLaunch func()
	if !unattended {
		beforeLaunch = func() {
			m.mu.RLock()
			clear := m.clearUpdateCache
			m.mu.RUnlock()
			if clear != nil {
				clear(packageName)
			}
		}
	}
	return m.startJob(JobUpdate, source, packageName, true, expectedName, previousPort, beforeLaunch)
}

func (m *Manager) startJob(kind JobKind, source, packageName string, force bool, expectedName string, previousPort int, beforeLaunch func()) (*Job, error) {
	m.mu.Lock()
	if m.active {
		m.mu.Unlock()
		return nil, ErrBusy
	}
	job := &Job{
		ID:           uuid.NewString(),
		Source:       source,
		Kind:         kind,
		Status:       StatusPending,
		PackageName:  packageName,
		Lines:        []string{"validating source"},
		expectedName: expectedName,
		previousPort: previousPort,
	}
	m.active = true
	m.activePackage = packageName
	m.jobs[job.ID] = job
	m.order = append(m.order, job.ID)
	m.evictLocked()
	result := cloneJob(job)
	m.mu.Unlock()

	if beforeLaunch != nil {
		beforeLaunch()
	}
	go m.run(job.ID, force)
	return result, nil
}

func (m *Manager) run(jobID string, force bool) {
	started := time.Now().UTC()
	m.mu.Lock()
	job := m.jobs[jobID]
	job.Status = StatusRunning
	job.StartedAt = &started
	source, kind, packageName := job.Source, job.Kind, job.PackageName
	originalPackageName := packageName
	expectedName, previousPort := job.expectedName, job.previousPort
	onUpdateState := m.onUpdateState
	m.mu.Unlock()
	if kind == JobUpdate && onUpdateState != nil {
		onUpdateState(packageName, true)
		defer onUpdateState(packageName, false)
	}

	m.appendLine(jobID, fmt.Sprintf("installing %s", source))
	var err error
	var wasRunning bool
	var replaceHookFired bool
	if kind == JobUpdate {
		wasRunning, err = m.isRunning(packageName)
	}

	before := m.installedNames()
	if err == nil {
		installSource, options := splitSubdir(source, force)
		options.ExpectedPackageName = expectedName
		if kind == JobUpdate {
			options.BeforeReplace = func() error {
				replaceHookFired = true
				if !wasRunning {
					return nil
				}
				m.appendLine(jobID, fmt.Sprintf("stopping %s", packageName))
				return m.stopForUpdate(packageName)
			}
		}
		if reporting, ok := m.installer.(resultInstaller); ok {
			var installedName string
			installedName, err = reporting.InstallPackageWithResult(installSource, options)
			// The installer is authoritative about what it installed, and an
			// update is where that matters most: a `superseded_by` redirect in
			// the recorded source can retire the package being updated and put
			// a differently-named successor in its place. Following the
			// installer here means the job reports — and restarts — the node
			// that now exists, rather than the name that went in and no longer
			// resolves.
			if err == nil && installedName != "" {
				packageName = installedName
			}
		} else {
			err = m.installer.InstallPackage(installSource, options)
		}
	}
	if err == nil && packageName == "" {
		packageName = m.discoverPackageName(before)
	}
	if kind == JobUpdate && err == nil && packageName != "" && packageName != originalPackageName {
		// A differently-named successor is a first install, so the installer's
		// same-name BeforeReplace boundary does not fire. Once the successor is
		// safely installed, stop and retire the old package here.
		if wasRunning && !replaceHookFired {
			m.appendLine(jobID, fmt.Sprintf("stopping %s", originalPackageName))
			if stopErr := m.stopForUpdate(originalPackageName); stopErr != nil {
				err = stopErr
			} else {
				replaceHookFired = true
			}
		}
		if err == nil && m.installedNames()[originalPackageName] {
			m.appendLine(jobID, fmt.Sprintf("retiring %s", originalPackageName))
			if uninstallErr := m.installer.UninstallPackage(originalPackageName); uninstallErr != nil {
				err = uninstallErr
			}
		}
	}
	if kind == JobUpdate && wasRunning && replaceHookFired {
		restartName := packageName
		if err != nil {
			restartName = job.PackageName
		}
		port := 0
		if previousPort > 0 && m.portAvailable != nil && m.portAvailable(previousPort) {
			port = previousPort
		}
		// A stop issued while the node was down for this update is the user's
		// last word: RunAgent would record running intent again and erase it.
		stoppedMeanwhile := false
		if entry, entryErr := m.registryEntry(restartName); entryErr == nil && entry.DesiredState == "stopped" {
			stoppedMeanwhile = true
			m.appendLine(jobID, fmt.Sprintf("not restarting %s: it was stopped during the update", restartName))
		}
		if !stoppedMeanwhile {
			m.appendLine(jobID, fmt.Sprintf("restarting %s", restartName))
			_, restartErr := m.agentService.RunAgent(restartName, domain.RunOptions{Detach: true, Port: port, PortIsPreference: true})
			if restartErr != nil {
				if err == nil {
					err = restartErr
				} else {
					err = fmt.Errorf("%v; restoring previous package: %w", err, restartErr)
				}
			}
		}
	}
	if err == nil {
		m.appendLine(jobID, fmt.Sprintf("install completed: %s", packageName))
		m.notifyRegistryChange()
		if kind == JobUpdate {
			m.mu.RLock()
			clear := m.clearUpdateCache
			m.mu.RUnlock()
			if clear != nil {
				if expectedName != "" {
					clear(job.PackageName)
				}
				if packageName != job.PackageName {
					clear(packageName)
				}
			}
		}
	}
	m.finish(jobID, packageName, err)
}

func (m *Manager) stopForUpdate(name string) error {
	if stopper, ok := m.agentService.(updateStopper); ok {
		return stopper.StopAgentForUpdate(name)
	}
	return m.agentService.StopAgent(name)
}

func splitSubdir(source string, force bool) (string, domain.InstallOptions) {
	options := domain.InstallOptions{Force: force}
	if info, err := packages.ParseGitURL(source); err == nil {
		options.Path = info.Subdir
		source = info.CloneURL
		if info.Ref != "" {
			source += "@" + info.Ref
		}
	}
	return source, options
}

func (m *Manager) installedNames() map[string]bool {
	names := make(map[string]bool)
	pkgs, err := m.installer.ListInstalledPackages()
	if err != nil {
		return names
	}
	for _, pkg := range pkgs {
		names[pkg.Name] = true
	}
	return names
}

func (m *Manager) discoverPackageName(before map[string]bool) string {
	pkgs, err := m.installer.ListInstalledPackages()
	if err != nil {
		return ""
	}
	for _, pkg := range pkgs {
		if !before[pkg.Name] {
			return pkg.Name
		}
	}
	return ""
}

func (m *Manager) isRunning(name string) (bool, error) {
	if m.agentService == nil {
		return false, nil
	}
	status, err := m.agentService.GetAgentStatus(name)
	if err != nil {
		return false, err
	}
	return status.IsRunning, nil
}

func (m *Manager) Uninstall(packageName string) error {
	m.mu.Lock()
	if m.active {
		m.mu.Unlock()
		return ErrBusy
	}
	m.active = true
	m.activePackage = packageName
	m.mu.Unlock()
	defer func() {
		m.mu.Lock()
		m.active = false
		m.activePackage = ""
		m.mu.Unlock()
	}()

	if _, err := m.installer.GetPackageInfo(packageName); err != nil {
		return ErrNotFound
	}
	running, err := m.isRunning(packageName)
	if err != nil {
		return err
	}
	if running {
		if err := m.agentService.StopAgent(packageName); err != nil {
			return err
		}
	}
	if err := m.installer.UninstallPackage(packageName); err != nil {
		return err
	}
	m.notifyRegistryChange()
	return nil
}

func (m *Manager) GetJob(id string) (*Job, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	job, ok := m.jobs[id]
	if !ok {
		return nil, false
	}
	return cloneJob(job), true
}

// ActiveFor reports whether the package currently owns the manager's single
// mutation slot. Restore uses it to avoid launching from files being replaced.
func (m *Manager) ActiveFor(name string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.active && m.activePackage != "" && packages.NodeIDsEquivalent(m.activePackage, name)
}

func (m *Manager) ListJobs() []*Job {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make([]*Job, 0, len(m.order))
	for i := len(m.order) - 1; i >= 0; i-- {
		result = append(result, cloneJob(m.jobs[m.order[i]]))
	}
	return result
}

func (m *Manager) appendLine(id, line string) {
	line = ansiRE.ReplaceAllString(line, "")
	m.mu.Lock()
	defer m.mu.Unlock()
	job, ok := m.jobs[id]
	if !ok {
		return
	}
	job.Lines = append(job.Lines, line)
	if len(job.Lines) > maxJobLines {
		job.Lines = append([]string(nil), job.Lines[len(job.Lines)-maxJobLines:]...)
	}
}

func (m *Manager) finish(id, packageName string, runErr error) {
	finished := time.Now().UTC()
	m.mu.Lock()
	defer m.mu.Unlock()
	job := m.jobs[id]
	job.PackageName = packageName
	job.FinishedAt = &finished
	if runErr != nil {
		job.Status = StatusFailed
		job.Error = runErr.Error()
		logger.Logger.Error().Err(runErr).Str("job_id", id).Msg("package job failed")
	} else {
		job.Status = StatusSucceeded
	}
	m.active = false
	m.activePackage = ""
}

func (m *Manager) evictLocked() {
	for len(m.order) > maxJobs {
		delete(m.jobs, m.order[0])
		m.order = m.order[1:]
	}
}

func cloneJob(job *Job) *Job {
	copy := *job
	copy.Lines = append([]string(nil), job.Lines...)
	return &copy
}

type registryPackage struct {
	SourcePath   string `yaml:"source_path"`
	DesiredState string `yaml:"desired_state"`
	Runtime      struct {
		Port *int `yaml:"port"`
	} `yaml:"runtime"`
}

func (m *Manager) registryEntry(name string) (*registryPackage, error) {
	data, err := os.ReadFile(filepath.Join(m.agentfieldHome, "installed.yaml"))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, ErrNotFound
		}
		return nil, err
	}
	var registry struct {
		Installed map[string]registryPackage `yaml:"installed"`
	}
	if err := yaml.Unmarshal(data, &registry); err != nil {
		return nil, err
	}
	entry, ok := registry.Installed[name]
	if !ok {
		return nil, ErrNotFound
	}
	return &entry, nil
}

func sourceFromRegistry(source string) (string, error) {
	source = strings.TrimSpace(source)
	if !strings.Contains(source, "://") && !strings.HasPrefix(source, "git@") {
		source = "https://github.com/" + source
	}
	if strings.HasPrefix(source, "https://github.com/") {
		return ValidateSource(source)
	}
	if strings.ContainsAny(source, " \t\r\n?#") || strings.HasPrefix(source, "-") || !packages.IsGitURL(source) {
		return "", ErrInvalidSource
	}
	info, err := packages.ParseGitURL(source)
	if err != nil || strings.HasPrefix(info.CloneURL, "-") || (info.Ref != "" && !safeRefRE.MatchString(info.Ref)) {
		return "", ErrInvalidSource
	}
	if info.Subdir != "" {
		if strings.Contains(info.Subdir, "..") || !subdirPartRE.MatchString(info.Subdir) {
			return "", ErrInvalidSource
		}
		for _, segment := range strings.Split(info.Subdir, "/") {
			if segment == "" || strings.HasPrefix(segment, "-") {
				return "", ErrInvalidSource
			}
		}
	}
	return source, nil
}

func isPortAvailable(port int) bool {
	listener, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		return false
	}
	_ = listener.Close()
	return true
}
