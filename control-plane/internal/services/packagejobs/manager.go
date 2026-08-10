package packagejobs

import (
	"errors"
	"fmt"
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
	ansiRE       = regexp.MustCompile(`\x1b\[[0-?]*[ -/]*[@-~]`)
)

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
	ID          string     `json:"id"`
	Source      string     `json:"source"`
	Kind        JobKind    `json:"kind"`
	Status      JobStatus  `json:"status"`
	PackageName string     `json:"package_name,omitempty"`
	Error       string     `json:"error,omitempty"`
	Lines       []string   `json:"lines"`
	StartedAt   *time.Time `json:"started_at,omitempty"`
	FinishedAt  *time.Time `json:"finished_at,omitempty"`
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

type Manager struct {
	mu             sync.RWMutex
	installer      installer
	agentService   interfaces.AgentService
	agentfieldHome string
	jobs           map[string]*Job
	order          []string
	active         bool
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
	}
}

func ValidateSource(source string) (string, error) {
	source = strings.TrimSpace(source)
	const prefix = "https://github.com/"
	if !strings.HasPrefix(source, prefix) || strings.ContainsAny(source, " \t\r\n?#") {
		return "", ErrInvalidSource
	}
	rest := strings.TrimPrefix(source, prefix)
	repoRaw, subdir, hasSubdir := rest, "", false
	if i := strings.Index(rest, "//"); i >= 0 {
		repoRaw, subdir, hasSubdir = rest[:i], rest[i+2:], true
	}
	repoRaw = strings.TrimSuffix(repoRaw, "/")
	parts := strings.Split(repoRaw, "/")
	if len(parts) != 2 || !repoPartRE.MatchString(parts[0]) || !repoPartRE.MatchString(parts[1]) ||
		parts[0] == "." || parts[0] == ".." || parts[1] == "." || parts[1] == ".." ||
		strings.HasPrefix(parts[0], "-") || strings.HasPrefix(parts[1], "-") {
		return "", ErrInvalidSource
	}
	normalized := prefix + strings.Join(parts, "/")
	if !hasSubdir {
		return normalized, nil
	}
	subdir = strings.TrimSuffix(subdir, "/")
	if subdir == "" || strings.Contains(subdir, "..") || strings.HasPrefix(subdir, "/") || !subdirPartRE.MatchString(subdir) {
		return "", ErrInvalidSource
	}
	for _, segment := range strings.Split(subdir, "/") {
		if segment == "" || strings.HasPrefix(segment, "-") {
			return "", ErrInvalidSource
		}
	}
	return normalized + "//" + subdir, nil
}

func (m *Manager) StartInstall(source string, force bool) (*Job, error) {
	normalized, err := ValidateSource(source)
	if err != nil {
		return nil, err
	}
	return m.startJob(JobInstall, normalized, "", force)
}

func (m *Manager) StartUpdate(packageName string) (*Job, error) {
	entry, err := m.registryEntry(packageName)
	if err != nil {
		return nil, err
	}
	source, err := sourceFromRegistry(entry.SourcePath)
	if err != nil {
		return nil, err
	}
	return m.startJob(JobUpdate, source, packageName, true)
}

func (m *Manager) startJob(kind JobKind, source, packageName string, force bool) (*Job, error) {
	m.mu.Lock()
	if m.active {
		m.mu.Unlock()
		return nil, ErrBusy
	}
	job := &Job{
		ID:          uuid.NewString(),
		Source:      source,
		Kind:        kind,
		Status:      StatusPending,
		PackageName: packageName,
		Lines:       []string{"validating source"},
	}
	m.active = true
	m.jobs[job.ID] = job
	m.order = append(m.order, job.ID)
	m.evictLocked()
	result := cloneJob(job)
	m.mu.Unlock()

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
	m.mu.Unlock()

	m.appendLine(jobID, fmt.Sprintf("installing %s", source))
	var err error
	var wasRunning bool
	if kind == JobUpdate {
		wasRunning, err = m.isRunning(packageName)
		if err == nil && wasRunning {
			m.appendLine(jobID, fmt.Sprintf("stopping %s", packageName))
			err = m.agentService.StopAgent(packageName)
		}
	}

	before := m.installedNames()
	if err == nil {
		installSource, options := splitSubdir(source, force)
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
	if err == nil && kind == JobUpdate && wasRunning {
		m.appendLine(jobID, fmt.Sprintf("restarting %s", packageName))
		_, err = m.agentService.RunAgent(packageName, domain.RunOptions{Detach: true})
	}
	if err == nil {
		m.appendLine(jobID, fmt.Sprintf("install completed: %s", packageName))
		m.notifyRegistryChange()
	}
	m.finish(jobID, packageName, err)
}

func splitSubdir(source string, force bool) (string, domain.InstallOptions) {
	options := domain.InstallOptions{Force: force}
	const prefix = "https://github.com/"
	rest := strings.TrimPrefix(source, prefix)
	if i := strings.Index(rest, "//"); i >= 0 {
		options.Path = rest[i+2:]
		source = prefix + rest[:i]
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
	m.mu.Unlock()
	defer func() {
		m.mu.Lock()
		m.active = false
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
	SourcePath string `yaml:"source_path"`
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
	if !strings.HasPrefix(source, "https://github.com/") {
		source = "https://github.com/" + source
	}
	// Registries retain the resolved ref as @branch; update intentionally tracks latest.
	if at := strings.LastIndex(source, "@"); at > strings.LastIndex(source, "/") {
		source = source[:at]
	}
	return ValidateSource(source)
}
