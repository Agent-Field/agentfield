package packages

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/fatih/color"
	"gopkg.in/yaml.v3"
)

// UserEnvironmentVar represents a user-configurable environment variable
type UserEnvironmentVar struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
	Type        string `yaml:"type"` // "string", "secret", "integer", "boolean", "float"
	Default     string `yaml:"default"`
	Optional    bool   `yaml:"optional"`
	Validation  string `yaml:"validation"` // regex pattern
	Scope       string `yaml:"scope"`      // "global" (shared across nodes, default) or "node"
}

// SecretScope returns the secret store scope for this variable given the node
// name. Variables default to global so shared keys (API tokens) are entered once.
func (v UserEnvironmentVar) SecretScope(nodeName string) string {
	if v.Scope == "node" {
		return nodeName
	}
	return globalScope
}

// UserEnvironmentConfig represents user-configurable environment variables
type UserEnvironmentConfig struct {
	Required []UserEnvironmentVar `yaml:"required"`
	Optional []UserEnvironmentVar `yaml:"optional"`
}

// PackageMetadata represents the structure of agentfield-package.yaml
type PackageMetadata struct {
	Name            string                 `yaml:"name"`
	Version         string                 `yaml:"version"`
	Description     string                 `yaml:"description"`
	Author          string                 `yaml:"author"`
	Type            string                 `yaml:"type"`
	Main            string                 `yaml:"main"`
	Entrypoint      EntrypointConfig       `yaml:"entrypoint"`
	AgentNode       AgentNodeConfig        `yaml:"agent_node"`
	Dependencies    DependencyConfig       `yaml:"dependencies"`
	Capabilities    CapabilityConfig       `yaml:"capabilities"`
	UserEnvironment UserEnvironmentConfig  `yaml:"user_environment"`
	Metadata        map[string]interface{} `yaml:"metadata"`
}

// EntrypointConfig describes how to start the agent node process.
type EntrypointConfig struct {
	// Start is the shell-free command used to launch the node, e.g.
	// "python -m pr_af.app". The first token is resolved against the package
	// venv when it is "python"/"python3". Empty falls back to "python main.py".
	Start string `yaml:"start"`
	// Healthcheck is the HTTP path polled to confirm readiness (default "/health").
	Healthcheck string `yaml:"healthcheck"`
}

// AgentNodeConfig represents agent node specific configuration
type AgentNodeConfig struct {
	NodeID      string `yaml:"node_id"`
	DefaultPort int    `yaml:"default_port"`
}

// DependencyConfig represents package dependencies
type DependencyConfig struct {
	Python []string `yaml:"python"`
	System []string `yaml:"system"`
	// Nodes lists other agent nodes this node depends on. Each entry is an
	// installable source: an "af://registry/<name>[@version]" ref or a git URL.
	// Installing this node installs its node dependencies recursively.
	Nodes []string `yaml:"nodes"`
}

// CapabilityConfig represents agent node capabilities
type CapabilityConfig struct {
	Reasoners []FunctionInfo `yaml:"reasoners"`
	Skills    []FunctionInfo `yaml:"skills"`
}

// FunctionInfo represents a reasoner or skill function
type FunctionInfo struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
}

// InstallationRegistry represents the global installation registry
type InstallationRegistry struct {
	Installed map[string]InstalledPackage `yaml:"installed"`
}

// InstalledPackage represents an installed package entry
type InstalledPackage struct {
	Name        string      `yaml:"name"`
	Version     string      `yaml:"version"`
	Description string      `yaml:"description"`
	Path        string      `yaml:"path"`
	Source      string      `yaml:"source"`
	SourcePath  string      `yaml:"source_path"`
	InstalledAt string      `yaml:"installed_at"`
	Status      string      `yaml:"status"`
	Runtime     RuntimeInfo `yaml:"runtime"`
}

// RuntimeInfo represents runtime information for a package
type RuntimeInfo struct {
	Port      *int    `yaml:"port"`
	PID       *int    `yaml:"pid"`
	StartedAt *string `yaml:"started_at"`
	LogFile   string  `yaml:"log_file"`
}

// PackageInstaller handles package installation
type PackageInstaller struct {
	AgentFieldHome string
	Verbose        bool
}

// Spinner represents a CLI spinner for progress indication
type Spinner struct {
	message string
	active  bool
	mu      sync.Mutex
	done    chan bool
}

// Professional CLI status symbols
const (
	StatusSuccess = "✓"
	StatusError   = "✗"
)

// Spinner characters for progress indication
var spinnerChars = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

// Color functions for professional output
var (
	Green  = color.New(color.FgGreen).SprintFunc()
	Red    = color.New(color.FgRed).SprintFunc()
	Yellow = color.New(color.FgYellow).SprintFunc()
	Blue   = color.New(color.FgBlue).SprintFunc()
	Cyan   = color.New(color.FgCyan).SprintFunc()
	Gray   = color.New(color.FgHiBlack).SprintFunc()
	Bold   = color.New(color.Bold).SprintFunc()
)

// newSpinner creates a new spinner with the given message
func (pi *PackageInstaller) newSpinner(message string) *Spinner {
	return &Spinner{
		message: message,
		done:    make(chan bool),
	}
}

// Start begins the spinner animation
func (s *Spinner) Start() {
	s.mu.Lock()
	s.active = true
	s.mu.Unlock()

	go func() {
		i := 0
		for {
			select {
			case <-s.done:
				return
			default:
				s.mu.Lock()
				if s.active {
					fmt.Printf("\r  %s %s", spinnerChars[i%len(spinnerChars)], s.message)
					i++
				}
				s.mu.Unlock()
				time.Sleep(100 * time.Millisecond)
			}
		}
	}()
}

// Stop stops the spinner and clears the line
func (s *Spinner) Stop() {
	s.mu.Lock()
	s.active = false
	s.mu.Unlock()
	s.done <- true
	fmt.Print("\r\033[K") // Clear the line
}

// Success stops the spinner and shows a success message
func (s *Spinner) Success(message string) {
	s.Stop()
	fmt.Printf("  %s %s\n", Green(StatusSuccess), message)
}

// Error stops the spinner and shows an error message
func (s *Spinner) Error(message string) {
	s.Stop()
	fmt.Printf("  %s %s\n", Red(StatusError), message)
}

// InstallPackage installs a package from the given source path
func (pi *PackageInstaller) InstallPackage(sourcePath string, force bool) error {
	// Import the CLI utilities
	// Note: We'll need to import this properly, but for now let's define local functions

	// Get package name first for better messaging
	metadata, err := pi.parsePackageMetadata(sourcePath)
	if err != nil {
		return fmt.Errorf("failed to parse package metadata: %w", err)
	}

	fmt.Printf("Installing %s...\n", metadata.Name)

	// 1. Validate source package
	spinner := pi.newSpinner("Validating package structure")
	spinner.Start()
	if err := pi.validatePackage(sourcePath); err != nil {
		spinner.Error("Package validation failed")
		return fmt.Errorf("package validation failed: %w", err)
	}
	spinner.Success("Package structure validated")

	// 2. Check if already installed
	if !force && pi.isPackageInstalled(metadata.Name) {
		return fmt.Errorf("package %s already installed (use --force to reinstall)", metadata.Name)
	}

	// 3. Copy package to global location
	destPath := filepath.Join(pi.AgentFieldHome, "packages", metadata.Name)
	spinner = pi.newSpinner("Setting up environment")
	spinner.Start()
	if err := pi.copyPackage(sourcePath, destPath); err != nil {
		spinner.Error("Failed to copy package")
		return fmt.Errorf("failed to copy package: %w", err)
	}
	spinner.Success("Environment configured")

	// 4. Install dependencies
	spinner = pi.newSpinner("Installing dependencies")
	spinner.Start()
	if err := pi.installDependencies(destPath, metadata); err != nil {
		spinner.Error("Failed to install dependencies")
		return fmt.Errorf("failed to install dependencies: %w", err)
	}
	spinner.Success("Dependencies installed")

	// 5. Update installation registry
	if err := pi.updateRegistry(metadata, sourcePath, destPath); err != nil {
		return fmt.Errorf("failed to update registry: %w", err)
	}

	fmt.Printf("%s Installed %s v%s\n", Green(StatusSuccess), Bold(metadata.Name), Gray(metadata.Version))
	fmt.Printf("  %s %s\n", Gray("Location:"), destPath)

	// 6. Check for required environment variables and provide guidance
	pi.checkEnvironmentVariables(metadata)

	fmt.Printf("\n%s %s\n", Blue("→"), Bold(fmt.Sprintf("Run: af run %s", metadata.Name)))

	return nil
}

// checkEnvironmentVariables checks for required environment variables and provides setup guidance
func (pi *PackageInstaller) checkEnvironmentVariables(metadata *PackageMetadata) {
	if len(metadata.UserEnvironment.Required) == 0 && len(metadata.UserEnvironment.Optional) == 0 {
		return // No user environment variables configured
	}

	// Check required environment variables
	missingRequired := []UserEnvironmentVar{}
	for _, envVar := range metadata.UserEnvironment.Required {
		if os.Getenv(envVar.Name) == "" {
			missingRequired = append(missingRequired, envVar)
		}
	}

	if len(missingRequired) > 0 {
		fmt.Printf("\n%s %s\n", Yellow("⚠"), Bold("Missing required environment variables:"))
		for _, envVar := range missingRequired {
			fmt.Printf("  %s\n", Cyan(fmt.Sprintf("af config %s --set %s=your-value-here", metadata.Name, envVar.Name)))
		}
	}

	// Show optional environment variables if any
	if len(metadata.UserEnvironment.Optional) > 0 {
		fmt.Printf("\n%s %s\n", Gray("ℹ"), Gray("Optional environment variables (with defaults):"))
		for _, envVar := range metadata.UserEnvironment.Optional {
			currentValue := os.Getenv(envVar.Name)
			if currentValue != "" {
				fmt.Printf("  %s: %s %s\n", Bold(envVar.Name), envVar.Description, Gray(fmt.Sprintf("(current: %s)", currentValue)))
			} else {
				fmt.Printf("  %s: %s %s\n", Bold(envVar.Name), envVar.Description, Gray(fmt.Sprintf("(default: %s)", envVar.Default)))
			}
		}
	}
}

// PackageUninstaller handles package uninstallation
type PackageUninstaller struct {
	AgentFieldHome string
	Force          bool
}

// UninstallPackage removes an installed package
func (pu *PackageUninstaller) UninstallPackage(packageName string) error {
	fmt.Printf("Uninstalling package: %s\n", packageName)

	// 1. Load registry
	registry, err := pu.loadRegistry()
	if err != nil {
		return fmt.Errorf("failed to load registry: %w", err)
	}

	// 2. Check if package exists
	agentNode, exists := registry.Installed[packageName]
	if !exists {
		return fmt.Errorf("package %s is not installed", packageName)
	}

	// 3. Check if package is running
	if agentNode.Status == "running" && !pu.Force {
		return fmt.Errorf("package %s is currently running (use --force to stop and uninstall)", packageName)
	}

	// 4. Stop the package if it's running
	if agentNode.Status == "running" {
		fmt.Printf("Stopping running agent node...\n")
		if err := pu.stopAgentNode(&agentNode); err != nil {
			fmt.Printf("Warning: Failed to stop agent node: %v\n", err)
		}
	}

	// 5. Remove package directory
	if err := os.RemoveAll(agentNode.Path); err != nil {
		return fmt.Errorf("failed to remove package directory: %w", err)
	}

	// 6. Remove log file
	if agentNode.Runtime.LogFile != "" {
		if err := os.Remove(agentNode.Runtime.LogFile); err != nil && !os.IsNotExist(err) {
			fmt.Printf("Warning: Failed to remove log file: %v\n", err)
		}
	}

	// 7. Update registry
	delete(registry.Installed, packageName)
	if err := pu.saveRegistry(registry); err != nil {
		return fmt.Errorf("failed to update registry: %w", err)
	}

	fmt.Printf("✓ Successfully uninstalled: %s\n", packageName)
	return nil
}

// stopAgentNode stops a running agent node
func (pu *PackageUninstaller) stopAgentNode(agentNode *InstalledPackage) error {
	if agentNode.Runtime.PID == nil {
		return fmt.Errorf("no PID found for agent node")
	}

	// Find and kill the process
	process, err := os.FindProcess(*agentNode.Runtime.PID)
	if err != nil {
		return fmt.Errorf("failed to find process: %w", err)
	}

	if err := process.Kill(); err != nil {
		return fmt.Errorf("failed to kill process: %w", err)
	}

	return nil
}

// loadRegistry loads the installation registry
func (pu *PackageUninstaller) loadRegistry() (*InstallationRegistry, error) {
	registryPath := filepath.Join(pu.AgentFieldHome, "installed.yaml")

	registry := &InstallationRegistry{
		Installed: make(map[string]InstalledPackage),
	}

	if data, err := os.ReadFile(registryPath); err == nil {
		if err := yaml.Unmarshal(data, registry); err != nil {
			return nil, fmt.Errorf("failed to parse registry: %w", err)
		}
	}

	return registry, nil
}

// saveRegistry saves the installation registry
func (pu *PackageUninstaller) saveRegistry(registry *InstallationRegistry) error {
	registryPath := filepath.Join(pu.AgentFieldHome, "installed.yaml")

	data, err := yaml.Marshal(registry)
	if err != nil {
		return fmt.Errorf("failed to marshal registry: %w", err)
	}

	if err := os.WriteFile(registryPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write registry: %w", err)
	}

	return nil
}

// validatePackage checks if the package has required files.
func (pi *PackageInstaller) validatePackage(sourcePath string) error {
	return ValidatePackage(sourcePath)
}

// ValidatePackage checks that a directory is an installable agent node: it must
// have an agentfield-package.yaml and declare how to start — either a manifest
// entrypoint.start (e.g. "python -m pr_af.app") or a top-level main.py. Real
// nodes use a module entrypoint and have no main.py, so main.py is not required.
func ValidatePackage(sourcePath string) error {
	packageYamlPath := filepath.Join(sourcePath, "agentfield-package.yaml")
	if _, err := os.Stat(packageYamlPath); os.IsNotExist(err) {
		return fmt.Errorf("agentfield-package.yaml not found in %s", sourcePath)
	}

	metadata, err := ParsePackageMetadata(sourcePath)
	if err != nil {
		return err
	}
	if metadata.Entrypoint.Start != "" {
		return nil
	}
	mainPyPath := filepath.Join(sourcePath, "main.py")
	if _, err := os.Stat(mainPyPath); os.IsNotExist(err) {
		return fmt.Errorf("package must declare entrypoint.start in agentfield-package.yaml or contain a main.py")
	}

	return nil
}

// StartCommand returns the tokens used to launch the node. It prefers the
// manifest entrypoint.start and falls back to "python <main>" (default main.py).
func (m *PackageMetadata) StartCommand() []string {
	if strings.TrimSpace(m.Entrypoint.Start) != "" {
		return strings.Fields(m.Entrypoint.Start)
	}
	main := m.Main
	if main == "" {
		main = "main.py"
	}
	return []string{"python", main}
}

// NodeDepName extracts the installed package name from a node dependency
// reference such as "af://registry/<name>@v" or a git URL. Returns "" when the
// name cannot be derived from the reference alone.
func NodeDepName(ref string) string {
	const afPrefix = "af://registry/"
	if strings.HasPrefix(ref, afPrefix) {
		spec := strings.TrimPrefix(ref, afPrefix)
		if at := strings.Index(spec, "@"); at >= 0 {
			spec = spec[:at]
		}
		return strings.Trim(spec, "/")
	}
	// Git URL: derive the repo name (last path segment, sans .git).
	trimmed := strings.TrimSuffix(strings.TrimSuffix(ref, "/"), ".git")
	if idx := strings.LastIndexAny(trimmed, "/:"); idx >= 0 {
		return trimmed[idx+1:]
	}
	return ""
}

// HealthcheckPath returns the readiness path, defaulting to "/health".
func (m *PackageMetadata) HealthcheckPath() string {
	if p := strings.TrimSpace(m.Entrypoint.Healthcheck); p != "" {
		return p
	}
	return "/health"
}

// ParsePackageMetadata parses agentfield-package.yaml from a package directory.
func ParsePackageMetadata(dir string) (*PackageMetadata, error) {
	packageYamlPath := filepath.Join(dir, "agentfield-package.yaml")

	data, err := os.ReadFile(packageYamlPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read agentfield-package.yaml: %w", err)
	}

	var metadata PackageMetadata
	if err := yaml.Unmarshal(data, &metadata); err != nil {
		return nil, fmt.Errorf("failed to parse agentfield-package.yaml: %w", err)
	}

	// Validate required fields
	if metadata.Name == "" {
		return nil, fmt.Errorf("package name is required in agentfield-package.yaml")
	}
	if metadata.Version == "" {
		return nil, fmt.Errorf("package version is required in agentfield-package.yaml")
	}
	if metadata.Main == "" {
		metadata.Main = "main.py" // Default
	}

	return &metadata, nil
}

// parsePackageMetadata parses the agentfield-package.yaml file.
func (pi *PackageInstaller) parsePackageMetadata(sourcePath string) (*PackageMetadata, error) {
	return ParsePackageMetadata(sourcePath)
}

// isPackageInstalled checks if a package is already installed
func (pi *PackageInstaller) isPackageInstalled(packageName string) bool {
	registryPath := filepath.Join(pi.AgentFieldHome, "installed.yaml")
	registry := &InstallationRegistry{
		Installed: make(map[string]InstalledPackage),
	}

	if data, err := os.ReadFile(registryPath); err == nil {
		if err := yaml.Unmarshal(data, registry); err != nil {
			return false
		}
	}

	_, exists := registry.Installed[packageName]
	return exists
}

// copyPackage copies all files from source to destination
func (pi *PackageInstaller) copyPackage(sourcePath, destPath string) error {
	// Remove existing destination if it exists
	if err := os.RemoveAll(destPath); err != nil {
		return fmt.Errorf("failed to remove existing package: %w", err)
	}

	// Create destination directory
	if err := os.MkdirAll(destPath, 0755); err != nil {
		return fmt.Errorf("failed to create destination directory: %w", err)
	}

	// Copy all files from source to destination
	return filepath.Walk(sourcePath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Calculate relative path
		relPath, err := filepath.Rel(sourcePath, path)
		if err != nil {
			return err
		}

		// Skip VCS, build artifacts, local venvs and plaintext secrets so they
		// never get copied into ~/.agentfield/packages.
		if shouldSkipCopy(relPath, info) {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		destFilePath := filepath.Join(destPath, relPath)

		if info.IsDir() {
			return os.MkdirAll(destFilePath, info.Mode())
		}

		// Copy file
		return pi.copyFile(path, destFilePath)
	})
}

// copyExcludedNames are directory/file names skipped during package copy.
var copyExcludedNames = map[string]bool{
	".git":          true,
	"venv":          true,
	".venv":         true,
	"__pycache__":   true,
	".env":          true,
	"node_modules":  true,
	".mypy_cache":   true,
	".pytest_cache": true,
}

// ShouldSkipCopy reports whether a walked path should be excluded when copying
// a package into ~/.agentfield/packages (VCS, venvs, caches, plaintext secrets).
func ShouldSkipCopy(relPath string, info os.FileInfo) bool {
	return shouldSkipCopy(relPath, info)
}

// shouldSkipCopy reports whether a walked path should be excluded from the copy.
func shouldSkipCopy(relPath string, info os.FileInfo) bool {
	if relPath == "." {
		return false
	}
	base := filepath.Base(relPath)
	if copyExcludedNames[base] {
		return true
	}
	// Skip stray .env.* local overrides but keep .env.example.
	if strings.HasPrefix(base, ".env.") && base != ".env.example" {
		return true
	}
	return false
}

// copyFile copies a single file from src to dst
func (pi *PackageInstaller) copyFile(src, dst string) error {
	sourceFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	destFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer destFile.Close()

	_, err = io.Copy(destFile, sourceFile)
	return err
}

// installDependencies installs package dependencies
func (pi *PackageInstaller) installDependencies(packagePath string, metadata *PackageMetadata) error {
	return InstallPythonDependencies(packagePath, metadata.Dependencies.Python, metadata.Dependencies.System)
}

// InstallPythonDependencies sets up a per-package virtual environment and
// installs the node's Python dependencies. A venv is created when the package
// has a requirements.txt, a pyproject.toml, or manifest-declared Python deps.
// Install sources, in order: requirements.txt, `pip install .` for a
// pyproject.toml/setup.py project, then any manifest-declared packages.
func InstallPythonDependencies(packagePath string, pyDeps, systemDeps []string) error {
	hasReq := fileExistsAt(packagePath, "requirements.txt")
	hasProject := fileExistsAt(packagePath, "pyproject.toml") || fileExistsAt(packagePath, "setup.py")

	if hasReq || hasProject || len(pyDeps) > 0 {
		venvPath := filepath.Join(packagePath, "venv")

		cmd := exec.Command("python3", "-m", "venv", venvPath)
		if _, err := cmd.CombinedOutput(); err != nil {
			cmd = exec.Command("python", "-m", "venv", venvPath)
			if output, err := cmd.CombinedOutput(); err != nil {
				return fmt.Errorf("failed to create virtual environment: %w\nOutput: %s", err, output)
			}
		}

		pipPath := filepath.Join(venvPath, "bin", "pip")
		if _, err := os.Stat(pipPath); err != nil {
			pipPath = filepath.Join(venvPath, "Scripts", "pip.exe") // Windows
		}

		// Upgrade pip first (ignore failures)
		_, _ = exec.Command(pipPath, "install", "--upgrade", "pip").CombinedOutput()

		// requirements.txt
		if hasReq {
			cmd = exec.Command(pipPath, "install", "-r", "requirements.txt")
			cmd.Dir = packagePath
			if output, err := cmd.CombinedOutput(); err != nil {
				return fmt.Errorf("failed to install requirements.txt dependencies: %w\nOutput: %s", err, output)
			}
		}

		// pyproject.toml / setup.py project (installs the project and its deps)
		if hasProject {
			cmd = exec.Command(pipPath, "install", ".")
			cmd.Dir = packagePath
			if output, err := cmd.CombinedOutput(); err != nil {
				return fmt.Errorf("failed to install project (pip install .): %w\nOutput: %s", err, output)
			}
		}

		// Manifest-declared Python packages
		for _, dep := range pyDeps {
			cmd = exec.Command(pipPath, "install", dep)
			cmd.Dir = packagePath
			if output, err := cmd.CombinedOutput(); err != nil {
				return fmt.Errorf("failed to install dependency %s: %w\nOutput: %s", dep, err, output)
			}
		}
	}

	for _, dep := range systemDeps {
		fmt.Printf("System dependency required: %s (please install manually)\n", dep)
	}

	return nil
}

// fileExistsAt reports whether name exists directly under dir.
func fileExistsAt(dir, name string) bool {
	_, err := os.Stat(filepath.Join(dir, name))
	return err == nil
}

// hasRequirementsFile checks if requirements.txt exists
func (pi *PackageInstaller) hasRequirementsFile(packagePath string) bool {
	requirementsPath := filepath.Join(packagePath, "requirements.txt")
	_, err := os.Stat(requirementsPath)
	return err == nil
}

// updateRegistry updates the installation registry with the new package
func (pi *PackageInstaller) updateRegistry(metadata *PackageMetadata, sourcePath, destPath string) error {
	registryPath := filepath.Join(pi.AgentFieldHome, "installed.yaml")

	// Load existing registry or create new one
	registry := &InstallationRegistry{
		Installed: make(map[string]InstalledPackage),
	}

	if data, err := os.ReadFile(registryPath); err == nil {
		if err := yaml.Unmarshal(data, registry); err != nil {
			return fmt.Errorf("failed to parse registry: %w", err)
		}
	}

	// Ensure logs directory exists before setting LogFile path
	logsDir := filepath.Join(pi.AgentFieldHome, "logs")
	if err := os.MkdirAll(logsDir, 0755); err != nil {
		return fmt.Errorf("failed to create logs directory: %w", err)
	}
	fmt.Printf("📁 Created logs directory: %s\n", logsDir)

	// Add/update package entry
	registry.Installed[metadata.Name] = InstalledPackage{
		Name:        metadata.Name,
		Version:     metadata.Version,
		Description: metadata.Description,
		Path:        destPath,
		Source:      "local",
		SourcePath:  sourcePath,
		InstalledAt: time.Now().Format(time.RFC3339),
		Status:      "stopped",
		Runtime: RuntimeInfo{
			Port:      nil,
			PID:       nil,
			StartedAt: nil,
			LogFile:   filepath.Join(pi.AgentFieldHome, "logs", metadata.Name+".log"),
		},
	}

	// Save registry
	data, err := yaml.Marshal(registry)
	if err != nil {
		return fmt.Errorf("failed to marshal registry: %w", err)
	}

	// Ensure directory exists
	if err := os.MkdirAll(filepath.Dir(registryPath), 0755); err != nil {
		return fmt.Errorf("failed to create registry directory: %w", err)
	}

	if err := os.WriteFile(registryPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write registry: %w", err)
	}

	return nil
}
