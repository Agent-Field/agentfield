// agentfield/internal/core/services/package_service.go
package services

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"

	"github.com/fatih/color"
	"github.com/your-org/agentfield/control-plane/internal/core/domain"
	"github.com/your-org/agentfield/control-plane/internal/core/interfaces"
	"github.com/your-org/agentfield/control-plane/internal/packages"
	"gopkg.in/yaml.v3"
)

// DefaultPackageService implements the PackageService interface
type DefaultPackageService struct {
	registryStorage interfaces.RegistryStorage
	fileSystem      interfaces.FileSystemAdapter
	agentfieldHome  string
}

// NewPackageService creates a new package service instance
func NewPackageService(
	registryStorage interfaces.RegistryStorage,
	fileSystem interfaces.FileSystemAdapter,
	agentfieldHome string,
) interfaces.PackageService {
	return &DefaultPackageService{
		registryStorage: registryStorage,
		fileSystem:      fileSystem,
		agentfieldHome:  agentfieldHome,
	}
}

// InstallPackage installs a package from the given source
func (ps *DefaultPackageService) InstallPackage(source string, options domain.InstallOptions) error {
	// Check if it's a Git URL (GitHub, GitLab, Bitbucket, etc.)
	if packages.IsGitURL(source) {
		installer := &packages.GitInstaller{
			AgentFieldHome: ps.agentfieldHome,
			Verbose:        options.Verbose,
		}
		return installer.InstallFromGit(source, options.Force)
	}

	// Handle local package installation
	return ps.installLocalPackage(source, options.Force, options.Verbose)
}

// installLocalPackage installs a package from a local source path
func (ps *DefaultPackageService) installLocalPackage(sourcePath string, force bool, verbose bool) error {
	// Get package name first for better messaging
	metadata, err := ps.parsePackageMetadata(sourcePath)
	if err != nil {
		return fmt.Errorf("failed to parse package metadata: %w", err)
	}

	fmt.Printf("Installing %s...\n", metadata.Name)

	// 1. Validate source package
	spinner := ps.newSpinner("Validating package structure")
	spinner.Start()
	if err := ps.validatePackage(sourcePath); err != nil {
		spinner.Error("Package validation failed")
		return fmt.Errorf("package validation failed: %w", err)
	}
	spinner.Success("Package structure validated")

	// 2. Check if already installed
	if !force && ps.isPackageInstalled(metadata.Name) {
		return fmt.Errorf("package %s already installed (use --force to reinstall)", metadata.Name)
	}

	// 3. Copy package to global location
	destPath := filepath.Join(ps.agentfieldHome, "packages", metadata.Name)
	spinner = ps.newSpinner("Setting up environment")
	spinner.Start()
	if err := ps.copyPackage(sourcePath, destPath); err != nil {
		spinner.Error("Failed to copy package")
		return fmt.Errorf("failed to copy package: %w", err)
	}
	spinner.Success("Environment configured")

	// 4. Install dependencies
	spinner = ps.newSpinner("Installing dependencies")
	spinner.Start()
	if err := ps.installDependencies(destPath, metadata); err != nil {
		spinner.Error("Failed to install dependencies")
		return fmt.Errorf("failed to install dependencies: %w", err)
	}
	spinner.Success("Dependencies installed")

	// 5. Update installation registry
	if err := ps.updateRegistry(metadata, sourcePath, destPath); err != nil {
		return fmt.Errorf("failed to update registry: %w", err)
	}

	fmt.Printf("%s Installed %s v%s\n", ps.green(ps.statusSuccess()), ps.bold(metadata.Name), ps.gray(metadata.Version))
	fmt.Printf("  %s %s\n", ps.gray("Location:"), destPath)

	// 6. Check for required environment variables and provide guidance
	ps.checkEnvironmentVariables(metadata)

	fmt.Printf("\n%s %s\n", ps.blue("→"), ps.bold(fmt.Sprintf("Run: af run %s", metadata.Name)))

	return nil
}

// UninstallPackage removes an installed package
func (ps *DefaultPackageService) UninstallPackage(name string) error {
	return ps.uninstallPackage(name, false) // Default to non-force
}

// uninstallPackage removes an installed package with force option
func (ps *DefaultPackageService) uninstallPackage(packageName string, force bool) error {
	fmt.Printf("Uninstalling package: %s\n", packageName)

	// 1. Load registry
	registry, err := ps.loadRegistryDirect()
	if err != nil {
		return fmt.Errorf("failed to load registry: %w", err)
	}

	// 2. Check if package exists
	agentNode, exists := registry.Installed[packageName]
	if !exists {
		return fmt.Errorf("package %s is not installed", packageName)
	}

	// 3. Check if package is running
	if agentNode.Status == "running" && !force {
		return fmt.Errorf("package %s is currently running (use --force to stop and uninstall)", packageName)
	}

	// 4. Stop the package if it's running
	if agentNode.Status == "running" {
		fmt.Printf("Stopping running agent node...\n")
		if err := ps.stopAgentNode(&agentNode); err != nil {
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
	if err := ps.saveRegistry(registry); err != nil {
		return fmt.Errorf("failed to update registry: %w", err)
	}

	fmt.Printf("✓ Successfully uninstalled: %s\n", packageName)
	return nil
}

// stopAgentNode stops a running agent node
func (ps *DefaultPackageService) stopAgentNode(agentNode *packages.InstalledPackage) error {
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

// saveRegistry saves the installation registry
func (ps *DefaultPackageService) saveRegistry(registry *packages.InstallationRegistry) error {
	registryPath := filepath.Join(ps.agentfieldHome, "installed.yaml")

	data, err := yaml.Marshal(registry)
	if err != nil {
		return fmt.Errorf("failed to marshal registry: %w", err)
	}

	if err := os.WriteFile(registryPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write registry: %w", err)
	}

	return nil
}

// ListInstalledPackages returns a list of all installed packages
func (ps *DefaultPackageService) ListInstalledPackages() ([]domain.InstalledPackage, error) {
	// Load registry using existing packages logic for now
	// TODO: Eventually migrate to use registryStorage interface
	registry, err := ps.loadRegistryDirect()
	if err != nil {
		return nil, err
	}

	var domainPackages []domain.InstalledPackage
	for _, pkg := range registry.Installed {
		domainPackages = append(domainPackages, ps.convertToDomainPackage(pkg))
	}

	return domainPackages, nil
}

// GetPackageInfo returns information about a specific installed package
func (ps *DefaultPackageService) GetPackageInfo(name string) (*domain.InstalledPackage, error) {
	// Load registry using existing packages logic for now
	registry, err := ps.loadRegistryDirect()
	if err != nil {
		return nil, err
	}

	pkg, exists := registry.Installed[name]
	if !exists {
		return nil, fmt.Errorf("package %s is not installed", name)
	}

	domainPackage := ps.convertToDomainPackage(pkg)
	return &domainPackage, nil
}

// loadRegistryDirect loads the registry using direct file system access
// TODO: Eventually replace with registryStorage interface usage
func (ps *DefaultPackageService) loadRegistryDirect() (*packages.InstallationRegistry, error) {
	registryPath := filepath.Join(ps.agentfieldHome, "installed.yaml")

	registry := &packages.InstallationRegistry{
		Installed: make(map[string]packages.InstalledPackage),
	}

	if data, err := os.ReadFile(registryPath); err == nil {
		if err := yaml.Unmarshal(data, registry); err != nil {
			return nil, fmt.Errorf("failed to parse registry: %w", err)
		}
	}

	return registry, nil
}

// convertToDomainPackage converts packages.InstalledPackage to domain.InstalledPackage
func (ps *DefaultPackageService) convertToDomainPackage(pkg packages.InstalledPackage) domain.InstalledPackage {
	// Parse the installed_at time
	var installedAt time.Time
	if pkg.InstalledAt != "" {
		if parsed, err := time.Parse(time.RFC3339, pkg.InstalledAt); err == nil {
			installedAt = parsed
		}
	}

	// Convert environment variables (for now, empty map as packages don't store this)
	environment := make(map[string]string)

	return domain.InstalledPackage{
		Name:        pkg.Name,
		Version:     pkg.Version,
		Path:        pkg.Path,
		Environment: environment,
		InstalledAt: installedAt,
	}
}

// Helper methods moved from packages/installer.go

// Professional CLI status symbols
const (
	statusSuccess = "✓"
	statusError   = "✗"
)

// Spinner characters for progress indication
var spinnerChars = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

// Color functions for professional output
var (
	green  = color.New(color.FgGreen).SprintFunc()
	red    = color.New(color.FgRed).SprintFunc()
	yellow = color.New(color.FgYellow).SprintFunc()
	blue   = color.New(color.FgBlue).SprintFunc()
	cyan   = color.New(color.FgCyan).SprintFunc()
	gray   = color.New(color.FgHiBlack).SprintFunc()
	bold   = color.New(color.Bold).SprintFunc()
)

// Spinner represents a CLI spinner for progress indication
type Spinner struct {
	message string
	active  bool
	mu      sync.Mutex
	done    chan bool
}

// Color helper methods
func (ps *DefaultPackageService) green(text string) string  { return green(text) }
func (ps *DefaultPackageService) red(text string) string    { return red(text) }
func (ps *DefaultPackageService) yellow(text string) string { return yellow(text) }
func (ps *DefaultPackageService) blue(text string) string   { return blue(text) }
func (ps *DefaultPackageService) cyan(text string) string   { return cyan(text) }
func (ps *DefaultPackageService) gray(text string) string   { return gray(text) }
func (ps *DefaultPackageService) bold(text string) string   { return bold(text) }
func (ps *DefaultPackageService) statusSuccess() string     { return statusSuccess }
func (ps *DefaultPackageService) statusError() string       { return statusError }

// newSpinner creates a new spinner with the given message
func (ps *DefaultPackageService) newSpinner(message string) *Spinner {
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
	fmt.Printf("  %s %s\n", green(statusSuccess), message)
}

// Error stops the spinner and shows an error message
func (s *Spinner) Error(message string) {
	s.Stop()
	fmt.Printf("  %s %s\n", red(statusError), message)
}

// validatePackage checks if the package has required files
func (ps *DefaultPackageService) validatePackage(sourcePath string) error {
	// Check if agentfield-package.yaml exists
	packageYamlPath := filepath.Join(sourcePath, "agentfield-package.yaml")
	if _, err := os.Stat(packageYamlPath); os.IsNotExist(err) {
		return fmt.Errorf("agentfield-package.yaml not found in %s", sourcePath)
	}

	// Check if main.py exists
	mainPyPath := filepath.Join(sourcePath, "main.py")
	if _, err := os.Stat(mainPyPath); os.IsNotExist(err) {
		return fmt.Errorf("main.py not found in %s", sourcePath)
	}

	return nil
}

// parsePackageMetadata parses the agentfield-package.yaml file
func (ps *DefaultPackageService) parsePackageMetadata(sourcePath string) (*packages.PackageMetadata, error) {
	packageYamlPath := filepath.Join(sourcePath, "agentfield-package.yaml")

	data, err := os.ReadFile(packageYamlPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read agentfield-package.yaml: %w", err)
	}

	var metadata packages.PackageMetadata
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

// isPackageInstalled checks if a package is already installed
func (ps *DefaultPackageService) isPackageInstalled(packageName string) bool {
	registryPath := filepath.Join(ps.agentfieldHome, "installed.yaml")
	registry := &packages.InstallationRegistry{
		Installed: make(map[string]packages.InstalledPackage),
	}

	if data, err := os.ReadFile(registryPath); err == nil {
		yaml.Unmarshal(data, registry)
	}

	_, exists := registry.Installed[packageName]
	return exists
}

// copyPackage copies all files from source to destination
func (ps *DefaultPackageService) copyPackage(sourcePath, destPath string) error {
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

		destFilePath := filepath.Join(destPath, relPath)

		if info.IsDir() {
			return os.MkdirAll(destFilePath, info.Mode())
		}

		// Copy file
		return ps.copyFile(path, destFilePath)
	})
}

// copyFile copies a single file from src to dst
func (ps *DefaultPackageService) copyFile(src, dst string) error {
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
func (ps *DefaultPackageService) installDependencies(packagePath string, metadata *packages.PackageMetadata) error {
	// Install Python dependencies in a virtual environment
	if len(metadata.Dependencies.Python) > 0 || ps.hasRequirementsFile(packagePath) {
		// Create virtual environment
		venvPath := filepath.Join(packagePath, "venv")

		cmd := exec.Command("python3", "-m", "venv", venvPath)
		if _, err := cmd.CombinedOutput(); err != nil {
			// Try with python if python3 fails
			cmd = exec.Command("python", "-m", "venv", venvPath)
			if output, err := cmd.CombinedOutput(); err != nil {
				return fmt.Errorf("failed to create virtual environment: %w\nOutput: %s", err, output)
			}
		}

		// Determine pip path
		var pipPath string
		if _, err := os.Stat(filepath.Join(venvPath, "bin", "pip")); err == nil {
			pipPath = filepath.Join(venvPath, "bin", "pip")
		} else {
			pipPath = filepath.Join(venvPath, "Scripts", "pip.exe") // Windows
		}

		// Upgrade pip first
		cmd = exec.Command(pipPath, "install", "--upgrade", "pip")
		if _, err := cmd.CombinedOutput(); err != nil {
			// Ignore pip upgrade failures
		}

		// Install from requirements.txt if it exists
		requirementsPath := filepath.Join(packagePath, "requirements.txt")
		if _, err := os.Stat(requirementsPath); err == nil {
			cmd = exec.Command(pipPath, "install", "-r", requirementsPath)
			cmd.Dir = packagePath
			if output, err := cmd.CombinedOutput(); err != nil {
				return fmt.Errorf("failed to install requirements.txt dependencies: %w\nOutput: %s", err, output)
			}
		}

		// Install dependencies from agentfield-package.yaml
		if len(metadata.Dependencies.Python) > 0 {
			for _, dep := range metadata.Dependencies.Python {
				cmd = exec.Command(pipPath, "install", dep)
				cmd.Dir = packagePath
				if output, err := cmd.CombinedOutput(); err != nil {
					return fmt.Errorf("failed to install dependency %s: %w\nOutput: %s", dep, err, output)
				}
			}
		}
	}

	// Install system dependencies (if any)
	for _, dep := range metadata.Dependencies.System {
		fmt.Printf("System dependency required: %s (please install manually)\n", dep)
	}

	return nil
}

// hasRequirementsFile checks if requirements.txt exists
func (ps *DefaultPackageService) hasRequirementsFile(packagePath string) bool {
	requirementsPath := filepath.Join(packagePath, "requirements.txt")
	_, err := os.Stat(requirementsPath)
	return err == nil
}

// updateRegistry updates the installation registry with the new package
func (ps *DefaultPackageService) updateRegistry(metadata *packages.PackageMetadata, sourcePath, destPath string) error {
	registryPath := filepath.Join(ps.agentfieldHome, "installed.yaml")

	// Load existing registry or create new one
	registry := &packages.InstallationRegistry{
		Installed: make(map[string]packages.InstalledPackage),
	}

	if data, err := os.ReadFile(registryPath); err == nil {
		yaml.Unmarshal(data, registry)
	}

	// Add/update package entry
	registry.Installed[metadata.Name] = packages.InstalledPackage{
		Name:        metadata.Name,
		Version:     metadata.Version,
		Description: metadata.Description,
		Path:        destPath,
		Source:      "local",
		SourcePath:  sourcePath,
		InstalledAt: time.Now().Format(time.RFC3339),
		Status:      "stopped",
		Runtime: packages.RuntimeInfo{
			Port:      nil,
			PID:       nil,
			StartedAt: nil,
			LogFile:   filepath.Join(ps.agentfieldHome, "logs", metadata.Name+".log"),
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

// checkEnvironmentVariables checks for required environment variables and provides setup guidance
func (ps *DefaultPackageService) checkEnvironmentVariables(metadata *packages.PackageMetadata) {
	if len(metadata.UserEnvironment.Required) == 0 && len(metadata.UserEnvironment.Optional) == 0 {
		return // No user environment variables configured
	}

	// Check required environment variables
	missingRequired := []packages.UserEnvironmentVar{}
	for _, envVar := range metadata.UserEnvironment.Required {
		if os.Getenv(envVar.Name) == "" {
			missingRequired = append(missingRequired, envVar)
		}
	}

	if len(missingRequired) > 0 {
		fmt.Printf("\n%s %s\n", ps.yellow("⚠"), ps.bold("Missing required environment variables:"))
		for _, envVar := range missingRequired {
			fmt.Printf("  %s\n", ps.cyan(fmt.Sprintf("af config %s --set %s=your-value-here", metadata.Name, envVar.Name)))
		}
	}

	// Show optional environment variables if any
	if len(metadata.UserEnvironment.Optional) > 0 {
		fmt.Printf("\n%s %s\n", ps.gray("ℹ"), ps.gray("Optional environment variables (with defaults):"))
		for _, envVar := range metadata.UserEnvironment.Optional {
			currentValue := os.Getenv(envVar.Name)
			if currentValue != "" {
				fmt.Printf("  %s: %s %s\n", ps.bold(envVar.Name), envVar.Description, ps.gray(fmt.Sprintf("(current: %s)", currentValue)))
			} else {
				fmt.Printf("  %s: %s %s\n", ps.bold(envVar.Name), envVar.Description, ps.gray(fmt.Sprintf("(default: %s)", envVar.Default)))
			}
		}
	}
}
