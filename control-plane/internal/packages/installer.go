package packages

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"

	"github.com/fatih/color"
	"gopkg.in/yaml.v3"
)

// UserEnvironmentVar represents a user-configurable environment variable
type UserEnvironmentVar struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
	Type        string `yaml:"type"`        // "string", "secret", "integer", "boolean", "float"
	Default     string `yaml:"default"`
	Optional    bool   `yaml:"optional"`
	Validation  string `yaml:"validation"`  // regex pattern
}

// UserEnvironmentConfig represents user-configurable environment variables
type UserEnvironmentConfig struct {
	Required []UserEnvironmentVar `yaml:"required"`
	Optional []UserEnvironmentVar `yaml:"optional"`
}

// PackageMetadata represents the structure of haxen-package.yaml
type PackageMetadata struct {
	Name            string                 `yaml:"name"`
	Version         string                 `yaml:"version"`
	Description     string                 `yaml:"description"`
	Author          string                 `yaml:"author"`
	Type            string                 `yaml:"type"`
	Main            string                 `yaml:"main"`
	AgentNode       AgentNodeConfig        `yaml:"agent_node"`
	Dependencies    DependencyConfig       `yaml:"dependencies"`
	Capabilities    CapabilityConfig       `yaml:"capabilities"`
	UserEnvironment UserEnvironmentConfig  `yaml:"user_environment"`
	Metadata        map[string]interface{} `yaml:"metadata"`
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
	HaxenHome string
	Verbose   bool
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
	destPath := filepath.Join(pi.HaxenHome, "packages", metadata.Name)
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
	
	fmt.Printf("\n%s %s\n", Blue("→"), Bold(fmt.Sprintf("Run: haxen run %s", metadata.Name)))

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
			fmt.Printf("  %s\n", Cyan(fmt.Sprintf("haxen config %s --set %s=your-value-here", metadata.Name, envVar.Name)))
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
	HaxenHome string
	Force     bool
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
	registryPath := filepath.Join(pu.HaxenHome, "installed.yaml")

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
	registryPath := filepath.Join(pu.HaxenHome, "installed.yaml")

	data, err := yaml.Marshal(registry)
	if err != nil {
		return fmt.Errorf("failed to marshal registry: %w", err)
	}

	if err := os.WriteFile(registryPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write registry: %w", err)
	}

	return nil
}

// validatePackage checks if the package has required files
func (pi *PackageInstaller) validatePackage(sourcePath string) error {
	// Check if haxen-package.yaml exists
	packageYamlPath := filepath.Join(sourcePath, "haxen-package.yaml")
	if _, err := os.Stat(packageYamlPath); os.IsNotExist(err) {
		return fmt.Errorf("haxen-package.yaml not found in %s", sourcePath)
	}

	// Check if main.py exists
	mainPyPath := filepath.Join(sourcePath, "main.py")
	if _, err := os.Stat(mainPyPath); os.IsNotExist(err) {
		return fmt.Errorf("main.py not found in %s", sourcePath)
	}

	return nil
}

// parsePackageMetadata parses the haxen-package.yaml file
func (pi *PackageInstaller) parsePackageMetadata(sourcePath string) (*PackageMetadata, error) {
	packageYamlPath := filepath.Join(sourcePath, "haxen-package.yaml")

	data, err := os.ReadFile(packageYamlPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read haxen-package.yaml: %w", err)
	}

	var metadata PackageMetadata
	if err := yaml.Unmarshal(data, &metadata); err != nil {
		return nil, fmt.Errorf("failed to parse haxen-package.yaml: %w", err)
	}

	// Validate required fields
	if metadata.Name == "" {
		return nil, fmt.Errorf("package name is required in haxen-package.yaml")
	}
	if metadata.Version == "" {
		return nil, fmt.Errorf("package version is required in haxen-package.yaml")
	}
	if metadata.Main == "" {
		metadata.Main = "main.py" // Default
	}

	return &metadata, nil
}

// isPackageInstalled checks if a package is already installed
func (pi *PackageInstaller) isPackageInstalled(packageName string) bool {
	registryPath := filepath.Join(pi.HaxenHome, "installed.yaml")
	registry := &InstallationRegistry{
		Installed: make(map[string]InstalledPackage),
	}

	if data, err := os.ReadFile(registryPath); err == nil {
		yaml.Unmarshal(data, registry)
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

		destFilePath := filepath.Join(destPath, relPath)

		if info.IsDir() {
			return os.MkdirAll(destFilePath, info.Mode())
		}

		// Copy file
		return pi.copyFile(path, destFilePath)
	})
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
	// Install Python dependencies in a virtual environment
	if len(metadata.Dependencies.Python) > 0 || pi.hasRequirementsFile(packagePath) {
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

		// Install dependencies from haxen-package.yaml
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
func (pi *PackageInstaller) hasRequirementsFile(packagePath string) bool {
	requirementsPath := filepath.Join(packagePath, "requirements.txt")
	_, err := os.Stat(requirementsPath)
	return err == nil
}

// updateRegistry updates the installation registry with the new package
func (pi *PackageInstaller) updateRegistry(metadata *PackageMetadata, sourcePath, destPath string) error {
	registryPath := filepath.Join(pi.HaxenHome, "installed.yaml")

	// Load existing registry or create new one
	registry := &InstallationRegistry{
		Installed: make(map[string]InstalledPackage),
	}

	if data, err := os.ReadFile(registryPath); err == nil {
		yaml.Unmarshal(data, registry)
	}

	// Ensure logs directory exists before setting LogFile path
	logsDir := filepath.Join(pi.HaxenHome, "logs")
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
			LogFile:   filepath.Join(pi.HaxenHome, "logs", metadata.Name+".log"),
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
