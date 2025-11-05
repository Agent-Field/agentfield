package packages

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/your-org/haxen/control-plane/internal/logger"
	"gopkg.in/yaml.v3"
)

// GitPackageInfo represents parsed Git package information
type GitPackageInfo struct {
	URL      string // Original URL provided by user
	Ref      string // branch, tag, or commit (optional)
	CloneURL string // URL for git clone (may be same as URL)
}

// GitInstaller handles Git package installation from any Git repository
type GitInstaller struct {
	HaxenHome string
	Verbose   bool
}

// newSpinner creates a new spinner with the given message
func (gi *GitInstaller) newSpinner(message string) *Spinner {
	return &Spinner{
		message: message,
		done:    make(chan bool),
	}
}

// IsGitURL checks if the given string is a Git URL
func IsGitURL(url string) bool {
	// Universal Git URL detection
	return strings.Contains(url, "github.com") ||
		strings.Contains(url, "gitlab.com") ||
		strings.Contains(url, "bitbucket.org") ||
		strings.Contains(url, "git.") ||
		strings.HasPrefix(url, "git@") ||
		strings.HasSuffix(url, ".git") ||
		isHTTPSGitURL(url)
}

// isHTTPSGitURL checks if it's an HTTPS URL that might be a Git repo
func isHTTPSGitURL(url string) bool {
	// Check if it's an HTTPS URL that might be a Git repo
	return strings.HasPrefix(url, "https://") && 
		strings.Contains(url, "/") && 
		!strings.HasSuffix(url, "/")
}

// ParseGitURL parses a Git URL into components
func ParseGitURL(url string) (*GitPackageInfo, error) {
	info := &GitPackageInfo{
		URL: url,
	}
	
	// Handle URLs with @ for branch/tag specification
	// e.g., https://github.com/owner/repo@branch
	// But not SSH URLs like git@github.com:owner/repo.git
	if strings.Contains(url, "@") && !strings.HasPrefix(url, "git@") {
		// Find the last @ that's not part of the domain
		parts := strings.Split(url, "@")
		if len(parts) >= 2 {
			// Check if the @ is part of authentication (like token:xxx@github.com)
			lastPart := parts[len(parts)-1]
			if !strings.Contains(lastPart, ".com") && !strings.Contains(lastPart, ".org") {
				// This @ is for branch/tag specification
				info.Ref = lastPart
				info.CloneURL = strings.Join(parts[:len(parts)-1], "@")
			} else {
				// This @ is part of authentication
				info.CloneURL = url
			}
		} else {
			info.CloneURL = url
		}
	} else {
		info.CloneURL = url
	}
	
	return info, nil
}

// checkGitAvailable checks if Git is available on the system
func checkGitAvailable() error {
	if _, err := exec.LookPath("git"); err != nil {
		return fmt.Errorf("git is required but not found in PATH\n\nPlease install Git:\n  • macOS: brew install git\n  • Ubuntu: sudo apt-get install git\n  • Windows: https://git-scm.com/download/win")
	}
	
	// Check git version (optional - ensure modern git)
	cmd := exec.Command("git", "--version")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("git installation appears to be broken")
	}
	
	return nil
}

// InstallFromGit installs a package from any Git repository
func (gi *GitInstaller) InstallFromGit(gitURL string, force bool) error {
	// Check if Git is available
	if err := checkGitAvailable(); err != nil {
		return err
	}

	// Parse Git URL
	info, err := ParseGitURL(gitURL)
	if err != nil {
		return fmt.Errorf("failed to parse Git URL: %w", err)
	}

	logger.Logger.Info().Msgf("Installing package from Git repository...")
	logger.Logger.Info().Msgf("  %s %s", Gray("Repository:"), info.URL)
	if info.Ref != "" {
		logger.Logger.Info().Msgf("  %s %s", Gray("Reference:"), info.Ref)
	}

	// 1. Clone repository
	spinner := gi.newSpinner("Cloning repository")
	spinner.Start()
	
	tempDir, err := gi.cloneRepository(info)
	if err != nil {
		spinner.Error("Failed to clone repository")
		return fmt.Errorf("failed to clone repository: %w", err)
	}
	defer os.RemoveAll(tempDir) // Always clean up
	
	spinner.Success("Repository cloned")

	// 2. Find and validate package structure
	spinner = gi.newSpinner("Validating package structure")
	spinner.Start()
	
	packagePath, err := gi.findPackageRoot(tempDir)
	if err != nil {
		spinner.Error("Invalid package structure")
		return fmt.Errorf("invalid package structure: %w", err)
	}
	
	spinner.Success("Package structure validated")

	// 3. Parse metadata to get package name
	metadata, err := gi.parsePackageMetadata(packagePath)
	if err != nil {
		return fmt.Errorf("failed to parse package metadata: %w", err)
	}

	// 4. Use existing installer for the rest
	installer := &PackageInstaller{
		HaxenHome: gi.HaxenHome,
		Verbose:   gi.Verbose,
	}

	// Check if already installed
	if !force && installer.isPackageInstalled(metadata.Name) {
		return fmt.Errorf("package %s already installed (use --force to reinstall)", metadata.Name)
	}

	// Install using existing flow
	destPath := filepath.Join(gi.HaxenHome, "packages", metadata.Name)
	
	spinner = gi.newSpinner("Setting up environment")
	spinner.Start()
	if err := installer.copyPackage(packagePath, destPath); err != nil {
		spinner.Error("Failed to copy package")
		return fmt.Errorf("failed to copy package: %w", err)
	}
	spinner.Success("Environment configured")

	spinner = gi.newSpinner("Installing dependencies")
	spinner.Start()
	if err := installer.installDependencies(destPath, metadata); err != nil {
		spinner.Error("Failed to install dependencies")
		return fmt.Errorf("failed to install dependencies: %w", err)
	}
	spinner.Success("Dependencies installed")

	// Update registry with Git source information
	if err := gi.updateRegistryWithGit(metadata, info, packagePath, destPath); err != nil {
		return fmt.Errorf("failed to update registry: %w", err)
	}

	logger.Logger.Info().Msgf("%s Installed %s v%s from Git", Green(StatusSuccess), Bold(metadata.Name), Gray(metadata.Version))
	logger.Logger.Info().Msgf("  %s %s", Gray("Source:"), info.URL)
	if info.Ref != "" {
		logger.Logger.Info().Msgf("  %s %s", Gray("Reference:"), info.Ref)
	}
	logger.Logger.Info().Msgf("  %s %s", Gray("Location:"), destPath)
	
	// Check for required environment variables
	installer.checkEnvironmentVariables(metadata)
	
	logger.Logger.Info().Msgf("\n%s %s", Blue("→"), Bold(fmt.Sprintf("Run: haxen run %s", metadata.Name)))

	return nil
}

// cloneRepository clones the Git repository with optimizations
func (gi *GitInstaller) cloneRepository(info *GitPackageInfo) (string, error) {
	// Create temporary directory
	tempDir, err := os.MkdirTemp("", "haxen-git-install-")
	if err != nil {
		return "", fmt.Errorf("failed to create temp directory: %w", err)
	}

	// Build git clone command with optimizations
	args := []string{"clone"}
	
	// Shallow clone for efficiency (only latest commit)
	args = append(args, "--depth", "1")
	
	// Clone specific branch/tag if specified
	if info.Ref != "" {
		args = append(args, "--branch", info.Ref)
	}
	
	// Add URLs
	args = append(args, info.CloneURL, tempDir)

	// Execute git clone
	cmd := exec.Command("git", args...)
	
	// Capture both stdout and stderr for better error messages
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	
	if gi.Verbose {
		logger.Logger.Info().Msgf("Executing: git %s", strings.Join(args, " "))
	}
	
	if err := cmd.Run(); err != nil {
		// Clean up temp directory on failure
		os.RemoveAll(tempDir)
		
		// Provide helpful error messages based on common failure scenarios
		stderrStr := stderr.String()
		
		if strings.Contains(stderrStr, "Authentication failed") || strings.Contains(stderrStr, "authentication failed") {
			return "", fmt.Errorf("authentication failed - please check your credentials\n\nFor private repositories, you can:\n  • Use SSH: git@github.com:owner/repo.git\n  • Use token: https://token:your_token@github.com/owner/repo\n  • Configure Git credentials: git config --global credential.helper")
		}
		if strings.Contains(stderrStr, "Repository not found") || strings.Contains(stderrStr, "repository not found") {
			return "", fmt.Errorf("repository not found - please check the URL and your access permissions")
		}
		if strings.Contains(stderrStr, "Remote branch") && strings.Contains(stderrStr, "not found") {
			return "", fmt.Errorf("branch/tag '%s' not found in repository", info.Ref)
		}
		if strings.Contains(stderrStr, "Could not resolve host") {
			return "", fmt.Errorf("could not resolve host - please check your internet connection and the repository URL")
		}
		
		return "", fmt.Errorf("git clone failed: %w\nError output: %s", err, stderrStr)
	}

	return tempDir, nil
}

// findPackageRoot finds the root directory containing haxen-package.yaml
func (gi *GitInstaller) findPackageRoot(cloneDir string) (string, error) {
	var packageRoot string

	err := filepath.Walk(cloneDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.Name() == "haxen-package.yaml" {
			packageRoot = filepath.Dir(path)
			return filepath.SkipDir // Found it, stop walking
		}

		return nil
	})

	if err != nil {
		return "", err
	}

	if packageRoot == "" {
		return "", fmt.Errorf("haxen-package.yaml not found in the repository")
	}

	// Also check for main.py
	mainPyPath := filepath.Join(packageRoot, "main.py")
	if _, err := os.Stat(mainPyPath); os.IsNotExist(err) {
		return "", fmt.Errorf("main.py not found in package root")
	}

	return packageRoot, nil
}

// parsePackageMetadata parses the haxen-package.yaml file (reuse from installer.go)
func (gi *GitInstaller) parsePackageMetadata(packagePath string) (*PackageMetadata, error) {
	installer := &PackageInstaller{
		HaxenHome: gi.HaxenHome,
		Verbose:   gi.Verbose,
	}
	return installer.parsePackageMetadata(packagePath)
}

// updateRegistryWithGit updates the installation registry with Git source info
func (gi *GitInstaller) updateRegistryWithGit(metadata *PackageMetadata, info *GitPackageInfo, sourcePath, destPath string) error {
	registryPath := filepath.Join(gi.HaxenHome, "installed.yaml")

	// Load existing registry or create new one
	registry := &InstallationRegistry{
		Installed: make(map[string]InstalledPackage),
	}

	if data, err := os.ReadFile(registryPath); err == nil {
		yaml.Unmarshal(data, registry)
	}

	// Determine source type based on URL
	sourceType := "git"
	if strings.Contains(info.URL, "github.com") {
		sourceType = "github"
	} else if strings.Contains(info.URL, "gitlab.com") {
		sourceType = "gitlab"
	} else if strings.Contains(info.URL, "bitbucket.org") {
		sourceType = "bitbucket"
	}

	// Build source path string
	sourcePathStr := info.URL
	if info.Ref != "" {
		sourcePathStr = fmt.Sprintf("%s@%s", info.URL, info.Ref)
	}

	// Add/update package entry with Git information
	registry.Installed[metadata.Name] = InstalledPackage{
		Name:        metadata.Name,
		Version:     metadata.Version,
		Description: metadata.Description,
		Path:        destPath,
		Source:      sourceType,
		SourcePath:  sourcePathStr,
		InstalledAt: time.Now().Format(time.RFC3339),
		Status:      "stopped",
		Runtime: RuntimeInfo{
			Port:      nil,
			PID:       nil,
			StartedAt: nil,
			LogFile:   filepath.Join(gi.HaxenHome, "logs", metadata.Name+".log"),
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
