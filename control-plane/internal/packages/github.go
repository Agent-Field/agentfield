package packages

import (
	"archive/zip"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/your-org/haxen/control-plane/internal/logger"
	"gopkg.in/yaml.v3"
)

// GitHubPackageInfo represents parsed GitHub package information
type GitHubPackageInfo struct {
	Owner      string
	Repo       string
	Ref        string // branch, tag, or commit
	FullURL    string
	ArchiveURL string
}

// GitHubInstaller handles GitHub package installation
type GitHubInstaller struct {
	HaxenHome string
	Verbose   bool
}

// newSpinner creates a new spinner with the given message
func (gi *GitHubInstaller) newSpinner(message string) *Spinner {
	return &Spinner{
		message: message,
		done:    make(chan bool),
	}
}

// GitHub URL patterns
var (
	// github.com/owner/repo[@ref]
	githubFullPattern = regexp.MustCompile(`^github\.com/([^/\.]+)/([^/@\.]+)(?:@(.+))?$`)
	// owner/repo[@ref] (but not starting with . or containing /)
	githubShortPattern = regexp.MustCompile(`^([^/\.][^/]*)/([^/@\.]+)(?:@(.+))?$`)
	// https://github.com/owner/repo[@ref]
	githubHTTPSPattern = regexp.MustCompile(`^https://github\.com/([^/]+)/([^/@]+)(?:@(.+))?$`)
)

// IsGitHubURL checks if the given string is a GitHub URL
func IsGitHubURL(url string) bool {
	return githubFullPattern.MatchString(url) ||
		githubShortPattern.MatchString(url) ||
		githubHTTPSPattern.MatchString(url)
}

// ParseGitHubURL parses a GitHub URL into components
func ParseGitHubURL(url string) (*GitHubPackageInfo, error) {
	var owner, repo, ref string
	var matches []string

	// Try different patterns
	if matches = githubFullPattern.FindStringSubmatch(url); matches != nil {
		owner, repo = matches[1], matches[2]
		if len(matches) > 3 {
			ref = matches[3]
		}
	} else if matches = githubShortPattern.FindStringSubmatch(url); matches != nil {
		owner, repo = matches[1], matches[2]
		if len(matches) > 3 {
			ref = matches[3]
		}
	} else if matches = githubHTTPSPattern.FindStringSubmatch(url); matches != nil {
		owner, repo = matches[1], matches[2]
		if len(matches) > 3 {
			ref = matches[3]
		}
	} else {
		return nil, fmt.Errorf("invalid GitHub URL format: %s", url)
	}

	// Default to main branch if no ref specified
	if ref == "" {
		ref = "main"
	}

	info := &GitHubPackageInfo{
		Owner:   owner,
		Repo:    repo,
		Ref:     ref,
		FullURL: fmt.Sprintf("https://github.com/%s/%s", owner, repo),
	}

	// Construct archive URL based on ref type
	if strings.HasPrefix(ref, "v") || isCommitHash(ref) {
		// Tag or commit
		info.ArchiveURL = fmt.Sprintf("https://github.com/%s/%s/archive/%s.zip", owner, repo, ref)
	} else {
		// Branch
		info.ArchiveURL = fmt.Sprintf("https://github.com/%s/%s/archive/refs/heads/%s.zip", owner, repo, ref)
	}

	return info, nil
}

// isCommitHash checks if a string looks like a commit hash
func isCommitHash(s string) bool {
	if len(s) < 7 || len(s) > 40 {
		return false
	}
	for _, c := range s {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}
	return true
}

// InstallFromGitHub installs a package from GitHub
func (gi *GitHubInstaller) InstallFromGitHub(githubURL string, force bool) error {
	// Parse GitHub URL
	info, err := ParseGitHubURL(githubURL)
	if err != nil {
		return fmt.Errorf("failed to parse GitHub URL: %w", err)
	}

	logger.Logger.Info().Msgf("Installing %s from GitHub...", Bold(fmt.Sprintf("%s/%s@%s", info.Owner, info.Repo, info.Ref)))
	logger.Logger.Info().Msgf("  %s %s", Gray("Repository:"), info.FullURL)

	// 1. Download archive
	spinner := gi.newSpinner("Downloading package from GitHub")
	spinner.Start()
	
	tempDir, err := gi.downloadAndExtract(info)
	if err != nil {
		spinner.Error("Failed to download package")
		return fmt.Errorf("failed to download package: %w", err)
	}
	defer os.RemoveAll(tempDir) // Clean up temp directory
	
	spinner.Success("Package downloaded and extracted")

	// 2. Validate package structure
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

	// Update registry with GitHub source information
	if err := gi.updateRegistryWithGitHub(metadata, info, packagePath, destPath); err != nil {
		return fmt.Errorf("failed to update registry: %w", err)
	}

	logger.Logger.Info().Msgf("%s Installed %s v%s from GitHub", Green(StatusSuccess), Bold(metadata.Name), Gray(metadata.Version))
	logger.Logger.Info().Msgf("  %s %s", Gray("Source:"), fmt.Sprintf("%s/%s@%s", info.Owner, info.Repo, info.Ref))
	logger.Logger.Info().Msgf("  %s %s", Gray("Location:"), destPath)
	
	// Check for required environment variables
	installer.checkEnvironmentVariables(metadata)
	
	logger.Logger.Info().Msgf("\n%s %s", Blue("→"), Bold(fmt.Sprintf("Run: haxen run %s", metadata.Name)))

	return nil
}

// downloadAndExtract downloads and extracts the GitHub archive
func (gi *GitHubInstaller) downloadAndExtract(info *GitHubPackageInfo) (string, error) {
	// Create temporary directory
	tempDir, err := os.MkdirTemp("", "haxen-github-install-")
	if err != nil {
		return "", fmt.Errorf("failed to create temp directory: %w", err)
	}

	// Download archive
	resp, err := http.Get(info.ArchiveURL)
	if err != nil {
		os.RemoveAll(tempDir)
		return "", fmt.Errorf("failed to download archive: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		os.RemoveAll(tempDir)
		if resp.StatusCode == http.StatusNotFound {
			return "", fmt.Errorf("repository or reference not found: %s/%s@%s", info.Owner, info.Repo, info.Ref)
		}
		return "", fmt.Errorf("failed to download archive: HTTP %d", resp.StatusCode)
	}

	// Save to temporary file
	zipPath := filepath.Join(tempDir, "archive.zip")
	zipFile, err := os.Create(zipPath)
	if err != nil {
		os.RemoveAll(tempDir)
		return "", fmt.Errorf("failed to create zip file: %w", err)
	}

	_, err = io.Copy(zipFile, resp.Body)
	zipFile.Close()
	if err != nil {
		os.RemoveAll(tempDir)
		return "", fmt.Errorf("failed to save archive: %w", err)
	}

	// Extract archive
	extractDir := filepath.Join(tempDir, "extracted")
	if err := gi.extractZip(zipPath, extractDir); err != nil {
		os.RemoveAll(tempDir)
		return "", fmt.Errorf("failed to extract archive: %w", err)
	}

	return extractDir, nil
}

// extractZip extracts a zip file to the specified directory
func (gi *GitHubInstaller) extractZip(src, dest string) error {
	reader, err := zip.OpenReader(src)
	if err != nil {
		return err
	}
	defer reader.Close()

	if err := os.MkdirAll(dest, 0755); err != nil {
		return err
	}

	for _, file := range reader.File {
		path := filepath.Join(dest, file.Name)
		
		// Security check: ensure path is within destination
		if !strings.HasPrefix(path, filepath.Clean(dest)+string(os.PathSeparator)) {
			return fmt.Errorf("invalid file path: %s", file.Name)
		}

		if file.FileInfo().IsDir() {
			os.MkdirAll(path, file.FileInfo().Mode())
			continue
		}

		// Create parent directories
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			return err
		}

		// Extract file
		fileReader, err := file.Open()
		if err != nil {
			return err
		}

		targetFile, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, file.FileInfo().Mode())
		if err != nil {
			fileReader.Close()
			return err
		}

		_, err = io.Copy(targetFile, fileReader)
		fileReader.Close()
		targetFile.Close()
		if err != nil {
			return err
		}
	}

	return nil
}

// findPackageRoot finds the root directory containing haxen-package.yaml
func (gi *GitHubInstaller) findPackageRoot(extractDir string) (string, error) {
	var packageRoot string

	err := filepath.Walk(extractDir, func(path string, info os.FileInfo, err error) error {
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
func (gi *GitHubInstaller) parsePackageMetadata(packagePath string) (*PackageMetadata, error) {
	installer := &PackageInstaller{
		HaxenHome: gi.HaxenHome,
		Verbose:   gi.Verbose,
	}
	return installer.parsePackageMetadata(packagePath)
}

// updateRegistryWithGitHub updates the installation registry with GitHub source info
func (gi *GitHubInstaller) updateRegistryWithGitHub(metadata *PackageMetadata, info *GitHubPackageInfo, sourcePath, destPath string) error {
	registryPath := filepath.Join(gi.HaxenHome, "installed.yaml")

	// Load existing registry or create new one
	registry := &InstallationRegistry{
		Installed: make(map[string]InstalledPackage),
	}

	if data, err := os.ReadFile(registryPath); err == nil {
		yaml.Unmarshal(data, registry)
	}

	// Add/update package entry with GitHub information
	registry.Installed[metadata.Name] = InstalledPackage{
		Name:        metadata.Name,
		Version:     metadata.Version,
		Description: metadata.Description,
		Path:        destPath,
		Source:      "github",
		SourcePath:  fmt.Sprintf("%s/%s@%s", info.Owner, info.Repo, info.Ref),
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
