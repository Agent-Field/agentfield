package packages

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/ui"
	"gopkg.in/yaml.v3"
)

// GitPackageInfo represents parsed Git package information
type GitPackageInfo struct {
	URL      string // Original URL provided by user
	Ref      string // branch, tag, or commit (optional)
	CloneURL string // URL for git clone (may be same as URL)
	// Subdir is an optional path inside the repository whose
	// agentfield-package.yaml is the package to install — the `//` selector
	// in e.g. https://github.com/Agent-Field/pr-af//go. Lets one repo ship
	// several installable nodes (a Python root and a Go port side by side).
	Subdir string
}

// GitInstaller handles Git package installation from any Git repository
type GitInstaller struct {
	AgentFieldHome string
	Verbose        bool
	// Subdir optionally selects a package subdirectory within the cloned repo
	// (the `--path` flag). Empty means the historical root-first walk. When set,
	// the manifest MUST live at <clone>/<Subdir>/agentfield-package.yaml and that
	// subdirectory becomes the package root that is copied and installed. It
	// composes with an @ref pin on the URL, which is parsed independently.
	Subdir string

	// redirects counts how many superseded_by hops led here, bounding a cycle
	// (A superseded by B, B superseded by A) instead of cloning forever.
	redirects int
	// installedName records the package name this installer actually installed.
	// A superseded_by redirect needs it to hand the old package's node-scoped
	// secrets to the successor, whose name it cannot know in advance.
	installedName string
}

// InstalledName returns the package name installed by the most recent
// successful InstallFromGit call. Redirects report the final successor name.
func (gi *GitInstaller) InstalledName() string {
	return gi.installedName
}

// maxSupersedeRedirects bounds a superseded_by chain. Three is generous for the
// real case (one hop) and still fails fast on a manifest cycle.
const maxSupersedeRedirects = 3

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

// splitSubdir separates a `//subdir` selector from a Git URL. The scheme's
// own `//` (https://…) is skipped; the first `//` after it marks the
// subdirectory. Returns the URL without the selector and the cleaned subdir
// ("" when none).
func splitSubdir(url string) (string, string) {
	rest := url
	offset := 0
	if i := strings.Index(url, "://"); i >= 0 {
		offset = i + 3
		rest = url[offset:]
	}
	j := strings.Index(rest, "//")
	if j < 0 {
		return url, ""
	}
	return url[:offset+j], strings.Trim(rest[j+2:], "/")
}

// ParseGitURL parses a Git URL into components
func ParseGitURL(url string) (*GitPackageInfo, error) {
	info := &GitPackageInfo{
		URL: url,
	}

	// Split a trailing `//subdir[@ref]` selector off first: the ref of
	// https://github.com/owner/repo//go@main belongs to the repo, not the dir.
	var subdir string
	url, subdir = splitSubdir(url)
	if at := strings.LastIndex(subdir, "@"); at >= 0 {
		info.Ref = subdir[at+1:]
		subdir = subdir[:at]
	}
	info.Subdir = subdir

	// Handle URLs with @ for branch/tag specification
	// e.g., https://github.com/owner/repo@branch
	// But not SSH URLs like git@github.com:owner/repo.git
	if info.Ref == "" && strings.Contains(url, "@") && !strings.HasPrefix(url, "git@") {
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
	// Reject a malformed --path selector (absolute / escaping) up front, before
	// any install work (clone, copy, registry mutation) happens.
	if err := validateSubdirSelector(gi.Subdir); err != nil {
		return err
	}

	// Check if Git is available
	if err := checkGitAvailable(); err != nil {
		return err
	}

	// Parse Git URL
	info, err := ParseGitURL(gitURL)
	if err != nil {
		return fmt.Errorf("failed to parse Git URL: %w", err)
	}

	fmt.Println(ui.Muted("  from " + installSourceLabel(info.URL, info.Ref)))

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

	// The subdirectory can be named two ways: the --path flag (gi.Subdir) or
	// the URL's //subdir selector (info.Subdir). The flag wins when both are
	// given; folding the URL form in here lets one resolver handle both.
	if strings.TrimSpace(gi.Subdir) == "" {
		gi.Subdir = info.Subdir
	}
	packagePath, err := gi.resolvePackageRoot(tempDir)
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

	// 4. A superseded package installs its successor instead. This runs before
	// the force check and before anything is copied, so a redirect never
	// half-installs the package it is redirecting away from.
	if target := strings.TrimSpace(metadata.SupersededBy); target != "" {
		return gi.followSupersededBy(metadata.Name, target, force)
	}
	gi.installedName = metadata.Name

	// 5. Use existing installer for the rest
	installer := &PackageInstaller{
		AgentFieldHome: gi.AgentFieldHome,
		Verbose:        gi.Verbose,
	}

	// Check if already installed
	if !force && installer.isPackageInstalled(metadata.Name) {
		return fmt.Errorf("package %s already installed (use --force to reinstall)", metadata.Name)
	}

	// Install using existing flow
	destPath := filepath.Join(gi.AgentFieldHome, "packages", metadata.Name)

	// Reinstalling clears the destination before the replacement is copied,
	// and long before its dependencies finish building — a missing toolchain
	// is enough to fail there. Without this the user would be left with
	// neither the package they had nor a working new one. Set the existing
	// directory aside instead, and put it back if anything below fails.
	backup, err := stashExistingPackage(destPath)
	if err != nil {
		return err
	}

	spinner = gi.newSpinner("Setting up environment")
	spinner.Start()
	if err := installer.copyPackage(packagePath, destPath); err != nil {
		spinner.Error("Failed to copy package")
		backup.restore()
		return fmt.Errorf("failed to copy package: %w", err)
	}
	spinner.Success("Environment configured")

	spinner = gi.newSpinner("Installing dependencies")
	spinner.Start()
	if err := installer.installDependencies(destPath, metadata); err != nil {
		spinner.Error("Failed to install dependencies")
		backup.restore()
		return fmt.Errorf("failed to install dependencies: %w", err)
	}
	spinner.Success("Dependencies installed")

	// Update registry with Git source information
	if err := gi.updateRegistryWithGit(metadata, info, packagePath, destPath); err != nil {
		backup.restore()
		return fmt.Errorf("failed to update registry: %w", err)
	}
	backup.discard()

	fmt.Println()
	fmt.Println(installSummaryPanel(metadata.Name, metadata.Version, info.URL, info.Ref, destPath))

	// Check for required environment variables
	installer.checkEnvironmentVariables(metadata)

	fmt.Println()
	fmt.Println(ui.Title("→ Run: af run " + metadata.Name))

	return nil
}

// safeGitRefPattern matches refs (branches, tags, commits) that cannot be
// mistaken for git options: they must start with an alphanumeric character.
// Ref values can originate from user input (CLI args or the HTTP install
// API), so anything option-like could otherwise smuggle flags such as
// --upload-pack into the git invocation.
var safeGitRefPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._/-]*$`)

// validateCloneArgs rejects ref/URL values that git would parse as options
// rather than positional arguments.
func validateCloneArgs(info *GitPackageInfo) error {
	if info.Ref != "" && !safeGitRefPattern.MatchString(info.Ref) {
		return fmt.Errorf("invalid git ref %q: must start with an alphanumeric character and contain only [A-Za-z0-9._/-]", info.Ref)
	}
	if strings.HasPrefix(info.CloneURL, "-") {
		return fmt.Errorf("invalid clone URL %q: must not start with '-'", info.CloneURL)
	}
	return nil
}

// cloneRepository clones the Git repository with optimizations
func (gi *GitInstaller) cloneRepository(info *GitPackageInfo) (string, error) {
	if err := validateCloneArgs(info); err != nil {
		return "", err
	}

	// Create temporary directory
	tempDir, err := os.MkdirTemp("", "agentfield-git-install-")
	if err != nil {
		return "", fmt.Errorf("failed to create temp directory: %w", err)
	}

	// Fixed-shape invocations: the user-influenced values (ref, clone URL)
	// only ever occupy option-value or post-"--" positional slots, so git can
	// never parse them as flags. The ref is re-validated inline against the
	// anchored safeGitRefPattern immediately before use.
	var cmd *exec.Cmd
	if ref := info.Ref; ref != "" && safeGitRefPattern.MatchString(ref) {
		cmd = exec.Command("git", "clone", "--depth", "1", "--branch", ref, "--", info.CloneURL, tempDir)
	} else {
		cmd = exec.Command("git", "clone", "--depth", "1", "--", info.CloneURL, tempDir)
	}
	args := cmd.Args[1:]

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

// resolvePackageRoot determines which directory of the cloned repository is the
// package to install. With no subdirectory selector (--path flag or the URL's
// //subdir form, folded into gi.Subdir by InstallFromGit) it defers to
// findPackageRoot's root-first walk (unchanged behavior). With a selector it
// resolves and validates <cloneDir>/<Subdir>, requiring the manifest to exist
// there, so one repo can ship multiple installable nodes selected explicitly.
func (gi *GitInstaller) resolvePackageRoot(cloneDir string) (string, error) {
	if strings.TrimSpace(gi.Subdir) == "" {
		return gi.findPackageRoot(cloneDir)
	}
	root, err := ResolvePackageSubdir(cloneDir, gi.Subdir)
	if err != nil {
		return "", err
	}
	// A selected subdir must still be a valid, startable agent node.
	if err := ValidatePackage(root); err != nil {
		return "", err
	}
	return root, nil
}

// findPackageRoot finds the root directory containing agentfield-package.yaml
func (gi *GitInstaller) findPackageRoot(cloneDir string) (string, error) {
	var packageRoot string

	err := filepath.Walk(cloneDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.Name() == "agentfield-package.yaml" {
			packageRoot = filepath.Dir(path)
			return filepath.SkipDir // Found it, stop walking
		}

		return nil
	})

	if err != nil {
		return "", err
	}

	if packageRoot == "" {
		return "", fmt.Errorf("agentfield-package.yaml not found in the repository")
	}

	// The node must declare how to start: a manifest entrypoint.start or a
	// top-level main.py. Real nodes use a module entrypoint and have no main.py.
	if err := ValidatePackage(packageRoot); err != nil {
		return "", err
	}

	return packageRoot, nil
}

// parsePackageMetadata parses the agentfield-package.yaml file (reuse from installer.go)
func (gi *GitInstaller) parsePackageMetadata(packagePath string) (*PackageMetadata, error) {
	installer := &PackageInstaller{
		AgentFieldHome: gi.AgentFieldHome,
		Verbose:        gi.Verbose,
	}
	return installer.parsePackageMetadata(packagePath)
}

// followSupersededBy installs the successor a manifest points at, then retires
// the superseded package when it was already installed. Order matters: the
// successor is installed FIRST, so a failure leaves the user's existing node
// exactly as it was rather than with nothing.
func (gi *GitInstaller) followSupersededBy(fromName, target string, force bool) error {
	if gi.redirects >= maxSupersedeRedirects {
		return fmt.Errorf(
			"superseded_by chain longer than %d hops (at %q → %q) — the manifests most likely point at each other",
			maxSupersedeRedirects, fromName, target)
	}

	installer := &PackageInstaller{AgentFieldHome: gi.AgentFieldHome}
	replacing := installer.isPackageInstalled(fromName)

	fmt.Println()
	fmt.Printf("⚠️  %s has been superseded by %s\n", fromName, target)
	if replacing {
		fmt.Printf("⚠️  %s is currently installed and WILL BE REPLACED. The successor is installed\n", fromName)
		fmt.Println("    first and node-scoped secrets are carried over; if that fails, what you")
		fmt.Println("    have now is left as it is.")
	}
	fmt.Println(ui.Muted("  installing the successor instead"))
	fmt.Println()

	successor := &GitInstaller{
		AgentFieldHome: gi.AgentFieldHome,
		Verbose:        gi.Verbose,
		redirects:      gi.redirects + 1,
	}
	// A successor may carry the same name as the package it retires — that is
	// a node renaming itself in place, and it is the shape a rename takes when
	// the old and new names are meant to converge. The warning above already
	// said this replaces the current install, so carry that consent into the
	// successor rather than failing the redirect with "already installed".
	if err := successor.InstallFromGit(target, force || replacing); err != nil {
		return fmt.Errorf("installing %s, the successor of %s: %w", target, fromName, err)
	}
	gi.installedName = successor.installedName

	if !replacing || successor.installedName == fromName {
		return nil
	}
	gi.retireSuperseded(fromName, successor.installedName)
	return nil
}

// retireSuperseded removes the old package once its successor is in place. It
// never fails the install: the successor is already working, so a stubborn
// leftover is a cleanup chore, not a reason to report failure.
func (gi *GitInstaller) retireSuperseded(oldName, newName string) {
	if store, err := NewSecretStore(gi.AgentFieldHome); err == nil {
		migrateNodeScopedSecrets(store, oldName, newName)
	}
	uninstaller := &PackageUninstaller{AgentFieldHome: gi.AgentFieldHome}
	if err := uninstaller.UninstallPackage(oldName); err != nil {
		fmt.Printf("⚠️  Could not remove the superseded %s: %v\n", oldName, err)
		fmt.Printf("    %s is installed and usable; remove the old one with: af uninstall %s\n", newName, oldName)
		return
	}
	fmt.Printf("✓ Replaced %s with %s\n", oldName, newName)
}

// packageBackup holds an installed package directory that a reinstall is about
// to overwrite, so it can be put back if the reinstall fails partway. The zero
// value is a valid no-op, which is what a first-time install gets.
type packageBackup struct {
	original string
	saved    string
}

// stashExistingPackage moves an installed package directory aside so a failed
// reinstall can restore it. A missing directory is not an error — there is
// simply nothing to protect.
func stashExistingPackage(destPath string) (*packageBackup, error) {
	if _, err := os.Stat(destPath); err != nil {
		if os.IsNotExist(err) {
			return &packageBackup{}, nil
		}
		return nil, fmt.Errorf("failed to inspect %s: %w", destPath, err)
	}
	// Dot-prefixed and alongside the original: same filesystem, so the move is
	// a rename rather than a copy, and it cannot be mistaken for a package.
	dir, name := filepath.Split(strings.TrimRight(destPath, string(os.PathSeparator)))
	saved := filepath.Join(dir, "."+name+".previous")
	// A leftover from an interrupted run would make the rename fail.
	if err := os.RemoveAll(saved); err != nil {
		return nil, fmt.Errorf("failed to clear a stale backup at %s: %w", saved, err)
	}
	if err := os.Rename(destPath, saved); err != nil {
		return nil, fmt.Errorf("failed to set the existing package aside: %w", err)
	}
	return &packageBackup{original: destPath, saved: saved}, nil
}

// restore puts the stashed package back, undoing a failed reinstall. It never
// returns an error: the install is already failing, and the caller's report of
// why is more useful than a cleanup problem layered on top. A backup that
// cannot be moved back is still on disk, so say where.
func (b *packageBackup) restore() {
	if b == nil || b.saved == "" {
		return
	}
	if err := os.RemoveAll(b.original); err != nil {
		fmt.Printf("⚠️  Could not clear the failed install at %s: %v\n", b.original, err)
		fmt.Printf("    your previous version is still on disk at %s\n", b.saved)
		return
	}
	if err := os.Rename(b.saved, b.original); err != nil {
		fmt.Printf("⚠️  Could not restore your previous version: %v\n", err)
		fmt.Printf("    it is still on disk at %s\n", b.saved)
		return
	}
	fmt.Printf("  restored the previously installed version at %s\n", b.original)
	b.saved = ""
}

// discard drops the stashed copy once the reinstall has succeeded.
func (b *packageBackup) discard() {
	if b == nil || b.saved == "" {
		return
	}
	os.RemoveAll(b.saved)
	b.saved = ""
}

// migrateNodeScopedSecrets hands node-scoped secrets to the successor before
// the old package is uninstalled, which deletes that scope outright. Without
// this every `af secrets set KEY --node <old>` value is silently lost in the
// swap. Global secrets are shared and untouched. A value already set on the
// successor wins: the user set that one deliberately, and later.
func migrateNodeScopedSecrets(store *SecretStore, oldName, newName string) {
	// Read the scope directly rather than via Get, which falls back to the
	// global scope and would copy shared secrets into the node scope.
	oldValues, err := store.load(oldName)
	if err != nil || len(oldValues) == 0 {
		return
	}
	newValues, err := store.load(newName)
	if err != nil {
		newValues = map[string]string{}
	}
	moved := 0
	for key, value := range oldValues {
		if _, exists := newValues[key]; exists {
			continue
		}
		if err := store.Set(newName, key, value); err != nil {
			fmt.Printf("⚠️  Could not move secret %s to %s: %v\n", key, newName, err)
			continue
		}
		moved++
	}
	if moved > 0 {
		fmt.Printf("  moved %d node-scoped secret(s) from %s to %s\n", moved, oldName, newName)
	}
}

// appendSubdirSelector rewrites "https://host/owner/repo[@ref]" into
// "https://host/owner/repo//subdir[@ref]" so a recorded source round-trips
// through ParseGitURL. It is needed whenever the subdirectory arrived by the
// --path flag (or the install API, which splits the selector off before
// calling): without it the registry records the REPO ROOT, and the next update
// installs whatever lives there instead of the package that is installed.
func appendSubdirSelector(url, subdir string) string {
	subdir = strings.Trim(strings.TrimSpace(subdir), "/")
	if subdir == "" {
		return url
	}
	base, ref := url, ""
	if at := strings.LastIndex(url, "@"); at > strings.LastIndex(url, "/") {
		base, ref = url[:at], url[at:]
	}
	return base + "//" + subdir + ref
}

// updateRegistryWithGit updates the installation registry with Git source info
func (gi *GitInstaller) updateRegistryWithGit(metadata *PackageMetadata, info *GitPackageInfo, sourcePath, destPath string) error {
	registryPath := filepath.Join(gi.AgentFieldHome, "installed.yaml")

	// Load existing registry or create new one
	registry := &InstallationRegistry{
		Installed: make(map[string]InstalledPackage),
	}

	if data, err := os.ReadFile(registryPath); err == nil {
		if err := yaml.Unmarshal(data, registry); err != nil {
			return fmt.Errorf("failed to parse registry %s: %w", registryPath, err)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("failed to read registry %s: %w", registryPath, err)
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

	// The original source string is already reproducible as-is — it carries
	// any @ref and //subdir the user gave. (Appending the ref again used to
	// produce doubled "…@main@main" entries.)
	sourcePathStr := info.URL
	// …except when the subdirectory came from --path (or the install API, which
	// splits `//subdir` off the URL before calling). Then info.URL is the bare
	// repo and the selector has to be put back, or this records a source that
	// resolves to the repo root and the next update installs a different package.
	if strings.TrimSpace(info.Subdir) == "" && strings.TrimSpace(gi.Subdir) != "" {
		sourcePathStr = appendSubdirSelector(sourcePathStr, gi.Subdir)
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
			LogFile:   filepath.Join(gi.AgentFieldHome, "logs", metadata.Name+".log"),
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

// installSourceLabel formats an install source for display: "<url>" or
// "<url> @ <ref>" when a ref is pinned.
func installSourceLabel(url, ref string) string {
	if ref != "" {
		return url + " @ " + ref
	}
	return url
}

// installSummaryPanel renders the post-install success panel showing the node
// name/version and its source and on-disk location.
func installSummaryPanel(name, version, source, ref, location string) string {
	details := [][2]string{{"Source", source}}
	if ref != "" {
		details = append(details, [2]string{"Reference", ref})
	}
	details = append(details, [2]string{"Location", location})
	return ui.SuccessPanel(fmt.Sprintf("Installed %s v%s", name, version), ui.KV(details))
}
