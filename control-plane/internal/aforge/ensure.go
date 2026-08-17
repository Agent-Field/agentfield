// Package aforge provisions the pinned aforge coding-harness binary that AgentField's harness providers spawn.
package aforge

import (
	"bufio"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const (
	// Version is deliberately pinned. Bump it only when af should distribute a
	// newer, reviewed aforge build.
	Version        = "v0.1.0"
	defaultBaseURL = "https://agentfield.ai/downloads/aforge/" + Version
	baseURLEnv     = "AGENTFIELD_AFORGE_BASE_URL"
	skipEnv        = "AGENTFIELD_SKIP_AFORGE"
	versionMarker  = ".aforge.version"
	// aforge is a ~35MB static Go binary; cap the DECOMPRESSED stream so a
	// hostile or wrong endpoint cannot gzip-bomb the process.
	maxDecompressedBytes = 96 << 20
)

// Options controls how Ensure selects and installs aforge.
type Options struct {
	GOOS, GOARCH string
	Home         string
	BaseURL      string
	Client       *http.Client
	// Force re-downloads even when the version marker already matches
	// (what `af aforge ensure --force` sets).
	Force bool
}

// AssetName returns the release asset name for a platform (no .gz suffix).
func AssetName(goos, goarch string) (string, bool) {
	switch goos + "/" + goarch {
	case "linux/amd64":
		return "aforge-linux-amd64", true
	case "linux/arm64":
		return "aforge-linux-arm64", true
	case "darwin/amd64":
		return "aforge-darwin-amd64", true
	case "darwin/arm64":
		return "aforge-darwin-arm64", true
	case "windows/amd64":
		return "aforge-windows-amd64.exe", true
	case "windows/arm64":
		return "aforge-windows-arm64.exe", true
	default:
		return "", false
	}
}

// BinaryName is the installed file name ("aforge", or "aforge.exe" on windows).
func BinaryName(goos string) string {
	if goos == "windows" {
		return "aforge.exe"
	}
	return "aforge"
}

// Ensure installs aforge if it is supported and not already executable.
func Ensure(opts Options) error {
	if os.Getenv(skipEnv) == "1" {
		return nil
	}
	goos, goarch := opts.GOOS, opts.GOARCH
	if goos == "" {
		goos = runtime.GOOS
	}
	if goarch == "" {
		goarch = runtime.GOARCH
	}
	asset, supported := AssetName(goos, goarch)
	if !supported {
		return nil
	}

	home, err := agentfieldHome(opts.Home)
	if err != nil {
		return err
	}
	destination := filepath.Join(home, "bin", BinaryName(goos))
	markerPath := filepath.Join(home, "bin", versionMarker)
	binDir := filepath.Dir(destination)
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		return fmt.Errorf("create aforge bin directory: %w", err)
	}
	unlock, err := lockAforge(filepath.Join(binDir, ".aforge.lock"))
	if err != nil {
		return fmt.Errorf("lock aforge installation: %w", err)
	}
	defer func() { _ = unlock() }()
	if !opts.Force && alreadyInstalled(destination, markerPath) {
		return nil
	}

	baseURL := strings.TrimRight(opts.BaseURL, "/")
	if baseURL == "" {
		baseURL = strings.TrimRight(os.Getenv(baseURLEnv), "/")
	}
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	client := opts.Client
	if client == nil {
		client = &http.Client{
			Transport: &http.Transport{
				DialContext:           (&net.Dialer{Timeout: 10 * time.Second}).DialContext,
				TLSHandshakeTimeout:   10 * time.Second,
				ResponseHeaderTimeout: 30 * time.Second,
			},
			// A ~35MB asset can take longer than three minutes over a slow link.
			Timeout: 10 * time.Minute,
		}
	}

	checksums, err := download(client, baseURL+"/checksums.txt")
	if err != nil {
		return fmt.Errorf("download checksums: %w", err)
	}
	want, err := checksumFor(checksums, asset)
	if err != nil {
		return err
	}
	binary, err := downloadGzip(client, baseURL+"/"+asset+".gz")
	if err != nil {
		return fmt.Errorf("download %s: %w", asset, err)
	}
	got := sha256.Sum256(binary)
	if !strings.EqualFold(hex.EncodeToString(got[:]), want) {
		return fmt.Errorf("checksum mismatch for %s", asset)
	}

	tmp, err := os.CreateTemp(binDir, ".aforge-*")
	if err != nil {
		return fmt.Errorf("create temporary aforge file: %w", err)
	}
	tmpName := tmp.Name()
	defer func() { _ = os.Remove(tmpName) }()
	if _, err = tmp.Write(binary); err == nil {
		err = tmp.Chmod(0o755)
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return fmt.Errorf("write temporary aforge file: %w", err)
	}
	if err := os.Rename(tmpName, destination); err != nil {
		return fmt.Errorf("install aforge: %w", err)
	}
	// Written after the binary is in place: a marker without a usable binary
	// would make the next Ensure skip a repair it should have done.
	if err := os.WriteFile(markerPath, []byte(Version+"\n"), 0o644); err != nil {
		return fmt.Errorf("record aforge version: %w", err)
	}
	return nil
}

func alreadyInstalled(destination, markerPath string) bool {
	info, err := os.Stat(destination)
	// The exec-bit check is meaningless on Windows, where NTFS carries no such
	// mode and Go reports 0666 for every regular file.
	if err != nil || !info.Mode().IsRegular() || (runtime.GOOS != "windows" && info.Mode().Perm()&0o111 == 0) {
		return false
	}
	installed, err := os.ReadFile(markerPath)
	return err == nil && strings.TrimSpace(string(installed)) == Version
}

// EnsureBestEffort is the install-path contract: provisioning can emit one
// short warning, but can never fail the operation that requested it.
func EnsureBestEffort(opts Options, warnings io.Writer) error {
	if err := Ensure(opts); err != nil && warnings != nil {
		_, _ = fmt.Fprintf(warnings, "warning: aforge was not installed: %v\n", err)
	}
	return nil
}

func agentfieldHome(override string) (string, error) {
	if override != "" {
		return override, nil
	}
	if home := os.Getenv("AGENTFIELD_HOME"); home != "" {
		return home, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolve home directory: %w", err)
	}
	return filepath.Join(home, ".agentfield"), nil
}

func download(client *http.Client, url string) ([]byte, error) {
	response, err := client.Get(url)
	if err != nil {
		return nil, err
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s", response.Status)
	}
	return io.ReadAll(io.LimitReader(response.Body, maxDecompressedBytes))
}

func downloadGzip(client *http.Client, url string) ([]byte, error) {
	response, err := client.Get(url)
	if err != nil {
		return nil, err
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s", response.Status)
	}
	gz, err := gzip.NewReader(response.Body)
	if err != nil {
		return nil, err
	}
	defer func() { _ = gz.Close() }()
	binary, err := io.ReadAll(io.LimitReader(gz, maxDecompressedBytes+1))
	if err != nil {
		return nil, err
	}
	if len(binary) > maxDecompressedBytes {
		return nil, fmt.Errorf("aforge payload exceeds %d bytes", maxDecompressedBytes)
	}
	return binary, nil
}

func checksumFor(data []byte, asset string) (string, error) {
	scanner := bufio.NewScanner(strings.NewReader(string(data)))
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) == 2 && strings.TrimPrefix(fields[1], "*") == asset {
			if _, err := hex.DecodeString(fields[0]); err != nil || len(fields[0]) != sha256.Size*2 {
				return "", fmt.Errorf("invalid checksum for %s", asset)
			}
			return fields[0], nil
		}
	}
	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("read checksums: %w", err)
	}
	return "", fmt.Errorf("checksum missing for %s", asset)
}
