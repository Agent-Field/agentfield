// Package furrow provisions the pinned furrow workspace client used by the
// agentfield-use skill.
package furrow

import (
	"bufio"
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
	// Version is deliberately pinned. Change it only when af should distribute
	// a newer, reviewed furrow release.
	Version        = "v0.1.0"
	defaultBaseURL = "https://github.com/Agent-Field/furrow/releases/download/" + Version
	baseURLEnv     = "AGENTFIELD_FURROW_BASE_URL"
	skipEnv        = "AGENTFIELD_SKIP_FURROW"

	// The installed version is recorded beside the binary so that bumping
	// Version actually upgrades an existing install. Keying the skip on the
	// file's existence alone would pin every machine to whatever it first got.
	versionMarker = ".furrow.version"

	// Generous next to a ~8MB binary, but bounded: an unexpected endpoint
	// should not be able to stream until the process runs out of memory.
	maxDownloadBytes = 64 << 20
)

type Options struct {
	GOOS, GOARCH string
	Home         string
	BaseURL      string
	Client       *http.Client
}

func AssetName(goos, goarch string) (string, bool) {
	switch goos + "/" + goarch {
	case "linux/amd64":
		return "furrow-linux-amd64", true
	case "darwin/arm64":
		return "furrow-darwin-arm64", true
	case "darwin/amd64":
		return "furrow-darwin-amd64", true
	default:
		return "", false
	}
}

// Ensure installs furrow if it is supported and not already executable.
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
	destination := filepath.Join(home, "bin", "furrow")
	markerPath := filepath.Join(home, "bin", versionMarker)
	binDir := filepath.Dir(destination)
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		return fmt.Errorf("create furrow bin directory: %w", err)
	}
	unlock, err := lockFurrow(filepath.Join(binDir, ".furrow.lock"))
	if err != nil {
		return fmt.Errorf("lock furrow installation: %w", err)
	}
	defer func() { _ = unlock() }()
	if alreadyInstalled(destination, markerPath) {
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
			Timeout: 3 * time.Minute,
		}
	}

	checksums, err := download(client, baseURL+"/SHA256SUMS")
	if err != nil {
		return fmt.Errorf("download checksums: %w", err)
	}
	want, err := checksumFor(checksums, asset)
	if err != nil {
		return err
	}
	binary, err := download(client, baseURL+"/"+asset)
	if err != nil {
		return fmt.Errorf("download %s: %w", asset, err)
	}
	got := sha256.Sum256(binary)
	if !strings.EqualFold(hex.EncodeToString(got[:]), want) {
		return fmt.Errorf("checksum mismatch for %s", asset)
	}

	tmp, err := os.CreateTemp(binDir, ".furrow-*")
	if err != nil {
		return fmt.Errorf("create temporary furrow file: %w", err)
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
		return fmt.Errorf("write temporary furrow file: %w", err)
	}
	if err := os.Rename(tmpName, destination); err != nil {
		return fmt.Errorf("install furrow: %w", err)
	}
	// Written after the binary is in place: a marker without a usable binary
	// would make the next Ensure skip a repair it should have done.
	if err := os.WriteFile(markerPath, []byte(Version+"\n"), 0o644); err != nil {
		return fmt.Errorf("record furrow version: %w", err)
	}
	return nil
}

func alreadyInstalled(destination, markerPath string) bool {
	info, err := os.Stat(destination)
	if err != nil || !info.Mode().IsRegular() || info.Mode().Perm()&0o111 == 0 {
		return false
	}
	installed, err := os.ReadFile(markerPath)
	return err == nil && strings.TrimSpace(string(installed)) == Version
}

// EnsureBestEffort is the install-path contract: provisioning can emit one
// short warning, but can never fail the operation that requested it.
func EnsureBestEffort(opts Options, warnings io.Writer) error {
	if err := Ensure(opts); err != nil && warnings != nil {
		_, _ = fmt.Fprintf(warnings, "warning: furrow was not installed: %v\n", err)
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
	return io.ReadAll(io.LimitReader(response.Body, maxDownloadBytes))
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
