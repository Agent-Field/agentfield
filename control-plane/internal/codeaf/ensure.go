// Package codeaf provisions AgentField's managed, pinned CodeAF harness binary.
package codeaf

import (
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
	"sync"
	"time"
)

const (
	Version          = "v0.8.2"
	defaultBaseURL   = "https://github.com/Agent-Field/CodeAF/releases/download/" + Version
	baseURLEnv       = "AGENTFIELD_CODEAF_BASE_URL"
	versionMarker    = ".codeaf.version"
	maxDownloadBytes = 128 << 20
)

// Options controls how Ensure selects and installs CodeAF.
type Options struct {
	GOOS, GOARCH string
	Home         string
	BaseURL      string
	Client       *http.Client
	Force        bool
}

// Embedded pins authenticate the raw executables, including when a mirror is used.
var checksums = map[string]string{
	"codeaf-darwin-amd64":      "a476f1ce44b21f8016583a65c49b21497d10713bcd72c33478b3bd12e52446d7",
	"codeaf-darwin-arm64":      "e61adac060bf2345c08a3e23ae83239871c3050eadac046879f5f3176270c4f1",
	"codeaf-linux-amd64":       "681d9813f2517addc86fb644d313e3f7cf0ab60be708c8e8b13d77fb16a6798c",
	"codeaf-linux-arm64":       "c6375d58d198d9e1a84880d8b4324ee6c067a0e5e351af115ca5506b1ffa1c78",
	"codeaf-windows-amd64.exe": "ff80f6f0c89c209fc809e960c70a1eb3518ccfc06775d587d5a187edd5d5222d",
	"codeaf-windows-arm64.exe": "bfc0b386e5248e7e6756f097b372f72e4e955ee1e19e9216ccafee47064b2b80",
}

// AssetName returns the raw release asset for a supported platform.
func AssetName(goos, goarch string) (string, bool) {
	asset := "codeaf-" + goos + "-" + goarch
	if goos == "windows" {
		asset += ".exe"
	}
	if _, supported := checksums[asset]; !supported {
		return "", false
	}
	return asset, true
}

// BinaryName is the managed executable's filename.
func BinaryName(goos string) string {
	if goos == "windows" {
		return "codeaf.exe"
	}
	return "codeaf"
}

// SkipRequested also honors the legacy harness opt-out.
func SkipRequested() bool {
	for _, name := range []string{"AGENTFIELD_SKIP_CODEAF", "AGENTFIELD_SKIP_AFORGE"} {
		switch strings.ToLower(strings.TrimSpace(os.Getenv(name))) {
		case "1", "true", "yes", "on":
			return true
		}
	}
	return false
}

// Ensure installs or repairs the pinned release in AgentField's own bin directory.
func Ensure(opts Options) error {
	return ensure(opts, checksums)
}

// Tests inject fixture hashes here without allowing production callers or mirrors
// to override the embedded release pins.
func ensure(opts Options, pins map[string]string) error {
	if SkipRequested() {
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
	binDir := filepath.Join(home, "bin")
	destination := filepath.Join(binDir, BinaryName(goos))
	markerPath := filepath.Join(binDir, versionMarker)
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		return fmt.Errorf("create codeaf bin directory: %w", err)
	}
	unlock, err := lockCodeaf(filepath.Join(binDir, ".codeaf.lock"))
	if err != nil {
		return fmt.Errorf("lock codeaf installation: %w", err)
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
			Timeout: 10 * time.Minute,
		}
	}
	tmp, err := os.CreateTemp(binDir, ".codeaf-*")
	if err != nil {
		return fmt.Errorf("create temporary codeaf file: %w", err)
	}
	defer func() { _ = os.Remove(tmp.Name()) }()
	err = download(client, baseURL+"/"+asset, tmp, pins[asset])
	if err == nil {
		err = tmp.Chmod(0o755)
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return fmt.Errorf("download %s: %w", asset, err)
	}
	if err := os.Rename(tmp.Name(), destination); err != nil {
		// Windows may refuse to replace an executable currently in use.
		if runtime.GOOS == "windows" && alreadyInstalled(destination, markerPath) {
			return nil
		}
		return fmt.Errorf("install codeaf: %w", err)
	}
	// Publish the marker last so an interrupted installation is repaired.
	if err := os.WriteFile(markerPath, []byte(Version+"\n"), 0o644); err != nil {
		return fmt.Errorf("record codeaf version: %w", err)
	}
	return nil
}

func alreadyInstalled(destination, markerPath string) bool {
	info, err := os.Stat(destination)
	if err != nil || !info.Mode().IsRegular() || (runtime.GOOS != "windows" && info.Mode().Perm()&0o111 == 0) {
		return false
	}
	installed, err := os.ReadFile(markerPath)
	return err == nil && strings.TrimSpace(string(installed)) == Version
}

var bestEffortOnce sync.Once

// EnsureBestEffort provisions at most once per process; a failure may warn but
// never fails the skill/package install or node launch that requested it.
func EnsureBestEffort(opts Options, warnings io.Writer) error {
	if SkipRequested() {
		return nil
	}
	bestEffortOnce.Do(func() {
		if err := Ensure(opts); err != nil && warnings != nil {
			_, _ = fmt.Fprintf(warnings, "warning: codeaf was not installed: %v\n", err)
		}
	})
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

func download(client *http.Client, url string, dst io.Writer, want string) error {
	response, err := client.Get(url)
	if err != nil {
		return err
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("%s", response.Status)
	}
	if response.ContentLength > maxDownloadBytes {
		return fmt.Errorf("codeaf payload exceeds %d bytes", maxDownloadBytes)
	}
	hash := sha256.New()
	size, err := io.Copy(io.MultiWriter(dst, hash), io.LimitReader(response.Body, maxDownloadBytes+1))
	if err != nil {
		return err
	}
	if size > maxDownloadBytes {
		return fmt.Errorf("codeaf payload exceeds %d bytes", maxDownloadBytes)
	}
	if hex.EncodeToString(hash.Sum(nil)) != want {
		return fmt.Errorf("checksum mismatch")
	}
	return nil
}
