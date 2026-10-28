package harness

import (
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

const (
	// CodeAFVersion is the managed CodeAF release, verified using embedded hashes.
	CodeAFVersion = "v0.8.2"
	// CodeAFBaseURL is the pinned release directory, overridable by AGENTFIELD_CODEAF_BASE_URL.
	CodeAFBaseURL         = "https://github.com/Agent-Field/CodeAF/releases/download/v0.8.2"
	maxCodeAFDownloadSize = 128 << 20
	codeAFVersionFilename = ".codeaf.version"
	codeAFDownloadRetry   = time.Minute
)

var codeAFHashes = map[string]string{
	"codeaf-darwin-amd64":      "a476f1ce44b21f8016583a65c49b21497d10713bcd72c33478b3bd12e52446d7",
	"codeaf-darwin-arm64":      "e61adac060bf2345c08a3e23ae83239871c3050eadac046879f5f3176270c4f1",
	"codeaf-linux-amd64":       "681d9813f2517addc86fb644d313e3f7cf0ab60be708c8e8b13d77fb16a6798c",
	"codeaf-linux-arm64":       "c6375d58d198d9e1a84880d8b4324ee6c067a0e5e351af115ca5506b1ffa1c78",
	"codeaf-windows-amd64.exe": "ff80f6f0c89c209fc809e960c70a1eb3518ccfc06775d587d5a187edd5d5222d",
	"codeaf-windows-arm64.exe": "bfc0b386e5248e7e6756f097b372f72e4e955ee1e19e9216ccafee47064b2b80",
}

type codeAFBinary struct {
	path   string
	legacy bool
}

type codeAFResolution struct {
	done   chan struct{}
	binary codeAFBinary
	err    error
}

// Only automatic resolution is shared. Explicit binaries bypass synchronization.
var codeAFResolver = struct {
	sync.Mutex
	inflight            *codeAFResolution
	cache               map[string]codeAFBinary
	lastDownloadAttempt time.Time
	lastDownloadError   error
}{cache: make(map[string]codeAFBinary)}

var (
	codeAFAliasOnce   sync.Once
	codeAFLegacyOnce  sync.Once
	codeAFCommandOnce sync.Once
)

func codeAFTruthy(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func codeAFSkipped() bool {
	return codeAFTruthy(os.Getenv("AGENTFIELD_SKIP_CODEAF")) || codeAFTruthy(os.Getenv("AGENTFIELD_SKIP_AFORGE"))
}

func agentFieldHome() (string, error) {
	if home := os.Getenv("AGENTFIELD_HOME"); home != "" {
		return home, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".agentfield"), nil
}

func codeAFExecutable(goos string) string {
	if goos == "windows" {
		return "codeaf.exe"
	}
	return "codeaf"
}

func codeAFAsset(goos, goarch string) (string, string) {
	asset := "codeaf-" + goos + "-" + goarch
	if goos == "windows" {
		asset += ".exe"
	}
	return asset, codeAFHashes[asset]
}

func codeAFManagedPaths() string {
	if home, err := agentFieldHome(); err == nil {
		return filepath.Join(home, "bin", codeAFExecutable(runtime.GOOS))
	}
	return ""
}

func codeAFInstalled(path string) bool {
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() || (runtime.GOOS != "windows" && info.Mode().Perm()&0o111 == 0) {
		return false
	}
	marker, err := os.ReadFile(filepath.Join(filepath.Dir(path), codeAFVersionFilename))
	return err == nil && strings.TrimSpace(string(marker)) == CodeAFVersion
}

func findManagedCodeAF() string {
	homePath := codeAFManagedPaths()
	if homePath != "" && codeAFInstalled(homePath) {
		return homePath
	}
	return ""
}

func resolveCodeAF(ctx context.Context, override string) (codeAFBinary, error) {
	if override == "" {
		override = strings.TrimSpace(os.Getenv("CODEAF_BIN"))
	}
	if override != "" {
		return findCodeAF(ctx, override)
	}
	codeAFResolver.Lock()
	if cached, ok := codeAFResolver.cache[""]; ok {
		codeAFResolver.Unlock()
		return cached, nil
	}
	pending := codeAFResolver.inflight
	if pending == nil {
		pending = &codeAFResolution{done: make(chan struct{})}
		codeAFResolver.inflight = pending
		go func() {
			// The installation outlives individual callers and their deadlines.
			binary, err := findCodeAF(context.Background(), "")
			codeAFResolver.Lock()
			pending.binary, pending.err = binary, err
			if err == nil {
				codeAFResolver.cache[""] = binary
			}
			codeAFResolver.inflight = nil
			close(pending.done)
			codeAFResolver.Unlock()
		}()
	}
	codeAFResolver.Unlock()
	select {
	case <-ctx.Done():
		return codeAFBinary{}, ctx.Err()
	case <-pending.done:
		return pending.binary, pending.err
	}
}

func findCodeAF(ctx context.Context, override string) (codeAFBinary, error) {
	if override != "" {
		path, err := exec.LookPath(override)
		if err != nil {
			return codeAFBinary{}, newProviderUnavailable(ProviderCodeAF, override)
		}
		return codeAFBinary{path: path}, nil
	}
	if path := findManagedCodeAF(); path != "" {
		return codeAFBinary{path: path}, nil
	}
	var downloadErr error
	if !codeAFSkipped() {
		codeAFResolver.Lock()
		attempt := time.Since(codeAFResolver.lastDownloadAttempt) >= codeAFDownloadRetry
		downloadErr = codeAFResolver.lastDownloadError
		if attempt {
			codeAFResolver.lastDownloadAttempt = time.Now()
		}
		codeAFResolver.Unlock()
		if attempt {
			path, err := downloadManagedCodeAF(ctx)
			codeAFResolver.Lock()
			codeAFResolver.lastDownloadError = err
			codeAFResolver.Unlock()
			if err == nil {
				return codeAFBinary{path: path}, nil
			}
			downloadErr = err
		}
	}
	if path, err := exec.LookPath("codeaf"); err == nil {
		return codeAFBinary{path: path}, nil
	}
	if home, err := agentFieldHome(); err == nil {
		legacy := "aforge"
		if runtime.GOOS == "windows" {
			legacy += ".exe"
		}
		if path, err := exec.LookPath(filepath.Join(home, "bin", legacy)); err == nil {
			return codeAFBinary{path: path, legacy: true}, nil
		}
	}
	if path, err := exec.LookPath("aforge"); err == nil {
		return codeAFBinary{path: path, legacy: true}, nil
	}
	unavailable := newProviderUnavailable(ProviderCodeAF, "codeaf")
	unavailable.DownloadError = downloadErr
	return codeAFBinary{}, unavailable
}

func downloadManagedCodeAF(ctx context.Context) (string, error) {
	asset, hash := codeAFAsset(runtime.GOOS, runtime.GOARCH)
	if hash == "" {
		return "", fmt.Errorf("CodeAF %s is not installable on %s/%s", CodeAFVersion, runtime.GOOS, runtime.GOARCH)
	}
	homePath := codeAFManagedPaths()
	return installCodeAF(ctx, homePath, asset, hash)
}

// installCodeAF accepts a hash explicitly for offline fixture tests. Production
// only calls it with the embedded pin selected by downloadManagedCodeAF.
func installCodeAF(ctx context.Context, homePath, asset, expectedHash string) (string, error) {
	target := homePath
	temp, err := prepareCodeAFDownload(target)
	if err != nil {
		return "", fmt.Errorf("cannot install CodeAF into %s: %w", filepath.Dir(target), err)
	}
	defer os.Remove(temp.Name())
	defer temp.Close()

	log.Printf("INFO downloading CodeAF %s (~70 MB) to %s", CodeAFVersion, target)
	base := strings.TrimSpace(os.Getenv("AGENTFIELD_CODEAF_BASE_URL"))
	if base == "" {
		base = CodeAFBaseURL
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(base, "/")+"/"+asset, nil)
	if err != nil {
		return "", err
	}
	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("CodeAF download returned HTTP %d", resp.StatusCode)
	}
	if resp.ContentLength > maxCodeAFDownloadSize {
		return "", fmt.Errorf("CodeAF download exceeds 128 MiB")
	}
	digest := sha256.New()
	n, err := io.Copy(io.MultiWriter(temp, digest), io.LimitReader(resp.Body, maxCodeAFDownloadSize+1))
	if err != nil {
		return "", fmt.Errorf("CodeAF download: %w", err)
	}
	if n > maxCodeAFDownloadSize {
		return "", fmt.Errorf("CodeAF download exceeds 128 MiB")
	}
	if actual := fmt.Sprintf("%x", digest.Sum(nil)); actual != expectedHash {
		return "", fmt.Errorf("CodeAF sha256 mismatch: expected %s, got %s", expectedHash, actual)
	}
	if err := temp.Chmod(0o755); err != nil {
		return "", err
	}
	if err := temp.Close(); err != nil {
		return "", err
	}
	if err := os.Rename(temp.Name(), target); err != nil {
		if runtime.GOOS == "windows" && codeAFInstalled(target) {
			return target, nil
		}
		return "", err
	}
	// The marker is the commit point: an interrupted or failed download can
	// never make a partial binary appear installed.
	if err := os.WriteFile(filepath.Join(filepath.Dir(target), codeAFVersionFilename), []byte(CodeAFVersion+"\n"), 0o644); err != nil {
		return "", err
	}
	return target, nil
}

func prepareCodeAFDownload(target string) (*os.File, error) {
	if target == "" {
		return nil, fmt.Errorf("CodeAF home path unavailable")
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return nil, err
	}
	return os.CreateTemp(filepath.Dir(target), ".codeaf-download-*")
}
