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
	CodeAFVersion = "v0.8.4"
	// CodeAFBaseURL is the pinned release directory, overridable by AGENTFIELD_CODEAF_BASE_URL.
	CodeAFBaseURL         = "https://github.com/Agent-Field/CodeAF/releases/download/v0.8.4"
	maxCodeAFDownloadSize = 128 << 20
	codeAFVersionFilename = ".codeaf.version"
	codeAFDownloadRetry   = time.Minute
)

var codeAFHashes = map[string]string{
	"codeaf-darwin-amd64":      "9a8db2d6cdb01affd5eab5d0259d2a8b773d2e7939ee3c519ee46eadff2b268d",
	"codeaf-darwin-arm64":      "5ba7992a61d1bcd4e222ba5de96b8221dc3e664ede78baf8524a2cbcafdacf64",
	"codeaf-linux-amd64":       "0af6241aa8822be7ed97c9cf14b302393ff0dbac481a74c213cceba94f2a4315",
	"codeaf-linux-arm64":       "bf4c485c77d98a40c7163e9ccdd452d3c06065491d9b045f4e83fad4df13a39e",
	"codeaf-windows-amd64.exe": "e1701579bc0d26ba60bd4c11be9181592e85ed8071ef0ba3bff785f637b75b91",
	"codeaf-windows-arm64.exe": "a54b3019e8bdd32894a3bf1ab956e592697299ed8ce10ea405cd9f74f0b6b83c",
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
