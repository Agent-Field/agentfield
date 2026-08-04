package packages

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

const goDownloadIndexURL = "https://go.dev/dl/?mode=json"

var (
	goProvisionHTTPClient     = http.DefaultClient
	goProvisionIndexURL       = goDownloadIndexURL
	goProvisionArchiveBaseURL = "https://go.dev/dl/"
)

type goRelease struct {
	Version string          `json:"version"`
	Stable  bool            `json:"stable"`
	Files   []goReleaseFile `json:"files"`
}

type goReleaseFile struct {
	Filename string `json:"filename"`
	OS       string `json:"os"`
	Arch     string `json:"arch"`
	Kind     string `json:"kind"`
	SHA256   string `json:"sha256"`
	Size     int64  `json:"size"`
}

// provisionGoToolchain returns an installed Go binary and whether it came from
// the cache rather than a fresh download, or an empty path for ordinary
// availability failures so the caller can retain its actionable install-Go
// guidance. Integrity and unsafe-archive failures are hard errors.
func provisionGoToolchain() (goCmd string, cached bool, err error) {
	if strings.EqualFold(strings.TrimSpace(os.Getenv("AGENTFIELD_DISABLE_GO_PROVISIONING")), "true") ||
		strings.TrimSpace(os.Getenv("AGENTFIELD_DISABLE_GO_PROVISIONING")) == "1" {
		return "", false, nil
	}
	home, err := AgentFieldHomeDir()
	if err != nil {
		return "", false, nil
	}
	release, file, err := discoverGoArchive()
	if err != nil {
		return "", false, nil
	}
	installDir := filepath.Join(home, "toolchains", release.Version)
	goCmd = filepath.Join(installDir, "go", "bin", goBinaryName())
	if usableGoBinary(goCmd) {
		return goCmd, true, nil
	}

	// Say so BEFORE the transfer, not after. This is ~60MB behind an install
	// spinner that cannot tick during it, so on a slow link the alternative is a
	// silent minute that reads as a hang — to exactly the users who have no Go
	// and least expect an install to fetch a compiler.
	fmt.Printf("%s  Downloading Go %s (%s) — no Go toolchain on this machine\n",
		clearLine(), strings.TrimPrefix(release.Version, "go"), humanBytes(file.Size))

	resp, err := goProvisionHTTPClient.Get(goProvisionArchiveBaseURL + file.Filename)
	if err != nil {
		return "", false, nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", false, nil
	}
	archive, err := os.CreateTemp("", "agentfield-go-archive-*"+filepath.Ext(file.Filename))
	if err != nil {
		return "", false, nil
	}
	archivePath := archive.Name()
	defer os.Remove(archivePath)
	h := sha256.New()
	if _, err := io.Copy(io.MultiWriter(archive, h), resp.Body); err != nil {
		archive.Close()
		return "", false, nil
	}
	if err := archive.Close(); err != nil {
		return "", false, nil
	}
	gotSum := hex.EncodeToString(h.Sum(nil))
	if !strings.EqualFold(gotSum, file.SHA256) {
		return "", false, fmt.Errorf("Go toolchain SHA256 mismatch for %s: index says %s, downloaded %s", file.Filename, file.SHA256, gotSum)
	}

	toolchainsDir := filepath.Dir(installDir)
	if err := os.MkdirAll(toolchainsDir, 0o755); err != nil {
		return "", false, nil
	}
	tmpDir, err := os.MkdirTemp(toolchainsDir, "."+release.Version+"-*")
	if err != nil {
		return "", false, nil
	}
	defer os.RemoveAll(tmpDir)
	if err := extractGoArchive(archivePath, tmpDir); err != nil {
		return "", false, fmt.Errorf("failed to safely extract Go toolchain: %w", err)
	}
	tmpGo := filepath.Join(tmpDir, "go", "bin", goBinaryName())
	if !usableGoBinary(tmpGo) {
		return "", false, fmt.Errorf("downloaded Go toolchain archive does not contain go/bin/%s", goBinaryName())
	}
	if err := os.Rename(tmpDir, installDir); err != nil {
		if usableGoBinary(goCmd) { // another concurrent installer won the race
			return goCmd, true, nil
		}
		return "", false, nil
	}
	return goCmd, false, nil
}

func discoverGoArchive() (goRelease, goReleaseFile, error) {
	resp, err := goProvisionHTTPClient.Get(goProvisionIndexURL)
	if err != nil {
		return goRelease{}, goReleaseFile{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return goRelease{}, goReleaseFile{}, fmt.Errorf("Go download index returned %s", resp.Status)
	}
	var releases []goRelease
	if err := json.NewDecoder(resp.Body).Decode(&releases); err != nil {
		return goRelease{}, goReleaseFile{}, err
	}
	for _, release := range releases {
		if !release.Stable {
			continue
		}
		for _, file := range release.Files {
			if file.OS == runtime.GOOS && file.Arch == runtime.GOARCH && file.Kind == "archive" && file.SHA256 != "" {
				return release, file, nil
			}
		}
	}
	return goRelease{}, goReleaseFile{}, fmt.Errorf("no stable Go archive for %s/%s", runtime.GOOS, runtime.GOARCH)
}

// humanBytes renders a download size the way a user would say it. An index that
// omits the size yields "unknown size" rather than a misleading "0 B".
func humanBytes(n int64) string {
	if n <= 0 {
		return "unknown size"
	}
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for m := n / unit; m >= unit && exp < 3; m /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.0f %cB", float64(n)/float64(div), "KMGT"[exp])
}

func goBinaryName() string {
	if runtime.GOOS == "windows" {
		return "go.exe"
	}
	return "go"
}

// usableGoBinary reports whether path is a Go toolchain that will actually run.
// Existence is deliberately not the test: a cached toolchain whose `go` lost its
// execute bit — a restore from backup, a umask, a copy across filesystems —
// would otherwise look available, be handed back from the cache on every
// install, and fail inside the build with a raw permission error that
// provisioning never retries. Probing it costs one `go version` and turns that
// into a re-provision.
func usableGoBinary(path string) bool {
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() {
		return false
	}
	if runtime.GOOS != "windows" && info.Mode().Perm()&0o111 == 0 {
		return false
	}
	// Only that it runs — deliberately not that its version parses. This file
	// treats an unparseable version as "unknown, don't gate" everywhere else,
	// and a toolchain we cannot label is still a toolchain that builds.
	return exec.Command(path, "version").Run() == nil
}

// safeArchiveName rejects an archive entry whose name is obviously hostile
// before it ever reaches the filesystem: empty, absolute, or climbing out with
// "..". This is a fast, legible first gate — it is NOT what makes extraction
// safe. That guarantee comes from os.Root below, which refuses to resolve any
// path outside its directory at the syscall level, symlinked parents included.
func safeArchiveName(name string) (string, error) {
	if name == "" || filepath.IsAbs(name) {
		return "", fmt.Errorf("unsafe archive path %q", name)
	}
	clean := filepath.Clean(filepath.FromSlash(name))
	if clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("unsafe archive path %q", name)
	}
	return clean, nil
}

func extractGoArchive(archivePath, dst string) error {
	if strings.HasSuffix(archivePath, ".zip") {
		return extractGoZip(archivePath, dst)
	}
	return extractGoTarGz(archivePath, dst)
}

func extractGoTarGz(archivePath, dst string) error {
	f, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()
	// Every write below goes through this root, which cannot be escaped: it
	// resolves each path inside dst and refuses "..", absolute paths, and
	// symlinked parents at the syscall level rather than by string comparison.
	root, err := os.OpenRoot(dst)
	if err != nil {
		return err
	}
	defer root.Close()
	tr := tar.NewReader(gz)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		name, err := safeArchiveName(h.Name)
		if err != nil {
			return err
		}
		switch h.Typeflag {
		case tar.TypeDir:
			if err := root.MkdirAll(name, 0o755); err != nil {
				return err
			}
		case tar.TypeReg, tar.TypeRegA:
			if dir := filepath.Dir(name); dir != "." {
				if err := root.MkdirAll(dir, 0o755); err != nil {
					return err
				}
			}
			out, err := root.OpenFile(name, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, os.FileMode(h.Mode)&0o777)
			if err != nil {
				return err
			}
			_, copyErr := io.Copy(out, tr)
			closeErr := out.Close()
			if copyErr != nil {
				return copyErr
			}
			if closeErr != nil {
				return closeErr
			}
		case tar.TypeSymlink, tar.TypeLink:
			return fmt.Errorf("archive links are not allowed: %q", h.Name)
		default:
			return fmt.Errorf("unsupported archive entry %q", h.Name)
		}
	}
}

func extractGoZip(archivePath, dst string) error {
	zr, err := zip.OpenReader(archivePath)
	if err != nil {
		return err
	}
	defer zr.Close()
	root, err := os.OpenRoot(dst)
	if err != nil {
		return err
	}
	defer root.Close()
	for _, f := range zr.File {
		name, err := safeArchiveName(f.Name)
		if err != nil {
			return err
		}
		if f.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("archive links are not allowed: %q", f.Name)
		}
		if f.FileInfo().IsDir() {
			if err := root.MkdirAll(name, 0o755); err != nil {
				return err
			}
			continue
		}
		if !f.Mode().IsRegular() {
			return fmt.Errorf("unsupported archive entry %q", f.Name)
		}
		if dir := filepath.Dir(name); dir != "." {
			if err := root.MkdirAll(dir, 0o755); err != nil {
				return err
			}
		}
		in, err := f.Open()
		if err != nil {
			return err
		}
		out, err := root.OpenFile(name, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, f.Mode().Perm())
		if err != nil {
			in.Close()
			return err
		}
		_, copyErr := io.Copy(out, in)
		inErr := in.Close()
		outErr := out.Close()
		if copyErr != nil {
			return copyErr
		}
		if inErr != nil {
			return inErr
		}
		if outErr != nil {
			return outErr
		}
	}
	return nil
}
