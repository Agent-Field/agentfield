package harness

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
)

var codeAFTempWarningOnce sync.Once

func codeAFTempBase() string {
	return filepath.Join(os.TempDir(), codeAFTempName())
}

// Check the base before touching a child: neither an existing public directory
// nor a symlink establishes a trusted place to read binaries or write state.
func prepareCodeAFTempDirectory(name string) (string, error) {
	base := codeAFTempBase()
	dir := filepath.Join(base, name)
	for _, path := range []string{base, dir} {
		if err := secureCodeAFTempDirectory(path); err != nil {
			err = fmt.Errorf("unusable CodeAF temp fallback at %s: %w", path, err)
			codeAFTempWarningOnce.Do(func() { log.Printf("WARNING %v", err) })
			return "", err
		}
	}
	return dir, nil
}

func secureCodeAFTempDirectory(path string) error {
	if err := os.Mkdir(path, 0o700); err != nil && !os.IsExist(err) {
		return err
	}
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("path is not a real directory")
	}
	return protectCodeAFTempDirectory(path, info)
}
