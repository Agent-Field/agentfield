package codeaf

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"sync"
)

var tempWarning sync.Once

func trustedTempDir() (dir string, err error) {
	defer func() {
		if err != nil {
			tempWarning.Do(func() { log.Printf("warning: CodeAF temp fallback is unusable: %v", err) })
		}
	}()
	name := "agentfield-codeaf"
	if runtime.GOOS != "windows" {
		name = fmt.Sprintf("%s-%d", name, os.Getuid())
	}
	base := filepath.Join(os.TempDir(), name)
	for _, path := range []string{base, filepath.Join(base, Version)} {
		if err := os.Mkdir(path, 0700); err != nil && !os.IsExist(err) {
			return "", err
		}
		info, err := os.Lstat(path)
		if err != nil {
			return "", err
		}
		if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
			return "", fmt.Errorf("%s is not a real directory", path)
		}
		if err := trustTempDirectory(path, info); err != nil {
			return "", err
		}
	}
	return filepath.Join(base, Version), nil
}
