//go:build unix

package codeaf

import (
	"fmt"
	"os"
	"syscall"
)

func trustTempDirectory(path string, info os.FileInfo) error {
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || stat.Uid != uint32(os.Getuid()) {
		return fmt.Errorf("%s is not owned by the current user", path)
	}
	if info.Mode()&(os.ModePerm|os.ModeSetuid|os.ModeSetgid|os.ModeSticky) != 0700 {
		return os.Chmod(path, 0700)
	}
	return nil
}
