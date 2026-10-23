//go:build !windows

package harness

import (
	"fmt"
	"os"
	"syscall"
)

func codeAFTempName() string {
	return fmt.Sprintf("agentfield-codeaf-%d", os.Getuid())
}

// Kept injectable so foreign ownership can be tested without chown privileges.
var codeAFTempOwnedByCurrentUser = func(info os.FileInfo) bool {
	stat, ok := info.Sys().(*syscall.Stat_t)
	return ok && stat.Uid == uint32(os.Getuid())
}

func protectCodeAFTempDirectory(path string, info os.FileInfo) error {
	if !codeAFTempOwnedByCurrentUser(info) {
		return fmt.Errorf("directory is not owned by uid %d", os.Getuid())
	}
	if info.Mode()&(os.ModePerm|os.ModeSetuid|os.ModeSetgid|os.ModeSticky) != 0o700 {
		return os.Chmod(path, 0o700)
	}
	return nil
}
