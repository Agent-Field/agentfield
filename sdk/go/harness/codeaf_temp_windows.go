//go:build windows

package harness

import (
	"fmt"
	"os"
	"syscall"
)

func codeAFTempName() string {
	return "agentfield-codeaf"
}

func protectCodeAFTempDirectory(_ string, info os.FileInfo) error {
	if data, ok := info.Sys().(*syscall.Win32FileAttributeData); ok && data.FileAttributes&syscall.FILE_ATTRIBUTE_REPARSE_POINT != 0 {
		return fmt.Errorf("directory is a reparse point")
	}
	return nil
}
