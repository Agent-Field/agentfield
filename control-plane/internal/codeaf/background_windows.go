package codeaf

import (
	"os/exec"
	"syscall"
)

func detach(cmd *exec.Cmd) {
	const detachedProcess = 0x00000008
	const newProcessGroup = 0x00000200
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: detachedProcess | newProcessGroup, HideWindow: true}
}
