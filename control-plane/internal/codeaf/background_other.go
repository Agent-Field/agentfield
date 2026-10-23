//go:build !unix && !windows

package codeaf

import "os/exec"

func detach(*exec.Cmd) {}
