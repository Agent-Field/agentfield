//go:build windows

package packages

import (
	"os/exec"
	"strconv"
	"strings"
)

func gracefulSignal(pid int) error {
	return exec.Command("taskkill", "/PID", strconv.Itoa(pid)).Run()
}

func forceKill(pid int) error {
	return exec.Command("taskkill", "/F", "/PID", strconv.Itoa(pid)).Run()
}

func processExists(pid int) bool {
	out, err := exec.Command(
		"tasklist", "/FI", "PID eq "+strconv.Itoa(pid), "/NH", "/FO", "CSV",
	).Output()
	if err != nil {
		return false
	}
	return strings.Contains(string(out), `"`+strconv.Itoa(pid)+`"`)
}
