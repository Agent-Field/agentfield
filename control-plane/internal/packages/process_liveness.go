package packages

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"
)

const bootIDPath = "/proc/sys/kernel/random/boot_id"

const processIdentityTimeout = 2 * time.Second

// RuntimeProcessState distinguishes a dead recorded process from a live one
// and from a process whose identity cannot be established safely. The unknown
// state is resolved through the package health endpoint before lifecycle code
// decides whether signalling the recorded PID is allowed.
type RuntimeProcessState uint8

const (
	RuntimeProcessDead RuntimeProcessState = iota
	RuntimeProcessAliveState
	RuntimeProcessUnknown
)

type processIdentityState uint8

const (
	processIdentityGone processIdentityState = iota
	processIdentityAvailable
	processIdentityUnavailable
)

// CurrentBootID returns the Linux kernel boot identifier when available. An
// empty value on other platforms preserves the legacy PID check.
func CurrentBootID() string {
	data, err := os.ReadFile(bootIDPath)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

// CurrentProcessStartTime returns a stable identity for one incarnation of a
// process. PID alone is unsafe because operating systems and fresh container
// namespaces recycle it.
func CurrentProcessStartTime(pid int) string {
	startTime, _ := currentProcessStartTime(pid)
	return startTime
}

func currentProcessStartTime(pid int) (string, processIdentityState) {
	if pid <= 0 {
		return "", processIdentityGone
	}
	switch runtime.GOOS {
	case "linux":
		data, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
		if err != nil {
			if os.IsNotExist(err) {
				return "", processIdentityGone
			}
			return "", processIdentityUnavailable
		}
		// comm is parenthesized and may contain spaces or parentheses. Fields
		// after the final ')' begin at proc stat field 3; starttime is field 22.
		closing := strings.LastIndexByte(string(data), ')')
		if closing < 0 {
			return "", processIdentityUnavailable
		}
		fields := strings.Fields(string(data)[closing+1:])
		if len(fields) <= 19 {
			return "", processIdentityUnavailable
		}
		bootTime := linuxBootTime()
		return "linux:" + bootTime + ":" + fields[19], processIdentityAvailable
	case "darwin":
		return commandProcessStartTimeResult(processIdentityTimeout, "ps", "-o", "lstart=", "-p", strconv.Itoa(pid))
	case "windows":
		command := fmt.Sprintf("(Get-Process -Id %d -ErrorAction Stop).StartTime.ToUniversalTime().ToString('o')", pid)
		return commandProcessStartTimeResult(processIdentityTimeout, "powershell", "-NoProfile", "-NonInteractive", "-Command", command)
	default:
		return "", processIdentityUnavailable
	}
}

func linuxBootTime() string {
	data, err := os.ReadFile("/proc/stat")
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "btime ") {
			return strings.TrimSpace(strings.TrimPrefix(line, "btime "))
		}
	}
	return ""
}

func commandProcessStartTime(name string, args ...string) string {
	startTime, _ := commandProcessStartTimeResult(processIdentityTimeout, name, args...)
	return startTime
}

func commandProcessStartTimeResult(timeout time.Duration, name string, args ...string) (string, processIdentityState) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	output, err := exec.CommandContext(ctx, name, args...).Output()
	if err != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return "", processIdentityUnavailable
		}
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			return "", processIdentityGone
		}
		return "", processIdentityUnavailable
	}
	startTime := strings.Join(strings.Fields(string(output)), " ")
	if startTime == "" {
		return "", processIdentityUnavailable
	}
	return startTime, processIdentityAvailable
}

// RuntimeProcessStatus performs the signal-safe identity check used before
// lifecycle decisions. A missing start token (legacy registry entry), an
// unsupported platform, or a timed-out platform probe is unknown rather than
// dead and must be resolved through the recorded health endpoint.
func RuntimeProcessStatus(info RuntimeInfo) RuntimeProcessState {
	return runtimeProcessStatus(info, currentProcessStartTime, processExists)
}

func runtimeProcessStatus(
	info RuntimeInfo,
	startTime func(int) (string, processIdentityState),
	alive func(int) bool,
) RuntimeProcessState {
	if info.PID == nil || *info.PID <= 0 {
		return RuntimeProcessDead
	}
	currentBootID := CurrentBootID()
	if info.BootID != "" && currentBootID != "" && info.BootID != currentBootID {
		return RuntimeProcessDead
	}
	if strings.TrimSpace(info.StartTime) == "" {
		return RuntimeProcessUnknown
	}
	current, state := startTime(*info.PID)
	switch state {
	case processIdentityUnavailable:
		return RuntimeProcessUnknown
	case processIdentityGone:
		return RuntimeProcessDead
	}
	if current != info.StartTime || !alive(*info.PID) {
		return RuntimeProcessDead
	}
	return RuntimeProcessAliveState
}

// RuntimeProcessAlive rejects a recycled PID before probing it with a signal.
// BootID remains a useful host-reboot guard, but process start time is the
// container-safe identity.
func RuntimeProcessAlive(info RuntimeInfo) bool {
	return runtimeProcessAlive(info, CurrentProcessStartTime, processExists)
}

// RuntimePIDAlive performs the cheap liveness probe used by status/dashboard
// reads. Process-start identity can require ps or PowerShell on non-Linux
// platforms, so it is reserved for maintenance and stop-before-signal paths.
func RuntimePIDAlive(info RuntimeInfo) bool {
	if info.PID == nil || *info.PID <= 0 {
		return false
	}
	currentBootID := CurrentBootID()
	if info.BootID != "" && currentBootID != "" && info.BootID != currentBootID {
		return false
	}
	return processExists(*info.PID)
}

func runtimeProcessAlive(info RuntimeInfo, startTime func(int) string, alive func(int) bool) bool {
	if info.PID == nil || *info.PID <= 0 {
		return false
	}
	if info.StartTime != "" {
		current := startTime(*info.PID)
		if current == "" || current != info.StartTime {
			return false
		}
	}
	currentBootID := CurrentBootID()
	if info.BootID != "" && currentBootID != "" && info.BootID != currentBootID {
		return false
	}
	return alive(*info.PID)
}
