package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"syscall"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
)

// healthProbeTimeout bounds the /health probe so a hung foreign service on the
// port cannot stall startup. Kept short — we already know startup failed and
// are only deciding whether to exit clean or fatal.
const healthProbeTimeout = 2 * time.Second

// isAddrInUse reports whether err (as returned by ListenAndServe, possibly
// wrapped by Start) is a "port already bound" failure. errors.Is unwraps the
// net.OpError -> os.SyscallError -> syscall.Errno chain. Used only to flavor
// the exit-clean log line — it is NOT the gate: an incumbent server kills a
// newcomer well before the bind (BoltDB file lock, SQLite lock during storage
// init), so keying exit-clean off EADDRINUSE alone would miss the real-world
// failure mode entirely.
func isAddrInUse(err error) bool {
	return errors.Is(err, syscall.EADDRINUSE)
}

// probeHealthyAgentField probes url and reports whether a *healthy AgentField*
// control plane answers there. It accepts only a body that looks like our own
// /health payload (see healthCheckHandler): a recognized "status" plus the
// "version"/"checks" shape. This is the same recognition the desktop app uses
// (desktop/src/main/agentfield.ts checkControlPlane) so an unrelated dev server
// squatting on port 8080 is never mistaken for us.
func probeHealthyAgentField(url string, timeout time.Duration) bool {
	client := &http.Client{Timeout: timeout}
	resp, err := client.Get(url)
	if err != nil {
		return false
	}
	defer func() { _ = resp.Body.Close() }()

	var body struct {
		Status  string          `json:"status"`
		Version string          `json:"version"`
		Checks  json.RawMessage `json:"checks"`
	}
	// Cap the read: a foreign service could stream forever within the timeout.
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&body); err != nil {
		return false
	}
	// Require the full AgentField health shape AND a healthy verdict. An
	// unhealthy (503) AgentField still owns the port, but "already running,
	// nothing to do" should only fire when the incumbent is actually serving.
	if body.Status != "healthy" {
		return false
	}
	return body.Version != "" && len(body.Checks) > 0
}

// decideExitClean is the pure decision behind ExitCleanIfAlreadyRunning, split
// out so it can be table-tested without opening a socket. Given the startup
// error and a probe that reports whether a healthy AgentField already owns the
// configured port, it returns true when the process should exit 0 (a clean,
// idempotent no-op) instead of treating the error as fatal.
//
// Deliberately NOT gated on the error's shape: an incumbent server manifests
// as whichever shared resource the newcomer hits first — BoltDB file-lock
// timeout or SQLite lock during storage init (before any bind), or EADDRINUSE
// if storage somehow succeeded. What makes "nothing to do" honest is not which
// collision killed us but that the supervisor's desired state — a healthy
// control plane on the port — is already true, which the probe verifies with
// strict payload recognition. Any failure with no healthy AgentField answering
// (foreign service, unhealthy incumbent, genuine startup bug) stays fatal.
func decideExitClean(startErr error, probe func() bool) bool {
	if startErr == nil {
		return false
	}
	return probe()
}

// ExitCleanIfAlreadyRunning inspects a failed server startup (storage init /
// server creation, or the HTTP bind). When a healthy AgentField control plane
// already answers on the configured port, it logs an honest line and returns
// true — the caller should exit 0. This makes `agentfield server` idempotent
// under supervisors: launchd's KeepAlive={SuccessfulExit:false} reads exit 0
// as a clean stop and does not relaunch, so a second instance racing the app's
// direct-spawned server stops instead of thrashing in a crash loop. For any
// other error (foreign service on the port, or a startup failure with no
// incumbent) it returns false and the caller keeps its non-zero exit.
func ExitCleanIfAlreadyRunning(startErr error, port int) bool {
	return decideExitClean(startErr, func() bool {
		url := fmt.Sprintf("http://localhost:%d/health", port)
		if !probeHealthyAgentField(url, healthProbeTimeout) {
			return false
		}
		cause := "startup failed"
		if isAddrInUse(startErr) {
			cause = "port already bound"
		}
		logger.Logger.Info().
			Int("port", port).
			Str("cause", cause).
			Err(startErr).
			Msgf("control plane already running on port %d — nothing to do", port)
		return true
	})
}
