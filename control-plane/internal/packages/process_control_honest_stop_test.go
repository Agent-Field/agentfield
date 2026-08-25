package packages

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A Go node (no /shutdown endpoint) that answers on its port while the
// recorded PID is a different process cannot be stopped by the control
// plane; Stop must say so rather than report success and erase the record.
func TestStopRefusesToReportSuccessForALiveGoNodeItCannotSignal(t *testing.T) {
	home := t.TempDir()
	pkgDir := filepath.Join(home, "packages", "demo")
	require.NoError(t, os.MkdirAll(pkgDir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"),
		[]byte("name: demo\nversion: 1.0.0\nlanguage: go\nagent_node:\n  node_id: demo\n"), 0o644))

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	node := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"status":"ok","node_id":"demo"}`)
	}))
	node.Listener = listener
	node.Start()
	t.Cleanup(node.Close)
	port := listener.Addr().(*net.TCPAddr).Port

	pid := os.Getpid() // alive, but recorded with a different identity
	entry := InstalledPackage{Name: "demo", Path: pkgDir, Status: "running",
		Runtime: RuntimeInfo{PID: &pid, Port: &port, StartTime: "a-different-process"}}

	assessment := AssessRecordedProcessWith(context.Background(), "demo", entry, RuntimeProcessStatus, ProbeHealthIdentity, ReadConfirmationPolicy())
	require.Equal(t, RecordedProcessOursHealthy, assessment.Ownership)
	require.False(t, assessment.SignalAllowed)

	result, err := StopRecordedProcessWithAssessment(context.Background(), "demo", entry, assessment)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "stop it manually")
	assert.False(t, result.InterruptSent)
	assert.False(t, result.ForceKillNeeded)
}
