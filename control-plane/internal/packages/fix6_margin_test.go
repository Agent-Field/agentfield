package packages

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHostingPlatformClassifiesRailwayDockerAndLocal(t *testing.T) {
	t.Setenv("RAILWAY_SERVICE_ID", "svc_123")
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	assert.Equal(t, HostingRailway, HostingPlatform())
	assert.True(t, HostedInContainer())

	t.Setenv("RAILWAY_SERVICE_ID", "")
	t.Setenv("AGENTFIELD_HOME", "/data")
	assert.Equal(t, HostingDocker, HostingPlatform())
	assert.True(t, HostedInContainer())

	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	if _, err := os.Stat("/.dockerenv"); err != nil {
		assert.Equal(t, HostingLocal, HostingPlatform())
		assert.False(t, HostedInContainer())
	}
}

// F1: the wall-clock start of the current process is knowable on Linux and
// is what a legacy started_at record is compared against.
func TestProcessStartWallClockOfTheCurrentProcess(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("wall-clock start via /proc is Linux-only; other platforms shell out")
	}
	started, ok := ProcessStartWallClock(os.Getpid())
	require.True(t, ok)
	assert.WithinDuration(t, time.Now(), started, time.Hour, "the test binary started recently")
	assert.False(t, started.After(time.Now().Add(time.Minute)), "a process cannot start in the future")

	_, ok = ProcessStartWallClock(0)
	assert.False(t, ok)
	_, ok = ProcessStartWallClock(1<<22 - 1)
	assert.False(t, ok, "a PID that does not exist has no start time")

	ticks, ok := linuxClockTicks()
	require.True(t, ok)
	assert.NotZero(t, ticks)
}

// F1 (E1/E2 through the public API): a legacy record with only started_at is
// ours when the process started when the record says, and a recycled PID when
// the process is younger than the record.
func TestRuntimeProcessStatusUsesStartedAtForLegacyRecords(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("relies on /proc start times")
	}
	pid := os.Getpid()
	started, ok := ProcessStartWallClock(pid)
	require.True(t, ok)

	recordedAtLaunch := started.Add(10 * time.Second).UTC().Format(time.RFC3339)
	assert.Equal(t, RuntimeProcessAliveState, RuntimeProcessStatus(RuntimeInfo{PID: &pid, StartedAt: &recordedAtLaunch}),
		"a process that started when the record was written is ours even without start_time")

	recordedLongBefore := started.Add(-time.Hour).UTC().Format(time.RFC3339)
	assert.Equal(t, RuntimeProcessDead, RuntimeProcessStatus(RuntimeInfo{PID: &pid, StartedAt: &recordedLongBefore}),
		"a process younger than the record is a recycled PID")

	recordedLongAfter := started.Add(time.Hour).UTC().Format(time.RFC3339)
	assert.Equal(t, RuntimeProcessDead, RuntimeProcessStatus(RuntimeInfo{PID: &pid, StartedAt: &recordedLongAfter}),
		"a process much older than the record is not the one recorded")

	garbage := "not-a-timestamp"
	assert.Equal(t, RuntimeProcessUnknown, RuntimeProcessStatus(RuntimeInfo{PID: &pid, StartedAt: &garbage}))

	gone := 1<<22 - 1
	assert.Equal(t, RuntimeProcessDead, RuntimeProcessStatus(RuntimeInfo{PID: &gone, StartedAt: &recordedAtLaunch}))
}

// F6: the registry store's error paths and its atomic write semantics.
func TestInstallationRegistryStoreEdgeCases(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "installed.yaml")

	registry, err := LoadInstallationRegistry(path)
	require.NoError(t, err, "a missing registry in an existing home is empty, not an error")
	assert.Empty(t, registry.Installed)

	require.NoError(t, os.WriteFile(path, []byte("installed: [not a map"), 0o600))
	_, err = LoadInstallationRegistry(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to parse registry")
	err = UpdateInstallationRegistry(path, func(*InstallationRegistry) error { return nil })
	require.Error(t, err, "an unreadable registry aborts the transaction")

	require.NoError(t, WriteInstallationRegistryAtomic(path, &InstallationRegistry{Installed: map[string]InstalledPackage{"a": {Name: "a", Status: "stopped"}}}))
	info, err := os.Stat(path)
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(0o600), info.Mode().Perm(), "the atomic write preserves the existing file mode")

	mutateErr := assert.AnError
	err = UpdateInstallationRegistryAtomic(path, func(r *InstallationRegistry) error {
		r.Installed["b"] = InstalledPackage{Name: "b"}
		return mutateErr
	})
	assert.ErrorIs(t, err, mutateErr)
	reloaded, err := LoadInstallationRegistry(path)
	require.NoError(t, err)
	_, leaked := reloaded.Installed["b"]
	assert.False(t, leaked, "a failed mutation leaves the file untouched")

	require.NoError(t, UpdateInstallationRegistryAtomic(path, func(r *InstallationRegistry) error {
		r.Installed["b"] = InstalledPackage{Name: "b"}
		return nil
	}))
	reloaded, err = LoadInstallationRegistry(path)
	require.NoError(t, err)
	assert.Len(t, reloaded.Installed, 2)

	nested := filepath.Join(dir, "missing", "home", "installed.yaml")
	require.NoError(t, WriteInstallationRegistryAtomic(nested, &InstallationRegistry{}), "the atomic write creates the directory")
	require.NoError(t, WriteInstallationRegistry(nested, &InstallationRegistry{Installed: map[string]InstalledPackage{"c": {Name: "c"}}}))
	reloaded, err = LoadInstallationRegistry(nested)
	require.NoError(t, err)
	assert.Contains(t, reloaded.Installed, "c")

	blocked := filepath.Join(path, "child.yaml") // parent is a regular file
	_, err = LoadInstallationRegistry(blocked)
	require.Error(t, err)
}
