package process

import (
	"fmt"
	"net"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func occupyPort(t *testing.T) (net.Listener, int) {
	t.Helper()

	ln, err := net.Listen("tcp", ":0")
	require.NoError(t, err)

	_, portStr, err := net.SplitHostPort(ln.Addr().String())
	require.NoError(t, err)

	port, err := strconv.Atoi(portStr)
	require.NoError(t, err)

	require.NoError(t, ln.Close())

	ln, err = net.Listen("tcp", fmt.Sprintf(":%d", port))
	require.NoError(t, err)

	return ln, port
}

func TestDefaultPortManager_FindFreePortSkipsBusyAndReserved(t *testing.T) {
	pm := NewPortManager()

	ln, busyPort := occupyPort(t)
	defer ln.Close()

	freePort, err := pm.FindFreePort(busyPort)
	require.NoError(t, err)
	assert.NotEqual(t, busyPort, freePort)

	require.NoError(t, pm.ReservePort(freePort))
	t.Cleanup(func() { _ = pm.ReleasePort(freePort) })

	nextPort, err := pm.FindFreePort(freePort)
	require.NoError(t, err)
	assert.NotEqual(t, freePort, nextPort, "reserved ports should be skipped")
}

func TestDefaultPortManager_ReserveAndReleaseLifecycle(t *testing.T) {
	pm := NewPortManager()

	freePort, err := pm.FindFreePort(35000)
	require.NoError(t, err)

	require.NoError(t, pm.ReservePort(freePort))

	err = pm.ReservePort(freePort)
	require.Error(t, err, "double reservation should fail")

	require.NoError(t, pm.ReleasePort(freePort))

	err = pm.ReleasePort(freePort)
	require.Error(t, err, "releasing an unreserved port should fail")
}

func TestDefaultPortManager_ReserveFailsWhenSystemPortBusy(t *testing.T) {
	pm := NewPortManager()

	ln, busyPort := occupyPort(t)
	defer ln.Close()

	err := pm.ReservePort(busyPort)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not available")
}
