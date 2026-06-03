package services

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestPresenceManagerHasFreshLease(t *testing.T) {
	pm, _ := setupPresenceManagerTest(t)
	pm.leases["fresh"] = &presenceLease{LastSeen: time.Now().Add(-time.Second)}
	pm.leases["stale"] = &presenceLease{LastSeen: time.Now().Add(-10 * time.Second)}
	pm.leases["offline"] = &presenceLease{LastSeen: time.Now(), MarkedOffline: true}

	require.True(t, pm.HasFreshLease("fresh"))
	require.False(t, pm.HasFreshLease("stale"))
	require.False(t, pm.HasFreshLease("offline"))
	require.False(t, pm.HasFreshLease("missing"))
}
