package packages

import (
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// E23: after a probe, the control plane — not the node — closes the
// connection. A node that closes first parks TIME_WAIT on its own port and
// the SDK then refuses that port on the next start.
func TestE23NodeHTTPClientsAreClosedByTheControlPlaneNotTheNode(t *testing.T) {
	var mu sync.Mutex
	states := map[net.Conn][]http.ConnState{}
	var sawConnectionClose bool
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Close || request.Header.Get("Connection") == "close" {
			mu.Lock()
			sawConnectionClose = true
			mu.Unlock()
		}
		response.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(response, `{"status":"ok","node_id":"demo"}`)
	}))
	server.Config.ConnState = func(conn net.Conn, state http.ConnState) {
		mu.Lock()
		states[conn] = append(states[conn], state)
		mu.Unlock()
	}
	server.Start()
	t.Cleanup(server.Close)

	for index := 0; index < 4; index++ {
		request, err := http.NewRequest(http.MethodGet, server.URL, nil)
		require.NoError(t, err)
		response, err := NewNodeHTTPClient(time.Second).Do(request)
		require.NoError(t, err)
		_, err = io.Copy(io.Discard, response.Body)
		require.NoError(t, err)
		require.NoError(t, response.Body.Close())
	}

	require.Eventually(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		if len(states) != 4 {
			return false
		}
		for _, sequence := range states {
			if sequence[len(sequence)-1] != http.StateClosed {
				return false
			}
		}
		return true
	}, 2*time.Second, 10*time.Millisecond, "every probe connection must end closed, none retained")

	mu.Lock()
	defer mu.Unlock()
	require.False(t, sawConnectionClose, "the probe must not ask the node to close the connection")
	for conn, sequence := range states {
		idle := false
		for _, state := range sequence {
			if state == http.StateIdle {
				idle = true
			}
		}
		require.True(t, idle, "connection %v was never idle: the node closed it, not the client (%v)", conn.RemoteAddr(), sequence)
	}
}
