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

// E23 (failure path): a round trip that fails must not leave a half-open
// connection in the client's pool either.
func TestE23NodeHTTPClientReleasesConnectionsWhenTheRoundTripFails(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		hijacker, ok := response.(http.Hijacker)
		require.True(t, ok)
		conn, _, err := hijacker.Hijack()
		require.NoError(t, err)
		_ = conn.Close() // slam the connection without a response
	}))
	t.Cleanup(server.Close)

	client := NewNodeHTTPClient(time.Second)
	request, err := http.NewRequest(http.MethodGet, server.URL, nil)
	require.NoError(t, err)
	response, err := client.Do(request)
	if err == nil {
		_ = response.Body.Close()
	}
	require.Error(t, err, "a slammed connection must surface as an error")

	transport, ok := client.Transport.(*clientCloseTransport)
	require.True(t, ok)
	// A second request on the same client must open a fresh connection and
	// succeed once the server behaves; the failed one was released.
	server.Config.Handler = http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(response, "ok")
	})
	request, err = http.NewRequest(http.MethodGet, server.URL, nil)
	require.NoError(t, err)
	response, err = client.Do(request)
	require.NoError(t, err)
	_, _ = io.Copy(io.Discard, response.Body)
	require.NoError(t, response.Body.Close())
	transport.base.CloseIdleConnections()
}
