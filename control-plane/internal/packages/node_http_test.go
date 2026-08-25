package packages

import (
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestE23NodeHTTPClientsCloseConnectionsWithoutReuse(t *testing.T) {
	var opened atomic.Int32
	var closed atomic.Int32
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(response, `{"status":"ok","node_id":"demo"}`)
	}))
	server.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		switch state {
		case http.StateNew:
			opened.Add(1)
		case http.StateClosed:
			closed.Add(1)
		}
	}
	server.Start()
	t.Cleanup(server.Close)

	for index := 0; index < 4; index++ {
		request, err := http.NewRequest(http.MethodGet, server.URL, nil)
		require.NoError(t, err)
		request.Close = true
		response, err := NewNodeHTTPClient(time.Second).Do(request)
		require.NoError(t, err)
		_, err = io.Copy(io.Discard, response.Body)
		require.NoError(t, err)
		require.NoError(t, response.Body.Close())
	}

	require.Eventually(t, func() bool {
		return opened.Load() == 4 && closed.Load() == 4
	}, time.Second, 10*time.Millisecond, "node probes reused or retained a TCP connection")
}
