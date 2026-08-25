package packages

import (
	"io"
	"net/http"
	"sync"
	"time"
)

// NewNodeHTTPClient builds a short-lived control-plane-to-node client for
// lifecycle traffic (readiness, identity, shutdown, capabilities).
//
// The connection must be closed by the control plane, never by the node:
// whichever side closes first keeps the socket in TIME_WAIT for ~60 s, and a
// TIME_WAIT socket on the node's port makes the Python SDK's availability
// check reject exactly the port the control plane assigned on the next start
// (the SDK's probe binds without SO_REUSEADDR). "Connection: close" would ask
// the node to close first, so the client keeps the connection alive for the
// request and then closes it from its own side as soon as the body is done.
func NewNodeHTTPClient(timeout time.Duration) *http.Client {
	base := &http.Transport{
		Proxy:               nil, // node traffic is loopback; never route it through a proxy
		MaxIdleConns:        4,
		MaxIdleConnsPerHost: 4,
		IdleConnTimeout:     5 * time.Second,
		DisableCompression:  true,
	}
	return &http.Client{Timeout: timeout, Transport: &clientCloseTransport{base: base}}
}

// clientCloseTransport releases the connection from the client side once the
// response body is closed (or the round trip failed).
type clientCloseTransport struct {
	base *http.Transport
}

func (t *clientCloseTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	response, err := t.base.RoundTrip(request)
	if err != nil {
		t.base.CloseIdleConnections()
		return nil, err
	}
	response.Body = &closeIdleBody{ReadCloser: response.Body, transport: t.base}
	return response, nil
}

type closeIdleBody struct {
	io.ReadCloser
	transport *http.Transport
	once      sync.Once
}

func (b *closeIdleBody) Close() error {
	err := b.ReadCloser.Close()
	b.once.Do(b.transport.CloseIdleConnections)
	return err
}
