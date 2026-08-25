package packages

import (
	"net/http"
	"time"
)

// NewNodeHTTPClient builds a short-lived CP-to-node client. Node lifecycle
// traffic must not leave an idle keep-alive socket behind on the node's port,
// because the SDK cannot immediately reclaim a port with that socket in
// TIME_WAIT after shutdown.
func NewNodeHTTPClient(timeout time.Duration) *http.Client {
	return &http.Client{
		Timeout: timeout,
		Transport: &http.Transport{
			DisableKeepAlives: true,
		},
	}
}
