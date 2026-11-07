package agenturl

import (
	"net/http"
	"strings"
	"sync/atomic"
)

// ── A NODE WITH NO ADDRESS, REACHED DOWN ITS LINK ───────────────────────────
//
// A node that opens the node link (internal/nodelink) and has no port anyone
// can dial registers a base_url on the reserved host "<node_id>.nodelink"
// (LinkHost). Every request the control plane sends a node — a dispatch, a
// nested call, a cancel, a steer, a health probe — goes through NewClient, and
// a request for a link host is handed to the link carrier instead of being
// dialed. THE LINK IS A CARRIER: the node answers the same request on its own
// handler and the caller reads the same response, so no caller learns which
// carrier brought it.

// LinkHostSuffix marks a base_url host that is reached only down a node link.
const LinkHostSuffix = ".nodelink"

// LinkHost is the base_url host a link-only node registers: node_id.nodelink.
func LinkHost(nodeID string) string { return nodeID + LinkHostSuffix }

// LinkCarrier carries one HTTP request to a node down its link.
type LinkCarrier interface {
	RoundTripLink(node string, req *http.Request) (*http.Response, error)
}

var carrier atomic.Value // holds carrierBox

type carrierBox struct{ c LinkCarrier }

// SetLinkCarrier installs the control plane's link hub as the carrier.
func SetLinkCarrier(c LinkCarrier) { carrier.Store(carrierBox{c}) }

// LinkNode is the node a URL host names when it is a link host.
func LinkNode(host string) (string, bool) {
	host = strings.ToLower(strings.TrimSpace(host))
	if i := strings.LastIndexByte(host, ':'); i >= 0 && !strings.Contains(host[i:], "]") {
		host = host[:i]
	}
	if !strings.HasSuffix(host, LinkHostSuffix) {
		return "", false
	}
	node := strings.TrimSuffix(host, LinkHostSuffix)
	return node, node != ""
}

// linkAware sends a link host's requests to the carrier and every other
// request to base.
type linkAware struct{ base http.RoundTripper }

func (l linkAware) RoundTrip(req *http.Request) (*http.Response, error) {
	if node, ok := LinkNode(req.URL.Host); ok {
		if box, _ := carrier.Load().(carrierBox); box.c != nil {
			return box.c.RoundTripLink(node, req)
		}
	}
	return l.base.RoundTrip(req)
}
