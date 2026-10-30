package agenturl

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync/atomic"
	"time"
)

type hooks struct {
	lookup func(context.Context, string) ([]string, error)
	dial   func(context.Context, string, string) (net.Conn, error)
}

var testHooks atomic.Pointer[hooks]

func networkHooks() hooks {
	if h := testHooks.Load(); h != nil {
		return *h
	}
	return hooks{lookup: net.DefaultResolver.LookupHost, dial: (&net.Dialer{}).DialContext}
}

// SetNetworkHooksForTesting injects DNS and TCP connection hooks for hermetic
// policy tests. It must not be used in production or by concurrent tests.
// A nil hook keeps the normal implementation. The returned function restores it.
func SetNetworkHooksForTesting(lookup func(context.Context, string) ([]string, error), dial func(context.Context, string, string) (net.Conn, error)) func() {
	h := networkHooks()
	if lookup != nil {
		h.lookup = lookup
	}
	if dial != nil {
		h.dial = dial
	}
	previous := testHooks.Swap(&h)
	return func() { testHooks.Store(previous) }
}

// NewTransport checks all DNS answers and dials a checked IP directly, avoiding
// a second lookup. Environment proxies are intentionally disabled: their DNS
// resolution would evade the destination policy. TLS still uses the URL hostname.
func NewTransport(connectTimeout time.Duration) *http.Transport {
	return &http.Transport{
		ForceAttemptHTTP2:   true,
		TLSHandshakeTimeout: 10 * time.Second,
		IdleConnTimeout:     90 * time.Second,
		DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
			if connectTimeout > 0 {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, connectTimeout)
				defer cancel()
			}
			host, port, err := net.SplitHostPort(address)
			if err != nil {
				return nil, fmt.Errorf("agent endpoint invalid dial address: %w", err)
			}
			p := currentPolicy()
			if err := p.check(host, nil); err != nil {
				return nil, err
			}
			h := networkHooks()
			ips := []string{host}
			if literalIP(host) == nil {
				ips, err = h.lookup(ctx, host)
				if err != nil {
					return nil, fmt.Errorf("agent endpoint DNS lookup failed: %w", err)
				}
			}
			if len(ips) == 0 {
				return nil, fmt.Errorf("agent endpoint DNS returned no addresses")
			}
			for _, s := range ips {
				ip := literalIP(s)
				if ip == nil {
					return nil, fmt.Errorf("agent endpoint DNS returned an invalid IP")
				}
				if err := p.check(host, ip); err != nil {
					return nil, err
				}
			}
			var lastErr error
			for i, ip := range ips {
				// Reserve a share of the remaining connect budget for each
				// checked address, including a fallback to the other IP family.
				attemptCtx := ctx
				cancel := func() {}
				if deadline, ok := ctx.Deadline(); ok && len(ips)-i > 1 {
					attemptCtx, cancel = context.WithTimeout(ctx, time.Until(deadline)/time.Duration(len(ips)-i))
				}
				conn, err := h.dial(attemptCtx, network, net.JoinHostPort(ip, port))
				cancel()
				if err == nil {
					return conn, nil
				}
				lastErr = err
			}
			return nil, lastErr
		},
	}
}

func origin(u *url.URL) string {
	port := u.Port()
	if port == "" {
		if strings.EqualFold(u.Scheme, "https") {
			port = "443"
		} else {
			port = "80"
		}
	}
	return strings.ToLower(u.Scheme) + "://" + net.JoinHostPort(strings.ToLower(u.Hostname()), port)
}
func checkRedirect(req *http.Request, via []*http.Request) error {
	if len(via) >= 10 {
		return fmt.Errorf("stopped after 10 redirects")
	}
	if len(via) > 0 && origin(req.URL) != origin(via[0].URL) {
		return fmt.Errorf("agent endpoint cross-origin redirect refused")
	}
	if req.URL.User != nil {
		return fmt.Errorf("agent endpoint redirect must not include user info")
	}
	return currentPolicy().check(req.URL.Hostname(), nil)
}

// NewClient preserves the caller's total timeout (zero permits streaming).
func NewClient(timeout time.Duration) *http.Client {
	return &http.Client{Timeout: timeout, Transport: NewTransport(30 * time.Second), CheckRedirect: checkRedirect}
}
