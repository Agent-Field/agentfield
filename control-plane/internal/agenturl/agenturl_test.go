package agenturl

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
)

// C1-C4: endpoint shape, default topology compatibility and every D4 embedding.
func TestEndpointValidation(t *testing.T) {
	defer SetNetworkHooksForTesting(func(context.Context, string) ([]string, error) { return nil, errors.New("DNS not ready") }, nil)()
	for _, raw := range []string{
		"http://127.0.0.1:18190/x?", "http://127.0.0.1/x?a=b", "http://127.0.0.1/x#", "http://127.0.0.1/x#frag",
		"", "http:opaque", "ftp://host/x", "http://", "http://user@host", " http://host", "http://host/white space", "http://host/\n", "http://host/%0a",
		"http://host/a/../b", "http://host/./b", "http://host/%2e%2e/b", "http://host/a%2fb", "http://host/a%5Cb", "http://host/%252f", "http://host/%252e%252e", "http://host/%25zz/%252e%252e", "http://host/%25zz/%252f", "http://host/%25zz/%250a", "http://host/%zz",
	} {
		t.Run(raw, func(t *testing.T) { _, err := Validate(context.Background(), raw); require.Error(t, err) })
	}
	for _, host := range deniedHosts() {
		t.Run(host, func(t *testing.T) {
			_, err := Validate(context.Background(), "http://"+host+"/prefix")
			require.Error(t, err)
		})
	}
	for _, raw := range []string{"http://127.0.0.1:18190", "http://localhost:18190", "http://10.1.2.3", "http://172.16.1.2", "http://172.31.2.3", "http://192.168.1.2", "http://100.64.1.2", "http://[fd12::1]", "http://[::]", "http://0.0.0.0", "http://0.250.250.254", "http://host.docker.internal", "http://host/prefix%25", "http://host/prefix%25zz", "http://agent-service", "https://example-agent.internal/prefix", "https://function.lambda-url.example/prefix"} {
		t.Run(raw, func(t *testing.T) {
			got, err := Validate(context.Background(), raw)
			require.NoError(t, err)
			require.Equal(t, raw, got)
		})
	}
	got, err := Normalize("HTTPS://EXAMPLE.INTERNAL/prefix/")
	require.NoError(t, err)
	require.Equal(t, "https://example.internal/prefix", got)
}
func deniedHosts() []string {
	return []string{"169.254.169.254", "169.254.0.1", "[fe80::1]", "[fe80::1%25eth0]", "[fd00:ec2::254]", "[::ffff:169.254.169.254]", "[::169.254.169.254]", "[64:ff9b::a9fe:a9fe]", "[64:ff9b:1:a9fe:a9:fe00::]", "[64:ff9b:1::a9fe:a9fe]", "[64:ff9b:1:a9:fe:a9fe::]", "[64:ff9b:1:0:a9:fea9:fe00:0]", "[2002:a9fe:a9fe::1]", "[2001:0:4136:e378:8000:63bf:5601:5601]", "[2001:0:a9fe:a9fe:8000:63bf:f5ff:fffe]", "224.0.0.1", "255.255.255.255", "[ff02::1]", "metadata", "MeTaDaTa.", "metadata.google.internal", "metadata.goog."}
}

// C4: strict exceptions follow webhook hostname/wildcard/CIDR semantics and
// can never override the always-on deny, even via an embedded IPv4 address.
func TestStrictPolicy(t *testing.T) {
	defer Configure(false, nil)
	strict := []string{"127.0.0.1", "::1", "10.0.0.1", "172.16.0.1", "192.168.0.1", "fd12::1", "100.64.0.1", "198.18.0.1", "0.0.0.0", "0.250.250.254", "::", "::10.0.0.1", "64:ff9b::a00:1", "64:ff9b:1:a00:0:100::", "2002:a00:1::", "2001:0:808:808::f5ff:fffe"}
	for _, ip := range strict {
		require.Error(t, (&policy{strict: true}).check("agent", net.ParseIP(ip)), ip)
	}
	require.NoError(t, (&policy{strict: true}).check("agent", net.ParseIP("8.8.8.8")))
	for _, tc := range []struct {
		host, ip string
		allowed  []string
		ok       bool
	}{
		{"Agent.INTERNAL", "10.0.0.1", []string{"agent.internal"}, true},
		{"pod.cluster.local", "10.0.0.1", []string{"*.cluster.local"}, true},
		{"cluster.local", "10.0.0.1", []string{"*.cluster.local"}, false},
		{"pod", "10.0.0.1", []string{"10.0.0.0/8"}, true},
		{"pod", "192.168.0.1", []string{"10.0.0.0/8"}, false},
		{"pod", "fd12::1", []string{"fc00::/7"}, true},
	} {
		err := (&policy{strict: true, allowed: tc.allowed}).check(tc.host, net.ParseIP(tc.ip))
		require.Equal(t, tc.ok, err == nil, tc)
	}
	Configure(true, []string{" localhost ", "", "127.0.0.0/8"})
	_, err := Normalize("http://localhost")
	require.NoError(t, err)
	_, err = Normalize("http://127.0.0.1")
	require.NoError(t, err)
	Configure(true, []string{"localhost"})
	_, err = Normalize("http://foo.localhost.")
	require.Error(t, err)
	for _, host := range deniedHosts() {
		raw := "http://" + host
		Configure(true, []string{strings.Trim(host, "[]"), "0.0.0.0/0", "::/0"})
		_, err := Normalize(raw)
		require.Error(t, err, raw)
	}
}

// C5-C6: prefix retention, no query/fragment confusion, segment confinement.
func TestJoin(t *testing.T) {
	for _, s := range []string{"", ".", "..", "a/b", "%2f", "%252f", "%2e%2e", "a\\b", "bad\n"} {
		_, err := Join("http://127.0.0.1/prefix", "reasoners", s)
		require.Error(t, err, s)
	}
	for _, s := range []string{"echo", "a?b", "a#b", "two words", "é", "%zz", "100%"} {
		u, err := Join("http://127.0.0.1/prefix/", "reasoners", s)
		require.NoError(t, err)
		require.Equal(t, "/prefix/reasoners/"+s, u.Path)
		require.Empty(t, u.RawQuery)
		require.Empty(t, u.Fragment)
		parsed, err := url.Parse(u.String())
		require.NoError(t, err)
		require.Equal(t, u.Path, parsed.Path)
	}
	_, err := Join("http://127.0.0.1/prefix?", "status")
	require.Error(t, err)
	u, err := Join("http://127.0.0.1/pre%3Ffix", "status")
	require.NoError(t, err)
	require.Equal(t, "http://127.0.0.1/pre%3Ffix/status", u.String())
}

// C3-C4, C7: registration checks every DNS answer, permits DNS not ready,
// and strict hostname exceptions still check the resolved IP for D4.
func TestRegistrationDNS(t *testing.T) {
	defer Configure(false, nil)
	for _, tc := range []struct {
		ips     []string
		strict  bool
		allowed []string
		ok      bool
	}{
		{[]string{"10.0.0.1"}, false, nil, true},
		{[]string{"127.0.0.1", "169.254.1.2"}, false, nil, false},
		{[]string{"10.0.0.1"}, true, nil, false},
		{[]string{"10.0.0.1"}, true, []string{"*.internal"}, true},
		{[]string{"169.254.1.2"}, true, []string{"*.internal"}, false},
		{nil, false, nil, true},
	} {
		Configure(tc.strict, tc.allowed)
		restore := SetNetworkHooksForTesting(func(context.Context, string) ([]string, error) { return tc.ips, nil }, nil)
		_, err := Validate(context.Background(), "http://pod.internal/prefix")
		restore()
		require.Equal(t, tc.ok, err == nil, tc)
	}
}

// C7: the checked address is the one dialed; mixed DNS answers, metadata names,
// lookup errors and rebinding are refused before even the fake dial is invoked.
func TestDialPolicy(t *testing.T) {
	var dialed string
	transport := NewTransport(time.Second)
	for _, ips := range [][]string{{"169.254.169.254"}, {"127.0.0.1", "169.254.1.2"}, {}, {"not-an-IP"}} {
		restore := SetNetworkHooksForTesting(func(context.Context, string) ([]string, error) { return ips, nil }, func(context.Context, string, string) (net.Conn, error) { t.Fatal("forbidden dial"); return nil, nil })
		_, err := transport.DialContext(context.Background(), "tcp", "agent.test:80")
		require.Error(t, err)
		restore()
	}
	restore := SetNetworkHooksForTesting(func(context.Context, string) ([]string, error) { return []string{"127.0.0.1"}, nil }, func(ctx context.Context, network, addr string) (net.Conn, error) {
		dialed = addr
		return nil, errors.New("fake connect")
	})
	_, err := transport.DialContext(context.Background(), "tcp", "agent.test:8080")
	require.ErrorContains(t, err, "fake connect")
	require.Equal(t, "127.0.0.1:8080", dialed)
	_, err = transport.DialContext(context.Background(), "tcp", "invalid address")
	require.Error(t, err)
	_, err = transport.DialContext(context.Background(), "tcp", "metadata:80")
	require.ErrorContains(t, err, "metadata")
	restore()
	restore = SetNetworkHooksForTesting(func(context.Context, string) ([]string, error) { return nil, errors.New("DNS failure") }, nil)
	_, err = transport.DialContext(context.Background(), "tcp", "agent.test:80")
	require.ErrorContains(t, err, "DNS failure")
	restore()
}

// C7/C7b: same-origin redirects work, origin changes (scheme/host/port) don't;
// a same-host redirect after DNS rebinding is checked again on the new dial.
func TestRedirectPolicy(t *testing.T) {
	var hits atomic.Int32
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		switch r.URL.Path {
		case "/start":
			http.Redirect(w, r, "/end", 302)
		case "/rebind":
			w.Header().Set("Connection", "close")
			http.Redirect(w, r, "/end", 302)
		case "/cross":
			http.Redirect(w, r, "http://other.test/end", 302)
		case "/loop":
			http.Redirect(w, r, "/loop", 302)
		default:
			if _, err := io.WriteString(w, "ok"); err != nil {
				t.Errorf("write response: %v", err)
			}
		}
	}))
	defer s.Close()
	c := NewClient(time.Second)
	defer c.CloseIdleConnections()
	resp, err := c.Get(s.URL + "/start")
	require.NoError(t, err)
	resp.Body.Close()
	require.EqualValues(t, 2, hits.Load())
	_, err = c.Get(s.URL + "/cross")
	require.ErrorContains(t, err, "cross-origin")
	require.EqualValues(t, 3, hits.Load())
	_, err = c.Get(s.URL + "/loop")
	require.ErrorContains(t, err, "10 redirects")
	u, _ := url.Parse(s.URL)
	var lookups atomic.Int32
	restore := SetNetworkHooksForTesting(func(context.Context, string) ([]string, error) {
		if lookups.Add(1) == 1 {
			return []string{"127.0.0.1"}, nil
		}
		return []string{"169.254.169.254"}, nil
	}, nil)
	defer restore()
	before := hits.Load()
	_, err = c.Get("http://agent.test:" + u.Port() + "/rebind")
	require.ErrorContains(t, err, "link-local")
	require.Equal(t, before+1, hits.Load())
	original, _ := http.NewRequest("GET", "http://agent.test/path", nil)
	for _, dest := range []string{"https://agent.test/path", "http://other.test/path", "http://agent.test:8080/path"} {
		req, _ := http.NewRequest("GET", dest, nil)
		require.Error(t, checkRedirect(req, []*http.Request{original}))
	}
	req, _ := http.NewRequest("GET", "http://agent.test:80/path", nil)
	require.NoError(t, checkRedirect(req, []*http.Request{original}))
}

// C9: invalid stored URLs warn once per node without affecting later valid use.
func TestStoredNodeWarningOnce(t *testing.T) {
	var output bytes.Buffer
	previous := logger.Logger
	logger.Logger = zerolog.New(&output)
	defer func() { logger.Logger = previous }()
	nodeID := "unit-legacy-node"
	warnedNodes.Delete(nodeID)
	defer warnedNodes.Delete(nodeID)
	for i := 0; i < 2; i++ {
		_, err := ForNode(nodeID, "http://127.0.0.1/prefix?bad", "status")
		require.ErrorContains(t, err, "query")
	}
	require.Equal(t, 1, strings.Count(output.String(), "agent endpoint refused"))
	u, err := ForNode(nodeID, "http://127.0.0.1/prefix", "status")
	require.NoError(t, err)
	require.Equal(t, "/prefix/status", u.Path)
	WarnNode(nodeID, nil)
	WarnNode("unrelated", errors.New("connection refused"))
	require.Equal(t, 1, strings.Count(output.String(), "agent endpoint refused"))
}

// Review finding 1: a stalled checked address must leave time for later answers.
func TestReviewAddressFallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, err := io.WriteString(w, "fallback reached")
		require.NoError(t, err)
	}))
	defer server.Close()
	var attempts []string
	defer SetNetworkHooksForTesting(func(context.Context, string) ([]string, error) {
		return []string{"2001:db8::1", "127.0.0.1"}, nil
	}, func(ctx context.Context, network, address string) (net.Conn, error) {
		host, _, err := net.SplitHostPort(address)
		require.NoError(t, err)
		attempts = append(attempts, host)
		if host == "2001:db8::1" {
			<-ctx.Done()
			return nil, ctx.Err()
		}
		return (&net.Dialer{}).DialContext(ctx, network, address)
	})()
	u, err := url.Parse(server.URL)
	require.NoError(t, err)
	client := NewClient(2 * time.Second)
	client.Transport = NewTransport(800 * time.Millisecond)
	defer client.CloseIdleConnections()
	started := time.Now()
	resp, err := client.Get("http://fallback.test:" + u.Port())
	require.NoError(t, err)
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equal(t, "fallback reached", string(body))
	require.Less(t, time.Since(started), 800*time.Millisecond)
	require.Equal(t, []string{"2001:db8::1", "127.0.0.1"}, attempts)
}

// Review finding 4: localhost CIDR exceptions apply after resolving every answer.
func TestReviewLocalhostCIDRExceptions(t *testing.T) {
	defer Configure(false, nil)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()
	u, err := url.Parse(server.URL)
	require.NoError(t, err)
	for _, allowed := range [][]string{{"127.0.0.0/8", "::1/128"}, {"localhost"}} {
		Configure(true, allowed)
		lookups := 0
		restore := SetNetworkHooksForTesting(func(context.Context, string) ([]string, error) {
			lookups++
			return []string{"127.0.0.1", "::1"}, nil
		}, nil)
		endpoint := "http://localhost:" + u.Port()
		_, validationErr := Validate(context.Background(), endpoint)
		client := NewClient(time.Second)
		resp, dialErr := client.Get(endpoint)
		if resp != nil {
			resp.Body.Close()
		}
		client.CloseIdleConnections()
		restore()
		require.NoError(t, validationErr)
		require.NoError(t, dialErr)
		require.Equal(t, 2, lookups)
	}
	Configure(true, []string{"127.0.0.0/8", "::1/128", "169.254.0.0/16", "localhost"})
	defer SetNetworkHooksForTesting(func(context.Context, string) ([]string, error) {
		return []string{"127.0.0.1", "169.254.169.254"}, nil
	}, func(context.Context, string, string) (net.Conn, error) {
		t.Error("always-denied DNS answer must prevent dialing")
		return nil, errors.New("unexpected dial")
	})()
	_, err = Validate(context.Background(), "http://localhost")
	require.ErrorContains(t, err, "link-local")
	_, err = NewClient(time.Second).Get("http://localhost")
	require.ErrorContains(t, err, "link-local")
}
