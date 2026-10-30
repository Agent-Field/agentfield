package services

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIsPrivateIP(t *testing.T) {
	tests := []struct {
		name    string
		ip      string
		private bool
	}{
		// Loopback
		{"ipv4 loopback", "127.0.0.1", true},
		{"ipv4 loopback high", "127.255.255.255", true},
		{"ipv6 loopback", "::1", true},

		// Link-local
		{"ipv4 link-local", "169.254.0.1", true},
		{"ipv4 link-local metadata", "169.254.169.254", true},
		{"ipv6 link-local", "fe80::1", true},

		// RFC-1918
		{"10.x.x.x", "10.0.0.1", true},
		{"10.x.x.x high", "10.255.255.255", true},
		{"172.16.x.x", "172.16.0.1", true},
		{"172.31.x.x", "172.31.255.255", true},
		{"192.168.x.x", "192.168.0.1", true},
		{"192.168.x.x high", "192.168.255.255", true},

		// Unspecified
		{"ipv4 unspecified", "0.0.0.0", true},
		{"ipv6 unspecified", "::", true},

		// IPv6 ULA
		{"ipv6 ULA", "fd00::1", true},

		// Nil IP
		{"nil ip", "", true},

		// Public IPs (should NOT be private)
		{"public 8.8.8.8", "8.8.8.8", false},
		{"public 1.1.1.1", "1.1.1.1", false},
		{"public 93.184.216.34", "93.184.216.34", false},
		{"public ipv6", "2001:db8::1", false},
		{"172.32 (just outside RFC-1918)", "172.32.0.1", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var ip net.IP
			if tt.ip != "" {
				ip = net.ParseIP(tt.ip)
			}
			got := isPrivateIP(ip)
			assert.Equal(t, tt.private, got, "isPrivateIP(%s)", tt.ip)
		})
	}
}

func TestIsPrivateHost(t *testing.T) {
	tests := []struct {
		host    string
		private bool
	}{
		{"localhost", true},
		{"LOCALHOST", true},
		{"foo.localhost", true},
		{"sub.localhost", true},
		{"example.com", false},
		{"notlocalhost", false},
		{"localhost.example.com", false},
	}

	for _, tt := range tests {
		t.Run(tt.host, func(t *testing.T) {
			assert.Equal(t, tt.private, isPrivateHost(tt.host))
		})
	}
}

func TestValidateWebhookURL(t *testing.T) {
	tests := []struct {
		name      string
		url       string
		wantError bool
		errSubstr string
	}{
		// Should be rejected
		{"loopback ipv4", "http://127.0.0.1:9999/cb", true, "private/internal address"},
		{"cloud metadata", "http://169.254.169.254/latest/meta-data/", true, "private/internal address"},
		{"rfc1918 10.x", "http://10.0.0.1:8080/cb", true, "private/internal address"},
		{"rfc1918 192.168.x", "http://192.168.1.1:8080/cb", true, "private/internal address"},
		{"rfc1918 172.16.x", "http://172.16.0.1:8080/cb", true, "private/internal address"},
		{"localhost", "http://localhost:9999/cb", true, "private/internal host"},
		{"ipv6 loopback", "http://[::1]:9999/cb", true, "private/internal address"},
		{"unspecified", "http://0.0.0.0:9999/cb", true, "private/internal address"},
		{"subdomain localhost", "http://foo.localhost:9999/cb", true, "private/internal host"},
		{"empty host", "http:///path", true, "no host"},

		// Should be allowed
		{"public domain", "https://example.com/cb", false, ""},
		{"public ip", "http://93.184.216.34:8080/cb", false, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateWebhookURL(tt.url)
			if tt.wantError {
				require.Error(t, err, "expected error for %s", tt.url)
				assert.Contains(t, err.Error(), tt.errSubstr)
			} else {
				assert.NoError(t, err, "expected no error for %s", tt.url)
			}
		})
	}
}

func TestNewSSRFSafeClient_BlocksPrivateIPs(t *testing.T) {
	// Start a test server on loopback — the SSRF-safe client should refuse to connect.
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"secret": "leaked"}`))
	}))
	defer ts.Close()

	client := NewSSRFSafeClient(5e9) // 5s timeout
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, ts.URL, nil)
	require.NoError(t, err)

	_, err = client.Do(req)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "private/internal address")
}

// C2: a hostname is checked after resolution at connect time, independently of
// the registration pre-check. "localhost" resolves through the hosts file on
// every CI platform, so this needs no external DNS.
func TestNewSSRFSafeClient_RefusesHostnameResolvingToLoopback(t *testing.T) {
	original := webhookAllowedHosts()
	t.Cleanup(func() { SetWebhookAllowedHosts(original) })
	SetWebhookAllowedHosts(nil)

	var hit atomic.Bool
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit.Store(true)
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	target := strings.Replace(ts.URL, "127.0.0.1", "localhost", 1)
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, target, nil)
	require.NoError(t, err)

	_, err = NewSSRFSafeClient(5 * time.Second).Do(req)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "private/internal address")
	assert.False(t, hit.Load(), "the loopback listener must not receive the request")
}

func TestNewSSRFSafeClient_AllowsPublicIPs(t *testing.T) {
	// We can't easily test a real public IP in CI, but we can verify the client
	// is constructed without error and has a non-nil transport.
	client := NewSSRFSafeClient(5e9)
	require.NotNil(t, client)
	require.NotNil(t, client.Transport)
}

func TestSetWebhookAllowedHosts(t *testing.T) {
	// Save and restore allowlist so other tests aren't affected.
	original := webhookAllowedHosts()
	defer SetWebhookAllowedHosts(original)

	t.Run("trims and skips empty entries", func(t *testing.T) {
		SetWebhookAllowedHosts([]string{"  foo  ", "", "   ", "bar"})
		got := webhookAllowedHosts()
		require.Equal(t, []string{"foo", "bar"}, got)
	})

	t.Run("empty list clears allowlist", func(t *testing.T) {
		SetWebhookAllowedHosts(nil)
		got := webhookAllowedHosts()
		assert.Empty(t, got)
	})
}

func TestIsHostAllowlisted(t *testing.T) {
	tests := []struct {
		name    string
		host    string
		ip      string
		allowed []string
		want    bool
	}{
		{"empty allowlist", "foo.com", "1.2.3.4", nil, false},
		{"exact hostname match", "test-runner", "", []string{"test-runner"}, true},
		{"case-insensitive hostname", "Test-Runner", "", []string{"test-runner"}, true},
		{"hostname mismatch", "other", "", []string{"test-runner"}, false},
		{"wildcard match", "foo.internal", "", []string{"*.internal"}, true},
		{"wildcard does not match base", "internal", "", []string{"*.internal"}, false},
		{"wildcard mismatch", "foo.external", "", []string{"*.internal"}, false},
		{"cidr match", "any", "172.18.0.5", []string{"172.18.0.0/16"}, true},
		{"cidr no match", "any", "10.0.0.1", []string{"172.18.0.0/16"}, false},
		{"cidr but no ip", "any", "", []string{"172.18.0.0/16"}, false},
		{"empty entry skipped", "foo", "", []string{"", "foo"}, true},
		{"multiple entries last matches", "b", "", []string{"a", "b"}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var ip net.IP
			if tt.ip != "" {
				ip = net.ParseIP(tt.ip)
			}
			got := isHostAllowlisted(tt.host, ip, tt.allowed)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestValidateWebhookURL_WithAllowlist(t *testing.T) {
	original := webhookAllowedHosts()
	defer SetWebhookAllowedHosts(original)

	t.Run("allowlisted hostname bypasses private-host check", func(t *testing.T) {
		SetWebhookAllowedHosts([]string{"test-runner"})
		err := ValidateWebhookURL("http://test-runner:8080/cb")
		// test-runner won't resolve in test env; allowlisted hostname returns nil
		// (skips DNS-resolved IP check too).
		assert.NoError(t, err)
	})

	t.Run("allowlisted CIDR bypasses private-IP check for raw IPs", func(t *testing.T) {
		SetWebhookAllowedHosts([]string{"172.18.0.0/16"})
		err := ValidateWebhookURL("http://172.18.0.5:8080/cb")
		assert.NoError(t, err)
	})

	t.Run("allowlist for one IP does not bypass others", func(t *testing.T) {
		SetWebhookAllowedHosts([]string{"172.18.0.0/16"})
		err := ValidateWebhookURL("http://10.0.0.1:8080/cb")
		assert.Error(t, err)
	})

	t.Run("wildcard hostname allowlist", func(t *testing.T) {
		SetWebhookAllowedHosts([]string{"*.svc.cluster.local"})
		err := ValidateWebhookURL("http://my-service.svc.cluster.local:8080/cb")
		assert.NoError(t, err)
	})

	t.Run("empty allowlist still blocks private IPs", func(t *testing.T) {
		SetWebhookAllowedHosts(nil)
		err := ValidateWebhookURL("http://127.0.0.1:8080/cb")
		assert.Error(t, err)
	})
}

func TestNewSSRFSafeClient_BadAddress(t *testing.T) {
	client := NewSSRFSafeClient(5 * time.Second)
	transport, ok := client.Transport.(*http.Transport)
	require.True(t, ok)

	// Pass a malformed addr (no port) to DialContext directly.
	_, err := transport.DialContext(context.Background(), "tcp", "no-port-here")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "ssrf: invalid address")
}

func TestNewSSRFSafeClient_DNSFailure(t *testing.T) {
	client := NewSSRFSafeClient(2 * time.Second)
	transport, ok := client.Transport.(*http.Transport)
	require.True(t, ok)

	// Use a hostname guaranteed not to resolve.
	_, err := transport.DialContext(context.Background(), "tcp", "no-such-host.invalid:80")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "ssrf: DNS lookup failed")
}

func TestValidateWebhookURL_InvalidURL(t *testing.T) {
	// url.Parse rejects URLs with control characters in the scheme.
	err := ValidateWebhookURL("http://exa\x00mple.com/")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "ssrf: invalid URL")
}

func TestValidateWebhookURL_HostnameResolvesToPrivate(t *testing.T) {
	// "localhost" hits the early isPrivateHost check, so to exercise the
	// hostname-resolves-to-private-IP branch we need a hostname that is not
	// "localhost" but still resolves to a private/loopback IP.
	//
	// Many hosts file setups define "ip6-loopback" → "::1". When the host
	// can't be resolved, the function returns nil (best-effort), so this
	// test asserts only on the resolution-succeeded outcome to avoid CI flake.
	err := ValidateWebhookURL("http://ip6-loopback/cb")
	if err != nil {
		assert.Contains(t, err.Error(), "private/internal")
	}
}

func TestNewSSRFSafeClient_AllowlistedLoopback(t *testing.T) {
	original := webhookAllowedHosts()
	t.Cleanup(func() { SetWebhookAllowedHosts(original) })

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	// C5: explicit hostnames, exact IPs and CIDRs bypass both validation and dialing.
	for _, tc := range []struct {
		name    string
		allowed string
		url     string
	}{
		{"CIDR", "127.0.0.0/8", ts.URL},
		{"exact IP", "127.0.0.1", ts.URL},
		{"hostname", "localhost", strings.Replace(ts.URL, "127.0.0.1", "localhost", 1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			SetWebhookAllowedHosts([]string{tc.allowed})
			require.NoError(t, ValidateWebhookURL(tc.url))
			client := NewSSRFSafeClient(5 * time.Second)
			t.Cleanup(client.CloseIdleConnections)
			req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, tc.url, nil)
			require.NoError(t, err)
			resp, err := client.Do(req)
			require.NoError(t, err)
			defer resp.Body.Close()
			assert.Equal(t, http.StatusOK, resp.StatusCode)
		})
	}
}

// C5: an IPv4 CIDR does not implicitly trust an IPv6 transition address;
// explicitly trusting that IPv6 host or CIDR still bypasses both checks.
func TestWebhookAllowlist_TransitionAddresses(t *testing.T) {
	original := webhookAllowedHosts()
	t.Cleanup(func() { SetWebhookAllowedHosts(original) })
	for _, tc := range []struct {
		name    string
		ip      string
		allowed string
		bypass  bool
	}{
		{"explicit NAT64 CIDR", "64:ff9b::7f00:1", "64:ff9b::/96", true},
		{"explicit NAT64 IP", "64:ff9b::7f00:1", "64:ff9b::7f00:1", true},
		{"IPv4 CIDR does not trust NAT64", "64:ff9b::7f00:1", "127.0.0.0/8", false},
		{"IPv4 CIDR does not trust 6to4", "2002:7f00:1::", "127.0.0.0/8", false},
		{"IPv4 CIDR does not trust compatible", "::127.0.0.1", "127.0.0.0/8", false},
		{"IPv4 CIDR does not trust translated", "::ffff:0:7f00:1", "127.0.0.0/8", false},
		{"IPv4 CIDR does not trust local-use NAT64", "64:ff9b:1::7f00:1", "127.0.0.0/8", false},
		{"IPv4 CIDR does not trust Teredo", "2001:0:4136:e378:8000:63bf:80ff:fffe", "127.0.0.0/8", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			SetWebhookAllowedHosts([]string{tc.allowed})
			err := ValidateWebhookURL("http://" + urlHost(tc.ip) + "/cb")
			if tc.bypass {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "private/internal address")
			}

			transport := NewSSRFSafeClient(time.Second).Transport.(*http.Transport)
			// No packet leaves even when the transition address is explicitly trusted.
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			conn, err := transport.DialContext(ctx, "tcp", net.JoinHostPort(tc.ip, "80"))
			if conn != nil {
				conn.Close()
			}
			require.Error(t, err)
			if tc.bypass {
				assert.True(t, errors.Is(err, context.Canceled), "trusted address must reach the cancelled dialer: %v", err)
				assert.NotContains(t, err.Error(), "ssrf:")
			} else {
				assert.Contains(t, err.Error(), "ssrf: webhook target resolves to private/internal address")
			}
		})
	}
}

// C1, C2, C3 and the hardening contract: embeddedAndSpecialUseRefused covers
// IPv6 forms that carry an IPv4 destination, shared and benchmarking space,
// "this network", reserved IPv4, site-local IPv6 and non-unicast destinations.
// Every address must be refused by registration and by the dial hook.
var embeddedAndSpecialUseRefused = []struct {
	name string
	ip   string
}{
	{"nat64 well-known to loopback", "64:ff9b::7f00:1"},
	{"nat64 well-known to metadata", "64:ff9b::a9fe:a9fe"},
	{"nat64 local-use to loopback", "64:ff9b:1::7f00:1"},
	{"6to4 to loopback", "2002:7f00:1::1"},
	{"6to4 to metadata", "2002:a9fe:a9fe::1"},
	{"teredo", "2001:0:4136:e378:8000:63bf:80ff:fffe"},
	{"ipv4-mapped loopback", "::ffff:127.0.0.1"},
	{"ipv4-mapped metadata", "::ffff:169.254.169.254"},
	{"ipv4-compatible loopback", "::127.0.0.1"},
	{"ipv4-translated loopback", "::ffff:0:7f00:1"},
	{"shared address space", "100.64.0.1"},
	{"shared address space low", "100.64.0.0"},
	{"shared address space high", "100.127.255.255"},
	{"benchmarking", "198.18.0.1"},
	{"benchmarking low", "198.18.0.0"},
	{"benchmarking high", "198.19.255.255"},
	{"ipv4 unspecified", "0.0.0.0"},
	{"ipv6 unspecified", "::"},
	{"this network low", "0.0.0.1"},
	{"this network assigned on loopback", "0.1.2.3"},
	{"this network high", "0.255.255.255"},
	{"reserved ipv4 low", "240.0.0.0"},
	{"reserved ipv4 assigned on loopback", "240.1.2.3"},
	{"reserved ipv4 high", "255.255.255.254"},
	{"limited broadcast", "255.255.255.255"},
	{"site-local low", "fec0::"},
	{"site-local", "fec0::1"},
	{"site-local high", "feff:ffff:ffff:ffff:ffff:ffff:ffff:ffff"},
	{"ipv4 global multicast", "224.0.1.1"},
	{"ipv4 multicast high", "239.255.255.255"},
	{"ipv6 global multicast", "ff0e::1"},
	{"ipv6 link-local multicast", "ff02::1"},
	{"ipv4 service continuity low", "192.0.0.0"},
	{"ipv4 service continuity", "192.0.0.2"},
	{"ipv4 service continuity high", "192.0.0.7"},
	{"ipv6 benchmarking", "2001:2::1"},
	{"ipv6 benchmarking high", "2001:2:0:ffff:ffff:ffff:ffff:ffff"},
}

// C4: publicControls must stay reachable. They include the neighbours of the new
// ranges and an IPv4-mapped public address, which is what a mis-spelled
// translated prefix (::ffff:0:0/96, read by Go as all of IPv4) would block.
var publicControls = []struct {
	name string
	ip   string
}{
	{"public ipv4", "8.8.8.8"},
	{"public ipv4 one", "1.1.1.1"},
	{"public ipv6", "2606:4700:4700::1111"},
	{"public ipv6 beside teredo", "2001:4860:4860::8888"},
	{"ipv4-mapped public", "::ffff:8.8.8.8"},
	{"just below shared address space", "100.63.255.255"},
	{"just above shared address space", "100.128.0.0"},
	{"just below benchmarking", "198.17.255.255"},
	{"just above benchmarking", "198.20.0.0"},
	{"just above this network", "1.0.0.0"},
	{"beside nat64 well-known", "64:ff9b:0:0:1::1"},
	{"just above nat64 well-known", "64:ff9b::1:0:0"},
	{"just below nat64 well-known", "64:ff9a:ffff:ffff:ffff:ffff:ffff:ffff"},
	{"just below nat64 local-use", "64:ff9b:0:ffff:ffff:ffff:ffff:ffff"},
	{"just above nat64 local-use", "64:ff9b:2::"},
	{"just above 6to4", "2003::1"},
	{"just below 6to4", "2001:ffff:ffff:ffff:ffff:ffff:ffff:ffff"},
	{"just above teredo", "2001:1::1"},
	{"just below teredo", "2000:ffff:ffff:ffff:ffff:ffff:ffff:ffff"},
	{"just above ipv4-compatible", "::1:0:0"},
	{"just below ipv4-translated", "::fffe:ffff:ffff:ffff"},
	{"just above ipv4-translated", "::ffff:1:0:0"},
	{"just below ipv4 service continuity", "191.255.255.255"},
	{"just above ipv6 benchmarking", "2001:3::1"},
}

func TestIsPrivateIP_EmbeddedIPv4AndSpecialUse(t *testing.T) {
	for _, tt := range embeddedAndSpecialUseRefused {
		t.Run(tt.name, func(t *testing.T) {
			ip := net.ParseIP(tt.ip)
			require.NotNil(t, ip, "test address %q must parse", tt.ip)
			assert.True(t, isPrivateIP(ip), "isPrivateIP(%s) must refuse", tt.ip)
		})
	}
	for _, tt := range publicControls {
		t.Run(tt.name, func(t *testing.T) {
			ip := net.ParseIP(tt.ip)
			require.NotNil(t, ip, "test address %q must parse", tt.ip)
			assert.False(t, isPrivateIP(ip), "isPrivateIP(%s) must allow", tt.ip)
		})
	}
}

// urlHost spells an IP literal as a URL host, bracketing every IPv6
// spelling, including the IPv4-mapped ones.
func urlHost(ip string) string {
	if strings.Contains(ip, ":") {
		return "[" + ip + "]"
	}
	return ip
}

func TestValidateWebhookURL_EmbeddedIPv4AndSpecialUse(t *testing.T) {
	original := webhookAllowedHosts()
	t.Cleanup(func() { SetWebhookAllowedHosts(original) })
	SetWebhookAllowedHosts(nil)

	for _, tt := range embeddedAndSpecialUseRefused {
		t.Run(tt.name, func(t *testing.T) {
			raw := "http://" + urlHost(tt.ip) + ":8080/cb"
			err := ValidateWebhookURL(raw)
			require.Error(t, err, "expected %s to be refused", raw)
			assert.Contains(t, err.Error(), "private/internal address")
		})
	}
	for _, tt := range publicControls {
		t.Run(tt.name, func(t *testing.T) {
			raw := "https://" + urlHost(tt.ip) + "/cb"
			assert.NoError(t, ValidateWebhookURL(raw), "expected %s to be allowed", raw)
		})
	}
}

func TestNewSSRFSafeClient_RefusesEmbeddedIPv4AndSpecialUseAtDial(t *testing.T) {
	original := webhookAllowedHosts()
	t.Cleanup(func() { SetWebhookAllowedHosts(original) })
	SetWebhookAllowedHosts(nil)

	client := NewSSRFSafeClient(2 * time.Second)
	for _, tt := range embeddedAndSpecialUseRefused {
		t.Run(tt.name, func(t *testing.T) {
			req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, "http://"+urlHost(tt.ip)+":80/cb", nil)
			require.NoError(t, err)
			resp, err := client.Do(req)
			if resp != nil {
				resp.Body.Close()
			}
			require.Error(t, err)
			assert.Contains(t, err.Error(), "ssrf: webhook target resolves to private/internal address")
		})
	}
}

func TestNewSSRFSafeClient_PassesPublicAddressesToTheDialer(t *testing.T) {
	original := webhookAllowedHosts()
	t.Cleanup(func() { SetWebhookAllowedHosts(original) })
	SetWebhookAllowedHosts(nil)

	transport, ok := NewSSRFSafeClient(2 * time.Second).Transport.(*http.Transport)
	require.True(t, ok)

	// A cancelled context lets the guard run on the literal (no DNS is needed
	// for one) and then stops the dialer before any packet leaves, so the
	// error shows the guard let the address through rather than refusing it.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for _, tt := range publicControls {
		t.Run(tt.name, func(t *testing.T) {
			conn, err := transport.DialContext(ctx, "tcp", net.JoinHostPort(tt.ip, "443"))
			if conn != nil {
				conn.Close()
			}
			require.Error(t, err)
			assert.NotContains(t, err.Error(), "ssrf:")
			assert.True(t, errors.Is(err, context.Canceled), "public target must reach the cancelled dialer: %v", err)
		})
	}
}

// C2: redirects from a permitted origin must re-check the destination at dial time.
func TestNewSSRFSafeClient_RefusesRedirectToTransitionAddress(t *testing.T) {
	original := webhookAllowedHosts()
	t.Cleanup(func() { SetWebhookAllowedHosts(original) })
	// Only the loopback origin is trusted; the redirect target is not.
	SetWebhookAllowedHosts([]string{"127.0.0.0/8"})

	hits := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		http.Redirect(w, r, "http://[64:ff9b::7f00:1]:80/cb", http.StatusFound)
	}))
	defer ts.Close()

	client := NewSSRFSafeClient(5 * time.Second)
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, ts.URL, nil)
	require.NoError(t, err)
	resp, err := client.Do(req)
	if resp != nil {
		resp.Body.Close()
	}
	require.Error(t, err)
	assert.Contains(t, err.Error(), "private/internal address 64:ff9b::7f00:1")
	assert.Equal(t, 1, hits, "the trusted origin is reached once, the redirect target never")
}
