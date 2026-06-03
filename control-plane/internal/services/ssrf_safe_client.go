package services

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

// allowedHostsList holds hosts/CIDRs/wildcards that are exempt from SSRF
// filtering. Set once at server startup from the AGENTFIELD_WEBHOOK_ALLOWED_HOSTS
// config, typically to trust hostnames on the deployment's internal network
// (e.g. Docker compose service names, Kubernetes cluster DNS).
var allowedHostsList atomic.Pointer[[]string]

// SetWebhookAllowedHosts configures which hosts bypass SSRF filtering. Entries
// may be hostnames ("test-runner"), wildcard subdomains ("*.internal"), or
// CIDR blocks ("172.18.0.0/16"). Empty list disables the bypass.
func SetWebhookAllowedHosts(hosts []string) {
	cleaned := make([]string, 0, len(hosts))
	for _, h := range hosts {
		trimmed := strings.TrimSpace(h)
		if trimmed != "" {
			cleaned = append(cleaned, trimmed)
		}
	}
	allowedHostsList.Store(&cleaned)
}

func webhookAllowedHosts() []string {
	if p := allowedHostsList.Load(); p != nil {
		return *p
	}
	return nil
}

// isHostAllowlisted reports whether host (hostname or IP literal) matches any
// entry in the configured allowlist. Matches CIDRs against the IP if the host
// resolves to one, hostnames case-insensitively, and wildcards like "*.foo".
func isHostAllowlisted(host string, ip net.IP, allowed []string) bool {
	if len(allowed) == 0 {
		return false
	}
	hostLower := strings.ToLower(host)
	for _, entry := range allowed {
		candidate := strings.ToLower(entry)
		if candidate == "" {
			continue
		}
		// CIDR match against the resolved IP.
		if _, network, err := net.ParseCIDR(candidate); err == nil {
			if ip != nil && network.Contains(ip) {
				return true
			}
			continue
		}
		// Wildcard suffix match ("*.foo" matches "bar.foo" but not "foo").
		if strings.HasPrefix(candidate, "*.") {
			suffix := candidate[1:] // includes the leading dot
			if strings.HasSuffix(hostLower, suffix) && hostLower != strings.TrimPrefix(suffix, ".") {
				return true
			}
			continue
		}
		// Exact hostname match.
		if hostLower == candidate {
			return true
		}
	}
	return false
}

// privateRanges contains all IP ranges that should be blocked for outbound webhook requests.
var privateRanges []*net.IPNet

func init() {
	cidrs := []string{
		"127.0.0.0/8",    // IPv4 loopback
		"10.0.0.0/8",     // RFC-1918
		"172.16.0.0/12",  // RFC-1918
		"192.168.0.0/16", // RFC-1918
		"169.254.0.0/16", // Link-local
		"::1/128",        // IPv6 loopback
		"fe80::/10",      // IPv6 link-local
		"fc00::/7",       // IPv6 unique local (ULA)
	}
	for _, cidr := range cidrs {
		_, network, err := net.ParseCIDR(cidr)
		if err != nil {
			panic(fmt.Sprintf("ssrf: bad CIDR %q: %v", cidr, err))
		}
		privateRanges = append(privateRanges, network)
	}
}

// isPrivateIP returns true if the IP belongs to a loopback, link-local,
// RFC-1918 private, or unspecified range.
func isPrivateIP(ip net.IP) bool {
	if ip == nil {
		return true // treat unparseable as private (deny by default)
	}
	if ip.IsLoopback() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsUnspecified() {
		return true
	}
	for _, network := range privateRanges {
		if network.Contains(ip) {
			return true
		}
	}
	return false
}

// NewSSRFSafeClient returns an *http.Client whose transport resolves DNS
// and rejects connections to private/internal IP addresses before the TCP
// connection is established. This prevents SSRF attacks including DNS
// rebinding, since the check happens at dial time after resolution.
// Hosts matching the configured allowlist (see SetWebhookAllowedHosts) bypass
// the private-IP check.
func NewSSRFSafeClient(timeout time.Duration) *http.Client {
	dialer := &net.Dialer{Timeout: timeout}
	transport := &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			host, port, err := net.SplitHostPort(addr)
			if err != nil {
				return nil, fmt.Errorf("ssrf: invalid address %q: %w", addr, err)
			}

			ips, err := net.DefaultResolver.LookupHost(ctx, host)
			if err != nil {
				return nil, fmt.Errorf("ssrf: DNS lookup failed for %q: %w", host, err)
			}
			if len(ips) == 0 {
				return nil, fmt.Errorf("ssrf: no addresses found for %q", host)
			}

			allowed := webhookAllowedHosts()
			for _, ipStr := range ips {
				ip := net.ParseIP(ipStr)
				if isHostAllowlisted(host, ip, allowed) {
					continue
				}
				if isPrivateIP(ip) {
					return nil, fmt.Errorf("ssrf: webhook target resolves to private/internal address %s", ipStr)
				}
			}

			// Connect to the first resolved IP to avoid TOCTOU with DNS rebinding.
			return dialer.DialContext(ctx, network, net.JoinHostPort(ips[0], port))
		},
	}
	return &http.Client{Timeout: timeout, Transport: transport}
}

// ValidateWebhookURL checks that a webhook URL does not point to a
// private/internal IP address. Call this at registration time to reject
// obviously-internal targets early, before the webhook is stored.
//
// Note: this is a best-effort pre-check. The SSRF-safe transport on the
// HTTP client is the authoritative enforcement (handles DNS rebinding).
func ValidateWebhookURL(rawURL string) error {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return fmt.Errorf("ssrf: invalid URL: %w", err)
	}
	host := parsed.Hostname()
	if host == "" {
		return fmt.Errorf("ssrf: URL has no host")
	}

	allowed := webhookAllowedHosts()

	// If the host is a raw IP, check it directly.
	if ip := net.ParseIP(host); ip != nil {
		if isHostAllowlisted(host, ip, allowed) {
			return nil
		}
		if isPrivateIP(ip) {
			return fmt.Errorf("webhook url must not target private/internal address %s", host)
		}
		return nil
	}

	// Hostname path. Allowlisted names bypass the private-host check.
	if isHostAllowlisted(host, nil, allowed) {
		return nil
	}
	if isPrivateHost(host) {
		return fmt.Errorf("webhook url must not target private/internal host %q", host)
	}

	// Resolve hostname and check all returned IPs.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	ips, err := net.DefaultResolver.LookupHost(ctx, host)
	if err != nil {
		// DNS failure at registration time is not necessarily SSRF — the host
		// may become resolvable later. The transport-level check will catch it.
		return nil
	}

	for _, ipStr := range ips {
		ip := net.ParseIP(ipStr)
		if isHostAllowlisted(host, ip, allowed) {
			continue
		}
		if isPrivateIP(ip) {
			return fmt.Errorf("webhook url host %q resolves to private/internal address %s", host, ipStr)
		}
	}

	return nil
}

// isPrivateHost is a quick syntactic check for obviously-internal hostnames
// (localhost and variants). Used in URL validation where DNS may not be needed.
func isPrivateHost(host string) bool {
	lower := strings.ToLower(strings.TrimSpace(host))
	if lower == "localhost" {
		return true
	}
	// localhost with any subdomain (e.g. foo.localhost)
	if strings.HasSuffix(lower, ".localhost") {
		return true
	}
	return false
}
