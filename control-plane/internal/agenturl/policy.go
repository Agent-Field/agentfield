package agenturl

import (
	"fmt"
	"net"
	"strings"
	"sync/atomic"
)

type policy struct {
	strict  bool
	allowed []string
}

var configured atomic.Pointer[policy]

// Configure sets the startup policy. The always-on deny cannot be overridden.
func Configure(strict bool, allowed []string) {
	p := &policy{strict: strict}
	for _, s := range allowed {
		if s = strings.TrimSpace(s); s != "" {
			p.allowed = append(p.allowed, s)
		}
	}
	configured.Store(p)
}
func currentPolicy() *policy {
	if p := configured.Load(); p != nil {
		return p
	}
	return &policy{}
}

func hostAllowed(host string, ip net.IP, allowed []string) bool {
	host = strings.ToLower(host)
	for _, entry := range allowed {
		candidate := strings.ToLower(entry)
		if _, n, err := net.ParseCIDR(candidate); err == nil {
			if ip != nil && n.Contains(ip) {
				return true
			}
			continue
		}
		if strings.HasPrefix(candidate, "*.") {
			if strings.HasSuffix(host, candidate[1:]) {
				return true
			}
		} else if host == candidate {
			return true
		}
	}
	return false
}

func hasCIDRException(allowed []string) bool {
	for _, entry := range allowed {
		if _, _, err := net.ParseCIDR(entry); err == nil {
			return true
		}
	}
	return false
}

func (p *policy) check(host string, ip net.IP) error {
	switch strings.ToLower(strings.TrimSuffix(host, ".")) {
	case "metadata.google.internal", "metadata.goog", "metadata":
		return fmt.Errorf("agent endpoint targets a metadata hostname")
	}
	if ip == nil {
		ip = literalIP(host)
	}
	if ip == nil {
		// CIDR exceptions require resolution, including for localhost. The
		// registration validator and transport still check every DNS answer.
		if p.strict && (strings.EqualFold(strings.TrimSuffix(host, "."), "localhost") || strings.HasSuffix(strings.ToLower(strings.TrimSuffix(host, ".")), ".localhost")) && !hostAllowed(host, nil, p.allowed) && !hasCIDRException(p.allowed) {
			return fmt.Errorf("agent endpoint targets loopback in strict mode")
		}
		return nil
	}
	addresses := embeddedAddresses(ip)
	for _, addr := range addresses {
		if alwaysDenied(addr) {
			return fmt.Errorf("agent endpoint targets link-local, metadata, multicast or broadcast address %s", addr)
		}
	}
	if p.strict {
		allowed := hostAllowed(host, ip, p.allowed)
		for _, addr := range addresses {
			if strictDenied(addr) && !allowed && !hostAllowed(host, addr, p.allowed) {
				return fmt.Errorf("agent endpoint targets private/internal address %s in strict mode", addr)
			}
		}
	}
	return nil
}

func cidrContains(cidr string, ip net.IP) bool { _, n, _ := net.ParseCIDR(cidr); return n.Contains(ip) }
func alwaysDenied(ip net.IP) bool {
	return ip.IsLinkLocalUnicast() || ip.IsMulticast() || ip.Equal(net.ParseIP("fd00:ec2::254")) || ip.Equal(net.IPv4bcast)
}
func strictDenied(ip net.IP) bool {
	return ip.IsLoopback() || ip.IsPrivate() || ip.IsUnspecified() || cidrContains("100.64.0.0/10", ip) || cidrContains("198.18.0.0/15", ip) || cidrContains("0.0.0.0/8", ip)
}

// embeddedAddresses checks both the IPv6 container and its embedded IPv4
// destinations. RFC 6052 /48 skips the reserved 'u' octet at byte 8.
func embeddedAddresses(ip net.IP) []net.IP {
	result := []net.IP{ip}
	if v4 := ip.To4(); v4 != nil {
		return []net.IP{v4}
	}
	v6 := ip.To16()
	if v6 == nil {
		return result
	}
	switch {
	case cidrContains("::/96", ip), cidrContains("64:ff9b::/96", ip):
		if !ip.IsUnspecified() && !ip.IsLoopback() {
			result = append(result, net.IPv4(v6[12], v6[13], v6[14], v6[15]))
		}
	case cidrContains("64:ff9b:1::/48", ip):
		// The local-use /48 may be delegated as /48, /56, /64 or /96
		// NAT64 prefixes. Check every RFC 6052 placement within that block.
		result = append(result,
			net.IPv4(v6[6], v6[7], v6[9], v6[10]),
			net.IPv4(v6[7], v6[9], v6[10], v6[11]),
			net.IPv4(v6[9], v6[10], v6[11], v6[12]),
			net.IPv4(v6[12], v6[13], v6[14], v6[15]))
	case cidrContains("2002::/16", ip):
		result = append(result, net.IPv4(v6[2], v6[3], v6[4], v6[5]))
	case cidrContains("2001::/32", ip):
		result = append(result, net.IPv4(v6[4], v6[5], v6[6], v6[7]), net.IPv4(^v6[12], ^v6[13], ^v6[14], ^v6[15]))
	}
	return result
}
