// Package agenturl validates agent endpoints and owns all control-plane to agent HTTP traffic.
package agenturl

import (
	"context"
	"fmt"
	"net"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
)

// Normalize checks endpoint shape and the literal destination policy without DNS.
// DNS is checked by Validate at registration and authoritatively by the transport.
func Normalize(raw string) (string, error) {
	if raw == "" {
		return "", fmt.Errorf("endpoint cannot be empty")
	}
	if strings.ContainsFunc(raw, func(r rune) bool { return unicode.IsSpace(r) || unicode.IsControl(r) }) {
		return "", fmt.Errorf("endpoint must not contain whitespace or control characters")
	}
	u, err := url.Parse(raw)
	if err != nil {
		return "", fmt.Errorf("invalid URL format: %w", err)
	}
	if u.Opaque != "" {
		return "", fmt.Errorf("endpoint must not be opaque")
	}
	u.Scheme = strings.ToLower(u.Scheme)
	if u.Scheme != "http" && u.Scheme != "https" {
		return "", fmt.Errorf("URL must use http or https scheme")
	}
	if u.User != nil {
		return "", fmt.Errorf("URL must not include user info")
	}
	if u.Hostname() == "" {
		return "", fmt.Errorf("URL must include a host")
	}
	if u.RawQuery != "" || u.ForceQuery {
		return "", fmt.Errorf("endpoint must not include a query")
	}
	// url.Parse drops a trailing empty fragment; check the delimiter as well.
	if strings.Contains(raw, "#") {
		return "", fmt.Errorf("endpoint must not include a fragment")
	}
	if err := checkPath(u.EscapedPath()); err != nil {
		return "", err
	}
	u.Host = strings.ToLower(u.Host)
	u.Path = strings.TrimSuffix(u.Path, "/")
	u.RawPath = strings.TrimSuffix(u.RawPath, "/")
	if err := currentPolicy().check(u.Hostname(), nil); err != nil {
		return "", err
	}
	return u.String(), nil
}

// decodeEscapes decodes valid percent triplets independently. A decoded literal
// percent may not itself start another escape (e.g. a prefix containing %25).
// Decode remaining valid triplets even alongside those literals so nested
// separators and traversal cannot hide behind an invalid second-layer escape.
var percentTriplet = regexp.MustCompile(`%[0-9a-fA-F]{2}`)

func decodeEscapes(s string) string {
	return percentTriplet.ReplaceAllStringFunc(s, func(triplet string) string {
		decoded, _ := url.PathUnescape(triplet)
		return decoded
	})
}

// checkPath also rejects nested escapes: downstream proxies must not turn an
// accepted prefix into a separator or traversal after another decoding pass.
func checkPath(escaped string) error {
	for {
		if strings.ContainsAny(escaped, "\\") || strings.ContainsFunc(escaped, func(r rune) bool { return unicode.IsSpace(r) || unicode.IsControl(r) }) {
			return fmt.Errorf("endpoint path must not contain whitespace, control characters or backslashes")
		}
		for _, s := range strings.Split(escaped, "/") {
			if s == "." || s == ".." {
				return fmt.Errorf("endpoint path must not contain dot-segments")
			}
		}
		lower := strings.ToLower(escaped)
		if strings.Contains(lower, "%2f") || strings.Contains(lower, "%5c") {
			return fmt.Errorf("endpoint path must not contain encoded path separators")
		}
		decoded := decodeEscapes(escaped)
		if decoded == escaped {
			return nil
		}
		escaped = decoded
	}
}

// Validate normalizes an endpoint and checks every resolved address. An agent
// may register before its DNS record exists; failure to resolve only warns.
func Validate(ctx context.Context, raw string) (string, error) {
	normalized, err := Normalize(raw)
	if err != nil {
		return "", err
	}
	u, _ := url.Parse(normalized)
	host := u.Hostname()
	if literalIP(host) != nil {
		return normalized, nil
	}
	lookupCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	ips, err := networkHooks().lookup(lookupCtx, host)
	if err != nil || len(ips) == 0 {
		logger.Logger.Warn().Err(err).Str("host", host).Msg("agent endpoint DNS unavailable at registration; destination will be checked at dial time")
		return normalized, nil
	}
	for _, ip := range ips {
		parsedIP := literalIP(ip)
		if parsedIP == nil {
			return "", fmt.Errorf("agent endpoint DNS returned an invalid IP")
		}
		if err := currentPolicy().check(host, parsedIP); err != nil {
			return "", err
		}
	}
	return normalized, nil
}

// Join appends individually escaped path segments while retaining a base prefix.
// Callers add query parameters to the returned URL using url.Values.
func Join(base string, segments ...string) (*url.URL, error) {
	normalized, err := Normalize(base)
	if err != nil {
		return nil, err
	}
	u, _ := url.Parse(normalized)
	escaped := u.EscapedPath()
	for _, s := range segments {
		decoded := s
		for {
			if decoded == "" || decoded == "." || decoded == ".." || strings.ContainsAny(decoded, "/\\") || strings.ContainsFunc(decoded, unicode.IsControl) {
				return nil, fmt.Errorf("invalid endpoint path segment")
			}
			next := decodeEscapes(decoded)
			if next == decoded {
				break
			}
			decoded = next
		}
		escaped += "/" + url.PathEscape(s)
	}
	u.Path, _ = url.PathUnescape(escaped)
	u.RawPath = escaped
	return u, nil
}

var warnedNodes sync.Map

// ForNode joins a stored endpoint and warns once per node when legacy data is
// refused. Storage reads and listing deliberately do not validate endpoints.
func ForNode(nodeID, base string, segments ...string) (*url.URL, error) {
	u, err := Join(base, segments...)
	if err != nil {
		err = fmt.Errorf("invalid agent endpoint: %w", err)
	}
	WarnNode(nodeID, err)
	return u, err
}

// WarnNode reports dispatch policy failures, including dial-time DNS failures.
func WarnNode(nodeID string, err error) {
	if err != nil && strings.Contains(err.Error(), "agent endpoint") {
		if _, loaded := warnedNodes.LoadOrStore(nodeID, struct{}{}); !loaded {
			logger.Logger.Warn().Err(err).Str("node_id", nodeID).Msg("agent endpoint refused; re-register with a valid endpoint")
		}
	}
}

func literalIP(host string) net.IP {
	// Preserve scoped IPv6 literals while applying the policy to their address.
	if i := strings.LastIndexByte(host, '%'); i >= 0 {
		host = host[:i]
	}
	return net.ParseIP(host)
}
