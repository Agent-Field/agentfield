package ai

import (
	"net/http"
	"os"
	"strings"
)

const (
	defaultOrcaRouterSiteURL = "https://agentfield.ai"
	defaultOrcaRouterAppName = "AgentField AI"
)

func orcaRouterAttributionEnabled() bool {
	value := strings.TrimSpace(os.Getenv("AGENTFIELD_ORCAROUTER_ATTRIBUTION"))
	if value == "" {
		return true
	}
	switch strings.ToLower(value) {
	case "0", "false", "no", "off":
		return false
	default:
		return true
	}
}

func resolveOrcaRouterAttribution(siteURL, siteName string) (string, string, bool) {
	if !orcaRouterAttributionEnabled() {
		return "", "", false
	}

	// OrcaRouter is OpenAI-compatible, so the HTTP-Referer / X-Title pair this
	// package already sends keeps working. The OpenRouter-scoped values are
	// honored as fallbacks so a deployment keeps its declared identity when it
	// moves gateways — but the opt-out travels with them: values a deployment
	// suppressed for one vendor (often because they name internal hosts or
	// products) must not be sent to another.
	inheritedURL, inheritedName := "", ""
	if openRouterAttributionEnabled() {
		inheritedURL = firstNonEmpty(
			os.Getenv("AGENTFIELD_OPENROUTER_SITE_URL"),
			os.Getenv("OR_SITE_URL"),
		)
		inheritedName = firstNonEmpty(
			os.Getenv("AGENTFIELD_OPENROUTER_APP_NAME"),
			os.Getenv("OR_APP_NAME"),
		)
	}

	resolvedURL := firstNonEmpty(
		siteURL,
		os.Getenv("AGENTFIELD_ORCAROUTER_SITE_URL"),
		inheritedURL,
		defaultOrcaRouterSiteURL,
	)
	resolvedName := firstNonEmpty(
		siteName,
		os.Getenv("AGENTFIELD_ORCAROUTER_APP_NAME"),
		inheritedName,
		defaultOrcaRouterAppName,
	)
	return resolvedURL, resolvedName, true
}

// applyOrcaRouterAttributionHeaders sets the app-attribution headers on an
// OrcaRouter request. OrcaRouter is OpenAI-compatible and accepts the
// HTTP-Referer / X-Title pair this package already sends, so a deployment that
// already identifies itself as "AgentField AI" keeps doing so after switching
// gateways.
func applyOrcaRouterAttributionHeaders(header http.Header, siteURL, siteName string) {
	resolvedURL, resolvedName, ok := resolveOrcaRouterAttribution(siteURL, siteName)
	if !ok {
		return
	}
	if resolvedURL != "" {
		header.Set("HTTP-Referer", resolvedURL)
	}
	if resolvedName != "" {
		header.Set("X-Title", resolvedName)
	}
}
