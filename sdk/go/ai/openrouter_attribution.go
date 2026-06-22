package ai

import (
	"net/http"
	"os"
	"strings"
)

const (
	defaultOpenRouterSiteURL = "https://agentfield.ai"
	defaultOpenRouterAppName = "AgentField AI"
)

func openRouterAttributionEnabled() bool {
	value := strings.TrimSpace(os.Getenv("AGENTFIELD_OPENROUTER_ATTRIBUTION"))
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

func resolveOpenRouterAttribution(siteURL, siteName string) (string, string, bool) {
	if !openRouterAttributionEnabled() {
		return "", "", false
	}

	resolvedURL := firstNonEmpty(
		siteURL,
		os.Getenv("AGENTFIELD_OPENROUTER_SITE_URL"),
		os.Getenv("OR_SITE_URL"),
		defaultOpenRouterSiteURL,
	)
	resolvedName := firstNonEmpty(
		siteName,
		os.Getenv("AGENTFIELD_OPENROUTER_APP_NAME"),
		os.Getenv("OR_APP_NAME"),
		defaultOpenRouterAppName,
	)
	return resolvedURL, resolvedName, true
}

func applyOpenRouterAttributionHeaders(header http.Header, siteURL, siteName string) {
	resolvedURL, resolvedName, ok := resolveOpenRouterAttribution(siteURL, siteName)
	if !ok {
		return
	}
	if resolvedURL != "" {
		header.Set("HTTP-Referer", resolvedURL)
	}
	if resolvedName != "" {
		header.Set("X-OpenRouter-Title", resolvedName)
		header.Set("X-Title", resolvedName)
	}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if cleaned := strings.TrimSpace(value); cleaned != "" {
			return cleaned
		}
	}
	return ""
}
