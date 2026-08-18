package ai

import (
	"net/http"
	"os"
	"strings"
)

const (
	defaultOpenRouterSiteURL    = "https://agentfield.ai"
	defaultOpenRouterAppName    = "AgentField AI"
	defaultOpenRouterCategories = "cli-agent,programming-app"
)

type resolvedOpenRouterAttribution struct {
	siteURL    string
	appName    string
	categories string
}

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

func resolveOpenRouterAttribution(siteURL, siteName string) (resolvedOpenRouterAttribution, bool) {
	if !openRouterAttributionEnabled() {
		return resolvedOpenRouterAttribution{}, false
	}

	return resolvedOpenRouterAttribution{
		siteURL: firstNonEmpty(
			siteURL,
			os.Getenv("AGENTFIELD_OPENROUTER_SITE_URL"),
			os.Getenv("OR_SITE_URL"),
			defaultOpenRouterSiteURL,
		),
		appName: firstNonEmpty(
			siteName,
			os.Getenv("AGENTFIELD_OPENROUTER_APP_NAME"),
			os.Getenv("OR_APP_NAME"),
			defaultOpenRouterAppName,
		),
		categories: firstNonEmpty(
			os.Getenv("AGENTFIELD_OPENROUTER_CATEGORIES"),
			os.Getenv("OR_CATEGORIES"),
			defaultOpenRouterCategories,
		),
	}, true
}

func applyOpenRouterAttributionHeaders(header http.Header, siteURL, siteName string) {
	resolved, ok := resolveOpenRouterAttribution(siteURL, siteName)
	if !ok {
		return
	}
	if resolved.siteURL != "" {
		header.Set("HTTP-Referer", resolved.siteURL)
	}
	if resolved.appName != "" {
		header.Set("X-OpenRouter-Title", resolved.appName)
		header.Set("X-Title", resolved.appName)
	}
	if resolved.categories != "" {
		header.Set("X-OpenRouter-Categories", resolved.categories)
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
