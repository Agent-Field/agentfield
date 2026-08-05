package ai

import (
	"net/http"
	"os"
	"strings"
)

const (
	defaultInfronSiteURL = "https://agentfield.ai"
	defaultInfronAppName = "AgentField AI"

	// infronModelPrefix marks a model as Infron-routed. It is stripped before
	// the request goes out; see stripInfronPrefix.
	infronModelPrefix = "infron/"
)

// stripInfronPrefix removes the routing-only "infron/" prefix from a model
// name, mirroring the prefix handling this package already does on the media
// path. The gateway serves the bare `<provider>/<model>` id, so the prefix must
// not reach the wire.
func stripInfronPrefix(model string) string {
	if len(model) >= len(infronModelPrefix) &&
		strings.EqualFold(model[:len(infronModelPrefix)], infronModelPrefix) {
		return model[len(infronModelPrefix):]
	}
	return model
}

func infronAttributionEnabled() bool {
	value := strings.TrimSpace(os.Getenv("AGENTFIELD_INFRON_ATTRIBUTION"))
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

func resolveInfronAttribution(siteURL, siteName string) (string, string, bool) {
	if !infronAttributionEnabled() {
		return "", "", false
	}

	// The OpenRouter-scoped values are honored as fallbacks so a deployment
	// keeps its declared identity when it moves gateways — but the opt-out
	// travels with them: values a deployment suppressed for one vendor (often
	// because they name internal hosts or products) must not be sent to
	// another.
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
		os.Getenv("AGENTFIELD_INFRON_SITE_URL"),
		inheritedURL,
		defaultInfronSiteURL,
	)
	resolvedName := firstNonEmpty(
		siteName,
		os.Getenv("AGENTFIELD_INFRON_APP_NAME"),
		inheritedName,
		defaultInfronAppName,
	)
	return resolvedURL, resolvedName, true
}

// applyInfronAttributionHeaders sets the app-attribution headers on an Infron
// request. Infron is OpenAI-compatible and accepts the HTTP-Referer / X-Title
// pair this package already sends, so a deployment that already identifies
// itself as "AgentField AI" keeps doing so after switching gateways — the
// attribution values already configured for the existing gateway are honored
// as fallbacks precisely so nobody has to re-declare their identity to move.
func applyInfronAttributionHeaders(header http.Header, siteURL, siteName string) {
	resolvedURL, resolvedName, ok := resolveInfronAttribution(siteURL, siteName)
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
