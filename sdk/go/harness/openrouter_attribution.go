package harness

import (
	"strings"
)

const (
	defaultOpenRouterSiteURL = "https://agentfield.ai"
	defaultOpenRouterAppName = "AgentField AI"
)

func openRouterAttributionEnabled(env map[string]string) bool {
	value := strings.TrimSpace(env["AGENTFIELD_OPENROUTER_ATTRIBUTION"])
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

func applyOpenRouterAttributionEnv(env map[string]string) {
	if !openRouterAttributionEnabled(env) {
		delete(env, "AGENTFIELD_OPENROUTER_SITE_URL")
		delete(env, "AGENTFIELD_OPENROUTER_APP_NAME")
		delete(env, "OR_SITE_URL")
		delete(env, "OR_APP_NAME")
		return
	}

	siteURL := firstNonEmpty(
		env["AGENTFIELD_OPENROUTER_SITE_URL"],
		env["OR_SITE_URL"],
		defaultOpenRouterSiteURL,
	)
	appName := firstNonEmpty(
		env["AGENTFIELD_OPENROUTER_APP_NAME"],
		env["OR_APP_NAME"],
		defaultOpenRouterAppName,
	)

	setDefaultEnv(env, "AGENTFIELD_OPENROUTER_SITE_URL", siteURL)
	setDefaultEnv(env, "AGENTFIELD_OPENROUTER_APP_NAME", appName)
	setDefaultEnv(env, "OR_SITE_URL", siteURL)
	setDefaultEnv(env, "OR_APP_NAME", appName)
}

func openRouterAttributionHeaders(env map[string]string) map[string]string {
	if !openRouterAttributionEnabled(env) {
		return map[string]string{}
	}
	siteURL := firstNonEmpty(
		env["AGENTFIELD_OPENROUTER_SITE_URL"],
		env["OR_SITE_URL"],
		defaultOpenRouterSiteURL,
	)
	appName := firstNonEmpty(
		env["AGENTFIELD_OPENROUTER_APP_NAME"],
		env["OR_APP_NAME"],
		defaultOpenRouterAppName,
	)
	return map[string]string{
		"HTTP-Referer":       siteURL,
		"X-OpenRouter-Title": appName,
		"X-Title":            appName,
	}
}

func setDefaultEnv(env map[string]string, key, value string) {
	if strings.TrimSpace(env[key]) == "" {
		env[key] = value
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
