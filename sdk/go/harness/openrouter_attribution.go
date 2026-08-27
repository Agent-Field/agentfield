package harness

import (
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

func resolveOpenRouterAttribution(env map[string]string) (resolvedOpenRouterAttribution, bool) {
	if !openRouterAttributionEnabled(env) {
		return resolvedOpenRouterAttribution{}, false
	}
	return resolvedOpenRouterAttribution{
		siteURL: firstNonEmpty(
			env["AGENTFIELD_OPENROUTER_SITE_URL"],
			env["OR_SITE_URL"],
			defaultOpenRouterSiteURL,
		),
		appName: firstNonEmpty(
			env["AGENTFIELD_OPENROUTER_APP_NAME"],
			env["OR_APP_NAME"],
			defaultOpenRouterAppName,
		),
		categories: firstNonEmpty(
			env["AGENTFIELD_OPENROUTER_CATEGORIES"],
			env["OR_CATEGORIES"],
			defaultOpenRouterCategories,
		),
	}, true
}

func applyOpenRouterAttributionEnv(env map[string]string) {
	resolved, ok := resolveOpenRouterAttribution(env)
	if !ok {
		delete(env, "AGENTFIELD_OPENROUTER_SITE_URL")
		delete(env, "AGENTFIELD_OPENROUTER_APP_NAME")
		delete(env, "AGENTFIELD_OPENROUTER_CATEGORIES")
		delete(env, "OR_SITE_URL")
		delete(env, "OR_APP_NAME")
		delete(env, "OR_CATEGORIES")
		return
	}

	setDefaultEnv(env, "AGENTFIELD_OPENROUTER_SITE_URL", resolved.siteURL)
	setDefaultEnv(env, "AGENTFIELD_OPENROUTER_APP_NAME", resolved.appName)
	setDefaultEnv(env, "AGENTFIELD_OPENROUTER_CATEGORIES", resolved.categories)
	setDefaultEnv(env, "OR_SITE_URL", resolved.siteURL)
	setDefaultEnv(env, "OR_APP_NAME", resolved.appName)
	setDefaultEnv(env, "OR_CATEGORIES", resolved.categories)
}

func openRouterAttributionHeaders(env map[string]string) map[string]string {
	resolved, ok := resolveOpenRouterAttribution(env)
	if !ok {
		return map[string]string{}
	}
	return map[string]string{
		"HTTP-Referer":            resolved.siteURL,
		"X-OpenRouter-Title":      resolved.appName,
		"X-Title":                 resolved.appName,
		"X-OpenRouter-Categories": resolved.categories,
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
