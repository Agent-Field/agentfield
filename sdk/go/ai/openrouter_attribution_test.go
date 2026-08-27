package ai

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestApplyOpenRouterAttributionHeadersIncludesCategories(t *testing.T) {
	t.Setenv("AGENTFIELD_OPENROUTER_SITE_URL", "")
	t.Setenv("AGENTFIELD_OPENROUTER_APP_NAME", "")
	t.Setenv("AGENTFIELD_OPENROUTER_CATEGORIES", "")
	t.Setenv("OR_SITE_URL", "")
	t.Setenv("OR_APP_NAME", "")
	t.Setenv("OR_CATEGORIES", "")

	header := make(http.Header)
	applyOpenRouterAttributionHeaders(header, "", "")

	assert.Equal(t, defaultOpenRouterSiteURL, header.Get("HTTP-Referer"))
	assert.Equal(t, defaultOpenRouterAppName, header.Get("X-OpenRouter-Title"))
	assert.Equal(t, defaultOpenRouterAppName, header.Get("X-Title"))
	assert.Equal(t, defaultOpenRouterCategories, header.Get("X-OpenRouter-Categories"))
}

func TestApplyOpenRouterAttributionHeadersCategoriesEnvOverride(t *testing.T) {
	t.Setenv("AGENTFIELD_OPENROUTER_CATEGORIES", "research,translation")

	header := make(http.Header)
	applyOpenRouterAttributionHeaders(header, "https://example.com", "Example")

	assert.Equal(t, "https://example.com", header.Get("HTTP-Referer"))
	assert.Equal(t, "Example", header.Get("X-OpenRouter-Title"))
	assert.Equal(t, "research,translation", header.Get("X-OpenRouter-Categories"))
}

func TestApplyOpenRouterAttributionHeadersDisabled(t *testing.T) {
	t.Setenv("AGENTFIELD_OPENROUTER_ATTRIBUTION", "false")

	header := make(http.Header)
	applyOpenRouterAttributionHeaders(header, "https://example.com", "Example")

	assert.Empty(t, header.Get("HTTP-Referer"))
	assert.Empty(t, header.Get("X-OpenRouter-Title"))
	assert.Empty(t, header.Get("X-Title"))
	assert.Empty(t, header.Get("X-OpenRouter-Categories"))
}
