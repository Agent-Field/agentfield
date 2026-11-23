package config

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestApplyEnvOverrides_CORS(t *testing.T) {
	t.Setenv("AGENTFIELD_API_CORS_ALLOWED_ORIGINS", "https://app.example.com, https://admin.example.com,")
	t.Setenv("AGENTFIELD_API_CORS_ALLOWED_METHODS", "GET,POST")
	t.Setenv("AGENTFIELD_API_CORS_ALLOWED_HEADERS", "Authorization, X-API-Key")
	t.Setenv("AGENTFIELD_API_CORS_EXPOSED_HEADERS", "Content-Length,X-Total-Count")
	t.Setenv("AGENTFIELD_API_CORS_ALLOW_CREDENTIALS", "true")

	cfg := &Config{}
	ApplyEnvOverrides(cfg)

	require.Equal(t, []string{"https://app.example.com", "https://admin.example.com"}, cfg.API.CORS.AllowedOrigins)
	require.Equal(t, []string{"GET", "POST"}, cfg.API.CORS.AllowedMethods)
	require.Equal(t, []string{"Authorization", "X-API-Key"}, cfg.API.CORS.AllowedHeaders)
	require.Equal(t, []string{"Content-Length", "X-Total-Count"}, cfg.API.CORS.ExposedHeaders)
	require.True(t, cfg.API.CORS.AllowCredentials)
}

func TestApplyEnvOverrides_CORSUnsetKeepsConfig(t *testing.T) {
	for _, name := range []string{
		"AGENTFIELD_API_CORS_ALLOWED_ORIGINS",
		"AGENTFIELD_API_CORS_ALLOWED_METHODS",
		"AGENTFIELD_API_CORS_ALLOWED_HEADERS",
		"AGENTFIELD_API_CORS_EXPOSED_HEADERS",
		"AGENTFIELD_API_CORS_ALLOW_CREDENTIALS",
	} {
		t.Setenv(name, "")
	}

	cfg := &Config{}
	cfg.API.CORS = CORSConfig{
		AllowedOrigins:   []string{"https://from-file.example.com"},
		AllowedMethods:   []string{"GET"},
		AllowCredentials: true,
	}
	ApplyEnvOverrides(cfg)

	require.Equal(t, []string{"https://from-file.example.com"}, cfg.API.CORS.AllowedOrigins)
	require.Equal(t, []string{"GET"}, cfg.API.CORS.AllowedMethods)
	require.Empty(t, cfg.API.CORS.AllowedHeaders)
	require.True(t, cfg.API.CORS.AllowCredentials)
}
