package ai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type capTestTransport struct {
	body map[string]any
}

func (rt *capTestTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	raw, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(raw, &rt.body); err != nil {
		return nil, err
	}
	resp := `{"id":"x","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(resp)),
		Request:    req,
	}, nil
}

func partialConfig() *Config {
	return &Config{
		APIKey:  "test-key",
		BaseURL: "https://api.example.com/v1",
		Model:   "gpt-4o",
	}
}

func TestNewClient_PartialConfigGetsBoundedDefaults(t *testing.T) {
	rt := &capTestTransport{}
	cfg := partialConfig()

	client, err := NewClient(cfg, WithTransport(rt))
	require.NoError(t, err)

	assert.Equal(t, defaultTimeout, client.httpClient.Timeout)
	assert.Equal(t, defaultMaxTokens, client.config.MaxTokens)

	_, err = client.Complete(context.Background(), "hi")
	require.NoError(t, err)
	assert.EqualValues(t, defaultMaxTokens, rt.body["max_tokens"])

	// The caller's struct is left as it was.
	assert.Zero(t, cfg.MaxTokens)
	assert.Zero(t, cfg.Timeout)
}

func TestNewClient_ExplicitLimitsAreHonored(t *testing.T) {
	rt := &capTestTransport{}
	cfg := partialConfig()
	cfg.MaxTokens = 256
	cfg.Timeout = 5 * time.Minute

	client, err := NewClient(cfg, WithTransport(rt))
	require.NoError(t, err)
	assert.Equal(t, 5*time.Minute, client.httpClient.Timeout)

	_, err = client.Complete(context.Background(), "hi")
	require.NoError(t, err)
	assert.EqualValues(t, 256, rt.body["max_tokens"])

	_, err = client.Complete(context.Background(), "hi", WithMaxTokens(32))
	require.NoError(t, err)
	assert.EqualValues(t, 32, rt.body["max_tokens"])
}

func TestNewClient_NegativeLimitsFallBackToDefaults(t *testing.T) {
	cfg := partialConfig()
	cfg.MaxTokens = -1
	cfg.Timeout = -time.Second

	client, err := NewClient(cfg)
	require.NoError(t, err)
	assert.Equal(t, defaultMaxTokens, client.config.MaxTokens)
	assert.Equal(t, defaultTimeout, client.httpClient.Timeout)
}
