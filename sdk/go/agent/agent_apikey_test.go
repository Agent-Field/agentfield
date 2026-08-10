package agent

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
)

// captureAPIKeyServer records the credential headers of the first request and
// answers with an empty JSON object so the client call completes.
func captureAPIKeyServer(t *testing.T, apiKey, authorization *string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*apiKey = r.Header.Get("X-API-Key")
		*authorization = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// An agent configured with an API key must present it on control-plane
// requests, so it can register against a control plane that has authentication
// enabled.
func TestNew_ConfigAPIKeyIsSentToControlPlane(t *testing.T) {
	var gotAPIKey, gotAuth string
	srv := captureAPIKeyServer(t, &gotAPIKey, &gotAuth)

	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: srv.URL,
		APIKey:        "config-key",
	})
	require.NoError(t, err)
	require.NotNil(t, a.client)

	_, err = a.client.GetNode(context.Background(), "node-1")
	require.NoError(t, err)
	require.Equal(t, "config-key", gotAPIKey)
}

// `af run` and `af dev` export AGENTFIELD_API_KEY for the agents they start, so
// an agent that names no key in code still authenticates.
func TestNew_APIKeyDefaultsFromEnvironment(t *testing.T) {
	var gotAPIKey, gotAuth string
	srv := captureAPIKeyServer(t, &gotAPIKey, &gotAuth)

	t.Setenv("AGENTFIELD_API_KEY", "env-key")

	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: srv.URL,
	})
	require.NoError(t, err)

	_, err = a.client.GetNode(context.Background(), "node-1")
	require.NoError(t, err)
	require.Equal(t, "env-key", gotAPIKey)
}

// An explicit key wins over the environment, so a caller can always override
// what the parent process exported.
func TestNew_ConfigAPIKeyOverridesEnvironment(t *testing.T) {
	var gotAPIKey, gotAuth string
	srv := captureAPIKeyServer(t, &gotAPIKey, &gotAuth)

	t.Setenv("AGENTFIELD_API_KEY", "env-key")

	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: srv.URL,
		APIKey:        "config-key",
	})
	require.NoError(t, err)

	_, err = a.client.GetNode(context.Background(), "node-1")
	require.NoError(t, err)
	require.Equal(t, "config-key", gotAPIKey)
}

// With no key anywhere the agent must stay exactly as it was: no credential
// header, which is the default local setup.
func TestNew_NoAPIKeySendsNoHeader(t *testing.T) {
	var gotAPIKey, gotAuth string
	srv := captureAPIKeyServer(t, &gotAPIKey, &gotAuth)

	t.Setenv("AGENTFIELD_API_KEY", "")

	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: srv.URL,
	})
	require.NoError(t, err)

	_, err = a.client.GetNode(context.Background(), "node-1")
	require.NoError(t, err)
	require.Empty(t, gotAPIKey)
}

// The API key is a separate credential from Token: setting one must not
// disturb the other, because Token also drives incoming-request auth.
func TestNew_APIKeyAndTokenCoexist(t *testing.T) {
	var gotAPIKey, gotAuth string
	srv := captureAPIKeyServer(t, &gotAPIKey, &gotAuth)

	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: srv.URL,
		APIKey:        "config-key",
		Token:         "bearer-token",
	})
	require.NoError(t, err)

	_, err = a.client.GetNode(context.Background(), "node-1")
	require.NoError(t, err)
	require.Equal(t, "config-key", gotAPIKey)
	require.Equal(t, "Bearer bearer-token", gotAuth)
}
