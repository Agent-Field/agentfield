package cli

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/stretchr/testify/require"
)

// isolateCredentials points credential storage at a throwaway home and clears
// the flag/env credentials, so a test never reads or writes the real
// ~/.agentfield and never inherits the developer's own key.
func isolateCredentials(t *testing.T) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("AGENTFIELD_API_KEY", "")

	oldServer, oldAPIKey := serverURL, apiKey
	serverURL, apiKey = "", ""
	t.Cleanup(func() {
		serverURL, apiKey = oldServer, oldAPIKey
	})
	return home
}

// Contract: the CLI sends the most specific credential the user supplied —
// flag, then environment, then the one `af auth login` stored for this server —
// and sends nothing at all when none exists.
func TestGetAPIKey_Precedence(t *testing.T) {
	home := isolateCredentials(t)
	t.Setenv("AGENTFIELD_SERVER", "http://localhost:8080")

	require.Equal(t, "", GetAPIKey(), "a machine with no key configured must send no credential")

	require.NoError(t, packages.NewCredentialStore(home).Save("http://localhost:8080", "stored-key"))
	key, source := resolveAPIKeyWithSource()
	require.Equal(t, "stored-key", key)
	require.Equal(t, apiKeySourceStored, source)

	t.Setenv("AGENTFIELD_API_KEY", "env-key")
	key, source = resolveAPIKeyWithSource()
	require.Equal(t, "env-key", key, "AGENTFIELD_API_KEY must win over a stored credential")
	require.Equal(t, apiKeySourceEnv, source)

	apiKey = "flag-key"
	key, source = resolveAPIKeyWithSource()
	require.Equal(t, "flag-key", key, "--api-key must win over everything")
	require.Equal(t, apiKeySourceFlag, source)
}

// Contract: a stored credential belongs to the control plane it was stored
// for. Pointing the CLI somewhere else must not replay the key at a server the
// user never logged into.
func TestGetAPIKey_StoredCredentialIsScopedToServer(t *testing.T) {
	home := isolateCredentials(t)
	require.NoError(t, packages.NewCredentialStore(home).Save("http://localhost:8080", "stored-key"))

	t.Setenv("AGENTFIELD_SERVER", "http://localhost:8080/")
	require.Equal(t, "stored-key", GetAPIKey(), "a trailing slash names the same control plane")

	t.Setenv("AGENTFIELD_SERVER", "https://someone-elses-cp.example.com")
	require.Equal(t, "", GetAPIKey())
}

// Contract: reading a credential never creates one. A user who never runs
// `af auth login` sees no new files.
func TestGetAPIKey_DoesNotCreateCredentialsFile(t *testing.T) {
	home := isolateCredentials(t)
	t.Setenv("AGENTFIELD_SERVER", "http://localhost:8080")

	require.Equal(t, "", GetAPIKey())

	store, err := credentialStore()
	require.NoError(t, err)
	require.NoFileExists(t, store.Path())
	require.DirExists(t, home)
}

// Contract: displayed keys are recognisable but not reusable — enough to tell
// two keys apart, never enough to authenticate with.
func TestMaskAPIKey(t *testing.T) {
	tests := []struct {
		name string
		key  string
		want string
	}{
		{name: "empty", key: "", want: ""},
		{name: "short key fully masked", key: "abcd", want: "••••"},
		{name: "boundary length fully masked", key: "abcdefgh", want: "••••••••"},
		{name: "vendor prefix kept", key: "af_live_0123456789abcdef", want: "af_live_…cdef"},
		{name: "dashed prefix kept", key: "sk-0123456789abcdef", want: "sk-…cdef"},
		{name: "no prefix shows three chars", key: "0123456789abcdef", want: "012…cdef"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := maskAPIKey(tc.key)
			require.Equal(t, tc.want, got)
			if len(tc.key) > 8 {
				require.NotContains(t, got, tc.key, "the full key must never be rendered")
			}
		})
	}
}

// Contract: a 401 tells the user what to run, naming the server they were
// actually talking to, and carries the control plane's own explanation.
func TestAuthRequiredError(t *testing.T) {
	err := authRequiredError("http://cp.example.com/", []byte(`{"error":"unauthorized","message":"invalid or missing API key"}`))
	require.ErrorContains(t, err, "authentication required by http://cp.example.com")
	require.ErrorContains(t, err, "invalid or missing API key")
	require.ErrorContains(t, err, "af auth login --server http://cp.example.com")

	// A body that is not the control plane's envelope still yields the remedy.
	err = authRequiredError("http://cp.example.com", []byte("not json"))
	require.ErrorContains(t, err, "authentication required by http://cp.example.com")
	require.ErrorContains(t, err, "af auth login --server http://cp.example.com")
}

// Contract: every command that talks to the control plane surfaces a 401 as an
// instruction, not as a bare status code.
func TestMakeRequest_UnauthorizedIsActionable(t *testing.T) {
	isolateCredentials(t)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":"unauthorized","message":"invalid or missing API key"}`))
	}))
	defer server.Close()
	t.Setenv("AGENTFIELD_SERVER", server.URL)

	resp, err := makeRequest(context.Background(), http.MethodGet, "/api/v1/nodes", nil, "application/json")
	require.Nil(t, resp)
	require.ErrorContains(t, err, "authentication required by "+server.URL)
	require.ErrorContains(t, err, "af auth login --server "+server.URL)
}

// Contract: `af auth login` verifies the key, stores it for the current server,
// and confirms with a masked value — never the key itself.
func TestAuthLogin_StoresVerifiedKey(t *testing.T) {
	isolateCredentials(t)

	var sawKey string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawKey = r.Header.Get("X-API-Key")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"nodes":[]}`))
	}))
	defer server.Close()
	t.Setenv("AGENTFIELD_SERVER", server.URL)

	apiKey = "af_live_0123456789abcdef"
	output := captureOutput(t, func() {
		cmd := NewAuthCommand()
		cmd.SetArgs([]string{"login"})
		require.NoError(t, cmd.Execute())
	})

	require.Equal(t, "af_live_0123456789abcdef", sawKey, "the key must be verified against the control plane")
	require.Contains(t, output, "af_live_…cdef")
	require.NotContains(t, output, "af_live_0123456789abcdef", "the full key must never be printed")

	apiKey = ""
	require.Equal(t, "af_live_0123456789abcdef", GetAPIKey(), "the stored key must be used by later commands")
}

// Contract: a key the control plane rejects is not saved. Storing it would
// leave the user with a credential that silently fails on every command.
func TestAuthLogin_RejectedKeyIsNotStored(t *testing.T) {
	isolateCredentials(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":"unauthorized"}`))
	}))
	defer server.Close()
	t.Setenv("AGENTFIELD_SERVER", server.URL)

	apiKey = "wrong-key-0123456789"
	cmd := NewAuthCommand()
	cmd.SetArgs([]string{"login"})
	err := cmd.Execute()
	require.ErrorContains(t, err, "rejected that API key")

	apiKey = ""
	require.Equal(t, "", GetAPIKey())
	store, storeErr := credentialStore()
	require.NoError(t, storeErr)
	require.NoFileExists(t, store.Path())
}

// Contract: --no-verify stores a key for a control plane that is not running,
// so an operator can provision credentials before starting the server.
func TestAuthLogin_NoVerifySkipsTheProbe(t *testing.T) {
	isolateCredentials(t)
	t.Setenv("AGENTFIELD_SERVER", "http://127.0.0.1:1")

	apiKey = "af_live_0123456789abcdef"
	captureOutput(t, func() {
		cmd := NewAuthCommand()
		cmd.SetArgs([]string{"login", "--no-verify"})
		require.NoError(t, cmd.Execute())
	})

	apiKey = ""
	require.Equal(t, "af_live_0123456789abcdef", GetAPIKey())
}

// Contract: `af auth status` reports the server, whether a key is in play and
// where it came from, always masked.
func TestAuthStatus_ReportsSourceWithoutRevealingKey(t *testing.T) {
	home := isolateCredentials(t)
	t.Setenv("AGENTFIELD_SERVER", "http://localhost:8080")

	output := captureOutput(t, func() {
		cmd := NewAuthCommand()
		cmd.SetArgs([]string{"status"})
		require.NoError(t, cmd.Execute())
	})
	require.Contains(t, output, "http://localhost:8080")
	require.Contains(t, output, "none")
	require.Contains(t, output, "af auth login --server http://localhost:8080")

	require.NoError(t, packages.NewCredentialStore(home).Save("http://localhost:8080", "af_live_0123456789abcdef"))
	output = captureOutput(t, func() {
		cmd := NewAuthCommand()
		cmd.SetArgs([]string{"status"})
		require.NoError(t, cmd.Execute())
	})
	require.Contains(t, output, "af_live_…cdef")
	require.Contains(t, output, apiKeySourceStored)
	require.NotContains(t, output, "af_live_0123456789abcdef")

	// An environment variable shadowing the stored key is called out, so the
	// user knows why the stored one is not being sent.
	t.Setenv("AGENTFIELD_API_KEY", "env-key-0123456789")
	output = captureOutput(t, func() {
		cmd := NewAuthCommand()
		cmd.SetArgs([]string{"status"})
		require.NoError(t, cmd.Execute())
	})
	require.Contains(t, output, apiKeySourceEnv)
	require.Contains(t, output, "overridden")
	require.NotContains(t, output, "env-key-0123456789")
}

// Contract: `af auth logout` removes the stored key for the current server and
// leaves other servers alone.
func TestAuthLogout_RemovesOnlyTheCurrentServer(t *testing.T) {
	home := isolateCredentials(t)
	store := packages.NewCredentialStore(home)
	require.NoError(t, store.Save("http://localhost:8080", "local-key-0123456789"))
	require.NoError(t, store.Save("https://cp.example.com", "remote-key-0123456789"))

	t.Setenv("AGENTFIELD_SERVER", "http://localhost:8080")
	captureOutput(t, func() {
		cmd := NewAuthCommand()
		cmd.SetArgs([]string{"logout"})
		require.NoError(t, cmd.Execute())
	})

	require.Equal(t, "", store.Lookup("http://localhost:8080"))
	require.Equal(t, "remote-key-0123456789", store.Lookup("https://cp.example.com"))

	// Logging out again is a no-op rather than an error.
	output := captureOutput(t, func() {
		cmd := NewAuthCommand()
		cmd.SetArgs([]string{"logout"})
		require.NoError(t, cmd.Execute())
	})
	require.Contains(t, output, "No stored API key")
}
