package ui

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

const agentSecretsTestScope = "test-node"

func newAgentSecretsTestRouter(t *testing.T) (*gin.Engine, string) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	agentfieldHome := t.TempDir()
	store := setupTestStorage(t)
	err := store.StoreAgentPackage(context.Background(), &types.AgentPackage{
		ID:   "agent-x",
		Name: agentSecretsTestScope,
		ConfigurationSchema: json.RawMessage(`{
			"user_environment": {
				"required": [{"name": "OPENAI_API_KEY", "type": "secret"}],
				"require_one_of": [{"options": [{"name": "ANTHROPIC_API_KEY", "type": "secret"}]}]
			}
		}`),
	})
	require.NoError(t, err)

	handler := NewAgentSecretsHandler(store, agentfieldHome)
	router := gin.New()
	router.GET("/agents/:agentId/secrets", handler.ListAgentSecretsHandler)
	router.PUT("/agents/:agentId/secrets", handler.SetAgentSecretHandler)
	router.DELETE("/agents/:agentId/secrets/:key", handler.DeleteAgentSecretHandler)
	return router, agentfieldHome
}

func agentSecretsRequest(t *testing.T, router http.Handler, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	request := httptest.NewRequest(method, path, strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	return response
}

// Validation contract 1: PUT writes the node scope consumed by runner-side resolution.
func TestAgentSecretsPutResolvesForRunner(t *testing.T) {
	router, home := newAgentSecretsTestRouter(t)
	response := agentSecretsRequest(t, router, http.MethodPut, "/agents/agent-x/secrets",
		`{"key":"OPENAI_API_KEY","value":"sk-test"}`)
	require.Equal(t, http.StatusNoContent, response.Code)

	store, err := packages.NewSecretStore(home)
	require.NoError(t, err)
	got, found, err := store.Get(agentSecretsTestScope, "OPENAI_API_KEY")
	require.NoError(t, err)
	require.True(t, found)
	require.Equal(t, "sk-test", got)

	resolved, err := (&packages.EnvResolver{
		Store:    store,
		NodeName: agentSecretsTestScope,
	}).Resolve(packages.UserEnvironmentConfig{
		Required: []packages.UserEnvironmentVar{{Name: "OPENAI_API_KEY"}},
	})
	require.NoError(t, err)
	require.Equal(t, "sk-test", resolved["OPENAI_API_KEY"])
}

// Validation contract 2: GET exposes set state and declared unset keys, never values.
func TestAgentSecretsListNamesOnly(t *testing.T) {
	router, _ := newAgentSecretsTestRouter(t)
	require.Equal(t, http.StatusNoContent, agentSecretsRequest(t, router, http.MethodPut,
		"/agents/agent-x/secrets", `{"key":"OPENAI_API_KEY","value":"sk-test"}`).Code)

	response := agentSecretsRequest(t, router, http.MethodGet, "/agents/agent-x/secrets", "")
	require.Equal(t, http.StatusOK, response.Code)
	require.NotContains(t, response.Body.String(), "sk-test")
	require.JSONEq(t, `{"secrets":[
		{"key":"ANTHROPIC_API_KEY","is_set":false},
		{"key":"OPENAI_API_KEY","is_set":true}
	]}`, response.Body.String())
}

// Validation contract 3: DELETE removes the secret and remains idempotent.
func TestAgentSecretsDeleteIsIdempotent(t *testing.T) {
	router, home := newAgentSecretsTestRouter(t)
	require.Equal(t, http.StatusNoContent, agentSecretsRequest(t, router, http.MethodPut,
		"/agents/agent-x/secrets", `{"key":"OPENAI_API_KEY","value":"sk-test"}`).Code)

	for range 2 {
		response := agentSecretsRequest(t, router, http.MethodDelete,
			"/agents/agent-x/secrets/OPENAI_API_KEY", "")
		require.Equal(t, http.StatusNoContent, response.Code)
	}
	store, err := packages.NewSecretStore(home)
	require.NoError(t, err)
	_, found, err := store.Get(agentSecretsTestScope, "OPENAI_API_KEY")
	require.NoError(t, err)
	require.False(t, found)
}

// Validation contract 4: invalid keys and empty values return 400 without writes.
func TestAgentSecretsRejectsInvalidPut(t *testing.T) {
	router, home := newAgentSecretsTestRouter(t)
	for _, body := range []string{
		`{"key":"lowercase","value":"secret"}`,
		`{"key":"BAD-KEY","value":"secret"}`,
		`{"key":"VALID_KEY","value":""}`,
	} {
		response := agentSecretsRequest(t, router, http.MethodPut, "/agents/agent-x/secrets", body)
		require.Equal(t, http.StatusBadRequest, response.Code)
	}
	store, err := packages.NewSecretStore(home)
	require.NoError(t, err)
	keys, err := store.List(agentSecretsTestScope)
	require.NoError(t, err)
	require.Empty(t, keys)
}

// Validation contract 5: every operation returns 404 for an unknown agent ID.
func TestAgentSecretsUnknownAgent(t *testing.T) {
	router, _ := newAgentSecretsTestRouter(t)
	tests := []struct {
		method string
		path   string
		body   string
	}{
		{http.MethodGet, "/agents/missing/secrets", ""},
		{http.MethodPut, "/agents/missing/secrets", `{"key":"KEY","value":"secret"}`},
		{http.MethodDelete, "/agents/missing/secrets/KEY", ""},
	}
	for _, test := range tests {
		response := agentSecretsRequest(t, router, test.method, test.path, test.body)
		require.Equal(t, http.StatusNotFound, response.Code)
	}
}

// Validation contract 6: JSON-special and Unicode content round-trips exactly.
func TestAgentSecretsComplexValueRoundTrip(t *testing.T) {
	router, home := newAgentSecretsTestRouter(t)
	want := "line \"one\"\n雪"
	body, err := json.Marshal(map[string]string{"key": "COMPLEX_VALUE", "value": want})
	require.NoError(t, err)
	response := agentSecretsRequest(t, router, http.MethodPut, "/agents/agent-x/secrets", string(body))
	require.Equal(t, http.StatusNoContent, response.Code)
	require.Empty(t, response.Body.Bytes())

	store, err := packages.NewSecretStore(home)
	require.NoError(t, err)
	got, found, err := store.Get(agentSecretsTestScope, "COMPLEX_VALUE")
	require.NoError(t, err)
	require.True(t, found)
	require.True(t, bytes.Equal([]byte(want), []byte(got)))
}
