package ui

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/packages"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

const agentSecretsTestScope = "test-node"

func newAgentSecretsTestRouter(t *testing.T) (*gin.Engine, string) {
	return newAgentSecretsTestRouterWithLookup(t, func(string) (string, bool) { return "", false })
}

func newAgentSecretsTestRouterWithLookup(
	t *testing.T,
	lookupEnv func(string) (string, bool),
) (*gin.Engine, string) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	agentfieldHome := t.TempDir()
	store := setupTestStorage(t)
	err := store.StoreAgentPackage(context.Background(), &types.AgentPackage{
		ID:   "agent-x",
		Name: agentSecretsTestScope,
		ConfigurationSchema: json.RawMessage(`{
			"user_environment": {
				"required": [
					{"name": "OPENAI_API_KEY", "description": "OpenAI key", "type": "secret"},
					{"name": "NODE_SCOPED_KEY", "type": "secret", "scope": "node"}
				],
				"require_one_of": [{"id":"llm_provider","description":"an LLM provider key","options": [{"name": "ANTHROPIC_API_KEY", "description":"Anthropic key", "type": "secret"}]}],
				"optional": [
					{"name":"SWE_DEFAULT_RUNTIME","description":"Coding runtime"},
					{"name":"AGENTFIELD_SERVER","description":"Control-plane URL","default":"http://localhost:8080"}
				]
			}
		}`),
	})
	require.NoError(t, err)

	handler := NewAgentSecretsHandler(store, agentfieldHome)
	handler.lookupEnv = lookupEnv
	router := gin.New()
	router.GET("/agents/:agentId/secrets", handler.ListAgentSecretsHandler)
	router.PUT("/agents/:agentId/secrets", handler.SetAgentSecretHandler)
	router.DELETE("/agents/:agentId/secrets/:key", handler.DeleteAgentSecretHandler)
	router.GET("/secrets", handler.ListAllSecretsHandler)
	return router, agentfieldHome
}

func TestAgentSecretsListProcessEnvironmentResolution(t *testing.T) {
	environment := map[string]string{
		"OPENAI_API_KEY":    "env-value",
		"ANTHROPIC_API_KEY": "",
		"NODE_SCOPED_KEY":   "env-value",
	}
	router, home := newAgentSecretsTestRouterWithLookup(t, func(key string) (string, bool) {
		value, ok := environment[key]
		return value, ok
	})
	store, err := packages.NewSecretStore(home)
	require.NoError(t, err)
	require.NoError(t, store.Set(agentSecretsTestScope, "NODE_SCOPED_KEY", "stored-value"))

	response := agentSecretsRequest(t, router, http.MethodGet, "/agents/agent-x/secrets?include=env", "")
	require.Equal(t, http.StatusOK, response.Code)
	require.NotContains(t, response.Body.String(), "env-value")
	require.NotContains(t, response.Body.String(), "stored-value")
	require.JSONEq(t, `{"secrets":[
		{"key":"AGENTFIELD_SERVER","is_set":false,"declared_scope":"global","description":"Control-plane URL","default":"http://localhost:8080","requirement":"optional"},
		{"key":"ANTHROPIC_API_KEY","is_set":false,"declared_scope":"global","description":"Anthropic key","secret":true,"requirement":"one_of","group":"llm_provider","group_description":"an LLM provider key"},
		{"key":"NODE_SCOPED_KEY","is_set":true,"env":true,"scope":"node","declared_scope":"node","secret":true,"requirement":"required"},
		{"key":"OPENAI_API_KEY","is_set":true,"env":true,"declared_scope":"global","description":"OpenAI key","secret":true,"requirement":"required"},
		{"key":"SWE_DEFAULT_RUNTIME","is_set":false,"declared_scope":"global","description":"Coding runtime","requirement":"optional"}
	]}`, response.Body.String())
}

// A handler built without an injected lookup (a zero-value struct rather than
// the constructor) must still consult the real process environment.
func TestAgentSecretsListDefaultsToProcessEnvironment(t *testing.T) {
	router, _ := newAgentSecretsTestRouterWithLookup(t, nil)
	t.Setenv("OPENAI_API_KEY", "from-process")
	t.Setenv("ANTHROPIC_API_KEY", "")

	response := agentSecretsRequest(t, router, http.MethodGet, "/agents/agent-x/secrets?include=env", "")
	require.Equal(t, http.StatusOK, response.Code)
	require.NotContains(t, response.Body.String(), "from-process")
	require.Contains(t, response.Body.String(), `{"key":"OPENAI_API_KEY","is_set":true,"env":true,`)
	require.Contains(t, response.Body.String(), `{"key":"ANTHROPIC_API_KEY","is_set":false,`)
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
	// EnvResolver prefers a non-empty process env value, so a developer
	// machine with this key exported would resolve the host value instead of
	// the stored one. Empty counts as unset; this keeps the test hermetic.
	t.Setenv("OPENAI_API_KEY", "")
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
		{"key":"NODE_SCOPED_KEY","is_set":false},
		{"key":"OPENAI_API_KEY","is_set":true,"scope":"global"}
	]}`, response.Body.String())
}

func TestAgentSecretsListIncludesEnvironmentMetadataOptIn(t *testing.T) {
	router, home := newAgentSecretsTestRouter(t)
	store, err := packages.NewSecretStore(home)
	require.NoError(t, err)
	require.NoError(t, store.Set(agentSecretsTestScope, "ANTHROPIC_API_KEY", "node-value"))
	require.NoError(t, store.Set(agentSecretsTestScope, "UNDECLARED_NODE", "value"))

	response := agentSecretsRequest(t, router, http.MethodGet, "/agents/agent-x/secrets?include=env", "")
	require.Equal(t, http.StatusOK, response.Code)
	require.NotContains(t, response.Body.String(), "node-value")
	require.JSONEq(t, `{"secrets":[
		{"key":"AGENTFIELD_SERVER","is_set":false,"declared_scope":"global","description":"Control-plane URL","default":"http://localhost:8080","requirement":"optional"},
		{"key":"ANTHROPIC_API_KEY","is_set":true,"scope":"node","declared_scope":"global","description":"Anthropic key","secret":true,"requirement":"one_of","group":"llm_provider","group_description":"an LLM provider key"},
		{"key":"NODE_SCOPED_KEY","is_set":false,"declared_scope":"node","secret":true,"requirement":"required"},
		{"key":"OPENAI_API_KEY","is_set":false,"declared_scope":"global","description":"OpenAI key","secret":true,"requirement":"required"},
		{"key":"SWE_DEFAULT_RUNTIME","is_set":false,"declared_scope":"global","description":"Coding runtime","requirement":"optional"},
		{"key":"UNDECLARED_NODE","is_set":true,"scope":"node"}
	]}`, response.Body.String())
}

func TestAgentSecretsListLegacyShapeIgnoresOtherIncludeValues(t *testing.T) {
	router, _ := newAgentSecretsTestRouter(t)
	for _, path := range []string{
		"/agents/agent-x/secrets",
		"/agents/agent-x/secrets?include=other",
	} {
		response := agentSecretsRequest(t, router, http.MethodGet, path, "")
		require.Equal(t, http.StatusOK, response.Code)
		require.JSONEq(t, `{"secrets":[
			{"key":"ANTHROPIC_API_KEY","is_set":false},
			{"key":"NODE_SCOPED_KEY","is_set":false},
			{"key":"OPENAI_API_KEY","is_set":false}
		]}`, response.Body.String())
		require.NotContains(t, response.Body.String(), "SWE_DEFAULT_RUNTIME")
		require.NotContains(t, response.Body.String(), "description")
	}
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

// Exercises malformed-JSON binding and the oversized-value validation branch.
func TestAgentSecretsRejectsMalformedAndOversizedPut(t *testing.T) {
	router, _ := newAgentSecretsTestRouter(t)
	for _, body := range []string{
		`{`,
		`{"key":"VALID_KEY","value":"` + strings.Repeat("x", maxAgentSecretValueBytes+1) + `"}`,
	} {
		response := agentSecretsRequest(t, router, http.MethodPut, "/agents/agent-x/secrets", body)
		require.Equal(t, http.StatusBadRequest, response.Code)
	}
}

// Exercises secret-store constructor failures in all three handlers.
func TestAgentSecretsStoreOpenFailures(t *testing.T) {
	_, home := newAgentSecretsTestRouter(t)
	blockedHome := filepath.Join(home, "not-a-directory")
	require.NoError(t, os.WriteFile(blockedHome, []byte("block"), 0o600))

	store := setupTestStorage(t)
	require.NoError(t, store.StoreAgentPackage(context.Background(), &types.AgentPackage{
		ID:                  "agent-x",
		Name:                agentSecretsTestScope,
		ConfigurationSchema: json.RawMessage(`{"user_environment":{}}`),
	}))
	handler := NewAgentSecretsHandler(store, blockedHome)
	failureRouter := gin.New()
	failureRouter.GET("/agents/:agentId/secrets", handler.ListAgentSecretsHandler)
	failureRouter.PUT("/agents/:agentId/secrets", handler.SetAgentSecretHandler)
	failureRouter.DELETE("/agents/:agentId/secrets/:key", handler.DeleteAgentSecretHandler)
	failureRouter.GET("/secrets", handler.ListAllSecretsHandler)

	for _, test := range []struct {
		name   string
		method string
		path   string
		body   string
	}{
		{"list", http.MethodGet, "/agents/agent-x/secrets", ""},
		{"set", http.MethodPut, "/agents/agent-x/secrets", `{"key":"KEY","value":"secret"}`},
		{"delete", http.MethodDelete, "/agents/agent-x/secrets/KEY", ""},
		{"list-all", http.MethodGet, "/secrets", ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			response := agentSecretsRequest(t, failureRouter, test.method, test.path, test.body)
			require.Equal(t, http.StatusInternalServerError, response.Code)
		})
	}
}

// Exercises List, Set, and Delete failures after a store opens successfully.
func TestAgentSecretsCorruptStoreFailures(t *testing.T) {
	router, home := newAgentSecretsTestRouter(t)
	store, err := packages.NewSecretStore(home)
	require.NoError(t, err)
	require.NoError(t, store.Set(agentSecretsTestScope, "KEY", "value"))
	require.NoError(t, os.WriteFile(
		filepath.Join(home, "secrets", agentSecretsTestScope+".enc"),
		[]byte("not encrypted data"),
		0o600,
	))

	for _, test := range []struct {
		method string
		path   string
		body   string
	}{
		{http.MethodGet, "/agents/agent-x/secrets", ""},
		{http.MethodPut, "/agents/agent-x/secrets", `{"key":"KEY","value":"secret","scope":"node"}`},
		{http.MethodDelete, "/agents/agent-x/secrets/KEY?scope=node", ""},
		{http.MethodGet, "/secrets", ""},
	} {
		response := agentSecretsRequest(t, router, test.method, test.path, test.body)
		require.Equal(t, http.StatusInternalServerError, response.Code)
	}
}

// Exercises the ID fallback used when a package has no name.
func TestAgentSecretScopeFallsBackToID(t *testing.T) {
	require.Equal(t, "agent-x", agentSecretScope(&types.AgentPackage{ID: "agent-x"}))
}

// Exercises invalid schema, de-duplication/empty-name branches, and the
// global-unless-node scope defaulting rule.
func TestDeclaredAgentSecretsEdgeCases(t *testing.T) {
	require.Nil(t, declaredAgentSecrets(json.RawMessage(`{`)))
	schema := json.RawMessage(`{"user_environment":{
		"required":[{"name":""},{"name":"KEY"},{"name":"NODE_KEY","scope":"node"}],
		"require_one_of":[{"options":[{"name":""},{"name":"KEY"},{"name":"WEIRD","scope":"bogus"}]}]
	}}`)
	require.Equal(t, map[string]string{
		"KEY":      "global",
		"NODE_KEY": "node",
		"WEIRD":    "global",
	}, declaredAgentSecrets(schema))
}

// Scope contract 1+2: undeclared keys default to the global scope; keys the
// manifest declares scope:node land in the node scope — mirroring af secrets.
func TestAgentSecretsScopeDefaults(t *testing.T) {
	router, home := newAgentSecretsTestRouter(t)
	require.Equal(t, http.StatusNoContent, agentSecretsRequest(t, router, http.MethodPut,
		"/agents/agent-x/secrets", `{"key":"UNDECLARED_KEY","value":"v1"}`).Code)
	require.Equal(t, http.StatusNoContent, agentSecretsRequest(t, router, http.MethodPut,
		"/agents/agent-x/secrets", `{"key":"NODE_SCOPED_KEY","value":"v2"}`).Code)

	store, err := packages.NewSecretStore(home)
	require.NoError(t, err)
	globalKeys, err := store.List("global")
	require.NoError(t, err)
	require.Equal(t, []string{"UNDECLARED_KEY"}, globalKeys)
	nodeKeys, err := store.List(agentSecretsTestScope)
	require.NoError(t, err)
	require.Equal(t, []string{"NODE_SCOPED_KEY"}, nodeKeys)
}

// Scope contract 3: explicit scope overrides the manifest default; anything
// other than node/global is rejected on both PUT and DELETE.
func TestAgentSecretsScopeOverrides(t *testing.T) {
	router, home := newAgentSecretsTestRouter(t)
	require.Equal(t, http.StatusNoContent, agentSecretsRequest(t, router, http.MethodPut,
		"/agents/agent-x/secrets", `{"key":"NODE_SCOPED_KEY","value":"v","scope":"global"}`).Code)
	require.Equal(t, http.StatusNoContent, agentSecretsRequest(t, router, http.MethodPut,
		"/agents/agent-x/secrets", `{"key":"UNDECLARED_KEY","value":"v","scope":"node"}`).Code)
	require.Equal(t, http.StatusBadRequest, agentSecretsRequest(t, router, http.MethodPut,
		"/agents/agent-x/secrets", `{"key":"UNDECLARED_KEY","value":"v","scope":"bogus"}`).Code)
	require.Equal(t, http.StatusBadRequest, agentSecretsRequest(t, router, http.MethodDelete,
		"/agents/agent-x/secrets/UNDECLARED_KEY?scope=bogus", "").Code)

	store, err := packages.NewSecretStore(home)
	require.NoError(t, err)
	globalKeys, err := store.List("global")
	require.NoError(t, err)
	require.Equal(t, []string{"NODE_SCOPED_KEY"}, globalKeys)
	nodeKeys, err := store.List(agentSecretsTestScope)
	require.NoError(t, err)
	require.Equal(t, []string{"UNDECLARED_KEY"}, nodeKeys)
}

// Scope contract 4+5: the listing reports effective resolution — node scope
// shadows global — and includes undeclared node-scoped keys (the runner
// injects them) while omitting undeclared global keys (it does not).
func TestAgentSecretsListEffectiveResolution(t *testing.T) {
	router, home := newAgentSecretsTestRouter(t)
	store, err := packages.NewSecretStore(home)
	require.NoError(t, err)
	require.NoError(t, store.Set("global", "OPENAI_API_KEY", "global-v"))
	require.NoError(t, store.Set(agentSecretsTestScope, "OPENAI_API_KEY", "node-v"))
	require.NoError(t, store.Set(agentSecretsTestScope, "UNDECLARED_NODE", "v"))
	require.NoError(t, store.Set("global", "UNDECLARED_GLOBAL", "v"))

	response := agentSecretsRequest(t, router, http.MethodGet, "/agents/agent-x/secrets", "")
	require.Equal(t, http.StatusOK, response.Code)
	require.JSONEq(t, `{"secrets":[
		{"key":"ANTHROPIC_API_KEY","is_set":false},
		{"key":"NODE_SCOPED_KEY","is_set":false},
		{"key":"OPENAI_API_KEY","is_set":true,"scope":"node"},
		{"key":"UNDECLARED_NODE","is_set":true,"scope":"node"}
	]}`, response.Body.String())
}

// Scope contract 6: DELETE uses the same defaulting rule as PUT and honors an
// explicit ?scope= override without touching the other scope's copy.
func TestAgentSecretsDeleteScopes(t *testing.T) {
	router, home := newAgentSecretsTestRouter(t)
	store, err := packages.NewSecretStore(home)
	require.NoError(t, err)
	require.NoError(t, store.Set("global", "OPENAI_API_KEY", "global-v"))
	require.NoError(t, store.Set(agentSecretsTestScope, "OPENAI_API_KEY", "node-v"))

	// Default for a global-declared key deletes the global copy only.
	require.Equal(t, http.StatusNoContent, agentSecretsRequest(t, router, http.MethodDelete,
		"/agents/agent-x/secrets/OPENAI_API_KEY", "").Code)
	nodeKeys, err := store.List(agentSecretsTestScope)
	require.NoError(t, err)
	require.Equal(t, []string{"OPENAI_API_KEY"}, nodeKeys)
	globalKeys, err := store.List("global")
	require.NoError(t, err)
	require.Empty(t, globalKeys)

	// Explicit node override removes the remaining node copy.
	require.Equal(t, http.StatusNoContent, agentSecretsRequest(t, router, http.MethodDelete,
		"/agents/agent-x/secrets/OPENAI_API_KEY?scope=node", "").Code)
	nodeKeys, err = store.List(agentSecretsTestScope)
	require.NoError(t, err)
	require.Empty(t, nodeKeys)
}

// Scope contract 7: the store-wide listing returns key+scope rows across all
// scopes and never values.
func TestListAllSecrets(t *testing.T) {
	router, home := newAgentSecretsTestRouter(t)
	store, err := packages.NewSecretStore(home)
	require.NoError(t, err)
	require.NoError(t, store.Set("global", "SHARED_KEY", "shh-1"))
	require.NoError(t, store.Set(agentSecretsTestScope, "NODE_KEY", "shh-2"))

	response := agentSecretsRequest(t, router, http.MethodGet, "/secrets", "")
	require.Equal(t, http.StatusOK, response.Code)
	require.NotContains(t, response.Body.String(), "shh-")
	require.JSONEq(t, `{"secrets":[
		{"key":"SHARED_KEY","scope":"global"},
		{"key":"NODE_KEY","scope":"test-node"}
	]}`, response.Body.String())
}

// Exercises the global-scope List failure branch after node List succeeds.
func TestAgentSecretsListGlobalScopeFailure(t *testing.T) {
	router, home := newAgentSecretsTestRouter(t)
	store, err := packages.NewSecretStore(home)
	require.NoError(t, err)
	require.NoError(t, store.Set(agentSecretsTestScope, "KEY", "value"))
	require.NoError(t, os.WriteFile(
		filepath.Join(home, "secrets", "global.enc"), []byte("not encrypted"), 0o600))

	response := agentSecretsRequest(t, router, http.MethodGet, "/agents/agent-x/secrets", "")
	require.Equal(t, http.StatusInternalServerError, response.Code)
}
