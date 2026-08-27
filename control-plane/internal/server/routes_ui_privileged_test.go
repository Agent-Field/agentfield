package server

import (
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/apicatalog"
	"github.com/Agent-Field/agentfield/control-plane/internal/services/packagemaint"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// remotePeer is any address that is not the local host. The control plane
// binds every interface, so this stands in for another machine on the LAN.
const remotePeer = "192.168.2.16:54321"

func newPrivilegedRoutesTestServer(t *testing.T, apiKey string) *AgentFieldServer {
	t.Helper()
	gin.SetMode(gin.TestMode)
	srv := &AgentFieldServer{
		Router:            gin.New(),
		storage:           newStubStorage(),
		payloadStore:      &stubPayloadStore{},
		webhookDispatcher: &stubWebhookDispatcher{},
		apiCatalog:        apicatalog.New(),
		agentfieldHome:    t.TempDir(),
		config: &config.Config{
			UI:  config.UIConfig{Enabled: true},
			API: config.APIConfig{Auth: config.AuthConfig{APIKey: apiKey}},
		},
	}
	srv.packageMaintenance = packagemaint.New(packagemaint.Config{AgentFieldHome: srv.agentfieldHome})
	srv.setupRoutes()
	return srv
}

func callFrom(t *testing.T, srv *AgentFieldServer, method, path, remoteAddr string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader("{}"))
	req.Header.Set("Content-Type", "application/json")
	req.RemoteAddr = remoteAddr
	rec := httptest.NewRecorder()
	srv.Router.ServeHTTP(rec, req)
	return rec
}

// concreteParams replaces gin's :param and *wildcard segments so a registered
// route pattern can be requested.
func concreteParams(routePath string) string {
	segments := strings.Split(routePath, "/")
	for i, seg := range segments {
		if strings.HasPrefix(seg, ":") || strings.HasPrefix(seg, "*") {
			segments[i] = "probe-value"
		}
	}
	return strings.Join(segments, "/")
}

// privilegedPathPattern matches the UI-API surface that installs code or
// handles credentials. Driving the test off the live route table means a new
// route in these areas is covered the moment it is registered.
var privilegedPathPattern = regexp.MustCompile(`^/api/ui/v1/(secrets|agents/(packages/(install|check-updates|maintenance/run)|packages/[^/]+/(uninstall|update|auto-update)|[^/]+/(secrets|env|config)))`)

// A route that matches the privileged pattern but is deliberately public.
func isPrivilegedExempt(routePath string) bool {
	// The config schema only describes which keys an agent accepts; it carries
	// no values, and the UI needs it to render the form before authenticating.
	return strings.HasSuffix(routePath, "/config/schema")
}

func privilegedRoutes(t *testing.T, srv *AgentFieldServer) []gin.RouteInfo {
	t.Helper()
	var found []gin.RouteInfo
	for _, route := range srv.Router.Routes() {
		if privilegedPathPattern.MatchString(route.Path) && !isPrivilegedExempt(route.Path) {
			found = append(found, route)
		}
	}
	return found
}

// Every code-executing or credential-handling UI-API route must reject a
// caller from another host while the control plane runs without an API key.
func TestUIAPI_PrivilegedRoutesRejectRemoteCallersWithoutAPIKey(t *testing.T) {
	srv := newPrivilegedRoutesTestServer(t, "")

	routes := privilegedRoutes(t, srv)
	require.NotEmpty(t, routes, "privileged route pattern matched nothing — did the UI API move?")

	// Guards against the pattern silently matching only a subset after a refactor.
	require.GreaterOrEqual(t, len(routes), 12,
		"expected the full install/secrets/env/config surface, got %d routes", len(routes))

	for _, route := range routes {
		t.Run(route.Method+" "+route.Path, func(t *testing.T) {
			rec := callFrom(t, srv, route.Method, concreteParams(route.Path), remotePeer)
			assert.Equal(t, http.StatusUnauthorized, rec.Code,
				"%s %s must not be reachable from another host without an API key", route.Method, route.Path)
		})
	}
}

// The same routes stay reachable from the local host, which is how the CLI,
// the desktop app and a browser on the same machine use them.
func TestUIAPI_PrivilegedRoutesAllowLoopbackWithoutAPIKey(t *testing.T) {
	srv := newPrivilegedRoutesTestServer(t, "")

	// A privileged route backed only by the temp agentfield home, so the
	// assertion is about the guard rather than the stub storage.
	rec := callFrom(t, srv, http.MethodGet, "/api/ui/v1/secrets", "127.0.0.1:54321")
	assert.NotEqual(t, http.StatusUnauthorized, rec.Code,
		"loopback callers must keep working with no API key configured")

	rec6 := callFrom(t, srv, http.MethodGet, "/api/ui/v1/secrets", "[::1]:54321")
	assert.NotEqual(t, http.StatusUnauthorized, rec6.Code,
		"IPv6 loopback callers must keep working with no API key configured")
}

// With a key configured the routes work from anywhere, provided the caller
// presents it — this is the documented way to manage a remote control plane.
func TestUIAPI_PrivilegedRoutesAcceptRemoteCallerWithAPIKey(t *testing.T) {
	const key = "configured-key"
	srv := newPrivilegedRoutesTestServer(t, key)

	req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/secrets", nil)
	req.RemoteAddr = remotePeer
	req.Header.Set("X-API-Key", key)
	rec := httptest.NewRecorder()
	srv.Router.ServeHTTP(rec, req)
	assert.NotEqual(t, http.StatusUnauthorized, rec.Code)

	// ...and reject it when the key is wrong.
	reqBad := httptest.NewRequest(http.MethodGet, "/api/ui/v1/secrets", nil)
	reqBad.RemoteAddr = remotePeer
	reqBad.Header.Set("X-API-Key", "wrong-key")
	recBad := httptest.NewRecorder()
	srv.Router.ServeHTTP(recBad, reqBad)
	assert.Equal(t, http.StatusUnauthorized, recBad.Code)
}

// The guard is deliberately narrow: read-only observability routes must not
// start requiring a local caller, or remote dashboards break.
func TestUIAPI_ReadOnlyRoutesRemainOpenToRemoteCallers(t *testing.T) {
	srv := newPrivilegedRoutesTestServer(t, "")

	openRoutes := []struct{ method, path string }{
		{http.MethodGet, "/api/ui/v1/agents/packages"},
	}

	for _, r := range openRoutes {
		t.Run(r.method+" "+r.path, func(t *testing.T) {
			rec := callFrom(t, srv, r.method, r.path, remotePeer)
			assert.NotEqual(t, http.StatusUnauthorized, rec.Code,
				"%s %s should not have been locked down", r.method, r.path)
		})
	}
}
