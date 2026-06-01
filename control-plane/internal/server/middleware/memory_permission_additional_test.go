package middleware

import (
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestResolveCallerIdentity(t *testing.T) {
	tests := []struct {
		name             string
		verifiedCallerDID string
		didResolver      DIDResolverInterface
		headers          map[string]string
		want             string
	}{
		{
			name:              "verified did takes precedence",
			verifiedCallerDID: "did:web:example.com:agents:caller",
			didResolver: &permissionDIDResolverStub{
				dids: map[string]string{"did:web:example.com:agents:caller": "caller-from-did"},
			},
			headers: map[string]string{
				"X-Caller-Agent-ID": "caller-from-header",
				"X-Agent-Node-ID":   "caller-from-node",
			},
			want: "caller-from-did",
		},
		{
			name:        "header caller agent fallback",
			didResolver: &permissionDIDResolverStub{dids: map[string]string{}},
			headers: map[string]string{
				"X-Caller-Agent-ID": "caller-from-header",
				"X-Agent-Node-ID":   "caller-from-node",
			},
			want: "caller-from-header",
		},
		{
			name:        "node id fallback",
			didResolver: &permissionDIDResolverStub{dids: map[string]string{}},
			headers: map[string]string{
				"X-Agent-Node-ID": "caller-from-node",
			},
			want: "caller-from-node",
		},
		{
			name:              "unresolved did falls back to headers",
			verifiedCallerDID: "did:web:example.com:agents:caller",
			didResolver:       &permissionDIDResolverStub{dids: map[string]string{}},
			headers: map[string]string{
				"X-Caller-Agent-ID": "caller-from-header",
			},
			want: "caller-from-header",
		},
		{
			name:              "nil resolver still falls back to headers",
			verifiedCallerDID: "did:web:example.com:agents:caller",
			headers: map[string]string{
				"X-Agent-Node-ID": "caller-from-node",
			},
			want: "caller-from-node",
		},
		{
			name:        "no identity present",
			didResolver: &permissionDIDResolverStub{dids: map[string]string{}},
			want:        "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gin.SetMode(gin.TestMode)
			c, _ := gin.CreateTestContext(httptest.NewRecorder())
			c.Request = httptest.NewRequest("GET", "/", nil)
			if tt.verifiedCallerDID != "" {
				c.Set(string(VerifiedCallerDIDKey), tt.verifiedCallerDID)
			}
			for k, v := range tt.headers {
				c.Request.Header.Set(k, v)
			}

			require.Equal(t, tt.want, resolveCallerIdentity(c, tt.didResolver))
		})
	}
}

func TestValidateScopeOwnership(t *testing.T) {
	tests := []struct {
		name         string
		scope        string
		callerAgentID string
		headers      map[string]string
		want         bool
	}{
		{
			name:          "actor scope owned by caller",
			scope:         "actor",
			callerAgentID: "agent-a",
			headers:       map[string]string{"X-Actor-ID": "agent-a"},
			want:          true,
		},
		{
			name:          "actor scope without actor id allowed",
			scope:         "actor",
			callerAgentID: "agent-a",
			want:          true,
		},
		{
			name:          "actor scope mismatched caller denied",
			scope:         "actor",
			callerAgentID: "agent-a",
			headers:       map[string]string{"X-Actor-ID": "agent-b"},
			want:          false,
		},
		{
			name:          "session scope requires session id",
			scope:         "session",
			callerAgentID: "agent-a",
			headers:       map[string]string{"X-Session-ID": "session-1"},
			want:          true,
		},
		{
			name:          "session scope missing session id denied",
			scope:         "session",
			callerAgentID: "agent-a",
			want:          false,
		},
		{
			name:          "workflow scope requires workflow id",
			scope:         "workflow",
			callerAgentID: "agent-a",
			headers:       map[string]string{"X-Workflow-ID": "wf-1"},
			want:          true,
		},
		{
			name:          "workflow scope missing workflow id denied",
			scope:         "workflow",
			callerAgentID: "agent-a",
			want:          false,
		},
		{
			name:          "global scope always allowed",
			scope:         "global",
			callerAgentID: "agent-a",
			want:          true,
		},
		{
			name:          "unknown scope denied",
			scope:         "tenant",
			callerAgentID: "agent-a",
			want:          false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gin.SetMode(gin.TestMode)
			c, _ := gin.CreateTestContext(httptest.NewRecorder())
			c.Request = httptest.NewRequest("GET", "/", nil)
			for k, v := range tt.headers {
				c.Request.Header.Set(k, v)
			}

			require.Equal(t, tt.want, validateScopeOwnership(c, tt.callerAgentID, tt.scope))
		})
	}
}
