package middleware

import (
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// C8: a verified DID is authoritative even when it does not resolve.
func TestResolveCallerIdentity(t *testing.T) {
	tests := []struct {
		name              string
		verifiedCallerDID string
		didResolver       DIDResolverInterface
		headers           map[string]string
		want              string
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
			name:              "unresolved verified did stays anonymous",
			verifiedCallerDID: "did:web:example.com:agents:caller",
			didResolver:       &permissionDIDResolverStub{dids: map[string]string{}},
			headers: map[string]string{
				"X-Caller-Agent-ID": "caller-from-header",
			},
			want: "",
		},
		{
			name:              "nil resolver with verified did stays anonymous",
			verifiedCallerDID: "did:web:example.com:agents:caller",
			headers: map[string]string{
				"X-Agent-Node-ID": "caller-from-node",
			},
			want: "",
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
