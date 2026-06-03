package middleware

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

// mockDIDService implements DIDWebServiceInterface for testing.
type mockDIDService struct {
	verifyFunc func(ctx context.Context, did string, message []byte, signature []byte) (bool, error)
}

func (m *mockDIDService) VerifyDIDOwnership(ctx context.Context, did string, message []byte, signature []byte) (bool, error) {
	if m.verifyFunc != nil {
		return m.verifyFunc(ctx, did, message, signature)
	}
	return true, nil
}

// resetReplayCache resets the global replay cache between tests.
func resetReplayCache() {
	globalReplayCacheOnce = sync.Once{}
	globalReplayCache = nil
}

// sigCounter generates unique signature bytes per call to avoid replay collisions.
var sigCounter atomic.Int64

func uniqueSig() string {
	n := sigCounter.Add(1)
	return base64.StdEncoding.EncodeToString([]byte(fmt.Sprintf("sig-%d", n)))
}

func newDIDAuthRouter(service DIDWebServiceInterface, config DIDAuthConfig) *gin.Engine {
	router := gin.New()
	router.Use(DIDAuthMiddleware(service, config))
	router.POST("/execute/:target", func(c *gin.Context) {
		did := GetVerifiedCallerDID(c)
		skipped := IsDIDAuthSkipped(c)
		c.JSON(http.StatusOK, gin.H{"did": did, "skipped": skipped})
	})
	return router
}

func TestDIDAuth_Disabled(t *testing.T) {
	resetReplayCache()
	router := newDIDAuthRouter(&mockDIDService{}, DIDAuthConfig{Enabled: false})

	req := httptest.NewRequest(http.MethodPost, "/execute/agent.func", strings.NewReader("{}"))
	req.Header.Set("X-Caller-DID", "did:web:example.com:agents:test")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestDIDAuth_NoDIDHeader_Skipped(t *testing.T) {
	resetReplayCache()
	router := newDIDAuthRouter(&mockDIDService{}, DIDAuthConfig{Enabled: true})

	req := httptest.NewRequest(http.MethodPost, "/execute/agent.func", strings.NewReader("{}"))
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), `"skipped":true`)
}

func TestDIDAuth_InvalidDIDFormat(t *testing.T) {
	resetReplayCache()
	router := newDIDAuthRouter(&mockDIDService{}, DIDAuthConfig{Enabled: true})

	tests := []struct {
		name string
		did  string
	}{
		{"no did: prefix", "web:example.com"},
		{"too short", "did:x"},
		{"too long", "did:" + strings.Repeat("x", 520)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/execute/agent.func", strings.NewReader("{}"))
			req.Header.Set("X-Caller-DID", tt.did)
			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)
			assert.Equal(t, http.StatusBadRequest, w.Code)
			assert.Contains(t, w.Body.String(), "invalid_did_format")
		})
	}
}

func TestDIDAuth_MissingSignature(t *testing.T) {
	resetReplayCache()
	router := newDIDAuthRouter(&mockDIDService{}, DIDAuthConfig{Enabled: true})

	req := httptest.NewRequest(http.MethodPost, "/execute/agent.func", strings.NewReader("{}"))
	req.Header.Set("X-Caller-DID", "did:web:example.com:agents:test")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Contains(t, w.Body.String(), "did_auth_required")
}

func TestDIDAuth_InvalidTimestamp(t *testing.T) {
	resetReplayCache()
	router := newDIDAuthRouter(&mockDIDService{}, DIDAuthConfig{Enabled: true})

	req := httptest.NewRequest(http.MethodPost, "/execute/agent.func", strings.NewReader("{}"))
	req.Header.Set("X-Caller-DID", "did:web:example.com:agents:test")
	req.Header.Set("X-DID-Signature", uniqueSig())
	req.Header.Set("X-DID-Timestamp", "not-a-number")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Contains(t, w.Body.String(), "invalid_timestamp")
}

func TestDIDAuth_ExpiredTimestamp(t *testing.T) {
	resetReplayCache()
	router := newDIDAuthRouter(&mockDIDService{}, DIDAuthConfig{
		Enabled:                true,
		TimestampWindowSeconds: 300,
	})

	req := httptest.NewRequest(http.MethodPost, "/execute/agent.func", strings.NewReader("{}"))
	req.Header.Set("X-Caller-DID", "did:web:example.com:agents:test")
	req.Header.Set("X-DID-Signature", uniqueSig())
	req.Header.Set("X-DID-Timestamp", fmt.Sprintf("%d", time.Now().Unix()-600))
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Contains(t, w.Body.String(), "timestamp_expired")
}

func TestDIDAuth_FutureTimestamp(t *testing.T) {
	resetReplayCache()
	router := newDIDAuthRouter(&mockDIDService{}, DIDAuthConfig{
		Enabled:                true,
		TimestampWindowSeconds: 300,
	})

	req := httptest.NewRequest(http.MethodPost, "/execute/agent.func", strings.NewReader("{}"))
	req.Header.Set("X-Caller-DID", "did:web:example.com:agents:test")
	req.Header.Set("X-DID-Signature", uniqueSig())
	req.Header.Set("X-DID-Timestamp", fmt.Sprintf("%d", time.Now().Unix()+600))
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Contains(t, w.Body.String(), "timestamp_expired")
}

func TestDIDAuth_InvalidBase64Signature(t *testing.T) {
	resetReplayCache()
	router := newDIDAuthRouter(&mockDIDService{}, DIDAuthConfig{Enabled: true})

	req := httptest.NewRequest(http.MethodPost, "/execute/agent.func", strings.NewReader("{}"))
	req.Header.Set("X-Caller-DID", "did:web:example.com:agents:test")
	req.Header.Set("X-DID-Signature", "not-valid-base64!!!")
	req.Header.Set("X-DID-Timestamp", fmt.Sprintf("%d", time.Now().Unix()))
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Contains(t, w.Body.String(), "invalid_signature_encoding")
}

func TestDIDAuth_VerificationError(t *testing.T) {
	resetReplayCache()
	svc := &mockDIDService{
		verifyFunc: func(_ context.Context, _ string, _ []byte, _ []byte) (bool, error) {
			return false, fmt.Errorf("DID document not found")
		},
	}
	router := newDIDAuthRouter(svc, DIDAuthConfig{Enabled: true})

	req := httptest.NewRequest(http.MethodPost, "/execute/agent.func", strings.NewReader("{}"))
	req.Header.Set("X-Caller-DID", "did:web:example.com:agents:test")
	req.Header.Set("X-DID-Signature", uniqueSig())
	req.Header.Set("X-DID-Timestamp", fmt.Sprintf("%d", time.Now().Unix()))
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Contains(t, w.Body.String(), "verification_error")
}

func TestDIDAuth_InvalidSignature(t *testing.T) {
	resetReplayCache()
	svc := &mockDIDService{
		verifyFunc: func(_ context.Context, _ string, _ []byte, _ []byte) (bool, error) {
			return false, nil
		},
	}
	router := newDIDAuthRouter(svc, DIDAuthConfig{Enabled: true})

	req := httptest.NewRequest(http.MethodPost, "/execute/agent.func", strings.NewReader("{}"))
	req.Header.Set("X-Caller-DID", "did:web:example.com:agents:test")
	req.Header.Set("X-DID-Signature", uniqueSig())
	req.Header.Set("X-DID-Timestamp", fmt.Sprintf("%d", time.Now().Unix()))
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Contains(t, w.Body.String(), "invalid_signature")
}

func TestDIDAuth_ValidSignature(t *testing.T) {
	resetReplayCache()
	svc := &mockDIDService{
		verifyFunc: func(_ context.Context, _ string, _ []byte, _ []byte) (bool, error) {
			return true, nil
		},
	}
	router := newDIDAuthRouter(svc, DIDAuthConfig{Enabled: true})

	req := httptest.NewRequest(http.MethodPost, "/execute/agent.func", strings.NewReader("{}"))
	req.Header.Set("X-Caller-DID", "did:web:example.com:agents:test")
	req.Header.Set("X-DID-Signature", uniqueSig())
	req.Header.Set("X-DID-Timestamp", fmt.Sprintf("%d", time.Now().Unix()))
	req.Header.Set("X-DID-Nonce", fmt.Sprintf("%d", time.Now().UnixNano()))
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "did:web:example.com:agents:test")
}

func TestDIDAuth_ReplayDetection(t *testing.T) {
	resetReplayCache()
	svc := &mockDIDService{
		verifyFunc: func(_ context.Context, _ string, _ []byte, _ []byte) (bool, error) {
			return true, nil
		},
	}
	router := newDIDAuthRouter(svc, DIDAuthConfig{Enabled: true})

	sig := uniqueSig()
	ts := fmt.Sprintf("%d", time.Now().Unix())

	// First request should succeed
	req1 := httptest.NewRequest(http.MethodPost, "/execute/agent.func", strings.NewReader("{}"))
	req1.Header.Set("X-Caller-DID", "did:web:example.com:agents:test")
	req1.Header.Set("X-DID-Signature", sig)
	req1.Header.Set("X-DID-Timestamp", ts)
	w1 := httptest.NewRecorder()
	router.ServeHTTP(w1, req1)
	assert.Equal(t, http.StatusOK, w1.Code)

	// Same signature should be rejected
	req2 := httptest.NewRequest(http.MethodPost, "/execute/agent.func", strings.NewReader("{}"))
	req2.Header.Set("X-Caller-DID", "did:web:example.com:agents:test")
	req2.Header.Set("X-DID-Signature", sig)
	req2.Header.Set("X-DID-Timestamp", ts)
	w2 := httptest.NewRecorder()
	router.ServeHTTP(w2, req2)
	assert.Equal(t, http.StatusUnauthorized, w2.Code)
	assert.Contains(t, w2.Body.String(), "replay_detected")
}

func TestDIDAuth_BodyTooLarge(t *testing.T) {
	resetReplayCache()
	router := newDIDAuthRouter(&mockDIDService{}, DIDAuthConfig{Enabled: true})

	largeBody := strings.Repeat("x", maxBodySize+1)
	req := httptest.NewRequest(http.MethodPost, "/execute/agent.func", strings.NewReader(largeBody))
	req.Header.Set("X-Caller-DID", "did:web:example.com:agents:test")
	req.Header.Set("X-DID-Signature", uniqueSig())
	req.Header.Set("X-DID-Timestamp", fmt.Sprintf("%d", time.Now().Unix()))
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusRequestEntityTooLarge, w.Code)
	assert.Contains(t, w.Body.String(), "body_too_large")
}

func TestDIDAuth_SkipPath(t *testing.T) {
	resetReplayCache()
	router := gin.New()
	router.Use(DIDAuthMiddleware(&mockDIDService{}, DIDAuthConfig{
		Enabled:   true,
		SkipPaths: []string{"/health"},
	}))
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"skipped": IsDIDAuthSkipped(c)})
	})

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	req.Header.Set("X-Caller-DID", "did:web:example.com:agents:test")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), `"skipped":true`)
}

func TestDIDAuth_PayloadFormat_WithNonce(t *testing.T) {
	resetReplayCache()
	var capturedMessage []byte
	svc := &mockDIDService{
		verifyFunc: func(_ context.Context, _ string, message []byte, _ []byte) (bool, error) {
			capturedMessage = message
			return true, nil
		},
	}
	router := newDIDAuthRouter(svc, DIDAuthConfig{Enabled: true})

	body := `{"input":"test"}`
	ts := fmt.Sprintf("%d", time.Now().Unix())
	nonce := "test-nonce-123"
	bodyHash := sha256.Sum256([]byte(body))
	expectedPayload := fmt.Sprintf("%s:%s:%x", ts, nonce, bodyHash)

	req := httptest.NewRequest(http.MethodPost, "/execute/agent.func", strings.NewReader(body))
	req.Header.Set("X-Caller-DID", "did:web:example.com:agents:test")
	req.Header.Set("X-DID-Signature", uniqueSig())
	req.Header.Set("X-DID-Timestamp", ts)
	req.Header.Set("X-DID-Nonce", nonce)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, expectedPayload, string(capturedMessage))
}

func TestDIDAuth_PayloadFormat_WithoutNonce(t *testing.T) {
	resetReplayCache()
	var capturedMessage []byte
	svc := &mockDIDService{
		verifyFunc: func(_ context.Context, _ string, message []byte, _ []byte) (bool, error) {
			capturedMessage = message
			return true, nil
		},
	}
	router := newDIDAuthRouter(svc, DIDAuthConfig{Enabled: true})

	body := `{"input":"test"}`
	ts := fmt.Sprintf("%d", time.Now().Unix())
	bodyHash := sha256.Sum256([]byte(body))
	expectedPayload := fmt.Sprintf("%s:%x", ts, bodyHash)

	req := httptest.NewRequest(http.MethodPost, "/execute/agent.func", strings.NewReader(body))
	req.Header.Set("X-Caller-DID", "did:web:example.com:agents:test")
	req.Header.Set("X-DID-Signature", uniqueSig())
	req.Header.Set("X-DID-Timestamp", ts)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, expectedPayload, string(capturedMessage))
}
