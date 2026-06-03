package client

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

// These tests cover GHSA-jp8j-g39q-qxwx: the SDK control-plane client must not
// replay any credential header it attaches (the static X-API-Key / Authorization
// and the per-request DID signature headers) to a host other than the configured
// baseURL when following a redirect.
//
// Validation contract:
//   - cross-host redirect  -> EVERY credential header MUST NOT reach the new host
//   - same-host redirect   -> credential headers MUST still be sent (no regression)
//   - no redirect          -> credential header MUST be sent (sanity)
//   - non-credential headers (e.g. Accept) MUST be preserved in all cases

// TestStripsCredentialsOnCrossHostRedirect mirrors the reporter's PoC: the
// operator endpoint 302-redirects to a different host, which records the headers
// it receives. The two httptest servers share 127.0.0.1 but differ by port, so
// url.Host (host:port) differs and the redirect is treated as cross-host.
func TestStripsCredentialsOnCrossHostRedirect(t *testing.T) {
	var reached bool
	var gotKey, gotAuth string

	attacker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reached = true
		gotKey = r.Header.Get("X-API-Key")
		gotAuth = r.Header.Get("Authorization")
		_, _ = w.Write([]byte(`{}`))
	}))
	defer attacker.Close()

	operator := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, attacker.URL+r.URL.Path, http.StatusFound)
	}))
	defer operator.Close()

	c, err := New(operator.URL, WithAPIKey("SECRET-AGENTFIELD-KEY"), WithBearerToken("SECRET-BEARER"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, _ = c.GetNode(context.Background(), "node-1")

	if !reached {
		t.Fatal("redirect target was never reached; test setup is wrong")
	}
	if gotKey != "" {
		t.Errorf("X-API-Key leaked to cross-host redirect target: got %q, want empty", gotKey)
	}
	if gotAuth != "" {
		t.Errorf("Authorization leaked to cross-host redirect target: got %q, want empty", gotAuth)
	}
}

// TestKeepsCredentialsOnSameHostRedirect ensures we did not break ordinary
// same-host redirect following: a path-only redirect on the same server must
// still carry the credential to the final handler.
func TestKeepsCredentialsOnSameHostRedirect(t *testing.T) {
	var finalReached bool
	var gotKey string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/final" {
			finalReached = true
			gotKey = r.Header.Get("X-API-Key")
			_, _ = w.Write([]byte(`{}`))
			return
		}
		http.Redirect(w, r, "/final", http.StatusFound)
	}))
	defer srv.Close()

	c, err := New(srv.URL, WithAPIKey("SECRET-AGENTFIELD-KEY"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, _ = c.GetNode(context.Background(), "node-1")

	if !finalReached {
		t.Fatal("same-host redirect target was never reached")
	}
	if gotKey != "SECRET-AGENTFIELD-KEY" {
		t.Errorf("X-API-Key dropped on same-host redirect: got %q, want %q", gotKey, "SECRET-AGENTFIELD-KEY")
	}
}

// TestSendsCredentialsWithoutRedirect is the baseline: without any redirect the
// configured credential reaches the server.
func TestSendsCredentialsWithoutRedirect(t *testing.T) {
	var gotKey string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get("X-API-Key")
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	c, err := New(srv.URL, WithAPIKey("SECRET-AGENTFIELD-KEY"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, _ = c.GetNode(context.Background(), "node-1")

	if gotKey != "SECRET-AGENTFIELD-KEY" {
		t.Errorf("X-API-Key not sent on direct request: got %q, want %q", gotKey, "SECRET-AGENTFIELD-KEY")
	}
}

// TestStripSensitiveHeadersOnRedirectUnit deterministically exercises the
// CheckRedirect hook over the FULL set of credential headers the client
// attaches (static creds + DID signature headers), without needing a live DID
// setup or TLS. It also guards that a non-credential header is preserved.
func TestStripSensitiveHeadersOnRedirectUnit(t *testing.T) {
	mustURL := func(s string) *url.URL {
		u, err := url.Parse(s)
		if err != nil {
			t.Fatalf("parse %q: %v", s, err)
		}
		return u
	}
	credHeaders := func() http.Header {
		h := http.Header{}
		h.Set("Authorization", "Bearer SECRET")
		h.Set("X-API-Key", "SECRET-KEY")
		h.Set(HeaderCallerDID, "did:web:example.com:agents:a")
		h.Set(HeaderDIDSignature, "sig")
		h.Set(HeaderDIDTimestamp, "123")
		h.Set(HeaderDIDNonce, "nonce")
		h.Set("Accept", "application/json") // non-credential: must survive
		return h
	}
	orig := &http.Request{URL: mustURL("https://cp.example.com/api/v1/nodes/n1")}

	t.Run("cross-host strips every credential header", func(t *testing.T) {
		req := &http.Request{
			URL:    mustURL("https://attacker.example.net/api/v1/nodes/n1"),
			Header: credHeaders(),
		}
		if err := stripSensitiveHeadersOnRedirect(req, []*http.Request{orig}); err != nil {
			t.Fatalf("hook returned error: %v", err)
		}
		for _, h := range sensitiveCrossHostHeaders {
			if v := req.Header.Get(h); v != "" {
				t.Errorf("credential header %q leaked cross-host: %q", h, v)
			}
		}
		if got := req.Header.Get("Accept"); got != "application/json" {
			t.Errorf("non-credential header Accept was dropped: %q", got)
		}
	})

	t.Run("same-host preserves every credential header", func(t *testing.T) {
		req := &http.Request{
			URL:    mustURL("https://cp.example.com/v2/nodes/n1"),
			Header: credHeaders(),
		}
		if err := stripSensitiveHeadersOnRedirect(req, []*http.Request{orig}); err != nil {
			t.Fatalf("hook returned error: %v", err)
		}
		for _, h := range sensitiveCrossHostHeaders {
			if req.Header.Get(h) == "" {
				t.Errorf("credential header %q was dropped on same-host redirect", h)
			}
		}
	})

	t.Run("no prior request is a no-op", func(t *testing.T) {
		req := &http.Request{
			URL:    mustURL("https://attacker.example.net/x"),
			Header: credHeaders(),
		}
		if err := stripSensitiveHeadersOnRedirect(req, nil); err != nil {
			t.Fatalf("hook returned error: %v", err)
		}
		if req.Header.Get("X-API-Key") == "" {
			t.Error("headers must be untouched when there is no prior request")
		}
	})
}
