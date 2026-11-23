package agent

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"
)

func TestRunOwnerProofContractB7(t *testing.T) {
	const expiry = "2026-05-01T12:00:00Z"
	type delegateRequest struct {
		Scopes     []string `json:"scopes"`
		TTLSeconds int      `json:"ttl_seconds"`
		Run        string   `json:"run"`
	}
	requests := make(chan delegateRequest, 4)
	hello := make(chan struct{})
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Error(err)
			return
		}
		defer conn.Close()
		for {
			var f linkFrame
			if err := conn.ReadJSON(&f); err != nil {
				return
			}
			if f.T == "hello" {
				close(hello)
				continue
			}
			if f.T != "delegate" || f.Exec != "" {
				t.Errorf("unexpected frame %s for %q", f.T, f.Exec)
				return
			}
			var req delegateRequest
			if err := json.Unmarshal(f.Body, &req); err != nil {
				t.Error(err)
				return
			}
			requests <- req
			ok := linkOK{Code: http.StatusOK, Token: "example-proof", ExpiresAt: expiry, ExpiresIn: 1800,
				Scopes: []string{"own", "read"}}
			switch {
			case len(req.Scopes) == 0:
				ok = linkOK{Code: http.StatusBadRequest, Error: "delegate requires at least one scope"}
			case req.Scopes[0] == "act":
				continue
			}
			raw, _ := json.Marshal(ok)
			if err := conn.WriteJSON(linkFrame{V: 1, T: "ok", Seq: f.Seq, Body: raw}); err != nil {
				t.Error(err)
				return
			}
		}
	}))
	defer server.Close()
	a, err := New(Config{NodeID: "my-app", Version: "1", AgentFieldURL: server.URL, Logger: log.New(io.Discard, "", 0)})
	require.NoError(t, err)
	proof, err := a.RunOwnerProof(context.Background(), RunOwnerProofRequest{Scopes: []string{"own"}})
	require.ErrorIs(t, err, errNoLink)
	require.ErrorContains(t, err, "no node link is up")
	require.Nil(t, proof)

	done := make(chan struct{})
	target, ok := a.linkURL()
	require.True(t, ok)
	go func() {
		defer close(done)
		a.linkOnce(target)
	}()
	defer func() {
		close(a.stopLease)
		<-done
	}()
	select {
	case <-hello:
	case <-time.After(5 * time.Second):
		t.Fatal("node did not send hello")
	}
	require.Eventually(t, func() bool {
		a.link.mu.Lock()
		defer a.link.mu.Unlock()
		return a.link.conn != nil
	}, time.Second, 5*time.Millisecond)
	takeRequest := func() delegateRequest {
		t.Helper()
		select {
		case req := <-requests:
			return req
		case <-time.After(5 * time.Second):
			t.Fatal("node did not send delegate")
			return delegateRequest{}
		}
	}
	ctx, timeout := context.WithTimeout(context.Background(), 5*time.Second)
	defer timeout()
	proof, err = a.RunOwnerProof(ctx, RunOwnerProofRequest{Scopes: []string{"read", "own"}, TTL: 30 * time.Minute, Run: "my-run"})
	require.NoError(t, err)
	require.Equal(t, delegateRequest{Scopes: []string{"read", "own"}, TTLSeconds: 1800, Run: "my-run"}, takeRequest())
	require.Equal(t, "example-proof", proof.Token)
	require.Equal(t, time.Date(2026, time.May, 1, 12, 0, 0, 0, time.UTC), proof.ExpiresAt)
	require.Equal(t, []string{"own", "read"}, proof.Scopes)

	proof, err = a.RunOwnerProof(ctx, RunOwnerProofRequest{})
	require.Nil(t, proof)
	require.ErrorContains(t, err, "delegate requires at least one scope")
	require.Empty(t, takeRequest().Scopes)

	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	proof, err = a.RunOwnerProof(cancelled, RunOwnerProofRequest{Scopes: []string{"act"}})
	require.Nil(t, proof)
	require.ErrorIs(t, err, context.Canceled)
	require.Empty(t, requests)

	waitCtx, cancelWait := context.WithCancel(ctx)
	defer cancelWait()
	failure := make(chan error, 1)
	go func() {
		_, err := a.RunOwnerProof(waitCtx, RunOwnerProofRequest{Scopes: []string{"act"}})
		failure <- err
	}()
	require.Equal(t, []string{"act"}, takeRequest().Scopes)
	cancelWait()
	select {
	case err := <-failure:
		require.ErrorIs(t, err, context.Canceled)
	case <-time.After(time.Second):
		t.Fatal("cancelled proof request kept waiting")
	}
}
