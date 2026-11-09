package nodelink

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"
)

type agents map[string]bool

func (a agents) GetAgent(_ context.Context, id string) (*types.AgentNode, error) {
	if a[id] {
		return &types.AgentNode{ID: id}, nil
	}
	return nil, nil
}

func testHub(t *testing.T, h *Hub) (*Hub, string) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	if h.Agents == nil {
		h.Agents = agents{"n1": true}
	}
	if h.Sweep == 0 {
		h.Sweep = 10 * time.Millisecond
	}
	r := gin.New()
	r.GET("/api/v1/nodes/:node_id/link", h.Handler())
	srv := httptest.NewServer(r)
	t.Cleanup(func() { h.Close(); srv.Close() })
	return h, "ws" + strings.TrimPrefix(srv.URL, "http") + "/api/v1/nodes/"
}

type client struct {
	t    *testing.T
	conn *websocket.Conn
}

func dial(t *testing.T, base, node string, hdr http.Header) (*client, *http.Response, error) {
	t.Helper()
	conn, resp, err := websocket.DefaultDialer.Dial(base+node+"/link", hdr)
	if err != nil {
		return nil, resp, err
	}
	c := &client{t: t, conn: conn}
	t.Cleanup(func() { _ = conn.Close() })
	f := c.read()
	require.Equal(t, TWelcome, f.T)
	c.send(Frame{T: THello}, Hello{InstanceID: "i1", Version: "1"})
	require.Equal(t, TOK, c.readType(TOK).T)
	return c, resp, nil
}

func (c *client) send(f Frame, body any) {
	raw, err := Encode(f, body)
	require.NoError(c.t, err)
	require.NoError(c.t, c.conn.WriteMessage(websocket.TextMessage, raw))
}

func (c *client) read() Frame {
	_ = c.conn.SetReadDeadline(time.Now().Add(3 * time.Second))
	_, raw, err := c.conn.ReadMessage()
	require.NoError(c.t, err)
	f, err := Decode(raw)
	require.NoError(c.t, err)
	return f
}

func (c *client) readType(t string) Frame {
	for {
		if f := c.read(); f.T == t {
			return f
		}
	}
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func TestFrameCodecRoundTrip(t *testing.T) {
	raw, err := Encode(Frame{T: TDeliver, ID: "dlv_1", Exec: "exec_1"}, DeliverBody{Kind: KindDispatch, Reasoner: "intake", Input: json.RawMessage(`{"event":{}}`), Headers: map[string]string{"X-Execution-ID": "exec_1"}, IdempotencyKey: "k1"})
	require.NoError(t, err)
	f, err := Decode(raw)
	require.NoError(t, err)
	require.Equal(t, Version, f.V)
	require.Equal(t, "dlv_1", f.ID)
	require.Equal(t, "exec_1", f.Exec)
	var b DeliverBody
	require.NoError(t, json.Unmarshal(f.Body, &b))
	require.Equal(t, "intake", b.Reasoner)
	require.Equal(t, "exec_1", b.Headers["X-Execution-ID"])
	require.Equal(t, "k1", b.IdempotencyKey)
	require.JSONEq(t, `{"event":{}}`, string(b.Input))
}

func TestFrameCodecRefuses(t *testing.T) {
	_, err := Decode([]byte(`{"v":2,"t":"ack"}`))
	require.ErrorContains(t, err, "version")
	_, err = Decode([]byte(`{"v":1,"t":"teleport"}`))
	require.ErrorContains(t, err, "unknown frame type")
	_, err = Decode([]byte(`not json`))
	require.Error(t, err)
	_, err = Decode(make([]byte, MaxFrame+1))
	require.ErrorContains(t, err, "1 MiB")
	_, err = Encode(Frame{T: "teleport"}, nil)
	require.Error(t, err)
	_, err = Encode(Frame{T: TLog}, map[string]string{"x": strings.Repeat("a", MaxFrame)})
	require.ErrorContains(t, err, "1 MiB")
}

func TestLinkAuthRefusals(t *testing.T) {
	_, base := testHub(t, &Hub{APIKey: "key", InternalToken: "itok"})
	for _, tc := range []struct {
		name string
		node string
		hdr  http.Header
		code int
	}{
		{"no credential", "n1", http.Header{}, http.StatusUnauthorized},
		{"wrong key", "n1", http.Header{"Authorization": {"Bearer nope"}}, http.StatusUnauthorized},
		{"browser origin", "n1", http.Header{"Authorization": {"Bearer key"}, "Origin": {"https://evil.example"}}, http.StatusForbidden},
		{"unregistered node", "ghost", http.Header{"Authorization": {"Bearer key"}}, http.StatusNotFound},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, resp, err := websocket.DefaultDialer.Dial(base+tc.node+"/link", tc.hdr)
			require.Error(t, err)
			require.NotNil(t, resp)
			require.Equal(t, tc.code, resp.StatusCode)
		})
	}
	// THE KEY IS NEVER TAKEN FROM THE QUERY STRING.
	_, resp, err := websocket.DefaultDialer.Dial(base+"n1/link?api_key=key", http.Header{})
	require.Error(t, err)
	require.Equal(t, http.StatusUnauthorized, resp.StatusCode)

	_, _, err = dial(t, base, "n1", http.Header{"Authorization": {"Bearer key"}})
	require.NoError(t, err, "the API key links")
	_, _, err = dial(t, base, "n1", http.Header{"Authorization": {"Bearer itok"}})
	require.NoError(t, err, "the internal token links")
}

func TestDeliveryWaitsForALinkAndIsRedeliveredUntilAcked(t *testing.T) {
	h, base := testHub(t, &Hub{RedeliverAfter: 80 * time.Millisecond})
	var settled atomic.Int32
	var got Outcome
	var mu sync.Mutex
	id := h.Deliver("n1", Delivery{Kind: KindDispatch, ExecutionID: "exec_1", Dispatch: &Dispatch{Reasoner: "intake"}, OnSettle: func(o Outcome) {
		settled.Add(1)
		mu.Lock()
		got = o
		mu.Unlock()
	}})
	require.Equal(t, 1, h.Pending("n1"), "no link yet: it waits")

	c, _, err := dial(t, base, "n1", nil)
	require.NoError(t, err)
	first := c.readType(TDeliver)
	require.Equal(t, id, first.ID)
	// Not acked: it comes again.
	again := c.readType(TDeliver)
	require.Equal(t, id, again.ID, "an unacked delivery is redelivered")

	// ACK DEDUPE: two acks of one delivery settle it once.
	c.send(Frame{T: TAck, ID: id}, Ack{Outcome: OutcomeAccepted})
	c.send(Frame{T: TAck, ID: id}, Ack{Outcome: OutcomeDone})
	waitFor(t, "the ack to settle", func() bool { return settled.Load() == 1 })
	time.Sleep(50 * time.Millisecond)
	require.EqualValues(t, 1, settled.Load(), "a second ack changes nothing")
	mu.Lock()
	require.Equal(t, OutcomeAccepted, got.Outcome)
	mu.Unlock()
	require.Equal(t, 0, h.Pending("n1"))
}

func TestReconnectRedeliversWhatWasNotAcked(t *testing.T) {
	h, base := testHub(t, &Hub{RedeliverAfter: time.Hour})
	done := make(chan Outcome, 1)
	c, _, err := dial(t, base, "n1", nil)
	require.NoError(t, err)
	id := h.Deliver("n1", Delivery{Kind: KindCancel, ExecutionID: "exec_9", OnSettle: func(o Outcome) { done <- o }})
	require.Equal(t, id, c.readType(TDeliver).ID)
	// The link drops before the node acks.
	_ = c.conn.Close()
	waitFor(t, "the link to drop", func() bool { return !h.Live("n1") })

	c2, _, err := dial(t, base, "n1", nil)
	require.NoError(t, err)
	f := c2.readType(TDeliver)
	require.Equal(t, id, f.ID, "a reconnect gets the unacked delivery again")
	var b DeliverBody
	require.NoError(t, json.Unmarshal(f.Body, &b))
	require.Equal(t, KindCancel, b.Kind)
	c2.send(Frame{T: TAck, ID: id}, Ack{Outcome: OutcomeAccepted})
	select {
	case o := <-done:
		require.Equal(t, OutcomeAccepted, o.Outcome)
	case <-time.After(3 * time.Second):
		t.Fatal("not settled")
	}
}

func TestUndeliveredExpires(t *testing.T) {
	h, _ := testHub(t, &Hub{})
	done := make(chan Outcome, 1)
	h.Deliver("n1", Delivery{Kind: KindDispatch, TTL: 30 * time.Millisecond, OnSettle: func(o Outcome) { done <- o }})
	select {
	case o := <-done:
		require.Equal(t, OutcomeExpired, o.Outcome)
	case <-time.After(3 * time.Second):
		t.Fatal("did not expire")
	}
}

func TestUpFramesReenterTheHTTPHandlers(t *testing.T) {
	gin.SetMode(gin.TestMode)
	type call struct{ method, path, key, body string }
	calls := make(chan call, 10)
	router := gin.New()
	record := func(c *gin.Context) {
		raw, _ := io.ReadAll(c.Request.Body)
		calls <- call{c.Request.Method, c.Request.URL.Path, c.GetHeader("X-API-Key"), string(raw)}
		if strings.HasSuffix(c.Request.URL.Path, "/exec_ended/status") {
			c.JSON(http.StatusConflict, gin.H{"error": "terminal state"})
			return
		}
		c.JSON(http.StatusOK, gin.H{})
	}
	router.POST("/api/v1/executions/:id/status", record)
	router.POST("/api/v1/executions/:id/logs", record)
	router.PATCH("/api/v1/nodes/:id/status", record)
	_, base := testHub(t, &Hub{Router: router, APIKey: "key"})

	c, _, err := dial(t, base, "n1", http.Header{"X-API-Key": {"key"}})
	require.NoError(t, err)
	hello := <-calls
	require.Equal(t, "PATCH", hello.method, "hello renews the lease")
	require.Equal(t, "key", hello.key)

	c.send(Frame{T: TStatus, Exec: "exec_1", Seq: 1}, map[string]any{"status": "succeeded", "result": map[string]any{"pr_url": "https://x/pull/1"}})
	got := <-calls
	require.Equal(t, "/api/v1/executions/exec_1/status", got.path)
	require.JSONEq(t, `{"status":"succeeded","result":{"pr_url":"https://x/pull/1"}}`, got.body)
	ok := c.readType(TOK)
	require.EqualValues(t, 1, ok.Seq)

	c.send(Frame{T: TProgress, Exec: "exec_1", Seq: 2}, map[string]any{"progress": 33})
	got = <-calls
	require.JSONEq(t, `{"status":"running","progress":33}`, got.body, "progress is a non-terminal status")
	c.readType(TOK)

	c.send(Frame{T: TLog, Exec: "exec_1", Seq: 3}, map[string]any{"entries": []any{}})
	require.Equal(t, "/api/v1/executions/exec_1/logs", (<-calls).path)
	c.readType(TOK)

	c.send(Frame{T: THeartbeat, Seq: 4}, Heartbeat{Inflight: []string{"exec_1"}})
	hb := <-calls
	require.Equal(t, "/api/v1/nodes/n1/status", hb.path)
	require.JSONEq(t, `{"phase":"ready","version":"1","inflight":["exec_1"]}`, hb.body)
	c.readType(TOK)

	// The node reads the re-entered handler's answer as it would read HTTP.
	c.send(Frame{T: TStatus, Exec: "exec_ended", Seq: 5}, map[string]any{"status": "succeeded"})
	<-calls
	ans := c.readType(TOK)
	var body OKBody
	require.NoError(t, json.Unmarshal(ans.Body, &body))
	require.Equal(t, http.StatusConflict, body.Code)
	require.Contains(t, body.Error, "terminal state")
}
