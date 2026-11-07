package services

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/nodelink"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"
)

// THE TRIGGER DISPATCHER SENDS THE INTERNAL TOKEN, as the execute controller
// and the cancel dispatcher do, and a node with no live link is reached over
// HTTP exactly as before even when a link hub exists.
func TestTriggerDispatchSendsBearerAndFallsBackToHTTP(t *testing.T) {
	var auth atomic.Value
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		auth.Store(r.Header.Get("Authorization"))
		w.WriteHeader(http.StatusAccepted)
	}))
	defer srv.Close()
	node, trig, ev := dispatcherTrigger(srv.URL, []types.ReasonerDefinition{{ID: "handle"}})
	ctx, provider, d := storeDispatchFixture(t, node, trig, ev)
	hub := &nodelink.Hub{}
	defer hub.Close()
	d.WithInternalToken("itok").WithLink(hub)

	d.DispatchEvent(ctx, trig, ev)

	require.EqualValues(t, 1, hits.Load(), "no link: the dispatch goes over HTTP")
	require.Equal(t, "Bearer itok", auth.Load())
	stored, err := provider.GetInboundEvent(ctx, ev.ID)
	require.NoError(t, err)
	require.Equal(t, types.InboundEventStatusDispatched, stored.Status)
	require.Equal(t, 0, hub.Pending(node.ID))
}

// A NODE WITH A LIVE LINK IS DISPATCHED DOWN IT, even when its health probe
// says it is unreachable, and its ack settles the execution the way an HTTP
// answer would.
func TestTriggerDispatchGoesDownALiveLink(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusAccepted)
	}))
	defer srv.Close()
	node, trig, ev := dispatcherTrigger(srv.URL, []types.ReasonerDefinition{{ID: "handle"}})
	node.HealthStatus = types.HealthStatusInactive // the probe cannot reach it
	ctx, provider, d := storeDispatchFixture(t, node, trig, ev)

	gin.SetMode(gin.TestMode)
	hub := &nodelink.Hub{Sweep: 10 * time.Millisecond}
	r := gin.New()
	r.GET("/api/v1/nodes/:node_id/link", hub.Handler())
	cp := httptest.NewServer(r)
	defer cp.Close()
	defer hub.Close()
	d.WithLink(hub)

	conn, _, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(cp.URL, "http")+"/api/v1/nodes/"+node.ID+"/link", nil)
	require.NoError(t, err)
	defer conn.Close()
	read := func() nodelink.Frame {
		_ = conn.SetReadDeadline(time.Now().Add(3 * time.Second))
		_, raw, err := conn.ReadMessage()
		require.NoError(t, err)
		f, err := nodelink.Decode(raw)
		require.NoError(t, err)
		return f
	}
	send := func(f nodelink.Frame, body any) {
		raw, err := nodelink.Encode(f, body)
		require.NoError(t, err)
		require.NoError(t, conn.WriteMessage(websocket.TextMessage, raw))
	}
	require.Equal(t, nodelink.TWelcome, read().T)
	send(nodelink.Frame{T: nodelink.THello}, nodelink.Hello{InstanceID: "i1"})
	require.Equal(t, nodelink.TOK, read().T)
	require.Eventually(t, func() bool { return hub.Live(node.ID) }, 3*time.Second, 5*time.Millisecond)

	d.DispatchEvent(ctx, trig, ev)

	f := read()
	require.Equal(t, nodelink.TDeliver, f.T)
	var b nodelink.DeliverBody
	require.NoError(t, json.Unmarshal(f.Body, &b))
	require.Equal(t, nodelink.KindDispatch, b.Kind)
	require.Equal(t, "handle", b.Reasoner)
	require.Equal(t, "idem-dispatch", b.IdempotencyKey)
	execID := b.Headers["X-Execution-ID"]
	require.NotEmpty(t, execID)
	require.Equal(t, f.Exec, execID)
	require.Equal(t, trig.ID, b.Headers["X-Trigger-ID"])
	var input map[string]any
	require.NoError(t, json.Unmarshal(b.Input, &input))
	require.Contains(t, input, "_meta")
	require.EqualValues(t, 0, hits.Load(), "a linked node is never dialled")

	send(nodelink.Frame{T: nodelink.TAck, ID: f.ID}, nodelink.Ack{Outcome: nodelink.OutcomeAccepted})
	require.Eventually(t, func() bool {
		stored, err := provider.GetInboundEvent(ctx, ev.ID)
		return err == nil && stored.Status == types.InboundEventStatusDispatched
	}, 3*time.Second, 10*time.Millisecond)
	exec, err := provider.GetExecutionRecord(ctx, execID)
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusRunning, exec.Status, "accepted is the 202 path")
}
