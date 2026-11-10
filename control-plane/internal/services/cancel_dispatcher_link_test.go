package services

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/agenturl"
	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/nodelink"
	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"
)

func dialTestNodeLink(t *testing.T, wsBase, node, bearer string, accepts []string) *websocket.Conn {
	t.Helper()
	hdr := http.Header{}
	if bearer != "" {
		hdr.Set("Authorization", "Bearer "+bearer)
	}
	conn, _, err := websocket.DefaultDialer.Dial(wsBase+node+"/link", hdr)
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })

	readFrame := func() nodelink.Frame {
		t.Helper()
		_ = conn.SetReadDeadline(time.Now().Add(3 * time.Second))
		_, raw, err := conn.ReadMessage()
		require.NoError(t, err)
		f, err := nodelink.Decode(raw)
		require.NoError(t, err)
		return f
	}

	welcome := readFrame()
	require.Equal(t, nodelink.TWelcome, welcome.T)

	helloRaw, err := nodelink.Encode(nodelink.Frame{T: nodelink.THello, Seq: 1}, nodelink.Hello{
		InstanceID: node + "-inst",
		Version:    "1.0.0",
		Accepts:    accepts,
	})
	require.NoError(t, err)
	require.NoError(t, conn.WriteMessage(websocket.TextMessage, helloRaw))

	for {
		if f := readFrame(); f.T == nodelink.TOK {
			break
		}
	}
	return conn
}

func readDeliverFrame(t *testing.T, conn *websocket.Conn) (nodelink.Frame, nodelink.DeliverBody) {
	t.Helper()
	for {
		_ = conn.SetReadDeadline(time.Now().Add(3 * time.Second))
		_, raw, err := conn.ReadMessage()
		require.NoError(t, err)
		f, err := nodelink.Decode(raw)
		require.NoError(t, err)
		if f.T != nodelink.TDeliver {
			continue
		}
		var body nodelink.DeliverBody
		require.NoError(t, json.Unmarshal(f.Body, &body))
		return f, body
	}
}

// Contract C1: CancelDispatcher sends a cancel down a node's live link when the
// node holds one instead of making a best-effort direct HTTP dial, and carries
// the bearer token and execution id when routing HTTP-over-link.
func TestCancelDispatcherLink_C1_DeliversCancelOverLiveNodeLink(t *testing.T) {
	gin.SetMode(gin.TestMode)

	var directHTTPHits atomic.Int32
	fallbackSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		directHTTPHits.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer fallbackSrv.Close()

	store := newFakeAgentStore()
	store.seed("node-cancel-link", fallbackSrv.URL)
	store.seed("node-http-link", "http://"+agenturl.LinkHost("node-http-link"))

	hub := &nodelink.Hub{
		Agents:        store,
		InternalToken: "internal-cancel-tok",
		Sweep:         10 * time.Millisecond,
		Router: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusOK)
		}),
	}
	agenturl.SetLinkCarrier(hub)
	t.Cleanup(func() {
		agenturl.SetLinkCarrier(nil)
		hub.Close()
	})

	router := gin.New()
	router.GET("/api/v1/nodes/:node_id/link", hub.Handler())
	hubSrv := httptest.NewServer(router)
	defer hubSrv.Close()
	wsBase := "ws" + strings.TrimPrefix(hubSrv.URL, "http") + "/api/v1/nodes/"

	// Connect both a native cancel-frame link peer and an HTTP-over-link peer,
	// authenticating each link upgrade with the internal bearer token.
	cancelPeer := dialTestNodeLink(t, wsBase, "node-cancel-link", "internal-cancel-tok", []string{nodelink.KindCancel})
	httpPeer := dialTestNodeLink(t, wsBase, "node-http-link", "internal-cancel-tok", []string{nodelink.KindHTTP})

	bus := events.NewExecutionEventBus()
	dispatcher := NewCancelDispatcher(store, CancelDispatcherConfig{
		Bus:           bus,
		InternalToken: "internal-cancel-tok",
		Link:          hub,
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	dispatcher.Start(ctx)
	defer dispatcher.Stop()

	require.Eventually(t, func() bool {
		return bus.GetSubscriberCount() >= 1
	}, time.Second, 5*time.Millisecond)

	// 1. A node whose live link accepts KindCancel receives a cancel delivery
	//    frame on its link (lines 190-197) without hitting its HTTP BaseURL.
	bus.Publish(events.ExecutionEvent{
		Type:        events.ExecutionCancelledEvent,
		ExecutionID: "exec-cancel-frame",
		WorkflowID:  "wf-cancel-1",
		AgentNodeID: "node-cancel-link",
		Status:      "cancelled",
		Timestamp:   time.Now().UTC(),
		Data:        map[string]interface{}{"reason": "stopped by owner"},
	})

	frame1, body1 := readDeliverFrame(t, cancelPeer)
	require.Equal(t, "exec-cancel-frame", frame1.Exec)
	require.Equal(t, nodelink.KindCancel, body1.Kind)
	require.Equal(t, "stopped by owner", body1.Reason)
	require.Zero(t, directHTTPHits.Load(), "live cancel link must bypass direct HTTP POST")

	ack1, err := nodelink.Encode(nodelink.Frame{T: nodelink.TAck, ID: frame1.ID, Exec: frame1.Exec}, nodelink.Ack{
		Outcome: nodelink.OutcomeAccepted,
	})
	require.NoError(t, err)
	require.NoError(t, cancelPeer.WriteMessage(websocket.TextMessage, ack1))

	// 2. A link-only node accepting KindHTTP receives the HTTP cancel request
	//    down its live link carrying the Bearer internal token and execution id.
	httpDelivered := make(chan nodelink.DeliverBody, 1)
	httpExecID := make(chan string, 1)
	go func() {
		f, b := readDeliverFrame(t, httpPeer)
		resRaw, _ := json.Marshal(nodelink.HTTPResult{Code: http.StatusOK, Body: []byte(`{}`)})
		ackRaw, _ := nodelink.Encode(nodelink.Frame{T: nodelink.TAck, ID: f.ID, Exec: f.Exec}, nodelink.Ack{
			Outcome: nodelink.OutcomeAccepted,
			Result:  resRaw,
		})
		_ = httpPeer.WriteMessage(websocket.TextMessage, ackRaw)
		httpExecID <- f.Exec
		httpDelivered <- b
	}()

	bus.Publish(events.ExecutionEvent{
		Type:        events.ExecutionCancelledEvent,
		ExecutionID: "exec-cancel-http-link",
		WorkflowID:  "wf-cancel-2",
		AgentNodeID: "node-http-link",
		Status:      "cancelled",
		Timestamp:   time.Now().UTC(),
		Data:        map[string]interface{}{"reason": "deadline exceeded"},
	})

	select {
	case gotBody := <-httpDelivered:
		require.Equal(t, "exec-cancel-http-link", <-httpExecID)
		require.Equal(t, nodelink.KindHTTP, gotBody.Kind)
		require.Equal(t, http.MethodPost, gotBody.Method)
		require.Equal(t, "/_internal/executions/exec-cancel-http-link/cancel", gotBody.Path)
		require.Equal(t, "Bearer internal-cancel-tok", gotBody.Headers["Authorization"])
		require.Equal(t, "exec-cancel-http-link", gotBody.Headers["X-Execution-ID"])
	case <-time.After(3 * time.Second):
		t.Fatal("did not receive HTTP-over-link cancel delivery")
	}
}
