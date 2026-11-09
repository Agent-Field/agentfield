package services

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/agenturl"
	"github.com/Agent-Field/agentfield/control-plane/internal/nodelink"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
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
	require.Equal(t, exec.RunID, b.Headers["X-Run-ID"], "the node is told the run its root is filed under")
}

// A LINK-ONLY GO SDK NODE (no base_url, hello accepts "http") IS DISPATCHED
// WITH THE HTTP REQUEST ITSELF CARRIED DOWN ITS LINK — never a dispatch
// frame it would refuse — and the node the trigger routed the run to owns it.
func TestTriggerDispatchCarriesHTTPToALinkOnlySDKNode(t *testing.T) {
	node, trig, ev := dispatcherTrigger("", []types.ReasonerDefinition{{ID: "handle"}})
	node.BaseURL = ""
	ctx, provider, d := storeDispatchFixture(t, node, trig, ev)

	gin.SetMode(gin.TestMode)
	hub := &nodelink.Hub{Sweep: 10 * time.Millisecond}
	r := gin.New()
	r.GET("/api/v1/nodes/:node_id/link", hub.Handler())
	cp := httptest.NewServer(r)
	defer cp.Close()
	defer hub.Close()
	agenturl.SetLinkCarrier(hub)
	defer agenturl.SetLinkCarrier(nil)
	d.WithInternalToken("itok").WithLink(hub)
	d.httpClient = agenturl.NewClient(5 * time.Second)

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
	send(nodelink.Frame{T: nodelink.THello}, nodelink.Hello{InstanceID: "i1", Accepts: []string{"http", "steer", "cancel"}})
	require.Equal(t, nodelink.TOK, read().T)
	require.Eventually(t, func() bool { return hub.LiveFor(node.ID, nodelink.KindHTTP) }, 3*time.Second, 5*time.Millisecond)

	done := make(chan struct{})
	go func() { defer close(done); d.DispatchEvent(ctx, trig, ev) }()

	f := read()
	require.Equal(t, nodelink.TDeliver, f.T)
	var b nodelink.DeliverBody
	require.NoError(t, json.Unmarshal(f.Body, &b))
	require.Equal(t, nodelink.KindHTTP, b.Kind)
	require.Equal(t, http.MethodPost, b.Method)
	require.Equal(t, "/reasoners/handle", b.Path)
	require.Equal(t, "Bearer itok", b.Headers["Authorization"])
	execID := b.Headers["X-Execution-ID"]
	require.NotEmpty(t, execID)
	res, _ := json.Marshal(nodelink.HTTPResult{Code: http.StatusAccepted})
	send(nodelink.Frame{T: nodelink.TAck, ID: f.ID}, nodelink.Ack{Outcome: nodelink.OutcomeAccepted, Result: res})
	<-done

	stored, err := provider.GetInboundEvent(ctx, ev.ID)
	require.NoError(t, err)
	require.Equal(t, types.InboundEventStatusDispatched, stored.Status)
	exec, err := provider.GetExecutionRecord(ctx, execID)
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusRunning, exec.Status)
	owners, ok := provider.(interface {
		RunOwner(ctx context.Context, runID string) (string, error)
	})
	require.True(t, ok)
	owner, err := owners.RunOwner(ctx, exec.RunID)
	require.NoError(t, err)
	require.Equal(t, node.ID, owner, "the node a trigger routed the run to owns it")
}

// unownedDispatchStore hides SetRunOwner and RunOwner from TriggerDispatcher so
// a test can exercise the dispatcher when a run has no recorded owner.
type unownedDispatchStore struct {
	storage.StorageProvider
}

// Contract C5: TRIGGER DISPATCH SETS X-AgentField-Run-Watched: 1 ON BOTH ITS
// HTTP AND LINK CARRIERS WHEN THE RUN HAS AN OWNER, AND OMITS IT WHEN NOT.
func TestTriggerDispatchSetsRunWatchedOnlyWhenRunHasOwner(t *testing.T) {
	var mu sync.Mutex
	var lastWatched, lastOwner []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		lastWatched = append([]string(nil), r.Header.Values(nodelink.RunWatchedHeader)...)
		lastOwner = append([]string(nil), r.Header.Values("X-AgentField-Run-Owner")...)
		mu.Unlock()
		w.WriteHeader(http.StatusAccepted)
	}))
	defer srv.Close()

	// 1. HTTP path with LocalStorage: createDispatchRecords sets the target node
	//    as the run's owner, so HTTP carries X-AgentField-Run-Watched: 1.
	node, trig, ev := dispatcherTrigger(srv.URL, []types.ReasonerDefinition{{ID: "handle"}})
	ctx, provider, d := storeDispatchFixture(t, node, trig, ev)
	d.DispatchEvent(ctx, trig, ev)
	mu.Lock()
	require.Equal(t, []string{"1"}, lastWatched)
	require.Empty(t, lastOwner)
	mu.Unlock()

	// 2. HTTP path when the store records no owner: X-AgentField-Run-Watched is absent.
	evUnowned := *ev
	evUnowned.ID = "ev-unowned"
	evUnowned.IdempotencyKey = "idem-unowned"
	require.NoError(t, provider.InsertInboundEvent(ctx, &evUnowned))
	dUnowned := NewTriggerDispatcher(unownedDispatchStore{StorageProvider: d.storage}, nil)
	dUnowned.DispatchEvent(ctx, trig, &evUnowned)
	mu.Lock()
	require.Empty(t, lastWatched)
	require.Empty(t, lastOwner)
	mu.Unlock()

	// 3. KindDispatch link path with owned run: headers carry X-AgentField-Run-Watched: 1.
	evLink := *ev
	evLink.ID = "ev-link"
	evLink.IdempotencyKey = "idem-link"
	require.NoError(t, provider.InsertInboundEvent(ctx, &evLink))

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

	d.DispatchEvent(ctx, trig, &evLink)
	f := read()
	require.Equal(t, nodelink.TDeliver, f.T)
	var b nodelink.DeliverBody
	require.NoError(t, json.Unmarshal(f.Body, &b))
	require.Equal(t, nodelink.KindDispatch, b.Kind)
	require.Equal(t, "1", b.Headers[nodelink.RunWatchedHeader])
	require.Empty(t, b.Headers["X-AgentField-Run-Owner"])
	send(nodelink.Frame{T: nodelink.TAck, ID: f.ID}, nodelink.Ack{Outcome: nodelink.OutcomeAccepted})
}
