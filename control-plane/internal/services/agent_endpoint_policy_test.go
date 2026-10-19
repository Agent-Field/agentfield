package services

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

// C5-C6: cancel treats execution IDs as one segment and keeps the base prefix.
func TestAgentEndpointCancelPaths(t *testing.T) {
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		require.Equal(t, "/prefix/_internal/executions/exec?x#y/cancel", r.URL.Path)
		require.Empty(t, r.URL.RawQuery)
	}))
	defer server.Close()
	store := newFakeAgentStore()
	store.seed("node", server.URL+"/prefix/")
	d := NewCancelDispatcher(store, CancelDispatcherConfig{})
	for _, id := range []string{"exec?x#y", "../status", "%2f", ".."} {
		d.dispatch(context.Background(), events.ExecutionEvent{AgentNodeID: "node", ExecutionID: id, Timestamp: time.Now()})
	}
	require.EqualValues(t, 1, hits.Load())
}

// C5-C6/C9: trigger dispatch preserves prefixes, escapes query/fragment in the
// reasoner name, and refuses traversal or invalid stored endpoints before I/O.
func TestAgentEndpointTriggerPaths(t *testing.T) {
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		require.Equal(t, "/prefix/reasoners/handle?x#y", r.URL.Path)
		require.Empty(t, r.URL.RawQuery)
		if _, err := w.Write([]byte(`{}`)); err != nil {
			t.Errorf("write response: %v", err)
		}
	}))
	defer server.Close()
	for _, tc := range []struct {
		name, base string
		ok         bool
	}{
		{"handle?x#y", server.URL + "/prefix/", true},
		{"../status", server.URL + "/prefix", false},
		{"%2f", server.URL + "/prefix", false},
		{"handle", server.URL + "/prefix?hidden", false},
		// Review finding 3: empty base_url retains trigger dispatch failure.
		{"handle", "", false},
	} {
		t.Run(tc.name+tc.base, func(t *testing.T) {
			node, trig, ev := dispatcherTrigger(tc.base, []types.ReasonerDefinition{{ID: tc.name}})
			trig.TargetReasoner = tc.name
			ctx, store, d := storeDispatchFixture(t, node, trig, ev)
			before := hits.Load()
			d.DispatchEvent(ctx, trig, ev)
			saved, err := store.GetInboundEvent(ctx, ev.ID)
			require.NoError(t, err)
			if tc.ok {
				require.Equal(t, before+1, hits.Load())
				require.NotEqual(t, types.InboundEventStatusFailed, saved.Status)
			} else {
				require.Equal(t, before, hits.Load())
				require.Equal(t, types.InboundEventStatusFailed, saved.Status)
			}
		})
	}
}

// C5-C6/C10: custom cancel templates retain their documented substring
// substitution while escaping the entire resulting segment beneath the base.
func TestAgentEndpointCancelCustomTemplate(t *testing.T) {
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		require.Equal(t, "/prefix/custom/exec-42?x#y", r.URL.Path)
		require.Empty(t, r.URL.RawQuery)
	}))
	defer server.Close()
	store := newFakeAgentStore()
	store.seed("node", server.URL+"/prefix")
	d := NewCancelDispatcher(store, CancelDispatcherConfig{CancelPathTemplate: "/custom/exec-:execution_id"})
	for _, id := range []string{"42?x#y", "../escape", "%2f"} {
		d.dispatch(context.Background(), events.ExecutionEvent{AgentNodeID: "node", ExecutionID: id, Timestamp: time.Now()})
	}
	require.EqualValues(t, 1, hits.Load())
}
