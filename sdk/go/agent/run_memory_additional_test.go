package agent

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRun_RoutesToServeAndCLI(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/api/v1/nodes":
			_, _ = io.WriteString(w, `{"id":"node-1","success":true}`)
		case r.Method == http.MethodPost && r.URL.Path == "/api/v1/nodes/node-1/status":
			_, _ = io.WriteString(w, `{"lease_seconds":120}`)
		case r.Method == http.MethodPost && r.URL.Path == "/api/v1/nodes/node-1/shutdown":
			_, _ = io.WriteString(w, `{"ok":true}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	serveAgent, err := New(Config{
		NodeID:           "node-1",
		Version:          "1.0.0",
		AgentFieldURL:    server.URL,
		ListenAddress:    "127.0.0.1:0",
		Logger:           log.New(io.Discard, "", 0),
		DisableLeaseLoop: true,
	})
	require.NoError(t, err)
	serveAgent.RegisterReasoner("demo", func(ctx context.Context, input map[string]any) (any, error) {
		return map[string]any{"ok": true}, nil
	})

	cliAgent, err := New(Config{
		NodeID:  "node-cli",
		Version: "1.0.0",
		Logger:  log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	cliAgent.RegisterReasoner("demo", func(ctx context.Context, input map[string]any) (any, error) {
		return map[string]any{"ok": true}, nil
	}, WithCLI())

	origArgs := os.Args
	defer func() { os.Args = origArgs }()

	t.Run("serve branch when explicitly requested", func(t *testing.T) {
		os.Args = []string{"agent", "serve"}
		ctx, cancel := context.WithCancel(context.Background())
		go func() {
			time.Sleep(50 * time.Millisecond)
			cancel()
		}()
		require.NoError(t, serveAgent.Run(ctx))
	})

	t.Run("cli branch for command args", func(t *testing.T) {
		os.Args = []string{"agent", "version"}
		require.NoError(t, cliAgent.Run(context.Background()))
	})
}

func TestMemory_UserScopeVectorAndTypedHelpers(t *testing.T) {
	backend := NewInMemoryBackend()
	memory := NewMemory(backend)

	ctxWithActor := contextWithExecution(context.Background(), ExecutionContext{
		RunID:     "run-1",
		SessionID: "session-1",
		ActorID:   "actor-1",
	})
	ctxWithSessionOnly := contextWithExecution(context.Background(), ExecutionContext{
		RunID:     "run-2",
		SessionID: "session-2",
	})
	ctxWithRunOnly := contextWithExecution(context.Background(), ExecutionContext{
		RunID: "run-3",
	})

	require.NoError(t, memory.UserScope().Set(ctxWithActor, "user-key", "actor-value"))
	val, err := memory.UserScope().Get(ctxWithActor, "user-key")
	require.NoError(t, err)
	assert.Equal(t, "actor-value", val)

	require.NoError(t, memory.UserScope().Set(ctxWithSessionOnly, "fallback-key", "session-value"))
	val, err = memory.UserScope().Get(ctxWithSessionOnly, "fallback-key")
	require.NoError(t, err)
	assert.Equal(t, "session-value", val)

	require.NoError(t, memory.UserScope().Set(ctxWithRunOnly, "run-key", "run-value"))
	val, err = memory.UserScope().Get(ctxWithRunOnly, "run-key")
	require.NoError(t, err)
	assert.Equal(t, "run-value", val)

	require.NoError(t, memory.UserScope().SetVector(ctxWithActor, "vec", []float64{1, 2}, map[string]any{"kind": "actor"}))
	embedding, metadata, err := memory.UserScope().GetVector(ctxWithActor, "vec")
	require.NoError(t, err)
	assert.Equal(t, []float64{1, 2}, embedding)
	assert.Equal(t, map[string]any{"kind": "actor"}, metadata)

	embedding, metadata, err = memory.UserScope().GetVector(ctxWithActor, "missing")
	require.NoError(t, err)
	assert.Nil(t, embedding)
	assert.Nil(t, metadata)

	type payload struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}

	require.NoError(t, backend.Set(ScopeSession, "typed", "bytes", []byte(`{"name":"bytes","age":1}`)))
	require.NoError(t, backend.Set(ScopeSession, "typed", "string", `{"name":"string","age":2}`))
	require.NoError(t, backend.Set(ScopeSession, "typed", "map", map[string]any{"name": "map", "age": 3}))

	scoped := memory.Scoped(ScopeSession, "typed")
	var got payload
	require.NoError(t, scoped.GetTyped(context.Background(), "bytes", &got))
	assert.Equal(t, payload{Name: "bytes", Age: 1}, got)
	require.NoError(t, scoped.GetTyped(context.Background(), "string", &got))
	assert.Equal(t, payload{Name: "string", Age: 2}, got)
	require.NoError(t, scoped.GetTyped(context.Background(), "map", &got))
	assert.Equal(t, payload{Name: "map", Age: 3}, got)
	require.NoError(t, scoped.GetTyped(context.Background(), "missing", &got))

	require.NoError(t, backend.Set(ScopeSession, "typed", "invalid", make(chan int)))
	err = scoped.GetTyped(context.Background(), "invalid", &got)
	require.Error(t, err)
}

func TestHandleDiscover_ReturnsPayloadAndRejectsWrongMethod(t *testing.T) {
	a, err := New(Config{
		NodeID:  "node-1",
		Version: "1.2.3",
		Logger:  log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	a.RegisterReasoner("demo", func(ctx context.Context, input map[string]any) (any, error) {
		return nil, nil
	}, WithInputSchema(json.RawMessage(`{"type":"object"}`)), WithOutputSchema(json.RawMessage(`{"type":"string"}`)))

	postReq := httptest.NewRequest(http.MethodPost, "/discover", nil)
	postResp := httptest.NewRecorder()
	a.handleDiscover(postResp, postReq)
	assert.Equal(t, http.StatusMethodNotAllowed, postResp.Code)

	getReq := httptest.NewRequest(http.MethodGet, "/discover", nil)
	getResp := httptest.NewRecorder()
	a.handleDiscover(getResp, getReq)
	require.Equal(t, http.StatusOK, getResp.Code)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(getResp.Body.Bytes(), &payload))
	assert.Equal(t, "node-1", payload["node_id"])
	assert.Equal(t, "1.2.3", payload["version"])
	assert.Equal(t, "long_running", payload["deployment_type"])
	reasoners := payload["reasoners"].([]any)
	require.Len(t, reasoners, 1)
	assert.Equal(t, "demo", reasoners[0].(map[string]any)["id"])
}
