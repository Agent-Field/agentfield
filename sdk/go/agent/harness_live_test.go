package agent

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/harness"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeCodeAF is a codeaf that streams two entries, waits for a person's note
// on its steer file, says it read it, and answers.
const fakeCodeAF = `#!/bin/sh
events=""; steer=""
while [ $# -gt 0 ]; do
  case "$1" in --events) events="$2"; shift;; --steer) steer="$2"; shift;; esac
  shift
done
cat >/dev/null
printf '%s\n' '{"kind":"status","text":"started"}' >> "$events"
printf '%s\n' '{"kind":"assistant","turn":1,"text":"looking at main.go"}' >> "$events"
i=0
while [ $i -lt 100 ] && [ ! -s "$steer" ]; do sleep 0.05; i=$((i+1)); done
printf '{"kind":"note","turn":1,"text":"steered: %s"}\n' "$(sed -e 's/.*"text":"//' -e 's/"}$//' "$steer")" >> "$events"
printf '%s\n' '{"ok":true,"stop":"done","answer":"wrote it"}'
`

type cpRecorder struct {
	mu   sync.Mutex
	logs []ExecutionLogEntry
}

func (c *cpRecorder) handler(t *testing.T) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/logs") {
			raw, _ := io.ReadAll(r.Body)
			var batch struct {
				Entries []ExecutionLogEntry `json:"entries"`
			}
			var one ExecutionLogEntry
			c.mu.Lock()
			if json.Unmarshal(raw, &batch) == nil && len(batch.Entries) > 0 {
				c.logs = append(c.logs, batch.Entries...)
			} else if json.Unmarshal(raw, &one) == nil {
				c.logs = append(c.logs, one)
			}
			c.mu.Unlock()
			w.WriteHeader(http.StatusAccepted)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{}`))
	})
}

func (c *cpRecorder) types() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []string
	for _, l := range c.logs {
		out = append(out, l.EventType)
	}
	return out
}

func steerRequest(a *Agent, id, message string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/_internal/executions/"+id+"/steer", strings.NewReader(`{"message":"`+message+`"}`))
	rec := httptest.NewRecorder()
	a.handleInternalExecution(rec, req)
	return rec
}

// THE DEVELOPER WRITES ONLY a.Harness(...): the run's transcript reaches its
// own execution's logs verbatim and in order, a person's steer reaches the
// running codeaf, and nothing else is steerable.
func TestHarnessOnCodeAFStreamsItsTranscriptAndTakesASteer(t *testing.T) {
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	t.Setenv("OPENROUTER_API_KEY", "test-key")
	t.Setenv("AGENTFIELD_LOG_STDOUT", "off")
	script := filepath.Join(t.TempDir(), "codeaf")
	require.NoError(t, os.WriteFile(script, []byte(fakeCodeAF), 0o755))

	cp := &cpRecorder{}
	server := httptest.NewServer(cp.handler(t))
	defer server.Close()
	a, err := New(Config{NodeID: "demo", Version: "1", AgentFieldURL: server.URL, Logger: log.New(io.Discard, "", 0),
		HarnessConfig: &HarnessConfig{Provider: "codeaf", BinPath: script}})
	require.NoError(t, err)

	ctx := contextWithExecution(context.Background(), ExecutionContext{ExecutionID: "exec-write", RunID: "run-1", WorkflowID: "run-1", AgentNodeID: "demo", ReasonerName: "write"})
	done := make(chan error, 1)
	go func() {
		_, err := a.Harness(ctx, "write a function", nil, nil, harness.Options{})
		done <- err
	}()

	// Not a codeaf run: refused, whatever is running beside it.
	rec := steerRequest(a, "exec-plan", "hello")
	assert.Equal(t, http.StatusConflict, rec.Code)
	assert.Contains(t, rec.Body.String(), "not running the codeaf harness")

	require.Eventually(t, func() bool { return a.steer.get("exec-write") != nil }, 5*time.Second, 10*time.Millisecond, "the run never became steerable")
	rec = steerRequest(a, "exec-write", "also add a doc comment")
	require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())

	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(15 * time.Second):
		t.Fatal("the harness run did not finish")
	}
	// Ended: no longer steerable.
	assert.Equal(t, http.StatusConflict, steerRequest(a, "exec-write", "too late").Code)

	require.Eventually(t, func() bool {
		ty := cp.types()
		return len(ty) >= 6
	}, 5*time.Second, 20*time.Millisecond)
	cp.mu.Lock()
	defer cp.mu.Unlock()
	var codeaf []string
	var order []string
	for _, l := range cp.logs {
		assert.Equal(t, "exec-write", l.ExecutionID)
		if l.EventType == harnessEventType {
			raw, _ := json.Marshal(l.Attributes["codeaf"])
			codeaf = append(codeaf, string(raw))
		}
		if l.EventType != harnessPersonType {
			order = append(order, l.EventType)
		}
	}
	assert.Equal(t, []string{harnessStartedType, harnessEventType, harnessEventType, harnessEventType, harnessEventType, harnessEndedType}, order)
	assert.Equal(t, `{"kind":"assistant","turn":1,"text":"looking at main.go"}`, codeaf[1], "the event was not carried verbatim")
	assert.Contains(t, codeaf[2], "steered: also add a doc comment", "codeaf never read the note")
	assert.Contains(t, cp.types(), harnessPersonType)
	started := cp.logs[0]
	assert.Equal(t, true, started.Attributes["steerable"])
	assert.Equal(t, "run-1", started.RunID)
}

// A harness call on another provider never opens a mailbox.
func TestHarnessOnAnotherProviderIsNeverSteerable(t *testing.T) {
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	a, err := New(Config{NodeID: "demo", Version: "1", AgentFieldURL: "http://127.0.0.1:1", Logger: log.New(io.Discard, "", 0),
		HarnessConfig: &HarnessConfig{Provider: "opencode", BinPath: "/nonexistent/opencode"}})
	require.NoError(t, err)
	ctx := contextWithExecution(context.Background(), ExecutionContext{ExecutionID: "exec-x"})
	_, _ = a.Harness(ctx, "x", nil, nil, harness.Options{Timeout: 2})
	assert.Nil(t, a.steer.get("exec-x"))
	assert.ErrorIs(t, a.Steer("exec-x", "hi"), ErrNotSteerable)
}

// The link: a steer for a codeaf run is acked accepted, one for anything else
// is acked rejected with the refusal's words, and hello says what it accepts.
func TestLinkServesSteerAndRefusesANonCodeAFExecution(t *testing.T) {
	t.Setenv("AGENTFIELD_LOG_STDOUT", "off")
	upgrader := websocket.Upgrader{}
	acks := make(chan map[string]any, 4)
	hello := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/api/v1/nodes/demo/link") {
			w.WriteHeader(http.StatusAccepted)
			return
		}
		assert.Equal(t, "k", r.Header.Get("X-API-Key"))
		conn, err := upgrader.Upgrade(w, r, nil)
		require.NoError(t, err)
		defer conn.Close()
		read := func() linkFrame {
			_, raw, err := conn.ReadMessage()
			require.NoError(t, err)
			var f linkFrame
			require.NoError(t, json.Unmarshal(raw, &f))
			return f
		}
		f := read()
		var h map[string]any
		_ = json.Unmarshal(f.Body, &h)
		hello <- h
		for _, d := range []struct{ id, exec string }{{"dlv_1", "exec-plan"}, {"dlv_2", "exec-write"}} {
			raw, _ := json.Marshal(linkFrame{V: 1, T: "deliver", ID: d.id, Exec: d.exec, Body: json.RawMessage(`{"kind":"steer","message":"also add a doc comment"}`)})
			require.NoError(t, conn.WriteMessage(websocket.TextMessage, raw))
			ack := read()
			var body map[string]any
			_ = json.Unmarshal(ack.Body, &body)
			body["id"] = ack.ID
			acks <- body
		}
	}))
	defer server.Close()
	a, err := New(Config{NodeID: "demo", Version: "1", AgentFieldURL: server.URL, APIKey: "k", Logger: log.New(io.Discard, "", 0)})
	require.NoError(t, err)
	box := make(chan string, 1)
	a.steer.open("exec-write", box)
	target, ok := a.linkURL()
	require.True(t, ok)
	go a.linkOnce(target)

	h := <-hello
	assert.Equal(t, []any{"steer", "cancel"}, h["accepts"])
	first, second := <-acks, <-acks
	assert.Equal(t, "dlv_1", first["id"])
	assert.Equal(t, "rejected", first["outcome"])
	assert.Contains(t, first["error"], "not running the codeaf harness")
	assert.Equal(t, "dlv_2", second["id"])
	assert.Equal(t, "accepted", second["outcome"])
	assert.Equal(t, "also add a doc comment", <-box)
}
