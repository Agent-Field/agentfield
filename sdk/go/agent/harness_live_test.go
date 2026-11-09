package agent

import (
	"context"
	"encoding/json"
	"fmt"
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
	mu         sync.Mutex
	entries    []RelayEntry
	seen       map[int64]bool
	down       bool
	relayPosts int
	completed  map[string]bool
}

func (c *cpRecorder) handler(t *testing.T) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/relay") {
			c.mu.Lock()
			defer c.mu.Unlock()
			c.relayPosts++
			if c.down {
				w.WriteHeader(http.StatusServiceUnavailable)
				return
			}
			var body struct {
				Entries []RelayEntry `json:"entries"`
			}
			raw, _ := io.ReadAll(r.Body)
			require.NoError(t, json.Unmarshal(raw, &body))
			if c.seen == nil {
				c.seen = map[int64]bool{}
			}
			var through int64
			for _, e := range body.Entries {
				if !c.seen[e.Seq] {
					c.seen[e.Seq] = true
					c.entries = append(c.entries, e)
				}
				through = e.Seq
			}
			_, _ = w.Write([]byte(fmt.Sprintf(`{"through":%d}`, through)))
			return
		}
		if strings.HasSuffix(r.URL.Path, "/status") {
			c.mu.Lock()
			if c.completed == nil {
				c.completed = map[string]bool{}
			}
			c.completed[r.URL.Path] = true
			c.mu.Unlock()
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{}`))
	})
}

func (c *cpRecorder) kinds() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []string
	for _, e := range c.entries {
		out = append(out, e.Kind)
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

	ctx := contextWithExecution(context.Background(), ExecutionContext{ExecutionID: "exec-write", RunID: "run-1", WorkflowID: "run-1", AgentNodeID: "demo", ReasonerName: "write", RunWatched: true})
	done := make(chan error, 1)
	go func() {
		res, err := a.Harness(ctx, "write a function", nil, nil, harness.Options{})
		if res != nil && res.IsError {
			t.Logf("harness result: %+v", res)
		}
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

	require.Eventually(t, func() bool { return len(cp.kinds()) >= 6 }, 5*time.Second, 20*time.Millisecond)
	cp.mu.Lock()
	defer cp.mu.Unlock()
	var order []string
	var lines []string
	for i, e := range cp.entries {
		assert.Equal(t, int64(i+1), e.Seq, "seq is the node's, from 1, in order")
		if e.Kind != RelayPerson {
			order = append(order, e.Kind)
		}
		if e.Kind == RelayLine {
			lines = append(lines, string(e.Data))
		}
	}
	assert.Equal(t, []string{RelayStarted, RelayLine, RelayLine, RelayLine, RelayEnded}, order)
	assert.Equal(t, `{"kind":"assistant","turn":1,"text":"looking at main.go"}`, lines[1], "the line was not carried verbatim")
	assert.Contains(t, lines[2], "steered: also add a doc comment", "codeaf never read the note")
	assert.Contains(t, cp.kindsLocked(), RelayPerson)
	assert.JSONEq(t, `{"harness":"codeaf","steerable":true}`, string(cp.entries[0].Data))
}

func (c *cpRecorder) kindsLocked() []string {
	var out []string
	for _, e := range c.entries {
		out = append(out, e.Kind)
	}
	return out
}

// A control plane that is down for a while loses nothing and doubles
// nothing: the node keeps every unanswered entry and replays it.
func TestRelaySurvivesAControlPlaneOutage(t *testing.T) {
	relayMaxBackoff = 50 * time.Millisecond
	t.Cleanup(func() { relayMaxBackoff = 5 * time.Second })
	cp := &cpRecorder{}
	server := httptest.NewServer(cp.handler(t))
	defer server.Close()
	a, err := New(Config{NodeID: "demo", Version: "1", AgentFieldURL: server.URL, Logger: log.New(io.Discard, "", 0)})
	require.NoError(t, err)
	r := a.relayFor("exec-1")
	r.send(RelayStarted, json.RawMessage(`{}`))
	require.Eventually(t, func() bool { return len(cp.kinds()) == 1 }, 3*time.Second, 10*time.Millisecond)
	cp.mu.Lock()
	cp.down = true
	cp.mu.Unlock()
	for i := 0; i < 50; i++ {
		r.send(RelayLine, json.RawMessage(fmt.Sprintf(`{"n":%d}`, i)))
	}
	time.Sleep(200 * time.Millisecond)
	assert.Equal(t, 50, r.relayPending(), "nothing is dropped while the control plane is down")
	cp.mu.Lock()
	cp.down = false
	cp.mu.Unlock()
	r.flush(3 * time.Second)
	assert.Equal(t, 0, r.relayPending())
	cp.mu.Lock()
	defer cp.mu.Unlock()
	require.Len(t, cp.entries, 51)
	for i, e := range cp.entries {
		assert.Equal(t, int64(i+1), e.Seq)
	}
}

// A harness call on another provider never opens a mailbox.
func TestHarnessOnAnotherProviderIsNeverSteerable(t *testing.T) {
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	a, err := New(Config{NodeID: "demo", Version: "1", AgentFieldURL: "http://127.0.0.1:1", Logger: log.New(io.Discard, "", 0),
		HarnessConfig: &HarnessConfig{Provider: "opencode", BinPath: "/nonexistent/opencode"}})
	require.NoError(t, err)
	ctx := contextWithExecution(context.Background(), ExecutionContext{ExecutionID: "exec-x", RunWatched: true})
	_, _ = a.Harness(ctx, "x", nil, nil, harness.Options{Timeout: 2})
	assert.Nil(t, a.steer.get("exec-x"))
	assert.ErrorIs(t, a.Steer("exec-x", "hi"), ErrNotSteerable)
}

// The link: a steer for a codeaf run is acked accepted, one for anything else
// is acked rejected with the refusal's words, and hello says what it accepts.
func TestLinkServesSteerAndRefusesANonCodeAFExecution(t *testing.T) {
	t.Setenv("AGENTFIELD_LOG_STDOUT", "off")
	upgrader := websocket.Upgrader{}
	acks := make(chan map[string]any, 6)
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
		next := func() linkFrame {
			_, raw, err := conn.ReadMessage()
			require.NoError(t, err)
			var f linkFrame
			require.NoError(t, json.Unmarshal(raw, &f))
			return f
		}
		// The steer's own record (a "person" entry) rides the link too, as an
		// entries frame; this test reads the acks.
		read := func() linkFrame {
			for {
				if f := next(); f.T != "entries" {
					return f
				}
			}
		}
		f := next()
		var h map[string]any
		_ = json.Unmarshal(f.Body, &h)
		hello <- h
		// A steer carried as http, for a plain step and a codeaf run; then
		// the second delivered again, as a reconnect would.
		for _, d := range []struct{ id, exec string }{{"dlv_1", "exec-plan"}, {"dlv_2", "exec-write"}, {"dlv_2", "exec-write"}} {
			body, _ := json.Marshal(map[string]any{"kind": "http", "method": "POST", "path": "/_internal/executions/" + d.exec + "/steer",
				"headers": map[string]string{"Content-Type": "application/json", "X-Execution-ID": d.exec}, "body": []byte(`{"message":"also add a doc comment"}`)})
			raw, _ := json.Marshal(linkFrame{V: 1, T: "deliver", ID: d.id, Exec: d.exec, Body: body})
			require.NoError(t, conn.WriteMessage(websocket.TextMessage, raw))
			ack := read()
			var out struct {
				Outcome string         `json:"outcome"`
				Result  linkHTTPResult `json:"result"`
			}
			_ = json.Unmarshal(ack.Body, &out)
			acks <- map[string]any{"id": ack.ID, "outcome": out.Outcome, "code": out.Result.Code, "body": string(out.Result.Body)}
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
	assert.Equal(t, []any{"http", "steer", "cancel"}, h["accepts"])
	first, second, again := <-acks, <-acks, <-acks
	assert.Equal(t, "dlv_1", first["id"])
	assert.Equal(t, 409, first["code"])
	assert.Contains(t, first["body"], "not running the codeaf harness")
	assert.Equal(t, "dlv_2", second["id"])
	assert.Equal(t, 200, second["code"])
	assert.Equal(t, second, again, "a redelivery is answered with the same ack")
	assert.Len(t, box, 1, "a redelivery is not a second steer")
	assert.Equal(t, "also add a doc comment", <-box)
}

// THE RELAY RIDES THE LINK. Each batch is an `entries` frame whose seq is the
// link's own counter (the reply key) and whose body carries the entries' own
// seqs; the node forgets what the ok's `through` covers. With the link gone the
// same entries go by POST /relay, and every entry is held once, in order.
func TestRelayRidesTheLinkAndFallsBackToHTTP(t *testing.T) {
	t.Setenv("AGENTFIELD_LOG_STDOUT", "off")
	relayMaxBackoff = 50 * time.Millisecond
	t.Cleanup(func() { relayMaxBackoff = 5 * time.Second })
	var mu sync.Mutex
	held := map[int64]string{}
	var frameSeqs []int64
	var overHTTP int
	keep := func(es []RelayEntry) (through int64) {
		for _, e := range es {
			if _, ok := held[e.Seq]; !ok {
				held[e.Seq] = e.Kind
			}
			through = e.Seq
		}
		return through
	}
	linkUp := make(chan *websocket.Conn, 1)
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/link"):
			conn, err := upgrader.Upgrade(w, r, nil)
			require.NoError(t, err)
			linkUp <- conn
			for {
				_, raw, err := conn.ReadMessage()
				if err != nil {
					return
				}
				var f linkFrame
				require.NoError(t, json.Unmarshal(raw, &f))
				if f.T != "entries" {
					continue
				}
				var body struct {
					Entries []RelayEntry `json:"entries"`
				}
				require.NoError(t, json.Unmarshal(f.Body, &body))
				mu.Lock()
				frameSeqs = append(frameSeqs, f.Seq)
				through := keep(body.Entries)
				mu.Unlock()
				okBody, _ := json.Marshal(map[string]any{"code": 200, "through": through})
				out, _ := json.Marshal(linkFrame{V: 1, T: "ok", Seq: f.Seq, Exec: f.Exec, Body: okBody})
				_ = conn.WriteMessage(websocket.TextMessage, out)
			}
		case strings.HasSuffix(r.URL.Path, "/relay"):
			var body struct {
				Entries []RelayEntry `json:"entries"`
			}
			raw, _ := io.ReadAll(r.Body)
			require.NoError(t, json.Unmarshal(raw, &body))
			mu.Lock()
			overHTTP++
			through := keep(body.Entries)
			mu.Unlock()
			_, _ = w.Write([]byte(fmt.Sprintf(`{"stored":%d,"through":%d}`, len(body.Entries), through)))
		default:
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{}`))
		}
	}))
	defer server.Close()
	a, err := New(Config{NodeID: "demo", Version: "1", AgentFieldURL: server.URL, Logger: log.New(io.Discard, "", 0)})
	require.NoError(t, err)
	target, ok := a.linkURL()
	require.True(t, ok)
	go a.linkOnce(target)
	conn := <-linkUp
	require.Eventually(t, func() bool { a.link.mu.Lock(); defer a.link.mu.Unlock(); return a.link.conn != nil }, 3*time.Second, 10*time.Millisecond)

	r := a.relayFor("exec-1")
	r.send(RelayStarted, json.RawMessage(`{"harness":"codeaf","steerable":true}`))
	r.send(RelayLine, json.RawMessage(`{"type":"message"}`))
	r.flush(3 * time.Second)
	mu.Lock()
	assert.Equal(t, 0, overHTTP, "with a link up nothing goes by HTTP")
	require.NotEmpty(t, frameSeqs)
	assert.Greater(t, frameSeqs[0], int64(1), "the frame seq is the link's counter after hello, not a data seq")
	mu.Unlock()

	// The link drops: the rest goes by HTTP.
	_ = conn.Close()
	require.Eventually(t, func() bool { a.link.mu.Lock(); defer a.link.mu.Unlock(); return a.link.conn == nil }, 3*time.Second, 10*time.Millisecond)
	r.send(RelayLine, json.RawMessage(`{"type":"message","n":2}`))
	r.send(RelayEnded, json.RawMessage(`{}`))
	r.flush(3 * time.Second)
	assert.Equal(t, 0, r.relayPending())
	mu.Lock()
	defer mu.Unlock()
	assert.Positive(t, overHTTP)
	assert.Equal(t, map[int64]string{1: RelayStarted, 2: RelayLine, 3: RelayLine, 4: RelayEnded}, held)
}

// recordingCodeAF records its argv and stdin into REC_DIR so a test can prove
// whether codeaf ran on the origin/main exec path or the watched chat path.
const recordingCodeAF = `#!/bin/sh
: > "$REC_DIR/argv"
for arg in "$@"; do
  printf '%s\n' "$arg" >> "$REC_DIR/argv"
done
cat > "$REC_DIR/stdin"
: > "$REC_DIR/running"
events=""; steer=""
while [ $# -gt 0 ]; do
  case "$1" in --events) events="$2"; shift;; --steer) steer="$2"; shift;; esac
  shift
done
if [ -n "$events" ]; then
  printf '%s\n' '{"kind":"assistant","turn":1,"text":"looking at main.go"}' >> "$events"
  if [ "$WAIT_FOR_STEER" = "1" ]; then
    i=0
    while [ $i -lt 100 ] && [ ! -s "$steer" ]; do sleep 0.05; i=$((i+1)); done
    printf '{"kind":"note","turn":1,"text":"steered: %s"}\n' "$(sed -e 's/.*"text":"//' -e 's/"}$//' "$steer")" >> "$events"
  fi
elif [ -n "$HOLD_FILE" ]; then
  i=0
  while [ $i -lt 100 ] && [ -e "$HOLD_FILE" ]; do sleep 0.05; i=$((i+1)); done
fi
printf '%s\n' '{"ok":true,"stop":"done","answer":"wrote it"}'
`

func readRecordedArgv(t *testing.T, dir string) []string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, "argv"))
	require.NoError(t, err)
	trimmed := strings.TrimSuffix(string(raw), "\n")
	if trimmed == "" {
		return nil
	}
	return strings.Split(trimmed, "\n")
}

// Contract C1, C2: ONLY AN EXECUTION MARKED X-AgentField-Run-Watched: 1 ARMS
// THE LIVE HARNESS. Without the header, the Agent runs plain `codeaf exec`,
// makes zero relay POSTs, and refuses steers with ErrNotSteerable; with the
// header on the same Agent and reasoner, it runs `chat --once` with --events
// and --steer, streams started/line/ended to the relay, and takes a steer.
func TestAgentStreamsOnlyWatchedRuns(t *testing.T) {
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	t.Setenv("OPENROUTER_API_KEY", "test-key")
	t.Setenv("AGENTFIELD_LOG_STDOUT", "off")
	script := filepath.Join(t.TempDir(), "codeaf")
	require.NoError(t, os.WriteFile(script, []byte(recordingCodeAF), 0o755))

	cp := &cpRecorder{}
	server := httptest.NewServer(cp.handler(t))
	defer server.Close()
	a, err := New(Config{
		NodeID:        "demo",
		Version:       "1",
		AgentFieldURL: server.URL,
		Logger:        log.New(io.Discard, "", 0),
		HarnessConfig: &HarnessConfig{Provider: "codeaf", BinPath: script},
	})
	require.NoError(t, err)
	a.RegisterReasoner("write", func(ctx context.Context, input map[string]any) (any, error) {
		env := map[string]string{"REC_DIR": input["rec_dir"].(string)}
		if hold, ok := input["hold_file"].(string); ok && hold != "" {
			env["HOLD_FILE"] = hold
		}
		if wait, ok := input["wait_for_steer"].(string); ok && wait != "" {
			env["WAIT_FOR_STEER"] = wait
		}
		res, err := a.Harness(ctx, "write a function", nil, nil, harness.Options{Env: env})
		if err != nil {
			return nil, err
		}
		return map[string]any{"answer": res.Result, "watched": executionContextFrom(ctx).ChildContext("demo", "nested").RunWatched}, nil
	})

	// C2: without X-AgentField-Run-Watched, /reasoners/write runs plain codeaf
	// exec, refuses steers while working, and never posts to /relay.
	unwatchedDir := t.TempDir()
	holdFile := filepath.Join(unwatchedDir, "hold")
	require.NoError(t, os.WriteFile(holdFile, []byte("1"), 0o600))
	body, _ := json.Marshal(map[string]any{"rec_dir": unwatchedDir, "hold_file": holdFile})
	req := httptest.NewRequest(http.MethodPost, "/reasoners/write", strings.NewReader(string(body)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Execution-ID", "exec-unwatched")
	req.Header.Set("X-Run-ID", "run-unwatched")
	rec := httptest.NewRecorder()
	a.ServeHTTP(rec, req)
	require.Equal(t, http.StatusAccepted, rec.Code, rec.Body.String())

	require.Eventually(t, func() bool {
		_, err := os.Stat(filepath.Join(unwatchedDir, "running"))
		return err == nil
	}, 5*time.Second, 10*time.Millisecond, "the unwatched codeaf invocation never started")

	assert.ErrorIs(t, a.Steer("exec-unwatched", "do not steer"), ErrNotSteerable)
	steerRec := steerRequest(a, "exec-unwatched", "do not steer")
	assert.Equal(t, http.StatusConflict, steerRec.Code)
	assert.Contains(t, steerRec.Body.String(), "not running the codeaf harness")

	require.NoError(t, os.Remove(holdFile))
	require.Eventually(t, func() bool {
		cp.mu.Lock()
		defer cp.mu.Unlock()
		return cp.completed["/api/v1/executions/exec-unwatched/status"]
	}, 5*time.Second, 10*time.Millisecond, "the unwatched execution never reported completion")

	unwatchedArgv := readRecordedArgv(t, unwatchedDir)
	require.GreaterOrEqual(t, len(unwatchedArgv), 8)
	assert.Equal(t, []string{"exec", "--json", "--dir", ".", "--model", harness.DefaultCodeAFModel, "--timeout"}, unwatchedArgv[:7])
	assert.NotContains(t, unwatchedArgv, "chat")
	assert.NotContains(t, unwatchedArgv, "--events")
	assert.NotContains(t, unwatchedArgv, "--steer")
	stdinBytes, err := os.ReadFile(filepath.Join(unwatchedDir, "stdin"))
	require.NoError(t, err)
	assert.Equal(t, "write a function", string(stdinBytes))

	// C2 on /execute/write (buildExecutionContextFromServerless): still plain exec and 0 relay POSTs.
	serverlessUnwatchedDir := t.TempDir()
	execPayload, _ := json.Marshal(map[string]any{"input": map[string]any{"rec_dir": serverlessUnwatchedDir}})
	execReq := httptest.NewRequest(http.MethodPost, "/execute/write", strings.NewReader(string(execPayload)))
	execReq.Header.Set("Content-Type", "application/json")
	execReq.Header.Set("X-Execution-ID", "exec-serverless-unwatched")
	execReq.Header.Set("X-Run-ID", "run-unwatched")
	execRec := httptest.NewRecorder()
	a.ServeHTTP(execRec, execReq)
	require.Equal(t, http.StatusOK, execRec.Code, execRec.Body.String())
	assert.Equal(t, "exec", readRecordedArgv(t, serverlessUnwatchedDir)[0])
	assert.Contains(t, execRec.Body.String(), `"watched":false`)

	cp.mu.Lock()
	assert.Equal(t, 0, cp.relayPosts, "an unwatched run must make zero relay requests")
	assert.Empty(t, cp.entries)
	cp.mu.Unlock()

	// C1: the same /reasoners/write request WITH X-AgentField-Run-Watched: 1
	// runs chat --once with --events/--steer, streams started/line/ended, and
	// accepts a steer while working.
	watchedDir := t.TempDir()
	watchedBody, _ := json.Marshal(map[string]any{"rec_dir": watchedDir, "wait_for_steer": "1"})
	wReq := httptest.NewRequest(http.MethodPost, "/reasoners/write", strings.NewReader(string(watchedBody)))
	wReq.Header.Set("Content-Type", "application/json")
	wReq.Header.Set("X-Execution-ID", "exec-watched")
	wReq.Header.Set("X-Run-ID", "run-watched")
	wReq.Header.Set(RunWatchedHeader, "1")
	wRec := httptest.NewRecorder()
	a.ServeHTTP(wRec, wReq)
	require.Equal(t, http.StatusAccepted, wRec.Code, wRec.Body.String())

	require.Eventually(t, func() bool { return a.steer.get("exec-watched") != nil }, 5*time.Second, 10*time.Millisecond, "the watched run never became steerable")
	require.Equal(t, http.StatusOK, steerRequest(a, "exec-watched", "also add a doc comment").Code)

	require.Eventually(t, func() bool {
		cp.mu.Lock()
		defer cp.mu.Unlock()
		return cp.completed["/api/v1/executions/exec-watched/status"] && len(cp.entries) >= 5
	}, 5*time.Second, 20*time.Millisecond, "the watched execution did not finish and flush its relay entries")

	watchedArgv := readRecordedArgv(t, watchedDir)
	require.GreaterOrEqual(t, len(watchedArgv), 4)
	assert.Equal(t, []string{"chat", "--once", "write a function"}, watchedArgv[:3])
	assert.Contains(t, watchedArgv, "--events")
	assert.Contains(t, watchedArgv, "--steer")

	kinds := cp.kinds()
	assert.Equal(t, RelayStarted, kinds[0])
	assert.Contains(t, kinds, RelayLine)
	assert.Equal(t, RelayEnded, kinds[len(kinds)-1])

	// C1 on /execute/write: X-AgentField-Run-Watched: 1 also arms buildExecutionContextFromServerless and ChildContext.
	serverlessWatchedDir := t.TempDir()
	execWatchedPayload, _ := json.Marshal(map[string]any{"input": map[string]any{"rec_dir": serverlessWatchedDir}})
	execWatchedReq := httptest.NewRequest(http.MethodPost, "/execute/write", strings.NewReader(string(execWatchedPayload)))
	execWatchedReq.Header.Set("Content-Type", "application/json")
	execWatchedReq.Header.Set("X-Execution-ID", "exec-serverless-watched")
	execWatchedReq.Header.Set("X-Run-ID", "run-watched")
	execWatchedReq.Header.Set(RunWatchedHeader, "1")
	execWatchedRec := httptest.NewRecorder()
	a.ServeHTTP(execWatchedRec, execWatchedReq)
	require.Equal(t, http.StatusOK, execWatchedRec.Code, execWatchedRec.Body.String())
	assert.Equal(t, "chat", readRecordedArgv(t, serverlessWatchedDir)[0])
	assert.Contains(t, execWatchedRec.Body.String(), `"watched":true`)
}

// Contract C3: A HARNESS CALL WITH NO EXECUTION CONTEXT LEAVES Live NIL,
// whether called on an Agent outside a reasoner or directly on harness.Runner.
func TestHarnessOutsideExecutionContextLeavesLiveNil(t *testing.T) {
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	t.Setenv("OPENROUTER_API_KEY", "test-key")
	t.Setenv("AGENTFIELD_LOG_STDOUT", "off")
	script := filepath.Join(t.TempDir(), "codeaf")
	require.NoError(t, os.WriteFile(script, []byte(recordingCodeAF), 0o755))

	cp := &cpRecorder{}
	server := httptest.NewServer(cp.handler(t))
	defer server.Close()
	a, err := New(Config{
		NodeID:        "demo",
		Version:       "1",
		AgentFieldURL: server.URL,
		Logger:        log.New(io.Discard, "", 0),
		HarnessConfig: &HarnessConfig{Provider: "codeaf", BinPath: script},
	})
	require.NoError(t, err)

	live, flush := a.liveHarness(context.Background())
	flush()
	assert.Nil(t, live)

	agentDir := t.TempDir()
	res, err := a.Harness(context.Background(), "outside reasoner", nil, nil, harness.Options{Env: map[string]string{"REC_DIR": agentDir}})
	require.NoError(t, err)
	assert.Equal(t, "wrote it", res.Result)
	assert.Equal(t, "exec", readRecordedArgv(t, agentDir)[0])
	assert.NotContains(t, readRecordedArgv(t, agentDir), "--events")

	runnerDir := t.TempDir()
	runner := harness.NewRunner(harness.Options{Provider: "codeaf", BinPath: script})
	res, err = runner.Run(context.Background(), "outside agent", nil, nil, harness.Options{Env: map[string]string{"REC_DIR": runnerDir}})
	require.NoError(t, err)
	assert.Equal(t, "wrote it", res.Result)
	assert.Equal(t, "exec", readRecordedArgv(t, runnerDir)[0])
	assert.NotContains(t, readRecordedArgv(t, runnerDir), "--events")

	cp.mu.Lock()
	assert.Equal(t, 0, cp.relayPosts)
	cp.mu.Unlock()
}
