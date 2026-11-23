package agent

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/harness"
	"github.com/stretchr/testify/require"
)

type testHarness func(context.Context, string, harness.Options) (*harness.RawResult, error)

func (f testHarness) Execute(ctx context.Context, prompt string, opts harness.Options) (*harness.RawResult, error) {
	return f(ctx, prompt, opts)
}

type testLiveHarness struct {
	testHarness
	info harness.LiveInfo
}

func (p testLiveHarness) LiveInfo() harness.LiveInfo { return p.info }

func watchTestAgent(t *testing.T) (*Agent, *cpRecorder) {
	t.Helper()
	t.Setenv("AGENTFIELD_LOG_STDOUT", "off")
	cp := &cpRecorder{}
	srv := httptest.NewServer(cp.handler(t))
	t.Cleanup(srv.Close)
	a, err := New(Config{NodeID: "my-app", Version: "1", AgentFieldURL: srv.URL, Logger: log.New(io.Discard, "", 0)})
	require.NoError(t, err)
	return a, cp
}

func watchedContext(ctx context.Context, id string) context.Context {
	return contextWithExecution(ctx, ExecutionContext{ExecutionID: id, RunWatched: true})
}

func registerTestHarness(t *testing.T, name string, provider harness.Provider) {
	t.Helper()
	require.NoError(t, harness.RegisterProvider(name, func(string) harness.Provider { return provider }))
	t.Cleanup(func() { harness.UnregisterProvider(name) })
}

func TestWatchedCustomHarnessContractB1(t *testing.T) {
	a, cp := watchTestAgent(t)
	const name, exec = "my-harness", "exec-custom"
	line := json.RawMessage(`{"progress":"working","step":1}`)
	p := testLiveHarness{info: harness.LiveInfo{Harness: name, Steerable: true}}
	p.testHarness = func(ctx context.Context, _ string, opts harness.Options) (*harness.RawResult, error) {
		if opts.Live == nil {
			return nil, errors.New("watched harness did not receive live callbacks")
		}
		opts.Live.Begin(p.LiveInfo())
		opts.Live.Event(line)
		defer opts.Live.End()
		select {
		case note := <-opts.Live.Steer:
			return &harness.RawResult{Result: note}, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	registerTestHarness(t, name, p)
	ctx, cancel := context.WithTimeout(watchedContext(context.Background(), exec), 10*time.Second)
	defer cancel()
	type callResult struct {
		value *harness.Result
		err   error
	}
	result := make(chan callResult, 1)
	go func() {
		value, err := a.Harness(ctx, "prepare the report", nil, nil, harness.Options{Provider: name})
		result <- callResult{value, err}
	}()
	require.Eventually(t, func() bool { return len(cp.kinds()) == 2 }, 5*time.Second, 10*time.Millisecond)
	require.Equal(t, http.StatusOK, steerRequest(a, exec, "check the figures").Code)
	select {
	case got := <-result:
		require.NoError(t, got.err)
		require.Equal(t, "check the figures", got.value.Result)
	case <-ctx.Done():
		t.Fatal("the watched harness did not finish")
	}

	require.Equal(t, []string{RelayStarted, RelayLine, RelayPerson, RelayEnded}, cp.kinds())
	cp.mu.Lock()
	defer cp.mu.Unlock()
	require.Equal(t, `{"harness":"my-harness","steerable":true}`, string(cp.entries[0].Data))
	require.Equal(t, line, cp.entries[1].Data)
	require.JSONEq(t, `{"text":"check the figures"}`, string(cp.entries[2].Data))
}

func TestLiveAbsentContractB3(t *testing.T) {
	a, cp := watchTestAgent(t)
	var options []harness.Options
	spy := testHarness(func(_ context.Context, _ string, opts harness.Options) (*harness.RawResult, error) {
		options = append(options, opts)
		return &harness.RawResult{}, nil
	})
	registerTestHarness(t, "non-live", spy)
	registerTestHarness(t, "live-capable", testLiveHarness{testHarness: spy, info: harness.LiveInfo{Harness: "live-capable"}})
	for _, tc := range []struct {
		provider string
		ctx      context.Context
	}{
		{"non-live", watchedContext(context.Background(), "watched")},
		{"live-capable", contextWithExecution(context.Background(), ExecutionContext{ExecutionID: "unwatched"})},
		{"live-capable", context.Background()},
	} {
		_, err := a.Harness(tc.ctx, "prepare the report", nil, nil, harness.Options{Provider: tc.provider})
		require.NoError(t, err)
	}
	require.Len(t, options, 3)
	for _, opts := range options {
		require.Nil(t, opts.Live)
	}
	cp.mu.Lock()
	defer cp.mu.Unlock()
	require.Zero(t, cp.relayPosts)
	require.Empty(t, cp.entries)
}

func TestPlainReasonerLiveRunContractB4B5B6(t *testing.T) {
	for _, tc := range []struct {
		name      string
		steerable bool
	}{
		{"takes notes", true},
		{"watch only", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, cp := watchTestAgent(t)
			const exec = "exec-plain"
			_, ok := a.LiveRun(context.Background())
			require.False(t, ok)
			_, ok = a.LiveRun(contextWithExecution(context.Background(), ExecutionContext{ExecutionID: exec}))
			require.False(t, ok)
			ready, finish := make(chan *LiveRun, 1), make(chan struct{})
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			a.RegisterReasoner("intake", func(ctx context.Context, _ map[string]any) (any, error) {
				live, ok := a.LiveRun(ctx)
				if !ok {
					return nil, errors.New("watched reasoner did not receive a live run")
				}
				live.Begin(harness.LiveInfo{Harness: "my-agent", Steerable: tc.steerable})
				defer live.End()
				live.Emit(json.RawMessage(`{"step":"collect inputs"}`))
				ready <- live
				select {
				case note := <-live.Notes():
					return map[string]string{"note": note}, nil
				case <-finish:
					return map[string]bool{"done": true}, nil
				case <-ctx.Done():
					return nil, ctx.Err()
				}
			})
			req := httptest.NewRequest(http.MethodPost, "/execute/intake", strings.NewReader(`{"input":{}}`)).WithContext(ctx)
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("X-Execution-ID", exec)
			req.Header.Set("X-Run-ID", "watch-run")
			req.Header.Set(RunWatchedHeader, "1")
			rec := httptest.NewRecorder()
			done := make(chan struct{})
			go func() {
				defer close(done)
				a.ServeHTTP(rec, req)
			}()
			var run *LiveRun
			select {
			case run = <-ready:
			case <-ctx.Done():
				t.Fatal("the plain reasoner did not start")
			}
			require.NotNil(t, run)
			if tc.steerable {
				body := json.RawMessage(`{"kind":"steer","message":"include the total"}`)
				a.linkServe(linkFrame{T: "deliver", ID: "steer-1", Exec: exec, Body: body})
			} else {
				require.ErrorIs(t, a.Steer(exec, "include the total"), ErrNotSteerable)
				rec := steerRequest(a, exec, "include the total")
				require.Equal(t, http.StatusConflict, rec.Code)
				require.Contains(t, rec.Body.String(), "this execution is not running anything that takes notes right now, so it cannot be steered")
				close(finish)
			}

			select {
			case <-done:
			case <-ctx.Done():
				t.Fatal("the plain reasoner did not finish")
			}
			require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())
			require.ErrorIs(t, a.Steer(exec, "too late"), ErrNotSteerable)
			if tc.steerable {
				require.Contains(t, rec.Body.String(), "include the total")
				require.Equal(t, []string{RelayStarted, RelayLine, RelayPerson, RelayEnded}, cp.kinds())
			} else {
				require.Contains(t, rec.Body.String(), `"done":true`)
				require.Equal(t, []string{RelayStarted, RelayLine, RelayEnded}, cp.kinds())
			}
			cp.mu.Lock()
			defer cp.mu.Unlock()
			var info harness.LiveInfo
			require.NoError(t, json.Unmarshal(cp.entries[0].Data, &info))
			require.Equal(t, harness.LiveInfo{Harness: "my-agent", Steerable: tc.steerable}, info)
			require.Equal(t, json.RawMessage(`{"step":"collect inputs"}`), cp.entries[1].Data)
		})
	}
}
