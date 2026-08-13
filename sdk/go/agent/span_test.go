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
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newSpanEventHarness(t *testing.T) (*Agent, chan types.WorkflowExecutionEvent, func()) {
	t.Helper()

	eventCh := make(chan types.WorkflowExecutionEvent, 32)
	eventServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		if !strings.Contains(r.URL.Path, "/workflow/executions/events") {
			w.WriteHeader(http.StatusOK)
			return
		}
		body, _ := io.ReadAll(r.Body)
		var event types.WorkflowExecutionEvent
		if err := json.Unmarshal(body, &event); err == nil {
			eventCh <- event
		}
		w.WriteHeader(http.StatusOK)
	}))

	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: eventServer.URL,
		Logger:        log.New(io.Discard, "", 0),
	}

	ag, err := New(cfg)
	require.NoError(t, err)

	return ag, eventCh, eventServer.Close
}

func spanParentContext() context.Context {
	return contextWithExecution(context.Background(), ExecutionContext{
		RunID:          "run-1",
		ExecutionID:    "exec-parent",
		WorkflowID:     "wf-1",
		RootWorkflowID: "wf-1",
		ReasonerName:   "parent",
		AgentNodeID:    "node-1",
	})
}

func collectSpanEvents(t *testing.T, eventCh chan types.WorkflowExecutionEvent, n int) []types.WorkflowExecutionEvent {
	t.Helper()
	var received []types.WorkflowExecutionEvent
	timeout := time.After(2 * time.Second)
	for len(received) < n {
		select {
		case evt := <-eventCh:
			received = append(received, evt)
		case <-timeout:
			t.Fatalf("timed out waiting for span events, received %d of %d", len(received), n)
		}
	}
	return received
}

func TestSpanEmitsEventsWithLineage(t *testing.T) {
	ag, eventCh, closeServer := newSpanEventHarness(t)
	defer closeServer()

	res, err := ag.Span(spanParentContext(), "understand_query", map[string]any{"query": "q"}, func(ctx context.Context) (any, error) {
		time.Sleep(2 * time.Millisecond)
		return map[string]any{"ok": true}, nil
	})
	require.NoError(t, err)
	require.NotNil(t, res)

	received := collectSpanEvents(t, eventCh, 2)

	statuses := map[string]types.WorkflowExecutionEvent{}
	for _, evt := range received {
		assert.Equal(t, "understand_query", evt.ReasonerID)
		assert.Equal(t, "node-1", evt.AgentNodeID)
		assert.Equal(t, "run-1", evt.RunID)
		assert.Equal(t, "wf-1", evt.WorkflowID)
		require.NotNil(t, evt.ParentExecutionID)
		assert.Equal(t, "exec-parent", *evt.ParentExecutionID)
		statuses[evt.Status] = evt
	}

	require.Contains(t, statuses, "running")
	require.Contains(t, statuses, "succeeded")
	assert.NotNil(t, statuses["succeeded"].DurationMS)
	assert.Equal(t, statuses["running"].ExecutionID, statuses["succeeded"].ExecutionID)
}

func TestSpanNestedLineage(t *testing.T) {
	ag, eventCh, closeServer := newSpanEventHarness(t)
	defer closeServer()

	_, err := ag.Span(spanParentContext(), "outer", nil, func(ctx context.Context) (any, error) {
		return ag.Span(ctx, "inner", nil, func(ctx context.Context) (any, error) {
			return "leaf", nil
		})
	})
	require.NoError(t, err)

	received := collectSpanEvents(t, eventCh, 4)

	var outerID string
	for _, evt := range received {
		if evt.ReasonerID == "outer" {
			outerID = evt.ExecutionID
		}
	}
	require.NotEmpty(t, outerID)

	for _, evt := range received {
		require.NotNil(t, evt.ParentExecutionID, "event %s/%s missing parent", evt.ReasonerID, evt.Status)
		switch evt.ReasonerID {
		case "outer":
			assert.Equal(t, "exec-parent", *evt.ParentExecutionID)
		case "inner":
			assert.Equal(t, outerID, *evt.ParentExecutionID)
		}
		assert.Equal(t, "run-1", evt.RunID)
	}
}

func TestSpanParallelSiblingsShareParent(t *testing.T) {
	ag, eventCh, closeServer := newSpanEventHarness(t)
	defer closeServer()

	parentCtx := spanParentContext()
	var wg sync.WaitGroup
	for _, name := range []string{"sibling_a", "sibling_b"} {
		wg.Add(1)
		go func(name string) {
			defer wg.Done()
			_, err := ag.Span(parentCtx, name, nil, func(ctx context.Context) (any, error) {
				return name, nil
			})
			assert.NoError(t, err)
		}(name)
	}
	wg.Wait()

	received := collectSpanEvents(t, eventCh, 4)

	execIDs := map[string]string{}
	for _, evt := range received {
		require.NotNil(t, evt.ParentExecutionID)
		assert.Equal(t, "exec-parent", *evt.ParentExecutionID)
		execIDs[evt.ReasonerID] = evt.ExecutionID
	}
	assert.Len(t, execIDs, 2)
	assert.NotEqual(t, execIDs["sibling_a"], execIDs["sibling_b"])
}

func TestSpanErrorEmitsFailed(t *testing.T) {
	ag, eventCh, closeServer := newSpanEventHarness(t)
	defer closeServer()

	sentinel := errors.New("stage exploded")
	_, err := ag.Span(spanParentContext(), "failing_stage", nil, func(ctx context.Context) (any, error) {
		return nil, sentinel
	})
	require.ErrorIs(t, err, sentinel)

	received := collectSpanEvents(t, eventCh, 2)
	var sawFailed bool
	for _, evt := range received {
		if evt.Status == "failed" {
			sawFailed = true
			assert.Contains(t, evt.Error, "stage exploded")
		}
	}
	assert.True(t, sawFailed)
}

func TestSpanPanicEmitsFailedAndRepanics(t *testing.T) {
	ag, eventCh, closeServer := newSpanEventHarness(t)
	defer closeServer()

	require.Panics(t, func() {
		_, _ = ag.Span(spanParentContext(), "panicking_stage", nil, func(ctx context.Context) (any, error) {
			panic("boom")
		})
	})

	received := collectSpanEvents(t, eventCh, 2)
	var sawFailed bool
	for _, evt := range received {
		if evt.Status == "failed" {
			sawFailed = true
			assert.Contains(t, evt.Error, "boom")
		}
	}
	assert.True(t, sawFailed)
}

func TestSpanTruncatesLargePayloads(t *testing.T) {
	ag, eventCh, closeServer := newSpanEventHarness(t)
	defer closeServer()

	large := strings.Repeat("x", spanPayloadMaxBytes*4)
	_, err := ag.Span(spanParentContext(), "big_result", nil, func(ctx context.Context) (any, error) {
		return map[string]any{"blob": large}, nil
	})
	require.NoError(t, err)

	received := collectSpanEvents(t, eventCh, 2)
	for _, evt := range received {
		if evt.Status != "succeeded" {
			continue
		}
		resultMap, ok := evt.Result.(map[string]any)
		require.True(t, ok)
		assert.Equal(t, true, resultMap["_truncated"])
		preview, ok := resultMap["preview"].(string)
		require.True(t, ok)
		assert.LessOrEqual(t, len(preview), spanPayloadPreviewBytes)
	}
}

func TestSpanNilAgentRunsUntraced(t *testing.T) {
	var ag *Agent
	res, err := ag.Span(context.Background(), "untraced", nil, func(ctx context.Context) (any, error) {
		return "ok", nil
	})
	require.NoError(t, err)
	assert.Equal(t, "ok", res)
}

func TestSpanWithoutControlPlaneStillRuns(t *testing.T) {
	cfg := Config{
		NodeID:  "node-1",
		Version: "1.0.0",
		Logger:  log.New(io.Discard, "", 0),
	}
	ag, err := New(cfg)
	require.NoError(t, err)

	res, err := ag.Span(context.Background(), "untraced", nil, func(ctx context.Context) (any, error) {
		return 42, nil
	})
	require.NoError(t, err)
	assert.Equal(t, 42, res)
}

func TestSpanWithoutParentMintsRootLineage(t *testing.T) {
	ag, eventCh, closeServer := newSpanEventHarness(t)
	defer closeServer()

	_, err := ag.Span(context.Background(), "rootless", nil, func(ctx context.Context) (any, error) {
		return nil, nil
	})
	require.NoError(t, err)

	received := collectSpanEvents(t, eventCh, 2)
	for _, evt := range received {
		assert.Nil(t, evt.ParentExecutionID)
		assert.NotEmpty(t, evt.RunID)
		assert.NotEmpty(t, evt.ExecutionID)
	}
}
